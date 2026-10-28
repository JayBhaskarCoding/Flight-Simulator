//! The async tile streamer: request de-duplication, bounded concurrent
//! fetches, cache-backed replay and broadcast events.
//!
//! Architecture — one worker task owns everything stateful:
//!
//! ```text
//!   TileStreamer (Clone handle)
//!      │  mpsc<Cmd>            requests / stats / shutdown
//!      │  broadcast<TileEvent>  resident-state transitions
//!      ▼
//!   worker task ── owns ── TileCache (SQLite)
//!      │        ├── JoinSet of fetches (bounded by a Semaphore)
//!      │        └── resident map (Arc<RwLock>, readable from handles)
//!      ▼
//!   TileSource (filesystem, ...)
//! ```
//!
//! Consistency rules:
//!
//! * Every request consults the cache first, so a warm database replays
//!   without touching the source — even right after a restart (the
//!   resident map is only an in-memory mirror for de-duplication).
//! * The resident map mirrors the cache: whenever an entry is evicted the
//!   worker drops the resident marker, so a later request refetches.
//! * Confirmed misses (`Ok(None)`) are negative-cached as `Unavailable`
//!   until evicted; transient `Err` fetches are *not* cached, so the next
//!   request retries immediately.
//! * Events fire only on resident-state *transitions* (nothing →
//!   available, nothing → unavailable). Plain cache hits answer through
//!   the request's own oneshot, keeping broadcasts quiet.
//! * A tile larger than the whole cache budget streams through to its
//!   waiters but is not retained (see [`crate::cache::PutOutcome`]).

use crate::cache::{CacheKey, CacheStats, TileCache};
use crate::source::{TileKind, TileSource};
use crate::WorldError;
use sim_geo::quadtree::TileId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::{broadcast, mpsc, oneshot, Semaphore};
use tokio::task::{JoinError, JoinSet};

/// Result of a completed [`TileStreamer::fetch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileFetch {
    /// Raw tile payload.
    pub bytes: Vec<u8>,
    /// `true` when served from the SQLite cache instead of the source.
    pub from_cache: bool,
}

/// Resident-state of one tile, as tracked by the streamer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentState {
    /// The tile is available (from source or cache).
    Available {
        /// Payload size in bytes.
        bytes: usize,
        /// Highest priority seen for this tile.
        priority: i32,
    },
    /// The source confirmed the tile does not exist.
    Unavailable,
}

/// Payload of a [`TileEvent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileEventPayload {
    /// The tile became available; carries the payload so late subscribers
    /// can use it without a second request.
    Available(Vec<u8>),
    /// The source confirmed the tile does not exist.
    Unavailable,
}

/// State-transition notification broadcast to subscribers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileEvent {
    /// Kind of the tile.
    pub kind: TileKind,
    /// Quadtree identity.
    pub id: TileId,
    /// What happened.
    pub payload: TileEventPayload,
}

/// Streamer tuning.
#[derive(Debug, Clone)]
pub struct StreamerConfig {
    /// Rolling cache byte budget (applied to the cache on spawn).
    pub max_cache_bytes: u64,
    /// Maximum concurrent source fetches (extra requests queue behind a
    /// tokio semaphore).
    pub max_inflight: usize,
    /// Capacity of the command channel.
    pub request_capacity: usize,
    /// Capacity of the event broadcast (oldest events drop on overflow).
    pub event_capacity: usize,
}

impl Default for StreamerConfig {
    fn default() -> Self {
        StreamerConfig {
            max_cache_bytes: 512 * 1024 * 1024,
            max_inflight: 8,
            request_capacity: 256,
            event_capacity: 256,
        }
    }
}

/// Combined statistics (cache counters + streamer bookkeeping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamerStats {
    /// SQLite cache counters.
    pub cache: CacheStats,
    /// Tiles currently marked available in the resident map.
    pub resident_available: usize,
    /// Tiles currently marked unavailable (negative entries).
    pub resident_unavailable: usize,
    /// Bytes of available tiles tracked by the resident map.
    pub resident_bytes: u64,
    /// Fetches spawned and not yet completed (queued + running).
    pub inflight: usize,
    /// Waiters attached to in-flight fetches (including fire-and-forget
    /// requests that only bumped priority).
    pub waiters: usize,
}

enum Cmd {
    Request {
        kind: TileKind,
        id: TileId,
        priority: i32,
        reply: Option<oneshot::Sender<Result<TileFetch, WorldError>>>,
    },
    Stats {
        reply: oneshot::Sender<StreamerStats>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

type ResidentMap = Arc<RwLock<HashMap<(TileKind, TileId), ResidentState>>>;
type FetchResult = (TileKind, TileId, crate::Result<Option<Vec<u8>>>);

/// Cloneable handle to a running streamer (see module docs).
#[derive(Clone)]
pub struct TileStreamer {
    cmd: mpsc::Sender<Cmd>,
    events: broadcast::Sender<TileEvent>,
    resident: ResidentMap,
}

impl TileStreamer {
    /// Spawn a worker task driving `cache` from `source`.
    ///
    /// Must be called from within a tokio runtime; the worker ends when
    /// every handle is dropped or [`Self::shutdown`] is called. The
    /// cache's byte budget is set from `config.max_cache_bytes`.
    pub fn spawn<S: TileSource>(source: S, mut cache: TileCache, config: StreamerConfig) -> Self {
        cache.set_max_bytes(config.max_cache_bytes.max(1));
        let (cmd_tx, cmd_rx) = mpsc::channel(config.request_capacity.max(1));
        let (ev_tx, _) = broadcast::channel(config.event_capacity.max(1));
        let resident: ResidentMap = Arc::new(RwLock::new(HashMap::new()));
        let worker = Worker {
            source: Arc::new(source),
            cache,
            rx: cmd_rx,
            events: ev_tx.clone(),
            resident: Arc::clone(&resident),
            join: JoinSet::new(),
            task_ids: HashMap::new(),
            inflight: HashMap::new(),
            priorities: HashMap::new(),
            sem: Arc::new(Semaphore::new(config.max_inflight.max(1))),
        };
        tokio::spawn(worker.run());
        TileStreamer {
            cmd: cmd_tx,
            events: ev_tx,
            resident,
        }
    }

    /// Fire-and-forget request: subscribers learn the result through
    /// [`TileEvent`]; use [`Self::fetch`] when you want the bytes back
    /// directly.
    ///
    /// Priority orders eviction: higher values are retained longer.
    pub async fn request(&self, kind: TileKind, id: TileId, priority: i32) -> crate::Result<()> {
        self.cmd
            .send(Cmd::Request {
                kind,
                id,
                priority,
                reply: None,
            })
            .await
            .map_err(|_| WorldError::Closed)
    }

    /// Request a tile and wait for its payload (cache or source).
    pub async fn fetch(
        &self,
        kind: TileKind,
        id: TileId,
        priority: i32,
    ) -> Result<TileFetch, WorldError> {
        let (tx, rx) = oneshot::channel();
        self.cmd
            .send(Cmd::Request {
                kind,
                id,
                priority,
                reply: Some(tx),
            })
            .await
            .map_err(|_| WorldError::Closed)?;
        rx.await.map_err(|_| WorldError::Closed)?
    }

    /// Subscribe to resident-state transitions.
    pub fn subscribe(&self) -> broadcast::Receiver<TileEvent> {
        self.events.subscribe()
    }

    /// Current resident marker for one tile, if any.
    pub fn resident_state(&self, kind: TileKind, id: TileId) -> Option<ResidentState> {
        self.resident.read().ok()?.get(&(kind, id)).copied()
    }

    /// Snapshot of cache and streamer counters.
    pub async fn stats(&self) -> crate::Result<StreamerStats> {
        let (tx, rx) = oneshot::channel();
        self.cmd
            .send(Cmd::Stats { reply: tx })
            .await
            .map_err(|_| WorldError::Closed)?;
        rx.await.map_err(|_| WorldError::Closed)
    }

    /// Ask the worker to exit and wait for the acknowledgement. Later
    /// requests fail with [`WorldError::Closed`].
    pub async fn shutdown(&self) -> crate::Result<()> {
        let (tx, rx) = oneshot::channel();
        self.cmd
            .send(Cmd::Shutdown { reply: tx })
            .await
            .map_err(|_| WorldError::Closed)?;
        rx.await.map_err(|_| WorldError::Closed)
    }
}

/// Clone an error for the remaining waiters of one failed fetch (the
/// first waiter consumes the original).
fn replicate_error(e: &WorldError) -> WorldError {
    match e {
        WorldError::Io(io) => WorldError::Io(std::io::Error::new(io.kind(), io.to_string())),
        WorldError::Dem(m) => WorldError::Dem(m.clone()),
        WorldError::Osm(o) => WorldError::Osm(o.clone()),
        WorldError::TileUnavailable { kind, id } => WorldError::TileUnavailable {
            kind: *kind,
            id: *id,
        },
        WorldError::FetchFailed(m) => WorldError::FetchFailed(m.clone()),
        WorldError::Closed => WorldError::Closed,
        WorldError::WrongTileFace {
            tile_face,
            dir_face,
        } => WorldError::WrongTileFace {
            tile_face: *tile_face,
            dir_face: *dir_face,
        },
        WorldError::Sql(sq) => WorldError::FetchFailed(format!("sqlite during fetch: {sq}")),
    }
}

type Waiter = oneshot::Sender<Result<TileFetch, WorldError>>;

struct Worker<S: TileSource> {
    source: Arc<S>,
    cache: TileCache,
    rx: mpsc::Receiver<Cmd>,
    events: broadcast::Sender<TileEvent>,
    resident: ResidentMap,
    join: JoinSet<FetchResult>,
    task_ids: HashMap<tokio::task::Id, (TileKind, TileId)>,
    inflight: HashMap<(TileKind, TileId), Vec<Waiter>>,
    priorities: HashMap<(TileKind, TileId), i32>,
    sem: Arc<Semaphore>,
}

impl<S: TileSource> Worker<S> {
    /// Worker entry point: serve commands and fetch completions until the
    /// command channel closes or a shutdown arrives.
    async fn run(mut self) {
        loop {
            tokio::select! {
                biased;
                cmd = self.rx.recv() => {
                    match cmd {
                        Some(c) => {
                            if !self.handle_cmd(c) {
                                break;
                            }
                        }
                        None => break,
                    }
                }
                // Refutable pattern: while no fetch is in flight this
                // branch is disabled instead of spinning on `None`.
                Some(res) = self.join.join_next() => {
                    self.handle_join(res);
                }
            }
        }
        // Dropping `self` drops the JoinSet, aborting still-running fetches.
    }

    /// Handle one command. Returns `false` to stop the worker.
    fn handle_cmd(&mut self, cmd: Cmd) -> bool {
        match cmd {
            Cmd::Request {
                kind,
                id,
                priority,
                reply,
            } => {
                self.handle_request(kind, id, priority, reply);
                true
            }
            Cmd::Stats { reply } => {
                let _ = reply.send(self.collect_stats());
                true
            }
            Cmd::Shutdown { reply } => {
                let _ = reply.send(());
                false
            }
        }
    }

    fn collect_stats(&self) -> StreamerStats {
        let (resident_available, resident_unavailable, resident_bytes) = match self.resident.read() {
            Ok(g) => {
                let mut a = 0usize;
                let mut u = 0usize;
                let mut b = 0u64;
                for state in g.values() {
                    match state {
                        ResidentState::Available { bytes, .. } => {
                            a += 1;
                            b += *bytes as u64;
                        }
                        ResidentState::Unavailable => u += 1,
                    }
                }
                (a, u, b)
            }
            // Defensive: this worker never panics while holding the lock.
            Err(_) => (0, 0, 0),
        };
        StreamerStats {
            cache: self.cache.stats(),
            resident_available,
            resident_unavailable,
            resident_bytes,
            inflight: self.join.len(),
            waiters: self.inflight.values().map(Vec::len).sum(),
        }
    }

    fn handle_request(
        &mut self,
        kind: TileKind,
        id: TileId,
        priority: i32,
        reply: Option<Waiter>,
    ) {
        let key = (kind, id);
        let state = self
            .resident
            .read()
            .map(|m| m.get(&key).copied())
            .unwrap_or(None);

        if let Some(ResidentState::Unavailable) = state {
            // Confirmed miss: answer locally without touching anything.
            if let Some(r) = reply {
                let _ = r.send(Err(WorldError::TileUnavailable { kind, id }));
            }
            return;
        }

        let was_resident = matches!(state, Some(ResidentState::Available { .. }));

        // Cache first — warm databases replay without a source round-trip.
        match self.cache.get(CacheKey::new(kind, id), priority) {
            Ok(Some(bytes)) => {
                let len = bytes.len();
                if let Ok(mut m) = self.resident.write() {
                    m.insert(
                        key,
                        ResidentState::Available {
                            bytes: len,
                            priority,
                        },
                    );
                }
                if !was_resident {
                    // Cold-start transition: let subscribers know.
                    let _ = self.events.send(TileEvent {
                        kind,
                        id,
                        payload: TileEventPayload::Available(bytes.clone()),
                    });
                }
                if let Some(r) = reply {
                    let _ = r.send(Ok(TileFetch {
                        bytes,
                        from_cache: true,
                    }));
                }
                return;
            }
            Ok(None) => {
                // Miss (or stale resident marker): drop the marker if the
                // cache lost the payload, then fetch from the source.
                if was_resident {
                    if let Ok(mut m) = self.resident.write() {
                        m.remove(&key);
                    }
                }
            }
            Err(e) => {
                if let Some(r) = reply {
                    let _ = r.send(Err(e));
                }
                return;
            }
        }

        // Already fetching: attach as a waiter and bump the priority.
        if self.inflight.contains_key(&key) {
            let slot = self.priorities.entry(key).or_insert(priority);
            *slot = (*slot).max(priority);
            if let Some(r) = reply {
                self.inflight.entry(key).or_default().push(r);
            }
            return;
        }

        // Fresh fetch from the source.
        self.inflight.insert(key, reply.into_iter().collect());
        self.priorities.insert(key, priority);
        self.spawn_fetch(kind, id);
    }

    fn spawn_fetch(&mut self, kind: TileKind, id: TileId) {
        let source = Arc::clone(&self.source);
        let sem = Arc::clone(&self.sem);
        let task = async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (kind, id, Err(WorldError::Closed)),
            };
            let result = source.fetch(kind, id).await;
            (kind, id, result)
        };
        let handle = self.join.spawn(task);
        self.task_ids.insert(handle.id(), (kind, id));
    }

    fn handle_join(&mut self, res: Result<FetchResult, JoinError>) {
        match res {
            Ok((kind, id, result)) => self.complete_fetch(kind, id, result),
            Err(join_err) => {
                // Task panicked or was cancelled: attribute it to its key
                // and fail the waiters without negative-caching, so the
                // next request retries.
                if let Some(key) = self.task_ids.remove(&join_err.id()) {
                    self.priorities.remove(&key);
                    let waiters = self.inflight.remove(&key).unwrap_or_default();
                    Self::reply_err(
                        waiters,
                        WorldError::FetchFailed(format!("fetch task failed: {join_err}")),
                    );
                }
            }
        }
    }

    fn complete_fetch(
        &mut self,
        kind: TileKind,
        id: TileId,
        result: crate::Result<Option<Vec<u8>>>,
    ) {
        let key = (kind, id);
        self.task_ids.retain(|_, k| *k != key);
        let waiters: Vec<Waiter> = self.inflight.remove(&key).unwrap_or_default();
        let priority = self.priorities.remove(&key).unwrap_or(0);

        match result {
            Ok(Some(bytes)) => match self.cache.put(CacheKey::new(kind, id), &bytes, priority) {
                Ok(outcome) => {
                    // Evicted keys (possibly this one, if oversized) must
                    // not remain marked resident.
                    if let Ok(mut m) = self.resident.write() {
                        for ek in &outcome.evicted {
                            m.remove(&(ek.kind, ek.id));
                        }
                    }
                    if outcome.stored {
                        if let Ok(mut m) = self.resident.write() {
                            m.insert(
                                key,
                                ResidentState::Available {
                                    bytes: bytes.len(),
                                    priority,
                                },
                            );
                        }
                    }
                    Self::reply_ok(
                        waiters,
                        TileFetch {
                            bytes: bytes.clone(),
                            from_cache: false,
                        },
                    );
                    if outcome.stored {
                        let _ = self.events.send(TileEvent {
                            kind,
                            id,
                            payload: TileEventPayload::Available(bytes),
                        });
                    }
                }
                Err(e) => {
                    // Cache write failed: the payload still reaches every
                    // waiter; nothing is marked resident, so the next
                    // request retries (and may retry the write).
                    Self::reply_ok(
                        waiters,
                        TileFetch {
                            bytes,
                            from_cache: false,
                        },
                    );
                    let _ = e; // counters live in the cache; the error
                               // surfaces on the next cache-backed call.
                }
            },
            Ok(None) => {
                if let Ok(mut m) = self.resident.write() {
                    m.insert(key, ResidentState::Unavailable);
                }
                Self::reply_err(waiters, WorldError::TileUnavailable { kind, id });
                let _ = self.events.send(TileEvent {
                    kind,
                    id,
                    payload: TileEventPayload::Unavailable,
                });
            }
            Err(e) => {
                // Transient failure: no negative entry — retry on the next
                // request. The original error goes to the first waiter,
                // clones to the rest.
                Self::reply_err(waiters, e);
            }
        }
    }

    fn reply_ok(waiters: Vec<Waiter>, fetch: TileFetch) {
        for w in waiters {
            let _ = w.send(Ok(fetch.clone()));
        }
    }

    fn reply_err(waiters: Vec<Waiter>, err: WorldError) {
        let mut it = waiters.into_iter();
        match (it.next(), it.next()) {
            (Some(first), None) => {
                // Common case: the only waiter consumes the original error.
                let _ = first.send(Err(err));
            }
            (Some(first), Some(second)) => {
                let _ = first.send(Err(replicate_error(&err)));
                let _ = second.send(Err(replicate_error(&err)));
                for w in it {
                    let _ = w.send(Err(replicate_error(&err)));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// In-memory source: a table of prepared tiles + call/fail controls
    /// that stay reachable after the source moves into the worker.
    struct TableSource {
        table: HashMap<(TileKind, TileId), Vec<u8>>,
        calls: Arc<AtomicUsize>,
        fail: Arc<AtomicBool>,
    }

    impl TableSource {
        fn new(entries: &[((TileKind, TileId), Vec<u8>)]) -> Self {
            TableSource {
                table: entries.iter().map(|(k, v)| (*k, v.clone())).collect(),
                calls: Arc::new(AtomicUsize::new(0)),
                fail: Arc::new(AtomicBool::new(false)),
            }
        }

        fn calls_handle(&self) -> Arc<AtomicUsize> {
            Arc::clone(&self.calls)
        }

        fn fail_handle(&self) -> Arc<AtomicBool> {
            Arc::clone(&self.fail)
        }
    }

    impl TileSource for TableSource {
        fn fetch(&self, kind: TileKind, id: TileId) -> crate::source::FetchFuture<'_> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if self.fail.load(Ordering::SeqCst) {
                    return Err(WorldError::Io(std::io::Error::other(
                        "simulated outage",
                    )));
                }
                Ok(self.table.get(&(kind, id)).cloned())
            })
        }
    }

    fn tile(x: u32) -> TileId {
        TileId {
            face: 1,
            level: 4,
            x,
            y: 0,
        }
    }

    fn start(source: TableSource, max_bytes: u64, max_inflight: usize) -> TileStreamer {
        let cache = TileCache::open_in_memory(max_bytes).unwrap();
        let cfg = StreamerConfig {
            max_cache_bytes: max_bytes,
            max_inflight,
            request_capacity: 16,
            event_capacity: 16,
        };
        TileStreamer::spawn(source, cache, cfg)
    }

    #[tokio::test]
    async fn fetch_serves_from_source_then_cache() {
        let src = TableSource::new(&[((TileKind::Dem, tile(1)), vec![9u8; 64])]);
        let calls = src.calls_handle();
        let s = start(src, 1 << 20, 4);

        let first = s.fetch(TileKind::Dem, tile(1), 5).await.unwrap();
        assert_eq!(first.bytes.len(), 64);
        assert!(!first.from_cache);
        assert!(matches!(
            s.resident_state(TileKind::Dem, tile(1)),
            Some(ResidentState::Available { bytes: 64, priority: 5 })
        ));

        let second = s.fetch(TileKind::Dem, tile(1), 5).await.unwrap();
        assert!(second.from_cache);
        assert_eq!(second.bytes, first.bytes);

        let stats = s.stats().await.unwrap();
        assert_eq!(stats.cache.hits, 1);
        assert_eq!(stats.cache.misses, 1);
        assert_eq!(stats.cache.entries, 1);
        assert_eq!(stats.resident_available, 1);
        assert_eq!(stats.inflight, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        s.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn concurrent_identical_requests_fetch_once() {
        let src = TableSource::new(&[((TileKind::Osm, tile(2)), b"shared".to_vec())]);
        let calls = src.calls_handle();
        let s = start(src, 1 << 20, 4);

        let mut set = tokio::task::JoinSet::new();
        for _ in 0..8 {
            let h = s.clone();
            set.spawn(async move { h.fetch(TileKind::Osm, tile(2), 1).await });
        }
        let mut n_ok = 0;
        while let Some(r) = set.join_next().await {
            let f = r.unwrap().unwrap();
            assert_eq!(f.bytes, b"shared");
            n_ok += 1;
        }
        assert_eq!(n_ok, 8);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "identical concurrent requests must dedupe to one source call"
        );

        let stats = s.stats().await.unwrap();
        assert_eq!(stats.cache.entries, 1);
        assert_eq!(stats.inflight, 0);
        s.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn confirmed_miss_is_negative_cached() {
        let src = TableSource::new(&[]);
        let calls = src.calls_handle();
        let s = start(src, 1 << 20, 4);
        let mut events = s.subscribe();

        let err = s.fetch(TileKind::Dem, tile(7), 0).await.unwrap_err();
        assert!(matches!(err, WorldError::TileUnavailable { .. }));
        assert!(matches!(
            s.resident_state(TileKind::Dem, tile(7)),
            Some(ResidentState::Unavailable)
        ));

        // The transition event arrived exactly once.
        let ev = events.recv().await.unwrap();
        assert_eq!(ev.id, tile(7));
        assert!(matches!(ev.payload, TileEventPayload::Unavailable));

        // Second request is answered from the negative entry: no new call.
        let err2 = s.fetch(TileKind::Dem, tile(7), 0).await.unwrap_err();
        assert!(matches!(err2, WorldError::TileUnavailable { .. }));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        s.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn eviction_drops_resident_marker_so_next_request_refetches() {
        let entries: Vec<_> = (0..4u32)
            .map(|i| ((TileKind::Dem, tile(i)), vec![i as u8; 48]))
            .collect();
        let src = TableSource::new(&entries);
        let calls = src.calls_handle();
        // Budget fits two 48-byte tiles.
        let s = start(src, 100, 4);

        s.fetch(TileKind::Dem, tile(0), 10).await.unwrap();
        s.fetch(TileKind::Dem, tile(1), 10).await.unwrap();
        s.fetch(TileKind::Dem, tile(2), 10).await.unwrap();

        let stats = s.stats().await.unwrap();
        assert_eq!(stats.cache.entries, 2);
        assert_eq!(stats.cache.evictions, 1);
        assert!(s.resident_state(TileKind::Dem, tile(0)).is_none());

        let again = s.fetch(TileKind::Dem, tile(0), 10).await.unwrap();
        assert!(!again.from_cache, "evicted tile must come from source");
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        s.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn broadcast_reports_available_transition() {
        let src = TableSource::new(&[((TileKind::Dem, tile(9)), vec![5u8; 8])]);
        let s = start(src, 1 << 20, 4);
        let mut rx = s.subscribe();

        s.request(TileKind::Dem, tile(9), 1).await.unwrap();
        let ev = rx.recv().await.unwrap();
        assert_eq!(ev.kind, TileKind::Dem);
        assert_eq!(ev.id, tile(9));
        assert!(matches!(ev.payload, TileEventPayload::Available(ref b) if b == &vec![5u8; 8]));

        // A cache hit on a resident tile produces no further event.
        let _ = s.fetch(TileKind::Dem, tile(9), 1).await.unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
                .await
                .is_err(),
            "cache hits must not broadcast"
        );
        s.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn transient_errors_retry_on_next_request() {
        let src = TableSource::new(&[((TileKind::Dem, tile(3)), vec![1, 2, 3])]);
        let calls = src.calls_handle();
        let fail = src.fail_handle();
        let s = start(src, 1 << 20, 4);

        fail.store(true, Ordering::SeqCst);
        let err = s.fetch(TileKind::Dem, tile(3), 0).await.unwrap_err();
        assert!(
            matches!(err, WorldError::Io(_)),
            "first waiter receives the original error, got: {err}"
        );
        // No negative entry: resident is empty for this key.
        assert!(s.resident_state(TileKind::Dem, tile(3)).is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Recover: the very next request reaches the source again.
        fail.store(false, Ordering::SeqCst);
        let ok = s.fetch(TileKind::Dem, tile(3), 0).await.unwrap();
        assert_eq!(ok.bytes, vec![1, 2, 3]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        s.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_then_requests_fail_closed() {
        let src = TableSource::new(&[]);
        let s = start(src, 1 << 20, 4);
        s.shutdown().await.unwrap();
        // The ack is sent just before the worker exits; wait for the exit
        // so the command channel is definitively closed.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let err = s.fetch(TileKind::Dem, tile(0), 0).await.unwrap_err();
        assert!(matches!(err, WorldError::Closed));
        assert!(matches!(
            s.request(TileKind::Dem, tile(0), 0).await,
            Err(WorldError::Closed)
        ));
    }

    #[tokio::test]
    async fn priority_orders_eviction_in_the_cache() {
        let entries: Vec<_> = (0..3u32)
            .map(|i| ((TileKind::Osm, tile(i)), vec![7u8; 40]))
            .collect();
        let src = TableSource::new(&entries);
        let s = start(src, 80, 4); // fits two 40-byte tiles

        s.fetch(TileKind::Osm, tile(0), 1).await.unwrap();
        s.fetch(TileKind::Osm, tile(1), 50).await.unwrap();
        s.fetch(TileKind::Osm, tile(2), 1).await.unwrap();

        assert!(matches!(
            s.resident_state(TileKind::Osm, tile(1)),
            Some(ResidentState::Available { .. })
        ));
        let stats = s.stats().await.unwrap();
        assert_eq!(stats.cache.entries, 2);
        s.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn cold_streamer_replays_warm_cache_without_source() {
        let dir = std::env::temp_dir().join(format!(
            "sim-world-streamer-warm-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("warm.db");

        let k = (TileKind::Osm, tile(5));
        // First worker: load from source into the persistent cache.
        {
            let src = TableSource::new(&[(k, b"warm-tile".to_vec())]);
            let calls = src.calls_handle();
            let cache = TileCache::open(&path, 1 << 20).unwrap();
            let s = TileStreamer::spawn(src, cache, StreamerConfig::default());
            let f = s.fetch(TileKind::Osm, tile(5), 3).await.unwrap();
            assert_eq!(f.bytes, b"warm-tile");
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            s.shutdown().await.unwrap();
        }

        // Second worker, same database, source WITHOUT the tile: replay.
        {
            let src = TableSource::new(&[]);
            let calls = src.calls_handle();
            let cache = TileCache::open(&path, 1 << 20).unwrap();
            let s = TileStreamer::spawn(src, cache, StreamerConfig::default());
            let f = s.fetch(TileKind::Osm, tile(5), 3).await.unwrap();
            assert!(f.from_cache, "warm cache must replay without source");
            assert_eq!(f.bytes, b"warm-tile");
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            s.shutdown().await.unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
