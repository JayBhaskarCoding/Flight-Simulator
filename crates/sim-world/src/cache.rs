//! SQLite rolling cache with priority-aware LRU eviction.
//!
//! The cache is deliberately synchronous (`&mut self` methods): it is owned
//! by the single streamer worker task, which serialises access without any
//! locking, and it stays trivially unit-testable without a tokio runtime.
//!
//! Eviction keeps the working set under a byte budget. Victims are chosen
//! by `(priority ASC, last_used ASC, key ASC)` — lowest priority first,
//! then least recently used, with the tile key as a deterministic tie
//! break. `last_used` is a logical clock (an ever-increasing counter),
//! not wall time, so behaviour is reproducible under test.
//!
//! Durability: files open in WAL mode with `synchronous=NORMAL`, which
//! survives process crashes at the SQLite level while keeping tile writes
//! cheap. The cache is a *rolling* cache: losing the newest writes only
//! costs a re-fetch.

use crate::source::TileKind;
use rusqlite::{params, Connection};
use sim_geo::quadtree::TileId;
use std::path::Path;

/// Identity of one cached tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheKey {
    /// Data category.
    pub kind: TileKind,
    /// Quadtree tile identity.
    pub id: TileId,
}

impl CacheKey {
    /// Convenience constructor.
    pub fn new(kind: TileKind, id: TileId) -> Self {
        CacheKey { kind, id }
    }
}

/// Snapshot of cache counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of tiles currently stored.
    pub entries: u64,
    /// Bytes of payload currently stored.
    pub bytes: u64,
    /// Byte budget for eviction.
    pub max_bytes: u64,
    /// Successful [`TileCache::get`] calls.
    pub hits: u64,
    /// Failed [`TileCache::get`] calls.
    pub misses: u64,
    /// Tiles dropped to stay inside the budget.
    pub evictions: u64,
}

/// Outcome of [`TileCache::put`].
#[derive(Debug, Default)]
pub struct PutOutcome {
    /// `true` when the payload was stored (small enough for the budget).
    pub stored: bool,
    /// Keys evicted to make room (includes the new key when a single tile
    /// is larger than the whole budget and is dropped again immediately).
    pub evicted: Vec<CacheKey>,
}

/// SQLite-backed tile store.
pub struct TileCache {
    conn: Connection,
    bytes: u64,
    entries: u64,
    max_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    clock: i64,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS tiles (
    kind      INTEGER NOT NULL,
    face      INTEGER NOT NULL,
    level     INTEGER NOT NULL,
    x         INTEGER NOT NULL,
    y         INTEGER NOT NULL,
    size      INTEGER NOT NULL,
    priority  INTEGER NOT NULL,
    last_used INTEGER NOT NULL,
    data      BLOB    NOT NULL,
    PRIMARY KEY (kind, face, level, x, y)
);
CREATE INDEX IF NOT EXISTS tiles_evict ON tiles (priority, last_used);
";

fn key_columns(id: CacheKey) -> [i64; 5] {
    [
        id.kind.as_db(),
        i64::from(id.id.face),
        i64::from(id.id.level),
        i64::from(id.id.x),
        i64::from(id.id.y),
    ]
}

impl TileCache {
    /// Open (creating if needed) a cache database at `path`, keeping at
    /// most `max_bytes` of payload resident.
    pub fn open<P: AsRef<Path>>(path: P, max_bytes: u64) -> crate::Result<Self> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        Self::from_connection(conn, max_bytes)
    }

    /// In-memory cache (no durability). Used by tests and by callers that
    /// only want de-duplication, not persistence.
    pub fn open_in_memory(max_bytes: u64) -> crate::Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(conn, max_bytes)
    }

    fn from_connection(conn: Connection, max_bytes: u64) -> crate::Result<Self> {
        conn.execute_batch(SCHEMA)?;
        let (entries, bytes, clock): (i64, i64, i64) = conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(size), 0), COALESCE(MAX(last_used), 0) FROM tiles",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        Ok(TileCache {
            conn,
            bytes: bytes.max(0) as u64,
            entries: entries.max(0) as u64,
            max_bytes,
            hits: 0,
            misses: 0,
            evictions: 0,
            clock,
        })
    }

    /// Byte budget this cache enforces.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Replace the byte budget (eviction runs on the next [`Self::put`]).
    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        self.max_bytes = max_bytes;
    }

    /// Current byte total (payload only).
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Current entry count.
    pub fn entries(&self) -> u64 {
        self.entries
    }

    fn tick(&mut self) -> i64 {
        self.clock += 1;
        self.clock
    }

    /// Fetch a tile's payload.
    ///
    /// On a hit the logical clock advances and the stored priority is
    /// raised to `priority` (so a re-request at higher priority shields
    /// the tile from eviction). `priority` never lowers a stored value.
    pub fn get(&mut self, key: CacheKey, priority: i32) -> crate::Result<Option<Vec<u8>>> {
        let k = key_columns(key);
        let row = self.conn.query_row(
            "SELECT data, priority FROM tiles \
             WHERE kind = ?1 AND face = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
            params![k[0], k[1], k[2], k[3], k[4]],
            |r| {
                let data: Vec<u8> = r.get(0)?;
                let prio: i64 = r.get(1)?;
                Ok((data, prio))
            },
        );
        match row {
            Ok((data, prio)) => {
                self.hits += 1;
                let t = self.tick();
                let new_prio = prio.max(i64::from(priority));
                self.conn.execute(
                    "UPDATE tiles SET last_used = ?1, priority = ?2 \
                     WHERE kind = ?3 AND face = ?4 AND level = ?5 AND x = ?6 AND y = ?7",
                    params![t, new_prio, k[0], k[1], k[2], k[3], k[4]],
                )?;
                Ok(Some(data))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                self.misses += 1;
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// True when `key` is present (does not touch counters or the clock).
    pub fn contains(&mut self, key: CacheKey) -> crate::Result<bool> {
        let k = key_columns(key);
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM tiles \
             WHERE kind = ?1 AND face = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
            params![k[0], k[1], k[2], k[3], k[4]],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Store a payload at `priority`, then evict until the budget holds.
    ///
    /// A payload larger than the entire budget is not retained: it is
    /// reported as evicted so the caller can drop its resident bookkeeping
    /// too (the data still reaches every waiter of the current request).
    pub fn put(&mut self, key: CacheKey, data: &[u8], priority: i32) -> crate::Result<PutOutcome> {
        let size = i64::try_from(data.len()).unwrap_or(i64::MAX);
        let k = key_columns(key);

        // Subtract the previous payload, if any.
        let prev: Option<i64> = match self.conn.query_row(
            "SELECT size FROM tiles \
             WHERE kind = ?1 AND face = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
            params![k[0], k[1], k[2], k[3], k[4]],
            |r| r.get(0),
        ) {
            Ok(v) => Some(v),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(e.into()),
        };
        let t = self.tick();
        self.conn.execute(
            "INSERT OR REPLACE INTO tiles (kind, face, level, x, y, size, priority, last_used, data) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![k[0], k[1], k[2], k[3], k[4], size, i64::from(priority), t, data],
        )?;
        self.bytes = self
            .bytes
            .saturating_sub(prev.unwrap_or(0).max(0) as u64)
            .saturating_add(size.max(0) as u64);
        self.entries = match prev {
            Some(_) => self.entries,
            None => self.entries + 1,
        };

        let mut evicted = Vec::new();
        while self.bytes > self.max_bytes {
            let victim = self.conn.query_row(
                "SELECT kind, face, level, x, y, size FROM tiles \
                 ORDER BY priority ASC, last_used ASC, kind ASC, face ASC, level ASC, x ASC, y ASC \
                 LIMIT 1",
                [],
                |r| {
                    let kind = TileKind::from_db(r.get(0)?).ok_or_else(|| {
                        rusqlite::Error::InvalidColumnType(
                            0,
                            "kind".to_string(),
                            rusqlite::types::Type::Integer,
                        )
                    })?;
                    let face: i64 = r.get(1)?;
                    let level: i64 = r.get(2)?;
                    let x: i64 = r.get(3)?;
                    let y: i64 = r.get(4)?;
                    let size: i64 = r.get(5)?;
                    Ok((
                        CacheKey {
                            kind,
                            id: TileId {
                                face: face.clamp(0, i64::from(u8::MAX)) as u8,
                                level: level.clamp(0, i64::from(u8::MAX)) as u8,
                                x: x.max(0) as u32,
                                y: y.max(0) as u32,
                            },
                        },
                        size.max(0) as u64,
                    ))
                },
            );
            let (victim, victim_bytes) = match victim {
                Ok(v) => v,
                Err(rusqlite::Error::QueryReturnedNoRows) => break,
                Err(e) => return Err(e.into()),
            };
            let vk = key_columns(victim);
            self.conn.execute(
                "DELETE FROM tiles WHERE kind = ?1 AND face = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
                params![vk[0], vk[1], vk[2], vk[3], vk[4]],
            )?;
            self.bytes = self.bytes.saturating_sub(victim_bytes);
            self.entries = self.entries.saturating_sub(1);
            self.evictions += 1;
            evicted.push(victim);
        }

        // Oversized single tile: it was evicted immediately (possibly by
        // evicting itself). Report accurately.
        let stored = self.contains_key_raw(k)?;
        Ok(PutOutcome { stored, evicted })
    }

    fn contains_key_raw(&mut self, k: [i64; 5]) -> crate::Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM tiles \
             WHERE kind = ?1 AND face = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
            params![k[0], k[1], k[2], k[3], k[4]],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Remove one tile (e.g. explicit invalidation). Returns whether it existed.
    pub fn remove(&mut self, key: CacheKey) -> crate::Result<bool> {
        let k = key_columns(key);
        let size: Option<i64> = match self.conn.query_row(
            "SELECT size FROM tiles \
             WHERE kind = ?1 AND face = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
            params![k[0], k[1], k[2], k[3], k[4]],
            |r| r.get(0),
        ) {
            Ok(v) => Some(v),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(e.into()),
        };
        let Some(size) = size else {
            return Ok(false);
        };
        self.conn.execute(
            "DELETE FROM tiles WHERE kind = ?1 AND face = ?2 AND level = ?3 AND x = ?4 AND y = ?5",
            params![k[0], k[1], k[2], k[3], k[4]],
        )?;
        self.bytes = self.bytes.saturating_sub(size.max(0) as u64);
        self.entries = self.entries.saturating_sub(1);
        Ok(true)
    }

    /// Drop every tile (keeps counters for hits/misses/evictions).
    pub fn clear(&mut self) -> crate::Result<()> {
        self.conn.execute("DELETE FROM tiles", [])?;
        self.bytes = 0;
        self.entries = 0;
        Ok(())
    }

    /// Snapshot of the counters.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries,
            bytes: self.bytes,
            max_bytes: self.max_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(kind: TileKind, face: u8, level: u8, x: u32, y: u32) -> CacheKey {
        CacheKey::new(
            kind,
            TileId {
                face,
                level,
                x,
                y,
            },
        )
    }

    #[test]
    fn open_schemata_are_idempotent_and_stats_start_zero() {
        let cache = TileCache::open_in_memory(1024).unwrap();
        let s = cache.stats();
        assert_eq!(s.entries, 0);
        assert_eq!(s.bytes, 0);
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.max_bytes, 1024);

        // Re-opening the same file works (CREATE ... IF NOT EXISTS).
        let dir = std::env::temp_dir().join(format!(
            "sim-world-cache-schema-{}-{}",
            std::process::id(),
            "a"
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let p = dir.join("cache.db");
        drop(TileCache::open(&p, 512).unwrap());
        drop(TileCache::open(&p, 512).unwrap());
        let reopened = TileCache::open(&p, 512).unwrap();
        assert_eq!(reopened.stats().entries, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn put_get_roundtrip_updates_hits_misses_and_clock() {
        let mut cache = TileCache::open_in_memory(1 << 20).unwrap();
        let k = key(TileKind::Dem, 2, 5, 11, 13);

        assert!(cache.get(k, 0).unwrap().is_none());
        assert_eq!(cache.stats().misses, 1);

        let out = cache.put(k, b"elevation-payload", 3).unwrap();
        assert!(out.stored);
        assert!(out.evicted.is_empty());

        let got = cache.get(k, 3).unwrap().unwrap();
        assert_eq!(got, b"elevation-payload");
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.entries, 1);
        assert_eq!(s.bytes, b"elevation-payload".len() as u64);
        assert!(cache.contains(k).unwrap());
    }

    #[test]
    fn eviction_prefers_lowest_priority_then_oldest() {
        // Budget fits exactly two 10-byte tiles.
        let mut cache = TileCache::open_in_memory(20).unwrap();
        let a = key(TileKind::Dem, 0, 1, 0, 0);
        let b = key(TileKind::Dem, 0, 1, 1, 0);
        let c = key(TileKind::Dem, 0, 1, 0, 1);

        cache.put(a, &[1u8; 10], 5).unwrap();
        cache.put(b, &[2u8; 10], 1).unwrap();
        // Third tile pushes over budget; b (priority 1) must go.
        let out = cache.put(c, &[3u8; 10], 5).unwrap();
        assert!(out.stored);
        assert_eq!(out.evicted, vec![b]);
        assert!(!cache.contains(b).unwrap());
        assert!(cache.contains(a).unwrap());
        assert!(cache.contains(c).unwrap());
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().bytes, 20);
    }

    #[test]
    fn eviction_breaks_priority_ties_by_lru() {
        let mut cache = TileCache::open_in_memory(20).unwrap();
        let a = key(TileKind::Osm, 1, 2, 0, 0);
        let b = key(TileKind::Osm, 1, 2, 1, 0);
        let c = key(TileKind::Osm, 1, 2, 2, 0);

        cache.put(a, &[1u8; 10], 2).unwrap();
        cache.put(b, &[2u8; 10], 2).unwrap();
        // Touch a so b becomes the oldest equal-priority tile.
        cache.get(a, 2).unwrap().unwrap();
        let out = cache.put(c, &[3u8; 10], 2).unwrap();
        assert_eq!(out.evicted, vec![b]);
        assert!(cache.contains(a).unwrap());
        assert!(cache.contains(c).unwrap());
    }

    #[test]
    fn get_raise_priority_shields_tile_from_eviction() {
        let mut cache = TileCache::open_in_memory(20).unwrap();
        let low = key(TileKind::Dem, 3, 4, 0, 0);
        let other = key(TileKind::Dem, 3, 4, 1, 0);
        let fresh = key(TileKind::Dem, 3, 4, 2, 0);

        cache.put(low, &[7u8; 10], 0).unwrap();
        cache.put(other, &[8u8; 10], 0).unwrap();
        // Re-request `low` at high priority: its stored priority jumps.
        cache.get(low, 9).unwrap().unwrap();
        let out = cache.put(fresh, &[9u8; 10], 0).unwrap();
        // `other` is older at the same low priority -> victim.
        assert_eq!(out.evicted, vec![other]);
        assert!(cache.contains(low).unwrap());
    }

    #[test]
    fn oversized_payload_is_not_retained_but_reported() {
        let mut cache = TileCache::open_in_memory(8).unwrap();
        let k = key(TileKind::Dem, 5, 0, 0, 0);
        let out = cache.put(k, &[0xAB; 64], 100).unwrap();
        assert!(!out.stored);
        assert_eq!(out.evicted, vec![k]);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().bytes, 0);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn re_put_same_key_keeps_single_entry_and_correct_bytes() {
        let mut cache = TileCache::open_in_memory(1 << 16).unwrap();
        let k = key(TileKind::Osm, 0, 3, 2, 2);
        cache.put(k, &[1u8; 100], 1).unwrap();
        cache.put(k, &[2u8; 40], 1).unwrap();
        let s = cache.stats();
        assert_eq!(s.entries, 1);
        assert_eq!(s.bytes, 40);
        assert_eq!(cache.get(k, 1).unwrap().unwrap(), vec![2u8; 40]);
    }

    #[test]
    fn remove_and_clear_keep_counters_honest() {
        let mut cache = TileCache::open_in_memory(1 << 16).unwrap();
        let a = key(TileKind::Dem, 0, 0, 0, 0);
        let b = key(TileKind::Osm, 0, 0, 0, 1);
        cache.put(a, &[1u8; 16], 1).unwrap();
        cache.put(b, &[2u8; 32], 1).unwrap();

        assert!(cache.remove(a).unwrap());
        assert!(!cache.remove(a).unwrap());
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().bytes, 32);

        cache.clear().unwrap();
        let s = cache.stats();
        assert_eq!(s.entries, 0);
        assert_eq!(s.bytes, 0);
    }

    #[test]
    fn file_cache_persists_across_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "sim-world-cache-persist-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("tiles.db");

        let k = key(TileKind::Dem, 2, 6, 40, 21);
        {
            let mut c = TileCache::open(&path, 1 << 20).unwrap();
            c.put(k, b"durable-bytes", 4).unwrap();
        }
        {
            let mut c = TileCache::open(&path, 1 << 20).unwrap();
            assert_eq!(c.get(k, 4).unwrap().unwrap(), b"durable-bytes");
            // Fresh connection: first access counts as a miss? No — data is
            // there, so it is a hit.
            assert_eq!(c.stats().hits, 1);
            assert_eq!(c.stats().misses, 0);
            assert_eq!(c.stats().entries, 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
