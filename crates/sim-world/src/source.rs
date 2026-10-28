//! Tile sources: where raw tile bytes come from.
//!
//! The [`TileSource`] trait is the single seam between the streamer and any
//! backing store. The bundled [`FileTileSource`] reads a local directory
//! tree so the engine stays air-gap friendly; a network implementation can
//! be added later without touching the streamer or the cache.
//!
//! On-disk layout for [`FileTileSource`] (one file per tile):
//!
//! ```text
//! root/
//!   dem/{face}/{level}/{x}/{y}.dem      # DEM1 elevation grid (see [`crate::dem`])
//!   osm/{face}/{level}/{x}/{y}.osm      # OSM XML fragment (see [`crate::osm`])
//! ```

use sim_geo::quadtree::TileId;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

/// Category of data carried by a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TileKind {
    /// Digital elevation model (f32 height grid, [`crate::dem::DemTile`]).
    Dem,
    /// OpenStreetMap vector fragment (XML, [`crate::osm`]).
    Osm,
}

impl TileKind {
    /// Stable directory / extension name used by [`FileTileSource`].
    pub fn as_str(self) -> &'static str {
        match self {
            TileKind::Dem => "dem",
            TileKind::Osm => "osm",
        }
    }

    /// Integer encoding used by the SQLite cache schema.
    pub fn as_db(self) -> i64 {
        match self {
            TileKind::Dem => 1,
            TileKind::Osm => 2,
        }
    }

    /// Inverse of [`TileKind::as_db`].
    pub fn from_db(v: i64) -> Option<TileKind> {
        match v {
            1 => Some(TileKind::Dem),
            2 => Some(TileKind::Osm),
            _ => None,
        }
    }
}

impl std::fmt::Display for TileKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A future returned by [`TileSource::fetch`].
///
/// Boxed so implementations stay object-safe while still promising `Send`
/// (the streamer spawns every fetch onto the tokio runtime).
pub type FetchFuture<'a> = Pin<Box<dyn Future<Output = crate::Result<Option<Vec<u8>>>> + Send + 'a>>;

/// Asynchronous producer of raw tile bytes.
///
/// Contract:
///
/// * `Ok(Some(bytes))` — the tile exists; `bytes` are handed to the cache
///   and to every waiter unchanged.
/// * `Ok(None)` — the source positively knows the tile does not exist.
///   The streamer records a negative entry so the miss is not retried
///   until it is evicted.
/// * `Err(_)` — transient failure; waiters receive the error and the tile
///   is recorded as unavailable (a later request fetches again only after
///   the negative entry is evicted — sources that want retry semantics
///   should return `Err` *without* letting the streamer cache it, which is
///   what [`crate::streamer`] does for panics only).
pub trait TileSource: Send + Sync + 'static {
    /// Fetch raw bytes for one tile.
    fn fetch(&self, kind: TileKind, id: TileId) -> FetchFuture<'_>;
}

/// Filesystem-backed [`TileSource`] reading `root/{kind}/{face}/{level}/{x}/{y}.{ext}`.
#[derive(Debug, Clone)]
pub struct FileTileSource {
    root: Arc<PathBuf>,
}

impl FileTileSource {
    /// Create a source rooted at `root` (the directory that contains the
    /// `dem/` and `osm/` trees). The directory need not exist yet.
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        FileTileSource {
            root: Arc::new(root.as_ref().to_path_buf()),
        }
    }

    /// Absolute path of the file backing one tile.
    pub fn path_for(&self, kind: TileKind, id: TileId) -> PathBuf {
        self.root
            .join(kind.as_str())
            .join(id.face.to_string())
            .join(id.level.to_string())
            .join(id.x.to_string())
            .join(format!("{}.{}", id.y, kind.as_str()))
    }
}

impl TileSource for FileTileSource {
    fn fetch(&self, kind: TileKind, id: TileId) -> FetchFuture<'_> {
        let path = self.path_for(kind, id);
        Box::pin(async move {
            match tokio::fs::read(&path).await {
                Ok(bytes) => Ok(Some(bytes)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e.into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let n = UNIQUE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("sim-world-src-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    static UNIQUE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[test]
    fn kind_roundtrips_through_db_encoding() {
        for kind in [TileKind::Dem, TileKind::Osm] {
            assert_eq!(TileKind::from_db(kind.as_db()), Some(kind));
            assert!(!kind.as_str().is_empty());
            assert!(kind.to_string() == kind.as_str());
        }
        assert_eq!(TileKind::from_db(0), None);
        assert_eq!(TileKind::from_db(99), None);
    }

    #[test]
    fn path_for_follows_documented_layout() {
        let src = FileTileSource::new("/tiles");
        let id = TileId {
            face: 4,
            level: 7,
            x: 99,
            y: 17,
        };
        assert_eq!(
            src.path_for(TileKind::Dem, id),
            PathBuf::from("/tiles/dem/4/7/99/17.dem")
        );
        assert_eq!(
            src.path_for(TileKind::Osm, id),
            PathBuf::from("/tiles/osm/4/7/99/17.osm")
        );
    }

    #[tokio::test]
    async fn fetch_returns_bytes_missing_none_and_errors_on_directory() {
        let root = unique_temp_dir("fetch");
        let id = TileId {
            face: 0,
            level: 1,
            x: 1,
            y: 0,
        };

        // Missing tree -> None (soft miss).
        let src = FileTileSource::new(&root);
        assert!(src.fetch(TileKind::Dem, id).await.unwrap().is_none());

        // Create the file -> Some(bytes).
        let path = src.path_for(TileKind::Dem, id);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"tile-bytes").unwrap();
        drop(f);
        let got = src.fetch(TileKind::Dem, id).await.unwrap().unwrap();
        assert_eq!(got, b"tile-bytes");

        // Reading a directory (same name, wrong type) surfaces an I/O error.
        let dir_path = src.path_for(TileKind::Osm, id);
        std::fs::create_dir_all(&dir_path).unwrap();
        let err = src.fetch(TileKind::Osm, id).await;
        assert!(matches!(err, Err(crate::WorldError::Io(_))));

        let _ = std::fs::remove_dir_all(&root);
    }
}
