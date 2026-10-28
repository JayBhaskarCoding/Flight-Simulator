//! Async world data services: a tokio tile streamer in front of a SQLite
//! rolling cache, with DEM elevation sampling and OSM vector extraction.
//!
//! Everything is keyed by [`sim_geo::quadtree::TileId`] — the six-face
//! spherical quadtree shared with the renderer's LOD selection — so the
//! cache, the streamer and the renderer all speak the same tile identity.
//!
//! Layering:
//!
//! * [`source`] — the [`source::TileSource`] trait (where raw bytes come
//!   from) and [`source::FileTileSource`], a clean-room filesystem layout
//!   (`root/{dem,osm}/{face}/{level}/{x}/{y}.{ext}`).
//! * [`cache`] — [`cache::TileCache`], a synchronous SQLite rolling cache
//!   with priority-aware LRU eviction to a byte budget. It is owned and
//!   driven by the streamer worker task; all methods are plain `&mut self`
//!   so it is trivially testable without a runtime.
//! * [`streamer`] — [`streamer::TileStreamer`], the async front end:
//!   request de-duplication, bounded concurrent fetches, cache-backed
//!   replay, broadcast events and stats.
//! * [`dem`] — [`dem::DemTile`]: an f32 elevation grid with a `DEM1`
//!   binary container, bilinear sampling through the cube-face projection.
//!   Samples are stored as `f32` (GPU-friendly) but every query result is
//!   `f64` — the 64-bit world law.
//! * [`osm`] — a dependency-free XML pull parser and the OSM model
//!   (`nodes`, `ways`, tags) with road / building extraction.
//!
//! No network access is assumed anywhere: the bundled [`source::FileTileSource`]
//! reads a local tile tree, which keeps the whole engine air-gap friendly.

pub mod cache;
pub mod dem;
pub mod osm;
pub mod source;
pub mod streamer;

pub use cache::{CacheKey, CacheStats, TileCache};
pub use dem::DemTile;
pub use osm::{Building, OsmBounds, OsmError, OsmMap, OsmNode, OsmWay, Road};
pub use source::{FileTileSource, TileKind, TileSource};
pub use streamer::{
    ResidentState, StreamerConfig, StreamerStats, TileEvent, TileEventPayload, TileFetch,
    TileStreamer,
};

use sim_geo::quadtree::TileId;
use std::fmt;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, WorldError>;

/// Errors produced by the world-data layer.
#[derive(Debug)]
pub enum WorldError {
    /// Underlying filesystem or I/O failure.
    Io(std::io::Error),
    /// SQLite failure from the rolling cache.
    Sql(rusqlite::Error),
    /// Malformed or out-of-contract DEM payload.
    Dem(String),
    /// OSM XML or model error.
    Osm(OsmError),
    /// The source has no data for this tile (checked and confirmed absent).
    TileUnavailable {
        /// Kind of the missing tile.
        kind: TileKind,
        /// Quadtree identity of the missing tile.
        id: TileId,
    },
    /// A background fetch task failed or panicked (message carries the
    /// join error text).
    FetchFailed(String),
    /// The streamer has been shut down (or was never reachable).
    Closed,
    /// A DEM tile does not cover the direction it was asked to sample.
    WrongTileFace {
        /// Face of the tile that was sampled.
        tile_face: u8,
        /// Face the direction belongs to.
        dir_face: u8,
    },
}

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorldError::Io(e) => write!(f, "i/o error: {e}"),
            WorldError::Sql(e) => write!(f, "sqlite error: {e}"),
            WorldError::Dem(m) => write!(f, "dem error: {m}"),
            WorldError::Osm(e) => write!(f, "osm error: {e}"),
            WorldError::TileUnavailable { kind, id } => write!(
                f,
                "tile unavailable: {:?} face={} level={} x={} y={}",
                kind, id.face, id.level, id.x, id.y
            ),
            WorldError::FetchFailed(m) => write!(f, "fetch task failed: {m}"),
            WorldError::Closed => write!(f, "tile streamer is closed"),
            WorldError::WrongTileFace {
                tile_face,
                dir_face,
            } => write!(
                f,
                "direction lies on cube face {dir_face} but tile belongs to face {tile_face}"
            ),
        }
    }
}

impl std::error::Error for WorldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WorldError::Io(e) => Some(e),
            WorldError::Sql(e) => Some(e),
            WorldError::Osm(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for WorldError {
    fn from(e: std::io::Error) -> Self {
        WorldError::Io(e)
    }
}

impl From<rusqlite::Error> for WorldError {
    fn from(e: rusqlite::Error) -> Self {
        WorldError::Sql(e)
    }
}

impl From<OsmError> for WorldError {
    fn from(e: OsmError) -> Self {
        WorldError::Osm(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn display_and_source_cover_variants() {
        let io = WorldError::Io(std::io::Error::other("boom"));
        assert!(io.to_string().contains("i/o error"));
        assert!(io.source().is_some());

        let unavailable = WorldError::TileUnavailable {
            kind: TileKind::Dem,
            id: TileId::root(3),
        };
        let s = unavailable.to_string();
        assert!(s.contains("face=3") && s.contains("level=0"));
        assert!(unavailable.source().is_none());

        let wrong = WorldError::WrongTileFace {
            tile_face: 1,
            dir_face: 4,
        };
        assert!(wrong.to_string().contains("face 4"));

        let osm = WorldError::Osm(OsmError::Xml {
            message: "unexpected end".into(),
            offset: 7,
        });
        assert!(osm.to_string().contains("byte 7"));
        assert!(osm.source().is_some());

        let sql = WorldError::Sql(rusqlite::Error::ExecuteReturnedResults);
        assert!(sql.source().is_some());

        let failed = WorldError::FetchFailed("task exploded".into());
        assert!(failed.to_string().contains("task exploded"));

        let closed = WorldError::Closed;
        assert!(closed.to_string().contains("closed"));

        let dem = WorldError::Dem("bad magic".into());
        assert!(dem.to_string().contains("bad magic"));
        assert!(dem.source().is_none());
    }
}
