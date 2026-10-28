//! End-to-end world-data pipeline: filesystem tiles → streamer → SQLite
//! cache → DEM sampling / OSM extraction, keyed by the sim-geo quadtree.

use sim_geo::quadtree::{is_watertight_partition, select_tiles, SelectionParams, TileId};
use sim_geo::WGS84;
use sim_world::{
    DemTile, FileTileSource, OsmMap, ResidentState, StreamerConfig, TileCache, TileEventPayload,
    TileKind, TileStreamer,
};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn unique_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "sim-world-it-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Marker-valued DEM: constant height = `marker as f32` everywhere.
fn marker_dem(id: TileId, marker: u32) -> DemTile {
    DemTile::new(id, 2, 2, vec![marker as f32; 4]).unwrap()
}

fn write_dem(root: &std::path::Path, id: TileId, marker: u32) {
    let src = FileTileSource::new(root);
    let path = src.path_for(TileKind::Dem, id);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, marker_dem(id, marker).encode()).unwrap();
}

fn write_osm(root: &std::path::Path, id: TileId, xml: &str) {
    let src = FileTileSource::new(root);
    let path = src.path_for(TileKind::Osm, id);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, xml).unwrap();
}

fn streamer_for(root: &std::path::Path, db: &std::path::Path, max_bytes: u64) -> TileStreamer {
    let source = FileTileSource::new(root);
    let cache = TileCache::open(db, max_bytes).unwrap();
    TileStreamer::spawn(
        source,
        cache,
        StreamerConfig {
            max_cache_bytes: max_bytes,
            max_inflight: 4,
            request_capacity: 32,
            event_capacity: 8,
        },
    )
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dem_pipeline_decodes_and_samples_through_the_quadtree() {
    let root = unique_dir("dem");
    let id = TileId {
        face: 5,
        level: 6,
        x: 21,
        y: 44,
    };
    write_dem(&root, id, 731); // constant 731 m field

    let s = streamer_for(&root, &root.join("cache.db"), 1 << 20);
    let fetch = s.fetch(TileKind::Dem, id, 4).await.unwrap();
    assert!(!fetch.from_cache);

    let dem = DemTile::decode_as(&fetch.bytes, id).unwrap();
    // Sample the tile centre through the cube-face projection: a constant
    // field must come back as its marker value.
    let centre = dem.sample_direction(id.center()).unwrap();
    assert!(
        (centre - 731.0).abs() < 1e-6,
        "centre sample {centre} != 731"
    );
    // A corner direction (still inside this tile) samples identically.
    let corner = dem.sample_direction(id.corners()[0]).unwrap();
    assert!((corner - 731.0).abs() < 1e-6);

    // Second fetch replays from the cache.
    let again = s.fetch(TileKind::Dem, id, 4).await.unwrap();
    assert!(again.from_cache);

    let stats = s.stats().await.unwrap();
    assert_eq!(stats.cache.entries, 1);
    assert_eq!(stats.cache.hits, 1);
    assert_eq!(stats.cache.misses, 1);
    s.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn osm_pipeline_extracts_roads_and_buildings_from_disk() {
    let root = unique_dir("osm");
    let id = TileId {
        face: 4,
        level: 3,
        x: 5,
        y: 2,
    };
    let xml = r#"<?xml version="1.0"?>
<osm version="0.6">
  <node id="1" lat="12.9700" lon="77.5900"/>
  <node id="2" lat="12.9700" lon="77.6000"/>
  <node id="3" lat="12.9750" lon="77.6000"/>
  <node id="4" lat="12.9750" lon="77.5900"/>
  <way id="10">
    <nd ref="1"/><nd ref="2"/>
    <tag k="highway" v="secondary"/>
    <tag k="name" v="Old Airport Road"/>
  </way>
  <way id="11">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="building" v="hangar"/>
    <tag k="height" v="15 m"/>
  </way>
</osm>
"#;
    write_osm(&root, id, xml);

    let s = streamer_for(&root, &root.join("cache.db"), 1 << 20);
    let fetch = s.fetch(TileKind::Osm, id, 2).await.unwrap();
    let map: OsmMap = sim_world::osm::parse_osm_bytes(&fetch.bytes).unwrap();

    let roads = map.roads().unwrap();
    assert_eq!(roads.len(), 1);
    assert_eq!(roads[0].kind, "secondary");
    assert_eq!(roads[0].name.as_deref(), Some("Old Airport Road"));
    assert_eq!(roads[0].points.len(), 2);

    let buildings = map.buildings().unwrap();
    assert_eq!(buildings.len(), 1);
    assert_eq!(buildings[0].height_m, Some(15.0));
    assert!(buildings[0].contains(12.9725, 77.5950));
    assert!(!buildings[0].contains(12.9725, 77.6100));

    s.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn restart_replays_from_persistent_cache() {
    let root = unique_dir("restart");
    let id = TileId {
        face: 2,
        level: 5,
        x: 9,
        y: 9,
    };
    write_dem(&root, id, 42);
    let db = root.join("cache.db");

    // Session 1: load from disk.
    let s1 = streamer_for(&root, &db, 1 << 20);
    let f1 = s1.fetch(TileKind::Dem, id, 3).await.unwrap();
    assert!(!f1.from_cache);
    s1.shutdown().await.unwrap();

    // Session 2: wipe the source tree — the warm SQLite cache must answer.
    std::fs::remove_dir_all(root.join("dem")).unwrap();
    let s2 = streamer_for(&root, &db, 1 << 20);
    let f2 = s2.fetch(TileKind::Dem, id, 3).await.unwrap();
    assert!(f2.from_cache, "restart must replay from the cache");
    assert_eq!(f2.bytes, f1.bytes);
    let dem = DemTile::decode_as(&f2.bytes, id).unwrap();
    assert_eq!(dem.min_max(), (42.0, 42.0));
    s2.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn missing_tile_gets_negative_entry_and_event() {
    let root = unique_dir("negative");
    let id = TileId {
        face: 0,
        level: 10,
        x: 100,
        y: 300,
    };
    let s = streamer_for(&root, &root.join("cache.db"), 1 << 20);
    let mut events = s.subscribe();

    let err = s.fetch(TileKind::Dem, id, 0).await.unwrap_err();
    assert!(
        matches!(err, sim_world::WorldError::TileUnavailable { .. }),
        "got {err}"
    );
    let ev = events.recv().await.unwrap();
    assert_eq!(ev.id, id);
    assert!(matches!(ev.payload, TileEventPayload::Unavailable));
    assert!(matches!(
        s.resident_state(TileKind::Dem, id),
        Some(ResidentState::Unavailable)
    ));

    // Negative answer is served locally from then on.
    let err2 = s.fetch(TileKind::Dem, id, 0).await.unwrap_err();
    assert!(matches!(err2, sim_world::WorldError::TileUnavailable { .. }));
    let stats = s.stats().await.unwrap();
    assert_eq!(stats.cache.entries, 0);
    s.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn priority_budget_keeps_the_important_tiles() {
    let root = unique_dir("priority");
    let ids: Vec<TileId> = (0..3u32)
        .map(|i| TileId {
            face: 1,
            level: 4,
            x: i,
            y: 7,
        })
        .collect();
    // DEM1 containers of a 2x2 constant grid are exactly 28 bytes.
    for (i, id) in ids.iter().enumerate() {
        write_dem(&root, *id, 100 + i as u32);
    }

    // Budget of 70 bytes fits exactly two tiles.
    let s = streamer_for(&root, &root.join("cache.db"), 70);
    s.fetch(TileKind::Dem, ids[0], 1).await.unwrap();
    s.fetch(TileKind::Dem, ids[1], 1).await.unwrap();
    s.fetch(TileKind::Dem, ids[2], 90).await.unwrap();

    let stats = s.stats().await.unwrap();
    assert_eq!(stats.cache.bytes, 56, "two 28-byte tiles resident");
    assert_eq!(stats.cache.entries, 2);
    assert_eq!(stats.cache.evictions, 1);

    // The high-priority tile survived; exactly one low-priority tile was
    // evicted (the oldest one, ids[0]).
    assert!(matches!(
        s.resident_state(TileKind::Dem, ids[2]),
        Some(ResidentState::Available { .. })
    ));
    assert!(s.resident_state(TileKind::Dem, ids[0]).is_none());
    assert!(matches!(
        s.resident_state(TileKind::Dem, ids[1]),
        Some(ResidentState::Available { .. })
    ));
    s.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn selected_quadtree_tiles_all_stream_and_agree() {
    // Camera 400 km above the Indian Ocean; refine a small patch.
    let camera = WGS84.geodetic_to_ecef(&sim_geo::LatLonAlt::new(
        -10f64.to_radians(),
        80f64.to_radians(),
        400_000.0,
    ));
    let params = SelectionParams {
        tolerance_rad: 0.003,
        max_level: 4,
        refine_radius_m: 80_000.0,
    };
    let tiles = select_tiles(camera.as_dvec3(), WGS84.semi_major_axis, params);
    assert!(!tiles.is_empty());
    assert!(is_watertight_partition(&tiles));

    let root = unique_dir("lod");
    for (i, id) in tiles.iter().enumerate() {
        // Distinct marker per tile = face*1000 + level*100 + index.
        let marker = u32::from(id.face) * 1000 + u32::from(id.level) * 100 + i as u32;
        write_dem(&root, *id, marker);
    }

    let s = streamer_for(&root, &root.join("cache.db"), 8 * 1024 * 1024);
    for (i, id) in tiles.iter().enumerate() {
        let marker = u32::from(id.face) * 1000 + u32::from(id.level) * 100 + i as u32;
        let fetch = s.fetch(TileKind::Dem, *id, 5).await.unwrap();
        let dem = DemTile::decode_as(&fetch.bytes, *id).unwrap();
        let centre = dem.sample_direction(id.center()).unwrap();
        assert!(
            (centre - marker as f64).abs() < 1e-6,
            "tile {id:?}: centre {centre} != marker {marker}"
        );
    }

    let stats = s.stats().await.unwrap();
    assert_eq!(stats.cache.entries as usize, tiles.len());
    assert_eq!(stats.resident_available, tiles.len());
    s.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(&root);
}
