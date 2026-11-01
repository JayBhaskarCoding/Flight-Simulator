//! Windowed mode: winit 0.30 event loop + wgpu surface presentation.
//!
//! The windowed path shares [`crate::sim::SimCore`] with headless mode
//! — the physics thread is identical; only presentation differs.

use crate::ecs::Ecs;
use crate::sim::{
    build_frame_input, camera_ecef_pose, spawn_aircraft_and_camera, system_build_draws_range,
    system_camera_follow, system_sync_aircraft, ControlCommand, SimCore, Snapshot,
};
use glam::{DMat3, DQuat, DVec3, Mat4, Quat, Vec3};
use sim_avionics::ApMode;
use sim_geo::quadtree::tile_for_direction;
use sim_geo::{Ecef, EnuFrame, FloatingOrigin, LatLonAlt, WGS84};
use sim_render::gltf::{DecodedImage, GltfInstance, GltfMaterial, GltfMesh, GltfModel};
use sim_render::types::GpuVertex;
use sim_render::Renderer;
use sim_world::{DemTile, FileTileSource, TerrainMesh, TileKind};
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use std::{
    collections::{HashMap, HashSet},
    iter,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

/// Preferred present formats (must be sRGB — the tonemap shader emits
/// linear ACES and relies on the target's transfer function).
const TARGET_FORMAT_PREFERENCE: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgba8UnormSrgb,
];

/// Throttle lever travel per key press (also repeats while held via
/// winit's key-repeat `Pressed` events), clamped to `0..=1`.
const THROTTLE_STEP: f64 = 0.05;

/// A normalized flight-control binding, shared by the logical-key
/// (`Key::Character` / arrows) and physical-key (`KeyCode`) paths so
/// both spellings drive the exact same `ControlCommand` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlBinding {
    /// Elevator nose-up (W / ArrowUp).
    PitchUp,
    /// Elevator nose-down (S / ArrowDown).
    PitchDown,
    /// Aileron roll left (A / ArrowLeft).
    RollLeft,
    /// Aileron roll right (D / ArrowRight).
    RollRight,
    /// Rudder + nosewheel steering left (Q).
    YawLeft,
    /// Rudder + nosewheel steering right (E).
    YawRight,
    /// Throttle lever up (R / T).
    ThrottleUp,
    /// Throttle lever down (F / Y).
    ThrottleDown,
    /// Wheel brakes (B).
    Brake,
    /// Flap notch cycle, 0 → 1 → 2 → 3 → 0 (G).
    Flaps,
    /// Autopilot altitude-hold toggle (P).
    ApToggle,
}

impl ControlBinding {
    /// Layout-dependent character mapping (kept as the test-facing and
    /// fallback path).
    fn from_str(ch: &str) -> Option<Self> {
        Some(match ch {
            "w" | "W" => Self::PitchUp,
            "s" | "S" => Self::PitchDown,
            "a" | "A" => Self::RollLeft,
            "d" | "D" => Self::RollRight,
            "q" | "Q" => Self::YawLeft,
            "e" | "E" => Self::YawRight,
            "r" | "R" => Self::ThrottleUp,
            "t" | "T" => Self::ThrottleUp,
            "f" | "F" => Self::ThrottleDown,
            "y" | "Y" => Self::ThrottleDown,
            "b" | "B" => Self::Brake,
            "g" | "G" => Self::Flaps,
            "p" | "P" => Self::ApToggle,
            _ => return None,
        })
    }

    /// Physical-position mapping: immune to keyboard layout so the
    /// keys the pilot presses always reach the same control surface.
    fn from_code(code: KeyCode) -> Option<Self> {
        Some(match code {
            KeyCode::KeyW | KeyCode::ArrowUp => Self::PitchUp,
            KeyCode::KeyS | KeyCode::ArrowDown => Self::PitchDown,
            KeyCode::KeyA | KeyCode::ArrowLeft => Self::RollLeft,
            KeyCode::KeyD | KeyCode::ArrowRight => Self::RollRight,
            KeyCode::KeyQ => Self::YawLeft,
            KeyCode::KeyE => Self::YawRight,
            KeyCode::KeyR | KeyCode::KeyT => Self::ThrottleUp,
            KeyCode::KeyF | KeyCode::KeyY => Self::ThrottleDown,
            KeyCode::KeyB => Self::Brake,
            KeyCode::KeyG => Self::Flaps,
            KeyCode::KeyP => Self::ApToggle,
            _ => return None,
        })
    }
}
const TERRAIN_PATCH_RADIUS: i32 = 7;
const TERRAIN_TEXTURE_SIZE: u32 = 768;

/// Shared inputs written from the winit thread.
pub struct WindowInput {
    pub command: Arc<Mutex<ControlCommand>>,
    pub snapshot: Arc<RwLock<Snapshot>>,
}

struct FrameResources {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    window: Arc<Window>,
}

pub struct App {
    input: WindowInput,
    bridge: crate::sim::FrameBridge,
    sky_sun_enu: DVec3,
    model: Option<sim_render::GltfModel>,
    aircraft_instances: Range<usize>,
    aircraft_base_worlds: Vec<Mat4>,
    cockpit_instances: Range<usize>,
    cockpit_digit_instances: Vec<usize>,
    cockpit_glyph_meshes: HashMap<char, usize>,
    terrain_instances: Option<Range<usize>>,
    terrain_root: Option<(DVec3, DQuat)>,
    ecs: Ecs,
    resources: Option<FrameResources>,
    start: Instant,
    last_title_update: Instant,
    pending_resize: Option<(u32, u32)>,
    camera_yaw: f64,
    camera_pitch: f64,
    camera_distance: f64,
    cockpit_view: bool,
    orbit_dragging: bool,
    last_cursor: Option<(f64, f64)>,
}

/// Decoded terrain tile plus the ECEF orientation of its local ENU mesh.
pub struct TerrainAsset {
    pub mesh: TerrainMesh,
    /// Regional terrain tiles with bundled elevation coverage at the same LOD.
    pub regional_meshes: Vec<TerrainMesh>,
    pub enu_to_ecef: DQuat,
    pub field_elevation_m: f64,
    pub osm_overlay: Vec<(GltfMesh, GltfMaterial)>,
}

/// Load a level-12 terrain patch around the selected origin, plus any runway
/// terrain tiles in the regional airport extract.
pub fn load_terrain_tile(root: &Path, origin: LatLonAlt) -> Result<TerrainAsset, String> {
    let lat = origin.latitude_rad();
    let lon = origin.longitude_rad();
    let dir = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
    let id = tile_for_direction(dir, 12);
    let source = FileTileSource::new(root);
    let path = source.path_for(TileKind::Dem, id);
    let mesh = if path.is_file() {
        let bytes =
            std::fs::read(&path).map_err(|e| format!("read DEM tile {}: {e}", path.display()))?;
        let dem = DemTile::decode_as(&bytes, id).map_err(|e| format!("decode DEM tile: {e}"))?;
        TerrainMesh::from_dem(&dem).map_err(|e| format!("build DEM mesh: {e}"))?
    } else {
        load_terrarium_terrain(root, id, lat, lon, 257)?
    };
    let regional_airports_path = root.join("osm/regions/new-york-airports.osm");
    let regional_airports = if regional_airports_path.is_file() {
        let bytes = std::fs::read(&regional_airports_path).map_err(|e| {
            format!(
                "read regional OSM airport data {}: {e}",
                regional_airports_path.display()
            )
        })?;
        Some(sim_world::osm::parse_osm_bytes(&bytes).map_err(|e| {
            format!(
                "parse regional OSM airport data {}: {e}",
                regional_airports_path.display()
            )
        })?)
    } else {
        None
    };
    let (u0, u1, v0, v1) = id.uv_range();
    let center_u = (u0 + u1) * 0.5;
    let center_v = (v0 + v1) * 0.5;
    let tile_u = (u1 - u0) * 0.5;
    let tile_v = (v1 - v0) * 0.5;
    let mut regional_meshes = Vec::new();
    let mut seen = HashSet::from([id]);
    for dy in -TERRAIN_PATCH_RADIUS..=TERRAIN_PATCH_RADIUS {
        for dx in -TERRAIN_PATCH_RADIUS..=TERRAIN_PATCH_RADIUS {
            if dx == 0 && dy == 0 {
                continue;
            }
            // Sampling beyond a cube-face edge and mapping the direction
            // back through the quadtree also handles tiles on face seams.
            let neighbor = tile_for_direction(
                id.direction_at(
                    center_u + 2.0 * tile_u * dx as f64,
                    center_v + 2.0 * tile_v * dy as f64,
                ),
                id.level,
            );
            if !seen.insert(neighbor) {
                continue;
            }
            let neighbor_path = source.path_for(TileKind::Dem, neighbor);
            let neighbor_center = neighbor.center_lat_lon();
            let n = 1u32 << 13;
            let (xyz_x, xyz_y) = web_mercator_tile(
                neighbor_center.latitude_rad(),
                neighbor_center.longitude_rad(),
                n,
            );
            let terrarium_path = root
                .join("terrarium")
                .join("13")
                .join(xyz_x.to_string())
                .join(format!("{xyz_y}.png"));
            if !neighbor_path.is_file() && !terrarium_path.is_file() {
                continue;
            }
            let neighbor_mesh = if neighbor_path.is_file() {
                let bytes = std::fs::read(&neighbor_path).map_err(|e| {
                    format!("read neighboring DEM tile {}: {e}", neighbor_path.display())
                })?;
                let dem = DemTile::decode_as(&bytes, neighbor)
                    .map_err(|e| format!("decode neighboring DEM tile: {e}"))?;
                TerrainMesh::from_dem(&dem)
                    .map_err(|e| format!("build neighboring DEM mesh: {e}"))?
            } else {
                match load_terrarium_terrain(
                    root,
                    neighbor,
                    neighbor_center.latitude_rad(),
                    neighbor_center.longitude_rad(),
                    65,
                ) {
                    Ok(mesh) => mesh,
                    // A centre Terrarium pixel may exist at the edge of the
                    // bundle while the surrounding pixels needed for a full
                    // cube tile do not. Leave that neighbor out cleanly.
                    Err(error)
                        if error
                            .contains("cube tile extends beyond the supplied Terrarium mosaic") =>
                    {
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            };
            regional_meshes.push(neighbor_mesh);
        }
    }
    if let Some(airport_map) = &regional_airports {
        for runway in airport_map
            .ways
            .iter()
            .filter(|way| way.tag("aeroway") == Some("runway"))
        {
            for (runway_lat, runway_lon) in airport_map
                .resolve_points(runway)
                .map_err(|e| format!("resolve OSM runway {}: {e}", runway.id))?
            {
                let runway_lat = runway_lat.to_radians();
                let runway_lon = runway_lon.to_radians();
                let direction = DVec3::new(
                    runway_lat.cos() * runway_lon.cos(),
                    runway_lat.cos() * runway_lon.sin(),
                    runway_lat.sin(),
                );
                let runway_tile = tile_for_direction(direction, id.level);
                if !seen.insert(runway_tile) {
                    continue;
                }
                let runway_center = runway_tile.center_lat_lon();
                let (xyz_x, xyz_y) = web_mercator_tile(
                    runway_center.latitude_rad(),
                    runway_center.longitude_rad(),
                    1 << 13,
                );
                let tile_path = root
                    .join("terrarium")
                    .join("13")
                    .join(xyz_x.to_string())
                    .join(format!("{xyz_y}.png"));
                if !tile_path.is_file() {
                    continue;
                }
                match load_terrarium_terrain(
                    root,
                    runway_tile,
                    runway_center.latitude_rad(),
                    runway_center.longitude_rad(),
                    65,
                ) {
                    Ok(runway_mesh) => regional_meshes.push(runway_mesh),
                    Err(error)
                        if error
                            .contains("cube tile extends beyond the supplied Terrarium mosaic") =>
                    {
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }
    let terrain_tiles: Vec<&TerrainMesh> =
        iter::once(&mesh).chain(regional_meshes.iter()).collect();
    let osm_path = source.path_for(TileKind::Osm, id);
    let mut osm_overlay = if osm_path.is_file() {
        let bytes = std::fs::read(&osm_path)
            .map_err(|e| format!("read OSM tile {}: {e}", osm_path.display()))?;
        build_osm_overlay_with_terrain(&mesh, &bytes, &terrain_tiles)?
    } else {
        Vec::new()
    };
    if let Some(airport_map) = &regional_airports {
        osm_overlay.extend(build_osm_overlay_from_map(
            &mesh,
            airport_map,
            &terrain_tiles,
        )?);
    }
    let regional_water_path = root.join("osm/regions/new-york-water.osm");
    if regional_water_path.is_file() {
        let bytes = std::fs::read(&regional_water_path).map_err(|e| {
            format!(
                "read regional OSM water data {}: {e}",
                regional_water_path.display()
            )
        })?;
        let water_map = sim_world::osm::parse_osm_bytes(&bytes).map_err(|e| {
            format!(
                "parse regional OSM water data {}: {e}",
                regional_water_path.display()
            )
        })?;
        osm_overlay.extend(build_osm_overlay_from_map(
            &mesh,
            &water_map,
            &terrain_tiles,
        )?);
    }
    let regional_roads = root.join("osm/regions/new-york-roads.osm");
    if regional_roads.is_file() {
        let bytes = std::fs::read(&regional_roads).map_err(|e| {
            format!(
                "read regional OSM road data {}: {e}",
                regional_roads.display()
            )
        })?;
        osm_overlay.extend(build_osm_overlay_with_terrain(
            &mesh,
            &bytes,
            &terrain_tiles,
        )?);
    }
    let regional_buildings = root.join("osm/regions/new-york-buildings.osm");
    if regional_buildings.is_file() {
        let bytes = std::fs::read(&regional_buildings).map_err(|e| {
            format!(
                "read regional OSM building data {}: {e}",
                regional_buildings.display()
            )
        })?;
        osm_overlay.extend(build_osm_overlay_with_terrain(
            &mesh,
            &bytes,
            &terrain_tiles,
        )?);
    }
    let enu_to_ecef = EnuFrame::new(id.center_lat_lon(), WGS84).enu_identity_ecef_quat();
    let field_elevation_m = mesh.elevation_near(lat, lon);
    Ok(TerrainAsset {
        mesh,
        regional_meshes,
        enu_to_ecef,
        field_elevation_m,
        osm_overlay,
    })
}

#[cfg(test)]
fn build_osm_overlay(
    terrain: &TerrainMesh,
    bytes: &[u8],
) -> Result<Vec<(GltfMesh, GltfMaterial)>, String> {
    let map = sim_world::osm::parse_osm_bytes(bytes).map_err(|e| format!("parse OSM tile: {e}"))?;
    build_osm_overlay_from_map(terrain, &map, &[terrain])
}

fn build_osm_overlay_with_terrain(
    terrain: &TerrainMesh,
    bytes: &[u8],
    terrain_tiles: &[&TerrainMesh],
) -> Result<Vec<(GltfMesh, GltfMaterial)>, String> {
    let map = sim_world::osm::parse_osm_bytes(bytes).map_err(|e| format!("parse OSM tile: {e}"))?;
    build_osm_overlay_from_map(terrain, &map, terrain_tiles)
}

fn build_osm_overlay_from_map(
    terrain: &TerrainMesh,
    map: &sim_world::osm::OsmMap,
    terrain_tiles: &[&TerrainMesh],
) -> Result<Vec<(GltfMesh, GltfMaterial)>, String> {
    let frame = EnuFrame::new(terrain.id.center_lat_lon(), WGS84);
    let terrain_by_id: HashMap<_, _> = terrain_tiles.iter().map(|tile| (tile.id, *tile)).collect();
    let local = |lat: f64, lon: f64| {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let direction = DVec3::new(
            lat_rad.cos() * lon_rad.cos(),
            lat_rad.cos() * lon_rad.sin(),
            lat_rad.sin(),
        );
        let tile_id = tile_for_direction(direction, terrain.id.level);
        let ground = terrain_by_id
            .get(&tile_id)
            .unwrap_or(&terrain)
            .elevation_near(lat_rad, lon_rad);
        let p = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(lat, lon, 0.0));
        let mut enu = frame.ecef_to_enu(&p).as_dvec3();
        enu.z += ground;
        enu.as_vec3()
    };

    let mut overlays = build_osm_water_overlays(map, &frame, &local)?;

    // Keep each GPU vertex buffer comfortably below wgpu's portable 256 MiB
    // limit; a regional city extract contains hundreds of thousands of roofs.
    const BUILDING_VERTEX_CHUNK: usize = 200_000;
    let mut building_meshes = Vec::new();
    let mut building_vertices = Vec::new();
    let mut building_indices = Vec::new();
    for building in map
        .buildings()
        .map_err(|e| format!("extract OSM buildings: {e}"))?
    {
        let base: Vec<Vec3> = building
            .points
            .iter()
            .map(|&(lat, lon)| local(lat, lon))
            .collect();
        if base.len() < 3 {
            continue;
        }
        let height = building.height_m.unwrap_or(9.0).clamp(2.5, 300.0) as f32;
        for i in 0..base.len() {
            let a = base[i];
            let b = base[(i + 1) % base.len()];
            let edge = b - a;
            let normal = Vec3::new(edge.y, -edge.x, 0.0).normalize_or_zero();
            let first = building_vertices.len() as u32;
            for (p, z) in [(a, 0.0), (b, 0.0), (b, height), (a, height)] {
                building_vertices.push(GpuVertex {
                    position: (p + Vec3::Z * z).to_array(),
                    normal: normal.to_array(),
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv: [0.0, z / height],
                });
            }
            building_indices.extend_from_slice(&[
                first,
                first + 1,
                first + 2,
                first,
                first + 2,
                first + 3,
            ]);
        }
        let top_start = building_vertices.len() as u32;
        for p in &base {
            building_vertices.push(GpuVertex {
                position: (*p + Vec3::Z * height).to_array(),
                normal: Vec3::Z.to_array(),
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: [p.x, p.y],
            });
        }
        for i in 1..base.len() - 1 {
            building_indices.extend_from_slice(&[
                top_start,
                top_start + i as u32,
                top_start + i as u32 + 1,
            ]);
        }
        if building_vertices.len() >= BUILDING_VERTEX_CHUNK {
            building_meshes.push(GltfMesh {
                vertices: std::mem::take(&mut building_vertices),
                indices: std::mem::take(&mut building_indices),
            });
        }
    }

    let mut road_vertices = Vec::new();
    let mut road_indices = Vec::new();
    for road in map.roads().map_err(|e| format!("extract OSM roads: {e}"))? {
        let width = match road.kind.as_str() {
            "motorway" | "trunk" => 10.0,
            "primary" => 8.0,
            "secondary" => 7.0,
            "tertiary" => 6.0,
            "residential" | "unclassified" => 5.0,
            "service" => 3.5,
            "footway" | "path" | "cycleway" => 1.8,
            _ => 4.0,
        };
        for segment in road.points.windows(2) {
            let (lat_a, lon_a) = segment[0];
            let (lat_b, lon_b) = segment[1];
            let a = local(lat_a, lon_a);
            let b = local(lat_b, lon_b);
            let direction = Vec3::new(b.x - a.x, b.y - a.y, 0.0);
            let length = direction.length();
            if length < 0.5 {
                continue;
            }
            let side = Vec3::new(-direction.y, direction.x, 0.0) * (0.5 * width / length);
            let lift = Vec3::Z * 0.25;
            let first = road_vertices.len() as u32;
            for p in [
                a + side + lift,
                a - side + lift,
                b - side + lift,
                b + side + lift,
            ] {
                road_vertices.push(GpuVertex {
                    position: p.to_array(),
                    normal: Vec3::Z.to_array(),
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                });
            }
            road_indices.extend_from_slice(&[
                first,
                first + 1,
                first + 2,
                first,
                first + 2,
                first + 3,
            ]);
        }
    }

    // OSM airport surfaces are commonly closed area ways, but some extracts
    // represent runways as centre lines. Render both forms so the same data
    // path supports airports in streamed regional tiles and this NYC patch.
    let mut runway_vertices = Vec::new();
    let mut runway_indices = Vec::new();
    let mut taxiway_vertices = Vec::new();
    let mut taxiway_indices = Vec::new();
    for way in &map.ways {
        let Some(kind) = way.tag("aeroway") else {
            continue;
        };
        let (vertices, indices, default_width) = match kind {
            "runway" => (&mut runway_vertices, &mut runway_indices, 45.0_f32),
            "taxiway" | "helipad" | "apron" => {
                (&mut taxiway_vertices, &mut taxiway_indices, 18.0_f32)
            }
            _ => continue,
        };
        let points = map
            .resolve_points(way)
            .map_err(|e| format!("resolve OSM aeroway {}: {e}", way.id))?;
        if points.len() < 2 {
            continue;
        }
        if points.len() >= 4 && points.first() == points.last() {
            let mut ring: Vec<Vec3> = points[..points.len() - 1]
                .iter()
                .map(|&(lat, lon)| local(lat, lon) + Vec3::Z * 0.45)
                .collect();
            if ring.len() < 3 {
                continue;
            }
            // The OSM surfaces used here are simple footprints. A fan gives
            // the usual rectangular runway/apron a single flat render mesh.
            let base = vertices.len() as u32;
            for p in &ring {
                vertices.push(GpuVertex {
                    position: p.to_array(),
                    normal: Vec3::Z.to_array(),
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv: [p.x, p.y],
                });
            }
            for i in 1..ring.len() - 1 {
                indices.extend_from_slice(&[base, base + i as u32, base + i as u32 + 1]);
            }
            ring.clear();
        } else {
            let width = way
                .tag("width")
                .and_then(sim_world::osm::parse_height)
                .map(|width| width as f32)
                .filter(|w| w.is_finite() && *w > 1.0 && *w < 300.0)
                .unwrap_or(default_width);
            for segment in points.windows(2) {
                let a = local(segment[0].0, segment[0].1) + Vec3::Z * 0.45;
                let b = local(segment[1].0, segment[1].1) + Vec3::Z * 0.45;
                let direction = Vec3::new(b.x - a.x, b.y - a.y, 0.0);
                let length = direction.length();
                if length < 1.0 {
                    continue;
                }
                let side = Vec3::new(-direction.y, direction.x, 0.0) * (0.5 * width / length);
                let base = vertices.len() as u32;
                for p in [a + side, a - side, b - side, b + side] {
                    vertices.push(GpuVertex {
                        position: p.to_array(),
                        normal: Vec3::Z.to_array(),
                        tangent: [1.0, 0.0, 0.0, 1.0],
                        uv: [0.0, 0.0],
                    });
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
    }

    if !building_indices.is_empty() {
        building_meshes.push(GltfMesh {
            vertices: building_vertices,
            indices: building_indices,
        });
    }
    overlays.extend(building_meshes.into_iter().map(|mesh| {
        (
            mesh,
            GltfMaterial {
                name: "OpenStreetMap building footprints".into(),
                base_color_factor: [0.46, 0.39, 0.32, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 0.9,
                double_sided: true,
                ..GltfMaterial::default()
            },
        )
    }));
    if !road_indices.is_empty() {
        overlays.push((
            GltfMesh {
                vertices: road_vertices,
                indices: road_indices,
            },
            GltfMaterial {
                name: "OpenStreetMap roads".into(),
                base_color_factor: [0.23, 0.24, 0.25, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                double_sided: true,
                ..GltfMaterial::default()
            },
        ));
    }
    if !runway_indices.is_empty() {
        overlays.push((
            GltfMesh {
                vertices: runway_vertices,
                indices: runway_indices,
            },
            GltfMaterial {
                name: "OpenStreetMap runways".into(),
                base_color_factor: [0.20, 0.21, 0.22, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                double_sided: true,
                ..GltfMaterial::default()
            },
        ));
    }
    if !taxiway_indices.is_empty() {
        overlays.push((
            GltfMesh {
                vertices: taxiway_vertices,
                indices: taxiway_indices,
            },
            GltfMaterial {
                name: "OpenStreetMap taxiways and helipads".into(),
                base_color_factor: [0.28, 0.29, 0.30, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                double_sided: true,
                ..GltfMaterial::default()
            },
        ));
    }
    Ok(overlays)
}

const COASTAL_WATER_BAND_M: f32 = 3_000.0;
const WATER_MESH_VERTEX_CHUNK: usize = 200_000;

fn build_osm_water_overlays(
    map: &sim_world::osm::OsmMap,
    frame: &EnuFrame,
    local: &impl Fn(f64, f64) -> Vec3,
) -> Result<Vec<(GltfMesh, GltfMaterial)>, String> {
    let sea_level = |lat: f64, lon: f64| {
        frame
            .ecef_to_enu(&WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(lat, lon, 0.0)))
            .as_dvec3()
            .as_vec3()
    };
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut meshes = Vec::new();
    let mut rendered_water_ways = HashSet::new();
    let mut triangulator = earcut::Earcut::<f64>::new();

    for way in &map.ways {
        if way.tag("natural") == Some("coastline") {
            let points = map
                .resolve_points(way)
                .map_err(|e| format!("resolve OSM coastline {}: {e}", way.id))?;
            for pair in points.windows(2) {
                let (lat_a, lon_a) = pair[0];
                let (lat_b, lon_b) = pair[1];
                let a = sea_level(lat_a, lon_a);
                let b = sea_level(lat_b, lon_b);
                let direction = Vec3::new(b.x - a.x, b.y - a.y, 0.0);
                let length = direction.length();
                if length < 1.0 {
                    continue;
                }

                // OSM coastlines keep land on their left and water on their
                // right. Offset the water edge geographically so its height
                // follows the ellipsoid instead of a flat ENU plane.
                let right = Vec3::new(direction.y, -direction.x, 0.0) / length;
                let mid_lat = (lat_a + lat_b).to_radians() * 0.5;
                let offset_lat = f64::from(right.y * COASTAL_WATER_BAND_M) / 6_378_137.0;
                let offset_lon = f64::from(right.x * COASTAL_WATER_BAND_M)
                    / (6_378_137.0 * mid_lat.cos().abs().max(0.01));
                let outer_a = sea_level(
                    lat_a + offset_lat.to_degrees(),
                    lon_a + offset_lon.to_degrees(),
                );
                let outer_b = sea_level(
                    lat_b + offset_lat.to_degrees(),
                    lon_b + offset_lon.to_degrees(),
                );
                let base = vertices.len() as u32;
                for p in [a, b, outer_b, outer_a] {
                    vertices.push(water_vertex(p + Vec3::Z * 0.75));
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        } else if way.tag("natural") == Some("water")
            || way.tag("waterway") == Some("riverbank")
            || way.tag("water").is_some()
        {
            if way.refs.len() < 4 || way.refs.first() != way.refs.last() {
                continue;
            }
            let points = map
                .resolve_points(way)
                .map_err(|e| format!("resolve OSM water area {}: {e}", way.id))?;
            append_water_polygon(
                &points,
                &[],
                frame,
                local,
                &sea_level,
                &mut WaterMesh {
                    triangulator: &mut triangulator,
                    vertices: &mut vertices,
                    indices: &mut indices,
                },
            );
            rendered_water_ways.insert(way.id);
        }

        if vertices.len() >= WATER_MESH_VERTEX_CHUNK {
            meshes.push(GltfMesh {
                vertices: std::mem::take(&mut vertices),
                indices: std::mem::take(&mut indices),
            });
        }
    }
    for relation in &map.relations {
        if !(relation.tag("natural") == Some("water") || relation.tag("water").is_some())
            || relation.tag("type") != Some("multipolygon")
        {
            continue;
        }
        for (outer_refs, member_ways, hole_refs) in relation_water_polygons(map, relation) {
            if member_ways
                .iter()
                .any(|way_id| rendered_water_ways.contains(way_id))
            {
                continue;
            }
            let outer = resolve_osm_refs(map, &outer_refs);
            let holes: Option<Vec<Vec<(f64, f64)>>> = hole_refs
                .iter()
                .map(|refs| resolve_osm_refs(map, refs))
                .collect();
            if let (Some(outer), Some(holes)) = (outer, holes) {
                append_water_polygon(
                    &outer,
                    &holes,
                    frame,
                    local,
                    &sea_level,
                    &mut WaterMesh {
                        triangulator: &mut triangulator,
                        vertices: &mut vertices,
                        indices: &mut indices,
                    },
                );
            }
            if vertices.len() >= WATER_MESH_VERTEX_CHUNK {
                meshes.push(GltfMesh {
                    vertices: std::mem::take(&mut vertices),
                    indices: std::mem::take(&mut indices),
                });
            }
        }
    }
    if !indices.is_empty() {
        meshes.push(GltfMesh { vertices, indices });
    }

    Ok(meshes
        .into_iter()
        .map(|mesh| {
            (
                mesh,
                GltfMaterial {
                    name: "OpenStreetMap water and shoreline".into(),
                    base_color_factor: [0.035, 0.20, 0.31, 1.0],
                    metallic_factor: 0.08,
                    roughness_factor: 0.24,
                    double_sided: true,
                    ..GltfMaterial::default()
                },
            )
        })
        .collect())
}

fn relation_rings(
    map: &sim_world::osm::OsmMap,
    relation: &sim_world::osm::OsmRelation,
    role: &str,
) -> Vec<(Vec<i64>, Vec<i64>)> {
    let mut segments: Vec<(i64, Vec<i64>)> = relation
        .members
        .iter()
        .filter(|member| member.member_type == "way" && member.role == role)
        .filter_map(|member| {
            map.ways
                .iter()
                .find(|way| way.id == member.reference)
                .map(|way| (way.id, way.refs.clone()))
        })
        .collect();
    let mut rings = Vec::new();
    while let Some((first_way, mut ring)) = segments.pop() {
        let mut ring_ways = vec![first_way];
        if ring.len() < 2 {
            continue;
        }
        let mut attempts = 0;
        while ring.first() != ring.last() && attempts < relation.members.len() {
            let start = *ring.first().unwrap();
            let end = *ring.last().unwrap();
            let matched = segments.iter().position(|(_, refs)| {
                refs.first() == Some(&end)
                    || refs.last() == Some(&end)
                    || refs.first() == Some(&start)
                    || refs.last() == Some(&start)
            });
            let Some(index) = matched else { break };
            let (way_id, mut refs) = segments.remove(index);
            ring_ways.push(way_id);
            if refs.first() == Some(&end) {
                ring.extend(refs.into_iter().skip(1));
            } else if refs.last() == Some(&end) {
                refs.reverse();
                ring.extend(refs.into_iter().skip(1));
            } else if refs.last() == Some(&start) {
                refs.pop();
                refs.extend(ring);
                ring = refs;
            } else {
                refs.reverse();
                refs.pop();
                refs.extend(ring);
                ring = refs;
            }
            attempts += 1;
        }
        if ring.len() >= 4 && ring.first() == ring.last() {
            rings.push((ring, ring_ways));
        }
    }
    rings
}

/// `(outer ring node ids, outer way ids, inner ring node ids)` —
/// a multipolygon's water geometry as assembled from OSM relations.
type RelationWaterPolygons = Vec<(Vec<i64>, Vec<i64>, Vec<Vec<i64>>)>;

fn relation_water_polygons(
    map: &sim_world::osm::OsmMap,
    relation: &sim_world::osm::OsmRelation,
) -> RelationWaterPolygons {
    let outers = relation_rings(map, relation, "outer");
    let expected_inner_ways = relation
        .members
        .iter()
        .filter(|member| member.member_type == "way" && member.role == "inner")
        .count();
    let inners = relation_rings(map, relation, "inner");
    let assembled_inner_ways = inners.iter().map(|(_, ways)| ways.len()).sum::<usize>();
    if outers.is_empty() || assembled_inner_ways != expected_inner_ways {
        return Vec::new();
    }
    let mut polygons: Vec<_> = outers
        .into_iter()
        .map(|(ring, ways)| (ring, ways, Vec::new()))
        .collect();
    for (hole, _) in inners {
        let Some(point) = hole
            .first()
            .and_then(|id| map.nodes.get(id))
            .map(|node| (node.lat, node.lon))
        else {
            return Vec::new();
        };
        let Some((_, _, holes)) = polygons.iter_mut().find(|(outer, _, _)| {
            resolve_osm_refs(map, outer).is_some_and(|ring| point_in_geographic_ring(point, &ring))
        }) else {
            // A missing or unassigned inner ring would silently fill land.
            return Vec::new();
        };
        holes.push(hole);
    }
    polygons
}

fn resolve_osm_refs(map: &sim_world::osm::OsmMap, refs: &[i64]) -> Option<Vec<(f64, f64)>> {
    refs.iter()
        .map(|reference| map.nodes.get(reference).map(|node| (node.lat, node.lon)))
        .collect()
}

fn point_in_geographic_ring(point: (f64, f64), ring: &[(f64, f64)]) -> bool {
    if ring.len() < 4 {
        return false;
    }
    let (lat, lon) = point;
    let mut inside = false;
    let mut previous = ring.len() - 1;
    for current in 0..ring.len() {
        let (lat_a, lon_a) = ring[current];
        let (lat_b, lon_b) = ring[previous];
        if (lat_a > lat) != (lat_b > lat)
            && lon < (lon_b - lon_a) * (lat - lat_a) / (lat_b - lat_a) + lon_a
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

/// Mutable triangulation target shared by the water-polygon builder.
struct WaterMesh<'a> {
    triangulator: &'a mut earcut::Earcut<f64>,
    vertices: &'a mut Vec<GpuVertex>,
    indices: &'a mut Vec<u32>,
}

fn append_water_polygon(
    outer: &[(f64, f64)],
    holes: &[Vec<(f64, f64)>],
    frame: &EnuFrame,
    local: &impl Fn(f64, f64) -> Vec3,
    sea_level: &impl Fn(f64, f64) -> Vec3,
    mesh: &mut WaterMesh<'_>,
) {
    if outer.len() < 4 || outer.first() != outer.last() {
        return;
    }
    if holes
        .iter()
        .any(|hole| hole.len() < 4 || hole.first() != hole.last())
    {
        return;
    }
    let outer = &outer[..outer.len() - 1];
    let holes: Vec<&[(f64, f64)]> = holes.iter().map(|hole| &hole[..hole.len() - 1]).collect();
    let mut all_points = outer.to_vec();
    let mut hole_starts = Vec::with_capacity(holes.len());
    for hole in &holes {
        hole_starts.push(all_points.len());
        all_points.extend_from_slice(hole);
    }
    let terrain_points: Vec<Vec3> = all_points
        .iter()
        .map(|&(lat, lon)| local(lat, lon))
        .collect();
    let Some(triangles) =
        triangulate_polygon_with_holes(&terrain_points, &hole_starts, mesh.triangulator)
    else {
        return;
    };
    let surface_alt_m = outer
        .iter()
        .zip(&terrain_points)
        .map(|(&(lat, lon), terrain_point)| terrain_point.z - sea_level(lat, lon).z)
        .sum::<f32>()
        / outer.len() as f32
        + 0.75;
    let first = mesh.vertices.len() as u32;
    mesh.vertices.extend(all_points.iter().map(|&(lat, lon)| {
        let p = frame
            .ecef_to_enu(&WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(
                lat,
                lon,
                f64::from(surface_alt_m),
            )))
            .as_dvec3()
            .as_vec3();
        water_vertex(p)
    }));
    mesh.indices.extend(triangles.into_iter().map(|i| first + i));
}

fn water_vertex(position: Vec3) -> GpuVertex {
    GpuVertex {
        position: position.to_array(),
        normal: Vec3::Z.to_array(),
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv: [position.x * 0.0005, position.y * 0.0005],
    }
}

/// Triangulate an outer ring and zero or more inner rings in ENU XY.
fn triangulate_polygon_with_holes(
    points: &[Vec3],
    hole_starts: &[usize],
    triangulator: &mut earcut::Earcut<f64>,
) -> Option<Vec<u32>> {
    if points.len() < 3
        || hole_starts
            .iter()
            .any(|&index| index < 3 || index >= points.len())
        || hole_starts.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let outer_end = hole_starts.first().copied().unwrap_or(points.len());
    if outer_end < 3 {
        return None;
    }
    let outer = &points[..outer_end];
    let signed_area = outer
        .iter()
        .zip(outer.iter().cycle().skip(1))
        .take(outer.len())
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f32>()
        * 0.5;
    if !signed_area.is_finite() || signed_area.abs() < 1e-3 {
        return None;
    }
    let coords: Vec<[f64; 2]> = points
        .iter()
        .map(|point| [f64::from(point.x), f64::from(point.y)])
        .collect();
    let mut triangles = Vec::new();
    triangulator.earcut(coords, hole_starts, &mut triangles);
    (!triangles.is_empty()
        && triangles.len() % 3 == 0
        && triangles.iter().all(|&index| index < points.len()))
    .then(|| triangles.into_iter().map(|index| index as u32).collect())
}

fn load_terrarium_terrain(
    root: &Path,
    id: sim_geo::quadtree::TileId,
    lat: f64,
    lon: f64,
    grid_size: u32,
) -> Result<TerrainMesh, String> {
    const ZOOM: u8 = 13;
    let n = 1u32 << ZOOM;
    let center_x = (((lon.to_degrees() + 180.0) / 360.0 * f64::from(n)).floor() as i64)
        .rem_euclid(n as i64) as u32;
    let center_y = (((1.0 - lat.tan().asinh() / std::f64::consts::PI) * 0.5 * f64::from(n)).floor()
        as i64)
        .clamp(0, n as i64 - 1) as u32;
    let mut tiles = std::collections::HashMap::new();
    for dy in -1i64..=1 {
        let y = center_y as i64 + dy;
        if !(0..n as i64).contains(&y) {
            continue;
        }
        for dx in -1i64..=1 {
            let x = (center_x as i64 + dx).rem_euclid(n as i64) as u32;
            let path = root
                .join("terrarium")
                .join(ZOOM.to_string())
                .join(x.to_string())
                .join(format!("{y}.png"));
            match std::fs::read(&path) {
                Ok(bytes) => {
                    tiles.insert((x, y as u32), bytes);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("read Terrarium tile {}: {e}", path.display())),
            }
        }
    }
    TerrainMesh::from_terrarium_xyz(id, ZOOM, &tiles, grid_size).map_err(|e| {
        format!(
            "build Terrarium elevation mesh from {}/terrarium/{ZOOM}: {e}",
            root.display()
        )
    })
}

fn terrain_patch_texture(
    tiles: &[&TerrainMesh],
    root_ecef: DVec3,
    root_rotation: DQuat,
) -> (DecodedImage, (f64, f64, f64, f64)) {
    let placements: Vec<_> = tiles
        .iter()
        .map(|tile| {
            let rotation = EnuFrame::new(tile.id.center_lat_lon(), WGS84).enu_identity_ecef_quat();
            (
                *tile,
                root_rotation.inverse() * rotation,
                root_rotation.inverse() * (tile.origin_ecef.as_dvec3() - root_ecef),
            )
        })
        .collect();
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (tile, rotation, offset) in &placements {
        for vertex in &tile.vertices {
            let p = *offset + *rotation * vertex.position_enu;
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }
    let range_x = (max_x - min_x).max(1.0);
    let range_y = (max_y - min_y).max(1.0);
    let size = TERRAIN_TEXTURE_SIZE as usize;
    let mut heights = vec![0.0f32; size * size];
    let mut light = vec![0.0f32; size * size];
    let mut samples = vec![0u8; size * size];

    for (tile, rotation, offset) in &placements {
        let (mut tile_min_x, mut tile_max_x, mut tile_min_y, mut tile_max_y) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for vertex in &tile.vertices {
            let p = *offset + *rotation * vertex.position_enu;
            tile_min_x = tile_min_x.min(p.x);
            tile_max_x = tile_max_x.max(p.x);
            tile_min_y = tile_min_y.min(p.y);
            tile_max_y = tile_max_y.max(p.y);
        }
        let cell_px = ((tile_max_x - tile_min_x) / f64::from(tile.width - 1) / range_x
            * size as f64)
            .max((tile_max_y - tile_min_y) / f64::from(tile.height - 1) / range_y * size as f64);
        let radius = (cell_px * 0.72).ceil().clamp(1.0, 3.0) as i32;
        for row in 0..tile.height {
            for col in 0..tile.width {
                let vertex = &tile.vertices[(row * tile.width + col) as usize];
                let p = *offset + *rotation * vertex.position_enu;
                let px = ((p.x - min_x) / range_x * (size - 1) as f64).round() as i32;
                let py = ((max_y - p.y) / range_y * (size - 1) as f64).round() as i32;
                let Some(elevation) = tile.elevation_at_grid(row, col) else {
                    continue;
                };
                let normal = (*rotation * vertex.normal_enu).normalize_or_zero();
                let illumination = (0.60
                    + 0.40
                        * normal
                            .dot(DVec3::new(0.36, 0.43, 0.83).normalize())
                            .max(0.0)) as f32;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy > radius * radius + 1 {
                            continue;
                        }
                        let x = px + dx;
                        let y = py + dy;
                        if x < 0 || y < 0 || x >= size as i32 || y >= size as i32 {
                            continue;
                        }
                        let i = y as usize * size + x as usize;
                        let count = u16::from(samples[i]);
                        heights[i] = (heights[i] * count as f32 + elevation) / (count + 1) as f32;
                        light[i] = (light[i] * count as f32 + illumination) / (count + 1) as f32;
                        samples[i] = samples[i].saturating_add(1);
                    }
                }
            }
        }
    }

    // Fill isolated raster gaps from neighboring terrain samples so the patch
    // remains continuous when low-resolution regional tiles meet.
    for _ in 0..5 {
        let old_heights = heights.clone();
        let old_light = light.clone();
        let old_samples = samples.clone();
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                let i = y * size + x;
                if old_samples[i] != 0 {
                    continue;
                }
                let neighbors = [i - 1, i + 1, i - size, i + size];
                let mut total_h = 0.0;
                let mut total_l = 0.0;
                let mut count = 0.0;
                for j in neighbors {
                    if old_samples[j] != 0 {
                        total_h += old_heights[j];
                        total_l += old_light[j];
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    heights[i] = total_h / count;
                    light[i] = total_l / count;
                    samples[i] = 1;
                }
            }
        }
    }

    let mut rgba = Vec::with_capacity(size * size * 4);
    for i in 0..size * size {
        let (mut r, mut g, mut b) = terrain_elevation_color(heights[i]);
        let illumination = if samples[i] == 0 { 0.78 } else { light[i] };
        let grain = ((i.wrapping_mul(1_103_515_245).wrapping_add(12_345) >> 16) % 13) as f32 - 6.0;
        for channel in [&mut r, &mut g, &mut b] {
            *channel = ((*channel as f32 * illumination + grain).round()).clamp(0.0, 255.0) as u8;
        }
        rgba.extend_from_slice(&[r, g, b, 255]);
    }
    (
        DecodedImage {
            width: TERRAIN_TEXTURE_SIZE,
            height: TERRAIN_TEXTURE_SIZE,
            rgba,
        },
        (min_x, max_x, min_y, max_y),
    )
}

fn terrain_elevation_color(elevation_m: f32) -> (u8, u8, u8) {
    if elevation_m < 0.0 {
        (34, 67, 83)
    } else if elevation_m < 20.0 {
        (38, 74, 42)
    } else if elevation_m < 80.0 {
        (65, 98, 49)
    } else if elevation_m < 200.0 {
        (111, 104, 72)
    } else {
        (133, 126, 101)
    }
}

fn web_mercator_tile(lat: f64, lon: f64, n: u32) -> (u32, u32) {
    let x = ((lon.to_degrees() + 180.0) / 360.0 * f64::from(n)).floor() as i64;
    let y = ((1.0 - lat.tan().asinh() / std::f64::consts::PI) * 0.5 * f64::from(n)).floor() as i64;
    (
        x.rem_euclid(n as i64) as u32,
        y.clamp(0, n as i64 - 1) as u32,
    )
}

impl App {
    /// Build the windowed application, seeding the ECS with one
    /// aircraft entity (driven by the physics snapshot) and one chase
    /// camera.
    pub fn new(
        input: WindowInput,
        core: &SimCore,
        sky_sun_enu: DVec3,
        model: Option<sim_render::GltfModel>,
        terrain: Option<TerrainAsset>,
        mut ecs: Ecs,
    ) -> Self {
        spawn_aircraft_and_camera(&mut ecs);

        // The clean-room C172 geometry is a built-in asset, so a fresh
        // checkout has a visible aircraft without requiring a download.
        let mut model = model.unwrap_or_else(cleanroom_c172_model);
        let aircraft_instances = 0..model.instances.len();
        let aircraft_base_worlds = model.instances.iter().map(|i| i.world).collect();
        let (cockpit_instances, cockpit_digit_instances, cockpit_glyph_meshes) =
            add_cockpit_instruments(&mut model);
        let (terrain_instances, terrain_root) = if let Some(terrain) = terrain {
            let root = (terrain.mesh.origin_ecef.as_dvec3(), terrain.enu_to_ecef);
            let tiles: Vec<&TerrainMesh> = iter::once(&terrain.mesh)
                .chain(terrain.regional_meshes.iter())
                .collect();
            let (albedo, terrain_bounds) = terrain_patch_texture(&tiles, root.0, root.1);
            let albedo_index = model.images.len();
            model.images.push(albedo);
            let first_instance = model.instances.len();
            let mesh_index = model.meshes.len();
            let material_index = model.materials.len();
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            for tile in iter::once(&terrain.mesh).chain(terrain.regional_meshes.iter()) {
                let tile_rotation =
                    EnuFrame::new(tile.id.center_lat_lon(), WGS84).enu_identity_ecef_quat();
                let to_root = root.1.inverse() * tile_rotation;
                let offset = root.1.inverse() * (tile.origin_ecef.as_dvec3() - root.0);
                let first_vertex = vertices.len() as u32;
                vertices.extend(tile.vertices.iter().map(|v| {
                    let position = offset + to_root * v.position_enu;
                    let normal = (to_root * v.normal_enu).normalize_or_zero();
                    GpuVertex {
                        position: position.as_vec3().to_array(),
                        normal: normal.as_vec3().to_array(),
                        tangent: [1.0, 0.0, 0.0, 1.0],
                        uv: [
                            ((position.x - terrain_bounds.0)
                                / (terrain_bounds.1 - terrain_bounds.0).max(1.0))
                                as f32,
                            ((terrain_bounds.3 - position.y)
                                / (terrain_bounds.3 - terrain_bounds.2).max(1.0))
                                as f32,
                        ],
                    }
                }));
                indices.extend(tile.indices.iter().map(|index| first_vertex + index));
            }
            model.meshes.push(GltfMesh { vertices, indices });
            model.materials.push(GltfMaterial {
                name: "Elevation terrain".into(),
                base_color_factor: [0.70, 0.80, 0.64, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                double_sided: true,
                base_color_texture: Some(albedo_index),
                ..GltfMaterial::default()
            });
            model.instances.push(GltfInstance {
                node_name: None,
                mesh: mesh_index,
                material_override: Some(material_index),
                world: Mat4::IDENTITY,
            });
            for (mesh, material) in terrain.osm_overlay {
                let mesh_index = model.meshes.len();
                let material_index = model.materials.len();
                model.meshes.push(mesh);
                model.materials.push(material);
                model.instances.push(GltfInstance {
                    node_name: None,
                    mesh: mesh_index,
                    material_override: Some(material_index),
                    world: Mat4::IDENTITY,
                });
            }
            (Some(first_instance..model.instances.len()), Some(root))
        } else {
            (None, None)
        };
        let model = Some(model);

        Self {
            input,
            bridge: core.bridge,
            sky_sun_enu,
            model,
            aircraft_instances,
            aircraft_base_worlds,
            cockpit_instances,
            cockpit_digit_instances,
            cockpit_glyph_meshes,
            terrain_instances,
            terrain_root,
            ecs,
            resources: None,
            start: Instant::now(),
            last_title_update: Instant::now(),
            pending_resize: None,
            camera_yaw: 0.0,
            camera_pitch: (4.0f64 / 212.0f64.sqrt()).asin(),
            camera_distance: 212.0f64.sqrt(),
            cockpit_view: false,
            orbit_dragging: false,
            last_cursor: None,
        }
    }

    /// Run the windowed event loop (blocks until the window closes).
    pub fn run(mut self) -> Result<(), String> {
        let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
        event_loop
            .run_app(&mut self)
            .map_err(|e| format!("run_app: {e}"))?;
        Ok(())
    }

    fn render_frame(&mut self) -> Result<(), String> {
        let snap = self
            .input
            .snapshot
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if self.last_title_update.elapsed() >= std::time::Duration::from_millis(500) {
            let autopilot = self
                .input
                .command
                .lock()
                .map(|command| command.autopilot.is_some_and(|mode| mode != ApMode::Off))
                .unwrap_or(false);
            if let Some(resources) = &self.resources {
                resources.window.set_title(&flight_window_title(
                    &snap,
                    autopilot,
                    self.cockpit_view,
                ));
            }
            self.last_title_update = Instant::now();
        }

        // ECS systems for this frame
        system_sync_aircraft(&mut self.ecs, &snap);
        system_camera_follow(&mut self.ecs);

        let cam_entity = self.ecs.camera_entities().first().copied();
        let (mut cam_pos_enu, mut cam_quat_enu) = cam_entity
            .and_then(|c| {
                self.ecs
                    .transform
                    .get(c)
                    .map(|t| (t.position, t.orientation))
            })
            .unwrap_or((snap.position_enu, snap.orientation_enu));
        if let Some(aircraft) = self.ecs.aircraft_entities().first().copied() {
            if let Some(aircraft_pose) = self.ecs.transform.get(aircraft) {
                if self.cockpit_view {
                    (cam_pos_enu, cam_quat_enu) = cockpit_camera_pose(
                        aircraft_pose.position,
                        aircraft_pose.orientation,
                        self.camera_yaw,
                        self.camera_pitch,
                    );
                } else {
                    let (offset, camera_orientation) =
                        orbit_camera_pose(self.camera_yaw, self.camera_pitch, self.camera_distance);
                    cam_pos_enu = aircraft_pose.position + aircraft_pose.orientation * offset;
                    cam_quat_enu = aircraft_pose.orientation * camera_orientation;
                }
            }
        }
        let (camera_ecef, camera_quat) = camera_ecef_pose(&self.bridge, cam_pos_enu, cam_quat_enu);

        let model = self
            .model
            .as_mut()
            .ok_or("aircraft model not initialised")?;
        update_c172_animation(
            &mut model.instances[self.aircraft_instances.clone()],
            &self.aircraft_base_worlds,
            snap.controls,
            snap.throttle,
            self.start.elapsed().as_secs_f64(),
        );
        let readouts = cockpit_readout_chars(&snap);
        for (&instance, character) in self.cockpit_digit_instances.iter().zip(readouts) {
            if let Some(&mesh) = self.cockpit_glyph_meshes.get(&character) {
                model.instances[instance].mesh = mesh;
            }
        }
        let mut draws = system_build_draws_range(
            &self.ecs,
            &self.bridge,
            camera_ecef,
            camera_quat,
            model,
            self.aircraft_instances.clone(),
        );
        if self.cockpit_view {
            draws.extend(system_build_draws_range(
                &self.ecs,
                &self.bridge,
                camera_ecef,
                camera_quat,
                model,
                self.cockpit_instances.clone(),
            ));
        }
        if let (Some(instances), Some((root_ecef, root_rotation))) =
            (&self.terrain_instances, self.terrain_root)
        {
            let origin = FloatingOrigin::new(Ecef::from(camera_ecef), camera_quat);
            draws.extend(Renderer::draws_from_gltf_range(
                model,
                instances.clone(),
                &origin,
                root_ecef,
                root_rotation,
                DVec3::ONE,
            ));
        }

        let time_s = self.start.elapsed().as_secs_f64();
        let mut frame = build_frame_input(
            &self.bridge,
            camera_ecef,
            camera_quat,
            &snap,
            self.sky_sun_enu,
            time_s,
        );
        frame.draws = &draws;

        let res = self
            .resources
            .as_ref()
            .ok_or("frame resources not initialised")?;
        let surface_tex = match res.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                res.surface
                    .configure(&res.renderer.device_handle(), &res.surface_config);
                return Ok(());
            }
            Err(e) => return Err(format!("surface: {e}")),
        };
        let view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        res.renderer
            .render(&view, res.surface_config.format, &frame)
            .map_err(|e| format!("render: {e}"))?;
        res.window.pre_present_notify();
        surface_tex.present();
        Ok(())
    }

    fn apply_resize(&mut self, width: u32, height: u32) {
        let Some(res) = self.resources.as_mut() else {
            self.pending_resize = Some((width, height));
            return;
        };
        res.surface_config.width = width.max(1);
        res.surface_config.height = height.max(1);
        res.surface
            .configure(&res.renderer.device_handle(), &res.surface_config);
        res.renderer.resize(width, height);
    }

    /// Fold a key event into the shared control command.
    ///
    /// Returns the *new* command state for a handled key: held keys
    /// deflect, released keys spring back to neutral. The update is
    /// written through to [`WindowInput::command`] inside the same lock
    /// that read it, so nothing can be dropped between "computed" and
    /// "written".
    fn handle_key(&self, key: &Key, pressed: bool) -> Option<ControlCommand> {
        let binding = match key {
            Key::Named(NamedKey::ArrowUp) => ControlBinding::PitchUp,
            Key::Named(NamedKey::ArrowDown) => ControlBinding::PitchDown,
            Key::Named(NamedKey::ArrowLeft) => ControlBinding::RollLeft,
            Key::Named(NamedKey::ArrowRight) => ControlBinding::RollRight,
            Key::Character(c) => ControlBinding::from_str(c.as_str())?,
            _ => return None,
        };
        self.apply_binding(binding, pressed)
    }

    /// Physical-position key path (`KeyCode`). winit always populates
    /// `physical_key`, so this reaches the bindings even when the
    /// layout turns W/S/A/D into different characters.
    fn handle_physical_key(&self, physical: &PhysicalKey, pressed: bool) -> Option<ControlCommand> {
        let PhysicalKey::Code(code) = physical else {
            return None;
        };
        let binding = ControlBinding::from_code(*code)?;
        self.apply_binding(binding, pressed)
    }

    /// Apply `binding` under a single `self.input.command` lock.
    ///
    /// The mutex is recovered from poisoning (a past panic can never
    /// brick the keyboard) and the read-modify-write is atomic, so the
    /// physics thread's 1 kHz read always observes a consistent state.
    fn apply_binding(&self, binding: ControlBinding, pressed: bool) -> Option<ControlCommand> {
        let mut guard = self.input.command.lock().unwrap_or_else(|e| e.into_inner());
        let mut cmd = *guard;
        let max = crate::sim::MAX_ELEVATOR_RAD;
        let ail = crate::sim::MAX_AILERON_RAD;
        let rud = crate::sim::MAX_RUDDER_RAD;
        match binding {
            ControlBinding::PitchUp => cmd.elevator_rad = if pressed { -max } else { 0.0 },
            ControlBinding::PitchDown => cmd.elevator_rad = if pressed { max } else { 0.0 },
            ControlBinding::RollLeft => cmd.aileron_rad = if pressed { -ail } else { 0.0 },
            ControlBinding::RollRight => cmd.aileron_rad = if pressed { ail } else { 0.0 },
            ControlBinding::YawLeft => {
                cmd.rudder_rad = if pressed { -rud } else { 0.0 };
                cmd.steering = if pressed { -1.0 } else { 0.0 };
            }
            ControlBinding::YawRight => {
                cmd.rudder_rad = if pressed { rud } else { 0.0 };
                cmd.steering = if pressed { 1.0 } else { 0.0 };
            }
            ControlBinding::Brake => cmd.brake = if pressed { 1.0 } else { 0.0 },
            ControlBinding::ThrottleUp => {
                if !pressed {
                    return None;
                }
                cmd.throttle = (cmd.throttle + THROTTLE_STEP).clamp(0.0, 1.0);
            }
            ControlBinding::ThrottleDown => {
                if !pressed {
                    return None;
                }
                cmd.throttle = (cmd.throttle - THROTTLE_STEP).clamp(0.0, 1.0);
            }
            ControlBinding::Flaps => {
                if !pressed {
                    return None;
                }
                cmd.flap_step = (cmd.flap_step + 1) % 4;
            }
            ControlBinding::ApToggle => {
                if !pressed {
                    return None;
                }
                if cmd.autopilot.is_some() {
                    cmd.autopilot = None;
                    cmd.ap_altitude_ft = None;
                    cmd.ap_heading_deg = None;
                } else {
                    let snap = self.input.snapshot.read().unwrap_or_else(|e| e.into_inner());
                    cmd.autopilot = Some(ApMode::AltitudeHold);
                    cmd.ap_altitude_ft = Some(snap.altitude_m * 3.280_839_895);
                    cmd.ap_heading_deg = None;
                }
            }
        }
        *guard = cmd;
        Some(cmd)
    }
}

/// Build a small original, untextured C172-like reference airframe.
/// Dimensions are metres and the body convention is +X forward, +Y
/// right, +Z down, matching the flight model and ECS.
fn cleanroom_c172_model() -> GltfModel {
    let materials = vec![
        GltfMaterial {
            name: "ivory paint".into(),
            base_color_factor: [0.86, 0.88, 0.82, 1.0],
            metallic_factor: 0.05,
            roughness_factor: 0.55,
            double_sided: true,
            ..GltfMaterial::default()
        },
        GltfMaterial {
            name: "blue glazing and trim".into(),
            base_color_factor: [0.035, 0.12, 0.19, 1.0],
            metallic_factor: 0.22,
            roughness_factor: 0.24,
            double_sided: true,
            ..GltfMaterial::default()
        },
        GltfMaterial {
            name: "rubber and propeller".into(),
            base_color_factor: [0.045, 0.05, 0.055, 1.0],
            metallic_factor: 0.0,
            roughness_factor: 0.82,
            double_sided: true,
            ..GltfMaterial::default()
        },
        GltfMaterial {
            name: "aluminum landing gear".into(),
            base_color_factor: [0.56, 0.59, 0.62, 1.0],
            metallic_factor: 0.72,
            roughness_factor: 0.32,
            double_sided: true,
            ..GltfMaterial::default()
        },
    ];

    let parts = vec![
        (fuselage_mesh(), 0usize),
        (airfoil_wing_mesh(11.0, 1.75, -0.15, -0.02), 0),
        (airfoil_wing_mesh(3.4, 0.9, -2.25, -0.04), 0),
        (box_mesh([-2.95, -0.10, -1.35], [-2.15, 0.10, 0.02]), 0),
        (ellipsoid_mesh([0.95, 0.0, -0.28], [0.82, 0.31, 0.32]), 1),
        (ellipsoid_mesh([3.20, 0.0, 0.0], [0.22, 0.13, 0.13]), 2),
        (propeller_blades_mesh(), 2),
        (box_mesh([-0.70, -0.58, 0.20], [-0.52, -0.48, 0.98]), 3),
        (box_mesh([-0.70, 0.48, 0.20], [-0.52, 0.58, 0.98]), 3),
        (ellipsoid_mesh([-0.60, -0.87, 1.0], [0.30, 0.13, 0.30]), 2),
        (ellipsoid_mesh([-0.60, 0.87, 1.0], [0.30, 0.13, 0.30]), 2),
        (box_mesh([2.22, -0.07, 0.18], [2.38, 0.07, 0.90]), 3),
        (ellipsoid_mesh([2.40, 0.0, 0.96], [0.23, 0.13, 0.25]), 2),
    ];
    let (meshes, part_materials): (Vec<GltfMesh>, Vec<usize>) = parts.into_iter().unzip();
    let instances = part_materials
        .iter()
        .enumerate()
        .map(|(mesh, material)| GltfInstance {
            node_name: None,
            mesh,
            material_override: Some(*material),
            world: Mat4::IDENTITY,
        })
        .collect();
    GltfModel {
        meshes,
        materials,
        images: Vec::new(),
        samplers: Vec::new(),
        instances,
        default_scene: 0,
    }
}

fn fuselage_mesh() -> GltfMesh {
    let stations = [
        (-3.2, 0.045, 0.06),
        (-2.85, 0.17, 0.18),
        (-2.2, 0.28, 0.27),
        (-1.0, 0.35, 0.32),
        (0.5, 0.37, 0.33),
        (1.8, 0.29, 0.27),
        (2.7, 0.18, 0.19),
        (3.25, 0.08, 0.10),
    ];
    let mut mesh = GltfMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    const SLICES: usize = 16;
    for pair in stations.windows(2) {
        for slice in 0..SLICES {
            let a0 = std::f32::consts::TAU * slice as f32 / SLICES as f32;
            let a1 = std::f32::consts::TAU * (slice + 1) as f32 / SLICES as f32;
            let point = |station: (f32, f32, f32), angle: f32| {
                Vec3::new(station.0, station.1 * angle.cos(), station.2 * angle.sin())
            };
            let mid = (a0 + a1) * 0.5;
            push_airframe_quad(
                &mut mesh,
                [
                    point(pair[0], a0),
                    point(pair[0], a1),
                    point(pair[1], a1),
                    point(pair[1], a0),
                ],
                Vec3::new(0.08, mid.cos(), mid.sin()).normalize(),
            );
        }
    }
    mesh
}

fn airfoil_wing_mesh(span: f32, root_chord: f32, leading_x: f32, center_z: f32) -> GltfMesh {
    let stations = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let chord_steps = 12usize;
    let mut mesh = GltfMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for side in [-1.0f32, 1.0] {
        for pair in stations.windows(2) {
            for step in 0..chord_steps {
                let u0 = step as f32 / chord_steps as f32;
                let u1 = (step + 1) as f32 / chord_steps as f32;
                for upper in [true, false] {
                    let point = |t: f32, u: f32| {
                        let chord = root_chord * (1.0 - 0.55 * t);
                        let thickness = 5.0
                            * 0.12
                            * chord
                            * (0.2969 * u.sqrt() - 0.1260 * u - 0.3516 * u * u
                                + 0.2843 * u.powi(3)
                                - 0.1015 * u.powi(4))
                            .max(0.0);
                        Vec3::new(
                            leading_x - 0.25 * t - chord * u,
                            side * span * 0.5 * t,
                            center_z + if upper { -thickness } else { thickness },
                        )
                    };
                    let normal = if upper { Vec3::NEG_Z } else { Vec3::Z };
                    let mut quad = [
                        point(pair[0], u0),
                        point(pair[0], u1),
                        point(pair[1], u1),
                        point(pair[1], u0),
                    ];
                    if side < 0.0 {
                        quad.swap(1, 3);
                    }
                    if !upper {
                        quad.swap(1, 3);
                    }
                    push_airframe_quad(&mut mesh, quad, normal);
                }
            }
        }
    }
    mesh
}

fn push_airframe_quad(mesh: &mut GltfMesh, corners: [Vec3; 4], normal: Vec3) {
    let base = mesh.vertices.len() as u32;
    for (index, position) in corners.into_iter().enumerate() {
        mesh.vertices.push(GpuVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv: [
                (index == 1 || index == 2) as u8 as f32,
                (index >= 2) as u8 as f32,
            ],
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn ellipsoid_mesh(center: [f32; 3], radii: [f32; 3]) -> GltfMesh {
    const LONGITUDES: usize = 16;
    const LATITUDES: usize = 8;
    let mut mesh = GltfMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for lat in 0..=LATITUDES {
        let phi = std::f32::consts::PI * lat as f32 / LATITUDES as f32;
        for lon in 0..=LONGITUDES {
            let theta = std::f32::consts::TAU * lon as f32 / LONGITUDES as f32;
            let unit = Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
            let normal = (unit / Vec3::from_array(radii)).normalize_or_zero();
            mesh.vertices.push(GpuVertex {
                position: (Vec3::from_array(center) + unit * Vec3::from_array(radii)).to_array(),
                normal: normal.to_array(),
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: [
                    lon as f32 / LONGITUDES as f32,
                    lat as f32 / LATITUDES as f32,
                ],
            });
        }
    }
    for lat in 0..LATITUDES {
        for lon in 0..LONGITUDES {
            let a = (lat * (LONGITUDES + 1) + lon) as u32;
            let b = a + LONGITUDES as u32 + 1;
            mesh.indices
                .extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    mesh
}

fn propeller_blades_mesh() -> GltfMesh {
    let axis = Vec3::new(3.18, 0.0, 0.0);
    let mut combined = GltfMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for angle in [0.28, 0.28 + std::f32::consts::PI] {
        let rotation = Quat::from_rotation_x(angle);
        let mut blade = ellipsoid_mesh([3.18, 0.0, 0.55], [0.055, 0.13, 0.53]);
        let base = combined.vertices.len() as u32;
        for vertex in &mut blade.vertices {
            vertex.position =
                (axis + rotation * (Vec3::from_array(vertex.position) - axis)).to_array();
            vertex.normal = (rotation * Vec3::from_array(vertex.normal)).to_array();
        }
        combined.vertices.extend(blade.vertices);
        combined
            .indices
            .extend(blade.indices.into_iter().map(|index| index + base));
    }
    combined
}

fn box_mesh(min: [f32; 3], max: [f32; 3]) -> GltfMesh {
    let p = [
        Vec3::new(min[0], min[1], min[2]),
        Vec3::new(max[0], min[1], min[2]),
        Vec3::new(min[0], max[1], min[2]),
        Vec3::new(max[0], max[1], min[2]),
        Vec3::new(min[0], min[1], max[2]),
        Vec3::new(max[0], min[1], max[2]),
        Vec3::new(min[0], max[1], max[2]),
        Vec3::new(max[0], max[1], max[2]),
    ];
    let faces: [([usize; 4], Vec3); 6] = [
        ([0, 2, 3, 1], Vec3::NEG_Z),
        ([4, 5, 7, 6], Vec3::Z),
        ([0, 1, 5, 4], Vec3::NEG_Y),
        ([2, 6, 7, 3], Vec3::Y),
        ([0, 4, 6, 2], Vec3::NEG_X),
        ([1, 3, 7, 5], Vec3::X),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (corners, normal) in faces {
        let base = vertices.len() as u32;
        for (uv, corner) in corners.into_iter().enumerate() {
            let uv = match uv {
                0 => [0.0, 0.0],
                1 => [1.0, 0.0],
                2 => [1.0, 1.0],
                _ => [0.0, 1.0],
            };
            vertices.push(GpuVertex {
                position: p[corner].to_array(),
                normal: normal.to_array(),
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    GltfMesh { vertices, indices }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.resources.is_some() {
            return;
        }
        let result = block_on_future(self.init_surface(event_loop));
        if let Err(e) = result {
            eprintln!("window init failed: {e}");
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.apply_resize(size.width, size.height),
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left | MouseButton::Right,
                ..
            } => {
                self.orbit_dragging = state == ElementState::Pressed;
                self.last_cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor = (position.x, position.y);
                if self.orbit_dragging {
                    if let Some((last_x, last_y)) = self.last_cursor {
                        (self.camera_yaw, self.camera_pitch) = orbit_camera_drag(
                            self.camera_yaw,
                            self.camera_pitch,
                            cursor.0 - last_x,
                            cursor.1 - last_y,
                        );
                    }
                }
                self.last_cursor = Some(cursor);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                self.camera_distance = zoom_camera_distance(self.camera_distance, amount);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && event.state == ElementState::Pressed
                {
                    event_loop.exit();
                    return;
                }
                if event.state == ElementState::Pressed {
                    if let Key::Character(character) = &event.logical_key {
                        if character.as_str().eq_ignore_ascii_case("c") {
                            self.cockpit_view = !self.cockpit_view;
                            return;
                        }
                    }
                }
                // Flight controls: try the physical key position first
                // (layout-proof), fall back to the logical character /
                // arrow key. Both paths write the new command through
                // to the shared mutex before returning, so the physics
                // thread picks every press and release up on its next
                // 1 kHz tick.
                let pressed = event.state == ElementState::Pressed;
                if self
                    .handle_physical_key(&event.physical_key, pressed)
                    .is_none()
                {
                    let _ = self.handle_key(&event.logical_key, pressed);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some((w, h)) = self.pending_resize.take() {
                    self.apply_resize(w, h);
                }
                if let Err(e) = self.render_frame() {
                    eprintln!("render error: {e}");
                }
                if let Some(res) = self.resources.as_ref() {
                    res.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

const COCKPIT_READOUT_SCALE: f32 = 0.70;

fn add_cockpit_instruments(
    model: &mut GltfModel,
) -> (Range<usize>, Vec<usize>, HashMap<char, usize>) {
    const PANEL_FACE_X: f32 = 1.35;
    const GLYPH_FACE_X: f32 = 1.33;
    const DISPLAY_CENTER_Y: f32 = 0.0;
    const DISPLAY_CENTER_Z: f32 = -0.65;
    let first_instance = model.instances.len();
    let panel_material = model.materials.len();
    model.materials.push(GltfMaterial {
        name: "live cockpit display glass".into(),
        base_color_factor: [0.008, 0.014, 0.012, 1.0],
        metallic_factor: 0.0,
        roughness_factor: 0.95,
        double_sided: true,
        ..GltfMaterial::default()
    });
    let digits_material = model.materials.len();
    model.materials.push(GltfMaterial {
        name: "live cockpit instrument phosphor".into(),
        base_color_factor: [0.28, 1.0, 0.42, 1.0],
        emissive_factor: [0.05, 0.38, 0.10],
        metallic_factor: 0.0,
        roughness_factor: 0.55,
        double_sided: true,
        ..GltfMaterial::default()
    });

    let panel_mesh = model.meshes.len();
    let mut panel = GltfMesh::default();
    push_cockpit_quad(
        &mut panel,
        -0.235 * COCKPIT_READOUT_SCALE,
        0.235 * COCKPIT_READOUT_SCALE,
        -0.105 * COCKPIT_READOUT_SCALE,
        0.105 * COCKPIT_READOUT_SCALE,
    );
    model.meshes.push(panel);
    model.instances.push(GltfInstance {
        node_name: Some("live cockpit display".into()),
        mesh: panel_mesh,
        material_override: Some(panel_material),
        world: Mat4::from_translation(Vec3::new(PANEL_FACE_X, DISPLAY_CENTER_Y, DISPLAY_CENTER_Z)),
    });

    let mut glyph_meshes = HashMap::new();
    for character in "0123456789+-IASLTHDGV".chars() {
        let Some(pattern) = cockpit_glyph_pattern(character) else {
            continue;
        };
        let mesh_index = model.meshes.len();
        model.meshes.push(cockpit_glyph_mesh(pattern));
        glyph_meshes.insert(character, mesh_index);
    }

    let label_specs = [
        ("IAS", -0.118, -0.078),
        ("ALT", 0.095, -0.078),
        ("HDG", -0.118, 0.020),
        ("VS", 0.095, 0.020),
    ];
    for (label, center_y, z) in label_specs {
        let chars: Vec<char> = label.chars().collect();
        let pitch = 0.019 * COCKPIT_READOUT_SCALE;
        for (i, character) in chars.iter().copied().enumerate() {
            let y = center_y * COCKPIT_READOUT_SCALE
                + (i as f32 - (chars.len() - 1) as f32 * 0.5) * pitch;
            push_cockpit_glyph_instance(
                model,
                glyph_meshes[&character],
                digits_material,
                Vec3::new(
                    GLYPH_FACE_X,
                    y + DISPLAY_CENTER_Y,
                    z * COCKPIT_READOUT_SCALE + DISPLAY_CENTER_Z,
                ),
                Some(format!("cockpit label {character}")),
            );
        }
    }

    let mut digit_instances = Vec::with_capacity(16);
    for (count, center_y, z) in [
        (3usize, -0.118f32, -0.040f32),
        (5, 0.095, -0.040),
        (3, -0.118, 0.058),
        (5, 0.095, 0.058),
    ] {
        let pitch = 0.026 * COCKPIT_READOUT_SCALE;
        for i in 0..count {
            let y =
                center_y * COCKPIT_READOUT_SCALE + (i as f32 - (count - 1) as f32 * 0.5) * pitch;
            let index = model.instances.len();
            push_cockpit_glyph_instance(
                model,
                glyph_meshes[&'+'],
                digits_material,
                Vec3::new(
                    GLYPH_FACE_X,
                    y + DISPLAY_CENTER_Y,
                    z * COCKPIT_READOUT_SCALE + DISPLAY_CENTER_Z,
                ),
                Some("live flight instrument".into()),
            );
            digit_instances.push(index);
        }
    }
    (
        first_instance..model.instances.len(),
        digit_instances,
        glyph_meshes,
    )
}

fn push_cockpit_glyph_instance(
    model: &mut GltfModel,
    mesh: usize,
    material: usize,
    position: Vec3,
    name: Option<String>,
) {
    model.instances.push(GltfInstance {
        node_name: name,
        mesh,
        material_override: Some(material),
        world: Mat4::from_translation(position),
    });
}

fn cockpit_glyph_pattern(character: char) -> Option<[u8; 7]> {
    Some(match character {
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        '+' => [0, 4, 4, 31, 4, 4, 0],
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [31, 4, 4, 4, 4, 4, 31],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        _ => return None,
    })
}

fn cockpit_glyph_mesh(pattern: [u8; 7]) -> GltfMesh {
    let mut mesh = GltfMesh::default();
    let cell_w = 0.0032f32 * COCKPIT_READOUT_SCALE;
    let cell_h = 0.0042f32 * COCKPIT_READOUT_SCALE;
    let gap = 0.00055f32 * COCKPIT_READOUT_SCALE;
    for (row, bits) in pattern.into_iter().enumerate() {
        for col in 0..5 {
            if bits & (1 << (4 - col)) == 0 {
                continue;
            }
            let y0 = -2.5 * cell_w + col as f32 * cell_w + gap * 0.5;
            let z0 = -3.5 * cell_h + row as f32 * cell_h + gap * 0.5;
            push_cockpit_quad(&mut mesh, y0, y0 + cell_w - gap, z0, z0 + cell_h - gap);
        }
    }
    mesh
}

fn push_cockpit_quad(mesh: &mut GltfMesh, y0: f32, y1: f32, z0: f32, z1: f32) {
    let base = mesh.vertices.len() as u32;
    for (position, uv) in [
        ([0.0, y0, z0], [0.0, 0.0]),
        ([0.0, y1, z0], [1.0, 0.0]),
        ([0.0, y1, z1], [1.0, 1.0]),
        ([0.0, y0, z1], [0.0, 1.0]),
    ] {
        mesh.vertices.push(GpuVertex {
            position,
            normal: [-1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0, 1.0],
            uv,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
}

fn cockpit_readout_chars(snapshot: &Snapshot) -> [char; 16] {
    let ias = (snapshot.indicated_airspeed_mps * 1.943_844)
        .round()
        .clamp(0.0, 999.0) as u32;
    let altitude = (snapshot.altitude_m * 3.280_839_895)
        .round()
        .clamp(0.0, 99_999.0) as u32;
    let heading = (snapshot.heading_deg.round() as i32).rem_euclid(360) as u32;
    let vertical_speed = (snapshot.vertical_speed_mps * 196.850_394)
        .round()
        .clamp(-9_999.0, 9_999.0) as i32;
    let mut result = [' '; 16];
    let values = [
        format!("{ias:03}"),
        format!("{altitude:05}"),
        format!("{heading:03}"),
        format!(
            "{}{:04}",
            if vertical_speed < 0 { '-' } else { '+' },
            vertical_speed.unsigned_abs().min(9999)
        ),
    ]
    .concat();
    for (slot, character) in result.iter_mut().zip(values.chars()) {
        *slot = character;
    }
    result
}

/// Apply visible C172 control-surface deflections to the named glTF nodes.
fn update_c172_animation(
    instances: &mut [GltfInstance],
    base_worlds: &[Mat4],
    controls: sim_physics::ControlInputs,
    throttle: f64,
    time_s: f64,
) {
    for (instance, &base_world) in instances.iter_mut().zip(base_worlds) {
        let Some(name) = instance.node_name.as_deref() else {
            instance.world = base_world;
            continue;
        };
        let (axis, angle) = match name {
            "elevatorleft" | "elevatorright" => (Vec3::Y, controls.elevator_rad.clamp(-0.5, 0.5)),
            "leftaileron" => (Vec3::Y, controls.aileron_rad.clamp(-0.4, 0.4)),
            "rightaileron" => (Vec3::Y, -controls.aileron_rad.clamp(-0.4, 0.4)),
            "rudder_1" => (Vec3::Z, -controls.rudder_rad.clamp(-0.5, 0.5)),
            "flaps" => (Vec3::Y, controls.flap_step.min(3) as f64 * 0.18),
            "Propeller" => (Vec3::X, time_s * 260.0 * throttle.clamp(0.0, 1.0).sqrt()),
            _ => {
                instance.world = base_world;
                continue;
            }
        };
        let pivot = base_world.transform_point3(Vec3::ZERO);
        let rotation = Mat4::from_quat(Quat::from_axis_angle(axis, angle as f32));
        instance.world =
            Mat4::from_translation(pivot) * rotation * Mat4::from_translation(-pivot) * base_world;
    }
}

fn flight_window_title(snapshot: &Snapshot, autopilot: bool, cockpit_view: bool) -> String {
    format!(
        "OpenMSFS | IAS {:>3.0} kt | ALT {:>5.0} ft | HDG {:03.0}° | PITCH {:+.1}° | BANK {:+.1}° | VS {:+.0} fpm | FLAP {}/3 | THR {:>3.0}% | {} | {} | C cockpit / arrows+drag view / wheel zoom",
        snapshot.indicated_airspeed_mps * 1.943_844,
        snapshot.altitude_m * 3.280_839_895,
        snapshot.heading_deg,
        snapshot.pitch_deg,
        snapshot.roll_deg,
        snapshot.vertical_speed_mps * 196.850_394,
        snapshot.controls.flap_step.min(3),
        snapshot.throttle.clamp(0.0, 1.0) * 100.0,
        if autopilot { "AP" } else { "MAN" },
        if cockpit_view { "COCKPIT" } else { "CHASE" },
    )
}

fn orbit_camera_drag(yaw: f64, pitch: f64, delta_x: f64, delta_y: f64) -> (f64, f64) {
    (
        yaw + delta_x * 0.006,
        (pitch - delta_y * 0.006).clamp(-0.15, 1.20),
    )
}

fn zoom_camera_distance(distance: f64, scroll_amount: f64) -> f64 {
    (distance * (-scroll_amount * 0.12).exp()).clamp(5.0, 80.0)
}

fn cockpit_camera_pose(
    position: DVec3,
    orientation: DQuat,
    look_yaw: f64,
    look_pitch: f64,
) -> (DVec3, DQuat) {
    let eye_in_body = DVec3::new(0.85, 0.0, -0.85);
    let look = DQuat::from_rotation_z(look_yaw) * DQuat::from_rotation_y(-look_pitch);
    (position + orientation * eye_in_body, orientation * look)
}

/// Camera offset and orientation in aircraft body coordinates. Drag/arrows
/// orbit; the wheel changes distance. The view aims slightly below the plane
/// so regional scenery remains in frame during the default airborne start.
fn orbit_camera_pose(yaw: f64, pitch: f64, distance: f64) -> (DVec3, DQuat) {
    let horizontal = pitch.cos() * distance;
    let offset = DVec3::new(
        -horizontal * yaw.cos(),
        horizontal * yaw.sin(),
        -pitch.sin() * distance,
    );
    let forward = (-offset - DVec3::Z * (distance * 0.14)).normalize();
    let mut right = forward.cross(DVec3::NEG_Z).normalize_or_zero();
    if right.length_squared() < 0.5 {
        right = DVec3::Y;
    }
    let up = right.cross(forward).normalize();
    let down = -up;
    let orientation = DQuat::from_mat3(&DMat3::from_cols(forward, right, down));
    (offset, orientation)
}

impl App {
    /// Async surface/adapter/device setup (driven by the small
    /// park-waker executor from `resumed`).
    async fn init_surface(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window: Arc<Window> = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("OpenMSFS — arrows or drag to orbit, wheel to zoom")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
                )
                .map_err(|e| format!("create_window: {e}"))?,
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: if cfg!(target_os = "windows") {
                wgpu::Backends::DX12
            } else {
                wgpu::Backends::PRIMARY
            },
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create_surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("no wgpu adapter for the window surface")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("open-msfs-window"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = TARGET_FORMAT_PREFERENCE
            .into_iter()
            .find(|f| caps.formats.contains(f))
            .ok_or_else(|| format!("surface offers no sRGB format (got {:?})", caps.formats))?;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let mut renderer =
            Renderer::with_device(device, queue, surface_config.width, surface_config.height)
                .map_err(|e| format!("renderer: {e}"))?;
        let sky_luts =
            sim_atmosphere::sky::SkyLuts::new(&sim_atmosphere::sky::SkyParams::default());
        renderer.set_sky_luts(&sky_luts);
        let noise = sim_render::generate_noise_volume(32, 42);
        renderer.set_noise(&noise);
        if let Some(model) = self.model.as_ref() {
            renderer
                .set_model(model)
                .map_err(|e| format!("aircraft model upload: {e}"))?;
        }
        window.request_redraw();
        self.resources = Some(FrameResources {
            surface,
            surface_config,
            renderer,
            window,
        });
        Ok(())
    }
}

/// Minimal future executor for `resumed` (the winit 0.30 callback API
/// is synchronous while surface/adapter acquisition is async).
fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    use std::sync::Arc as StdArc;
    use std::task::{Context, RawWaker, RawWakerVTable, Waker};
    struct Park(std::thread::Thread);
    const VT: RawWakerVTable = RawWakerVTable::new(
        |ptr| {
            let arc = unsafe { StdArc::from_raw(ptr as *const Park) };
            let c = StdArc::into_raw(arc.clone()) as *const ();
            let _ = StdArc::into_raw(arc);
            RawWaker::new(c, &VT)
        },
        |ptr| {
            let arc = unsafe { StdArc::from_raw(ptr as *const Park) };
            arc.0.unpark();
        },
        |ptr| {
            let arc = unsafe { StdArc::from_raw(ptr as *const Park) };
            arc.0.unpark();
            let _ = StdArc::into_raw(arc);
        },
        |ptr| drop(unsafe { StdArc::from_raw(ptr as *const Park) }),
    );
    let mut fut = Box::pin(fut);
    let waker = {
        let arc = StdArc::new(Park(std::thread::current()));
        unsafe { Waker::from_raw(RawWaker::new(StdArc::into_raw(arc) as *const (), &VT)) }
    };
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => {
                std::thread::park_timeout(std::time::Duration::from_millis(5))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimConfig;

    #[test]
    fn water_meshes_fill_concave_areas_and_follow_osm_coastline_water_side() {
        let id = sim_geo::quadtree::TileId {
            face: 0,
            level: 4,
            x: 8,
            y: 8,
        };
        let center = id.center_lat_lon();
        let lat = center.latitude_rad().to_degrees();
        let lon = center.longitude_rad().to_degrees();
        let dem = sim_world::DemTile::new(id, 2, 2, vec![0.0; 4]).expect("DEM");
        let terrain = TerrainMesh::from_dem(&dem).expect("terrain");
        let frame = EnuFrame::new(center, WGS84);
        let terrain_by_id = HashMap::from([(id, &terrain)]);
        let local = |latitude: f64, longitude: f64| {
            let lat_rad = latitude.to_radians();
            let lon_rad = longitude.to_radians();
            let direction = DVec3::new(
                lat_rad.cos() * lon_rad.cos(),
                lat_rad.cos() * lon_rad.sin(),
                lat_rad.sin(),
            );
            let tile_id = tile_for_direction(direction, terrain.id.level);
            let ground = terrain_by_id
                .get(&tile_id)
                .map(|tile| tile.elevation_near(lat_rad, lon_rad))
                .unwrap_or(0.0);
            let p = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(latitude, longitude, 0.0));
            let mut enu = frame.ecef_to_enu(&p).as_dvec3();
            enu.z += ground;
            enu.as_vec3()
        };
        let xml = format!(
            r#"<osm>
              <node id="1" lat="{}" lon="{}"/><node id="2" lat="{}" lon="{}"/>
              <node id="3" lat="{}" lon="{}"/><node id="4" lat="{}" lon="{}"/>
              <node id="5" lat="{}" lon="{}"/>
              <node id="6" lat="{}" lon="{}"/><node id="7" lat="{}" lon="{}"/>
              <node id="8" lat="{}" lon="{}"/><node id="9" lat="{}" lon="{}"/>
              <node id="10" lat="{}" lon="{}"/><node id="11" lat="{}" lon="{}"/>
              <node id="12" lat="{}" lon="{}"/><node id="13" lat="{}" lon="{}"/>
              <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="5"/><nd ref="1"/><tag k="natural" v="water"/></way>
              <way id="11"><nd ref="1"/><nd ref="2"/><tag k="natural" v="coastline"/></way>
              <way id="20"><nd ref="6"/><nd ref="7"/><nd ref="8"/></way>
              <way id="21"><nd ref="9"/><nd ref="8"/></way>
              <way id="22"><nd ref="6"/><nd ref="9"/></way>
              <way id="23"><nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="13"/><nd ref="10"/></way>
              <relation id="30"><member type="way" ref="20" role="outer"/><member type="way" ref="21" role="outer"/><member type="way" ref="22" role="outer"/><member type="way" ref="23" role="inner"/><tag k="type" v="multipolygon"/><tag k="natural" v="water"/></relation>
            </osm>"#,
            lat - 0.001,
            lon - 0.001,
            lat - 0.001,
            lon + 0.001,
            lat,
            lon,
            lat + 0.001,
            lon + 0.001,
            lat + 0.001,
            lon - 0.001,
            lat - 0.002,
            lon + 0.002,
            lat - 0.002,
            lon + 0.003,
            lat,
            lon + 0.003,
            lat + 0.001,
            lon + 0.002,
            lat - 0.0015,
            lon + 0.0022,
            lat - 0.0015,
            lon + 0.0028,
            lat - 0.0005,
            lon + 0.0028,
            lat - 0.0005,
            lon + 0.0022
        );
        let map = sim_world::osm::parse_osm(&xml).expect("OSM");
        let relation_polygons = relation_water_polygons(&map, &map.relations[0]);
        assert_eq!(relation_polygons.len(), 1);
        assert_eq!(relation_polygons[0].2.len(), 1);
        let overlays = build_osm_water_overlays(&map, &frame, &local).expect("water overlays");
        assert_eq!(overlays.len(), 1);
        let (mesh, material) = &overlays[0];
        assert_eq!(material.name, "OpenStreetMap water and shoreline");
        // Concave area + holed relation multipolygon + coastline segment.
        assert_eq!(mesh.indices.len(), 39);
        assert!(mesh
            .vertices
            .iter()
            .all(|vertex| vertex.position[2].is_finite()));

        let square_with_hole = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 4.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(3.0, 3.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
        ];
        let triangles = triangulate_polygon_with_holes(
            &square_with_hole,
            &[4],
            &mut earcut::Earcut::<f64>::new(),
        )
        .expect("outer and hole ring triangulate");
        let area = triangles
            .chunks_exact(3)
            .map(|triangle| {
                let a = square_with_hole[triangle[0] as usize];
                let b = square_with_hole[triangle[1] as usize];
                let c = square_with_hole[triangle[2] as usize];
                ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() * 0.5
            })
            .sum::<f32>();
        assert!((area - 12.0).abs() < 1e-4, "hole was filled: area={area}");
    }

    #[test]
    fn orbit_camera_starts_behind_aircraft_and_tracks_it_when_rotated() {
        let distance = 212.0f64.sqrt();
        let pitch = (4.0f64 / distance).asin();
        let (offset, orientation) = orbit_camera_pose(0.0, pitch, distance);
        assert!(offset.abs_diff_eq(DVec3::new(-14.0, 0.0, -4.0), 1e-9));
        let forward = orientation * DVec3::X;
        assert!(forward.x > 0.0 && forward.z < (-offset).normalize().z - 0.04);

        let (right_offset, right_orientation) =
            orbit_camera_pose(std::f64::consts::FRAC_PI_2, pitch, distance);
        assert!(right_offset.y > 13.9);
        let right_forward = right_orientation * DVec3::X;
        assert!(right_forward.y < 0.0);
        assert!(right_forward.z < (-right_offset).normalize().z);
    }

    #[test]
    fn camera_drag_orbits_and_wheel_zoom_stays_in_range() {
        let (yaw, pitch) = orbit_camera_drag(0.0, 0.2, 100.0, 50.0);
        assert!((yaw - 0.6).abs() < 1e-12);
        assert!((pitch + 0.1).abs() < 1e-12);
        assert_eq!(orbit_camera_drag(0.0, 1.19, 0.0, -100.0).1, 1.20);
        assert_eq!(orbit_camera_drag(0.0, -0.14, 0.0, 100.0).1, -0.15);

        let distance = 14.5;
        assert!(zoom_camera_distance(distance, 1.0) < distance);
        assert!(zoom_camera_distance(distance, -1.0) > distance);
        assert_eq!(zoom_camera_distance(distance, 100.0), 5.0);
        assert_eq!(zoom_camera_distance(distance, -100.0), 80.0);
    }

    #[test]
    fn terrain_texture_palette_distinguishes_water_and_elevation_bands() {
        let water = terrain_elevation_color(-5.0);
        let lowland = terrain_elevation_color(5.0);
        let upland = terrain_elevation_color(100.0);
        assert!(water.2 > water.1);
        assert!(lowland.1 > lowland.2);
        assert_ne!(lowland, upland);
    }

    #[test]
    fn key_press_deflects_and_release_centers() {
        let input = WindowInput {
            command: Arc::new(Mutex::new(ControlCommand::default())),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let cfg = SimConfig::default();
        let core = SimCore::new(cfg).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );

        let cmd = app
            .handle_key(&Key::Character("w".into()), true)
            .expect("handled");
        assert!((cmd.elevator_rad + crate::sim::MAX_ELEVATOR_RAD).abs() < 1e-9);
        let cmd = app
            .handle_key(&Key::Character("w".into()), false)
            .expect("handled");
        assert_eq!(cmd.elevator_rad, 0.0);
        assert!(app.handle_key(&Key::Character("z".into()), true).is_none());
    }

    #[test]
    fn rudder_keys_also_steer_the_nosewheel_and_release_to_center() {
        let command = Arc::new(Mutex::new(ControlCommand::default()));
        let app = App::new(
            WindowInput {
                command: Arc::clone(&command),
                snapshot: Arc::new(RwLock::new(Snapshot::default())),
            },
            &SimCore::new(SimConfig::default()).expect("core"),
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );
        for (key, sign) in [("q", -1.0), ("e", 1.0)] {
            let pressed = app
                .handle_key(&Key::Character(key.into()), true)
                .expect("rudder key");
            assert_eq!(pressed.steering, sign);
            assert!((pressed.rudder_rad - sign * crate::sim::MAX_RUDDER_RAD).abs() < 1e-12);
            *command.lock().unwrap() = pressed;
            let released = app
                .handle_key(&Key::Character(key.into()), false)
                .expect("release rudder key");
            assert_eq!(released.steering, 0.0);
            assert_eq!(released.rudder_rad, 0.0);
            *command.lock().unwrap() = released;
        }
    }

    #[test]
    fn throttle_keys_change_and_clamp_engine_setting() {
        let command = Arc::new(Mutex::new(ControlCommand::default()));
        let input = WindowInput {
            command: Arc::clone(&command),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );
        for _ in 0..25 {
            *command.lock().unwrap() = app.handle_key(&Key::Character("r".into()), true).unwrap();
        }
        assert_eq!(command.lock().unwrap().throttle, 1.0);
        assert!(app.handle_key(&Key::Character("r".into()), false).is_none());
        for _ in 0..25 {
            *command.lock().unwrap() = app.handle_key(&Key::Character("f".into()), true).unwrap();
        }
        assert_eq!(command.lock().unwrap().throttle, 0.0);
    }

    #[test]
    fn flap_key_cycles_through_extension_and_retraction() {
        let command = Arc::new(Mutex::new(ControlCommand::default()));
        let input = WindowInput {
            command: Arc::clone(&command),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );
        for expected in [1, 2, 3, 0] {
            *command.lock().unwrap() = app
                .handle_key(&Key::Character("g".into()), true)
                .expect("flap key");
            assert_eq!(command.lock().unwrap().flap_step, expected);
        }
    }

    #[test]
    fn p_toggles_altitude_hold_and_returns_to_manual_controls() {
        let command = Arc::new(Mutex::new(ControlCommand::default()));
        let snapshot = Arc::new(RwLock::new(Snapshot::default()));
        snapshot.write().unwrap().altitude_m = 300.0;
        let input = WindowInput {
            command: Arc::clone(&command),
            snapshot,
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );

        let engaged = app
            .handle_key(&Key::Character("p".into()), true)
            .expect("handled");
        assert_eq!(engaged.autopilot, Some(ApMode::AltitudeHold));
        assert_eq!(engaged.ap_altitude_ft, Some(300.0 * 3.280_839_895));
        assert!(app.handle_key(&Key::Character("p".into()), false).is_none());
        *command.lock().unwrap() = engaged;
        let manual = app
            .handle_key(&Key::Character("p".into()), true)
            .expect("handled");
        assert!(manual.autopilot.is_none());
    }

    #[test]
    fn c172_control_inputs_animate_the_named_model_parts() {
        let names = [
            "elevatorleft",
            "elevatorright",
            "leftaileron",
            "rightaileron",
            "rudder_1",
            "flaps",
            "Propeller",
        ];
        let base = vec![Mat4::IDENTITY; names.len()];
        let mut instances: Vec<_> = names
            .iter()
            .map(|name| GltfInstance {
                node_name: Some((*name).into()),
                mesh: 0,
                material_override: None,
                world: Mat4::IDENTITY,
            })
            .collect();
        update_c172_animation(
            &mut instances,
            &base,
            sim_physics::ControlInputs {
                elevator_rad: 0.2,
                aileron_rad: 0.1,
                rudder_rad: 0.2,
                flap_step: 2,
                ..Default::default()
            },
            0.65,
            1.0,
        );
        assert!(instances
            .iter()
            .all(|instance| instance.world != Mat4::IDENTITY));
        let trailing_edge = |name: &str| {
            instances[names.iter().position(|n| *n == name).unwrap()]
                .world
                .transform_vector3(-Vec3::X)
        };
        assert!(trailing_edge("elevatorleft").z > 0.0);
        assert!(trailing_edge("leftaileron").z > 0.0);
        assert!(trailing_edge("rightaileron").z < 0.0);
        assert!(trailing_edge("rudder_1").y > 0.0);
        assert!(trailing_edge("flaps").z > 0.0);

        update_c172_animation(
            &mut instances,
            &base,
            sim_physics::ControlInputs::default(),
            0.0,
            1.0,
        );
        assert!(instances
            .iter()
            .all(|instance| instance.world == Mat4::IDENTITY));
    }

    #[test]
    fn flight_title_reports_live_pilot_flight_state() {
        let snapshot = Snapshot {
            indicated_airspeed_mps: 50.0,
            altitude_m: 1_000.0,
            heading_deg: 92.4,
            vertical_speed_mps: 2.0,
            controls: sim_physics::ControlInputs {
                flap_step: 2,
                ..Default::default()
            },
            throttle: 0.65,
            ..Default::default()
        };
        let title = flight_window_title(&snapshot, true, true);
        assert!(title.contains("97 kt"));
        assert!(title.contains("3281 ft"));
        assert!(title.contains("HDG 092°"));
        assert!(title.contains("PITCH +0.0°"));
        assert!(title.contains("BANK +0.0°"));
        assert!(title.contains("VS +394 fpm"));
        assert!(title.contains("FLAP 2/3"));
        assert!(title.contains("THR  65%"));
        assert!(title.contains("| AP |"));
        assert!(title.contains("| COCKPIT |"));
    }

    #[test]
    fn cockpit_camera_sits_inside_and_looks_forward_with_the_aircraft() {
        let aircraft = DVec3::new(10.0, 20.0, 30.0);
        let attitude = DQuat::from_rotation_z(0.5);
        let (eye, camera) = cockpit_camera_pose(aircraft, attitude, 0.0, 0.0);
        let expected = aircraft + attitude * DVec3::new(0.85, 0.0, -0.85);
        assert!(eye.abs_diff_eq(expected, 1e-12));
        assert!(camera.abs_diff_eq(attitude, 1e-12));
    }

    #[test]
    fn cockpit_instruments_build_and_follow_live_flight_readings() {
        let mut model = cleanroom_c172_model();
        let (instances, digit_instances, glyphs) = add_cockpit_instruments(&mut model);
        assert_eq!(instances.len(), 28);
        assert_eq!(digit_instances.len(), 16);
        let panel_x = model.instances[instances.start].world.w_axis.x;
        assert!(digit_instances
            .iter()
            .all(|&index| model.instances[index].world.w_axis.x < panel_x));
        assert!("0123456789+-IASLTHDGV"
            .chars()
            .all(|character| glyphs.contains_key(&character)));
        assert!(instances.clone().all(|index| {
            let instance = &model.instances[index];
            (instance.mesh < model.meshes.len()) && instance.material_override.is_some()
        }));

        let snapshot = Snapshot {
            indicated_airspeed_mps: 50.0,
            altitude_m: 1_000.0,
            heading_deg: 92.4,
            vertical_speed_mps: 2.0,
            ..Default::default()
        };
        let readout: String = cockpit_readout_chars(&snapshot).into_iter().collect();
        assert_eq!(readout, "09703281092+0394");
    }

    #[test]
    fn seeds_ecs_with_aircraft_and_chase_camera() {
        let input = WindowInput {
            command: Arc::new(Mutex::new(ControlCommand::default())),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );
        assert_eq!(app.ecs.aircraft_entities().len(), 1);
        assert_eq!(app.ecs.camera_entities().len(), 1);
        assert_eq!(app.ecs.renderable_entities().len(), 1);
    }

    #[test]
    fn cleanroom_aircraft_has_valid_drawable_geometry() {
        let model = cleanroom_c172_model();
        assert_eq!(model.meshes.len(), model.instances.len());
        assert_eq!(model.materials.len(), 4);
        assert!(
            model.meshes[0].vertices.len() > 400,
            "fuselage should be lofted"
        );
        assert!(
            model.meshes[1].vertices.len() > 700,
            "wing should use airfoil sections"
        );
        assert!(
            model.meshes[9].vertices.len() > 100,
            "landing wheels should be rounded"
        );
        for mesh in &model.meshes {
            assert_eq!(mesh.indices.len() % 3, 0);
            assert!(mesh
                .indices
                .iter()
                .all(|&i| (i as usize) < mesh.vertices.len()));
            assert!(mesh.vertices.iter().all(|v| {
                Vec3::from_array(v.normal).is_normalized()
                    && v.position.iter().all(|x| x.is_finite())
            }));
        }
    }

    #[test]
    fn local_terrain_loader_reads_origin_dem_tile() {
        use sim_geo::quadtree::tile_for_direction;
        use sim_world::{DemTile, FileTileSource, TileKind};
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "open-msfs-world-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let origin = LatLonAlt::from_degrees(40.7128, -74.0060, 0.0);
        let lat = origin.latitude_rad();
        let lon = origin.longitude_rad();
        let dir = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
        let id = tile_for_direction(dir, 12);
        let source = FileTileSource::new(&root);
        let path = source.path_for(TileKind::Dem, id);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let dem = DemTile::new(id, 2, 2, vec![3.0; 4]).unwrap();
        std::fs::write(&path, dem.encode()).unwrap();

        let loaded = load_terrain_tile(&root, origin).unwrap();
        assert_eq!(loaded.mesh.id, id);
        assert_eq!(loaded.mesh.vertices.len(), 4);
        assert!((loaded.field_elevation_m - 3.0).abs() < 0.01);
        assert!(loaded
            .mesh
            .vertices
            .iter()
            .all(|v| v.position_enu.is_finite()));
    }

    #[test]
    fn bundled_nyc_terrarium_tiles_build_a_renderable_world_patch() {
        use std::collections::HashSet;

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/world");
        let origin = LatLonAlt::from_degrees(40.7128, -74.0060, 0.0);
        let asset = load_terrain_tile(&root, origin).expect("bundled NYC elevation tiles");
        let tiles: Vec<&TerrainMesh> = iter::once(&asset.mesh)
            .chain(asset.regional_meshes.iter())
            .collect();
        let (albedo, bounds) =
            terrain_patch_texture(&tiles, asset.mesh.origin_ecef.as_dvec3(), asset.enu_to_ecef);
        assert_eq!(albedo.width, TERRAIN_TEXTURE_SIZE);
        assert_eq!(albedo.height, TERRAIN_TEXTURE_SIZE);
        assert_eq!(
            albedo.rgba.len(),
            TERRAIN_TEXTURE_SIZE as usize * TERRAIN_TEXTURE_SIZE as usize * 4
        );
        assert!(bounds.0 < bounds.1 && bounds.2 < bounds.3);
        assert!(albedo
            .rgba
            .chunks_exact(4)
            .any(|pixel| pixel[0] != pixel[1]));
        assert!(
            asset.regional_meshes.len() >= (2 * TERRAIN_PATCH_RADIUS + 1).pow(2) as usize - 1,
            "all surrounding region tiles should load from the continuous elevation block"
        );
        let airport_osm = std::fs::read(root.join("osm/regions/new-york-airports.osm"))
            .expect("regional NYC airport OSM data");
        let airport_map = sim_world::osm::parse_osm_bytes(&airport_osm).unwrap();
        let mut runway_tiles = HashSet::new();
        for runway in airport_map
            .ways
            .iter()
            .filter(|way| way.tag("aeroway") == Some("runway"))
        {
            for (lat, lon) in airport_map.resolve_points(runway).unwrap() {
                let (lat, lon) = (lat.to_radians(), lon.to_radians());
                runway_tiles.insert(tile_for_direction(
                    DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()),
                    asset.mesh.id.level,
                ));
            }
        }
        let loaded_tiles: HashSet<_> = std::iter::once(asset.mesh.id)
            .chain(asset.regional_meshes.iter().map(|mesh| mesh.id))
            .collect();
        let (u0, u1, v0, v1) = asset.mesh.id.uv_range();
        let center_u = (u0 + u1) * 0.5;
        let center_v = (v0 + v1) * 0.5;
        let tile_u = (u1 - u0) * 0.5;
        let tile_v = (v1 - v0) * 0.5;
        let mut regional_tile_ids = HashSet::new();
        for dy in -TERRAIN_PATCH_RADIUS..=TERRAIN_PATCH_RADIUS {
            for dx in -TERRAIN_PATCH_RADIUS..=TERRAIN_PATCH_RADIUS {
                regional_tile_ids.insert(tile_for_direction(
                    asset.mesh.id.direction_at(
                        center_u + 2.0 * tile_u * dx as f64,
                        center_v + 2.0 * tile_v * dy as f64,
                    ),
                    asset.mesh.id.level,
                ));
            }
        }
        assert!(
            regional_tile_ids.is_subset(&loaded_tiles),
            "missing tiles in the continuous NYC terrain patch: {:?}",
            regional_tile_ids
                .difference(&loaded_tiles)
                .collect::<Vec<_>>()
        );
        let terrain_by_id: std::collections::HashMap<_, _> = std::iter::once(&asset.mesh)
            .chain(asset.regional_meshes.iter())
            .map(|mesh| (mesh.id, mesh))
            .collect();
        let vertex_ecef = |mesh: &TerrainMesh, index: usize| {
            let rotation = EnuFrame::new(mesh.id.center_lat_lon(), WGS84).enu_identity_ecef_quat();
            mesh.origin_ecef.as_dvec3() + rotation * mesh.vertices[index].position_enu
        };
        let mut maximum_seam_gap_m: f64 = 0.0;
        for (&tile_id, tile) in &terrain_by_id {
            for neighbor_id in [
                sim_geo::quadtree::TileId {
                    x: tile_id.x + 1,
                    ..tile_id
                },
                sim_geo::quadtree::TileId {
                    y: tile_id.y + 1,
                    ..tile_id
                },
            ] {
                let Some(neighbor) = terrain_by_id.get(&neighbor_id) else {
                    continue;
                };
                for step in 0..=64_u32 {
                    let tile_x = step * (tile.width - 1) / 64;
                    let tile_y = step * (tile.height - 1) / 64;
                    let neighbor_x = step * (neighbor.width - 1) / 64;
                    let neighbor_y = step * (neighbor.height - 1) / 64;
                    let (tile_index, neighbor_index) = if neighbor_id.x == tile_id.x + 1 {
                        (
                            (tile_y * tile.width + tile.width - 1) as usize,
                            (neighbor_y * neighbor.width) as usize,
                        )
                    } else {
                        (
                            ((tile.height - 1) * tile.width + tile_x) as usize,
                            neighbor_x as usize,
                        )
                    };
                    let gap = vertex_ecef(tile, tile_index)
                        .distance(vertex_ecef(neighbor, neighbor_index));
                    maximum_seam_gap_m = maximum_seam_gap_m.max(gap);
                }
            }
        }
        assert!(
            maximum_seam_gap_m < 0.05,
            "regional tile edges differ by up to {maximum_seam_gap_m:.3} m"
        );
        assert!(
            runway_tiles.is_subset(&loaded_tiles),
            "missing runway terrain tiles: {:?}",
            runway_tiles.difference(&loaded_tiles).collect::<Vec<_>>()
        );
        assert_eq!(asset.mesh.width, 257);
        assert_eq!(asset.mesh.height, 257);
        assert_eq!(asset.mesh.vertices.len(), 257 * 257);
        assert_eq!(asset.mesh.indices.len(), 256 * 256 * 6);
        assert!(asset.field_elevation_m.is_finite());
        assert!(
            !asset.osm_overlay.is_empty(),
            "the bundled OSM extract should produce roads/buildings"
        );
        assert!(asset
            .osm_overlay
            .iter()
            .all(|(mesh, _)| !mesh.vertices.is_empty() && !mesh.indices.is_empty()));
        assert!(
            asset
                .osm_overlay
                .iter()
                .filter(|(_, mat)| mat.name == "OpenStreetMap roads")
                .count()
                >= 2,
            "the downtown tile and regional extract should both produce road meshes"
        );
        assert!(asset
            .osm_overlay
            .iter()
            .any(|(_, mat)| mat.name == "OpenStreetMap building footprints"));
        assert!(asset
            .osm_overlay
            .iter()
            .any(|(_, mat)| mat.name == "OpenStreetMap runways"));
        assert!(asset
            .osm_overlay
            .iter()
            .any(|(_, mat)| mat.name == "OpenStreetMap water and shoreline"));
        let origin_local = EnuFrame::new(asset.mesh.id.center_lat_lon(), WGS84)
            .ecef_to_enu(&WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(40.7128, -74.0060, 0.0)))
            .as_dvec3()
            .as_vec3();
        let closest_water_vertex_m = asset
            .osm_overlay
            .iter()
            .filter(|(_, mat)| mat.name == "OpenStreetMap water and shoreline")
            .flat_map(|(mesh, _)| &mesh.vertices)
            .map(|vertex| {
                let p = Vec3::from_array(vertex.position);
                ((p.x - origin_local.x).powi(2) + (p.y - origin_local.y).powi(2)).sqrt()
            })
            .fold(f32::INFINITY, f32::min);
        assert!(
            closest_water_vertex_m < 4_000.0,
            "NYC water geometry should reach Manhattan; nearest water is {closest_water_vertex_m:.0} m away"
        );
        assert!(asset
            .osm_overlay
            .iter()
            .any(|(_, mat)| mat.name == "OpenStreetMap roads"));
        assert!(asset
            .mesh
            .vertices
            .iter()
            .all(|v| v.position_enu.is_finite() && v.normal_enu.is_finite()));
    }

    #[test]
    fn osm_runway_areas_and_centerlines_become_airport_surfaces() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/world");
        let origin = LatLonAlt::from_degrees(40.7128, -74.0060, 0.0);
        let terrain = load_terrain_tile(&root, origin).expect("NYC terrain");
        let osm = br#"<osm>
            <node id="1" lat="40.7000" lon="-74.0100"/>
            <node id="2" lat="40.7000" lon="-74.0000"/>
            <node id="3" lat="40.7020" lon="-74.0000"/>
            <node id="4" lat="40.7020" lon="-74.0100"/>
            <node id="5" lat="40.7030" lon="-74.0100"/>
            <node id="6" lat="40.7040" lon="-74.0100"/>
            <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
                <tag k="aeroway" v="runway"/></way>
            <way id="11"><nd ref="5"/><nd ref="6"/>
                <tag k="aeroway" v="taxiway"/><tag k="width" v="20"/></way>
        </osm>"#;
        let overlays = build_osm_overlay(&terrain.mesh, osm).expect("airport surface mesh");
        let runway = overlays
            .iter()
            .find(|(_, material)| material.name == "OpenStreetMap runways")
            .expect("runway area mesh");
        assert_eq!(runway.0.vertices.len(), 4);
        assert_eq!(runway.0.indices.len(), 6);
        let taxiway = overlays
            .iter()
            .find(|(_, material)| material.name == "OpenStreetMap taxiways and helipads")
            .expect("taxiway centerline mesh");
        assert_eq!(taxiway.0.vertices.len(), 4);
        assert_eq!(taxiway.0.indices.len(), 6);
        assert!(runway
            .0
            .vertices
            .iter()
            .chain(&taxiway.0.vertices)
            .all(|vertex| vertex.position.iter().all(|value| value.is_finite())));
    }

    #[test]
    fn physical_keys_deflect_and_center_via_shared_command() {
        let command = Arc::new(Mutex::new(ControlCommand::default()));
        let input = WindowInput {
            command: Arc::clone(&command),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );
        let shared =
            || command.lock().unwrap_or_else(|e| e.into_inner()).elevator_rad;
        let elev = crate::sim::MAX_ELEVATOR_RAD;
        let ail = crate::sim::MAX_AILERON_RAD;

        // Physical W deflects the elevator AND writes through to the
        // shared command without any manual store.
        let cmd = app
            .handle_physical_key(&PhysicalKey::Code(KeyCode::KeyW), true)
            .expect("physical w");
        assert!((cmd.elevator_rad + elev).abs() < 1e-9);
        assert!((shared() + elev).abs() < 1e-9);
        // Release springs back to neutral, again write-through.
        app.handle_physical_key(&PhysicalKey::Code(KeyCode::KeyW), false)
            .expect("physical w release");
        assert_eq!(shared(), 0.0);

        // A and the left arrow both roll left.
        let cmd = app
            .handle_physical_key(&PhysicalKey::Code(KeyCode::KeyA), true)
            .expect("physical a");
        assert!((cmd.aileron_rad + ail).abs() < 1e-12);
        app.handle_physical_key(&PhysicalKey::Code(KeyCode::KeyA), false)
            .expect("physical a release");
        let cmd = app
            .handle_physical_key(&PhysicalKey::Code(KeyCode::ArrowLeft), true)
            .expect("physical left");
        assert!((cmd.aileron_rad + ail).abs() < 1e-12);
        app.handle_physical_key(&PhysicalKey::Code(KeyCode::ArrowLeft), false)
            .expect("physical left release");
        assert_eq!(
            command.lock().unwrap_or_else(|e| e.into_inner()).aileron_rad,
            0.0
        );

        // Q/E drive rudder + nosewheel steering together.
        let cmd = app
            .handle_physical_key(&PhysicalKey::Code(KeyCode::KeyQ), true)
            .expect("physical q");
        assert_eq!(cmd.steering, -1.0);
        assert!((cmd.rudder_rad + crate::sim::MAX_RUDDER_RAD).abs() < 1e-12);
        app.handle_physical_key(&PhysicalKey::Code(KeyCode::KeyQ), false)
            .expect("physical q release");
        assert_eq!(
            command.lock().unwrap_or_else(|e| e.into_inner()).steering,
            0.0
        );

        // B brakes while held.
        app.handle_physical_key(&PhysicalKey::Code(KeyCode::KeyB), true)
            .expect("physical b");
        assert_eq!(
            command.lock().unwrap_or_else(|e| e.into_inner()).brake,
            1.0
        );
        app.handle_physical_key(&PhysicalKey::Code(KeyCode::KeyB), false)
            .expect("physical b release");
        assert_eq!(
            command.lock().unwrap_or_else(|e| e.into_inner()).brake,
            0.0
        );

        // Unidentified / unmapped keys stay out of the command path.
        assert!(app
            .handle_physical_key(&PhysicalKey::Code(KeyCode::Escape), true)
            .is_none());
        assert!(app
            .handle_physical_key(
                &PhysicalKey::Unidentified(winit::keyboard::NativeKeyCode::Unidentified),
                true
            )
            .is_none());
    }

    #[test]
    fn arrow_and_alias_keys_share_the_flight_bindings() {
        let command = Arc::new(Mutex::new(ControlCommand::default()));
        let input = WindowInput {
            command: Arc::clone(&command),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );
        let elev = crate::sim::MAX_ELEVATOR_RAD;
        let ail = crate::sim::MAX_AILERON_RAD;

        // Logical arrow keys deflect like W/S/A/D and center on release.
        let cmd = app
            .handle_key(&Key::Named(NamedKey::ArrowUp), true)
            .expect("arrow up");
        assert!((cmd.elevator_rad + elev).abs() < 1e-9);
        app.handle_key(&Key::Named(NamedKey::ArrowUp), false)
            .expect("arrow up release");
        let cmd = app
            .handle_key(&Key::Named(NamedKey::ArrowDown), true)
            .expect("arrow down");
        assert!((cmd.elevator_rad - elev).abs() < 1e-9);
        app.handle_key(&Key::Named(NamedKey::ArrowDown), false)
            .expect("arrow down release");
        let cmd = app
            .handle_key(&Key::Named(NamedKey::ArrowRight), true)
            .expect("arrow right");
        assert!((cmd.aileron_rad - ail).abs() < 1e-12);
        app.handle_key(&Key::Named(NamedKey::ArrowRight), false)
            .expect("arrow right release");
        assert_eq!(command.lock().unwrap().aileron_rad, 0.0);

        // T/Y are throttle aliases for R/F.
        app.handle_key(&Key::Character("t".into()), true)
            .expect("throttle up alias");
        assert_eq!(command.lock().unwrap().throttle, THROTTLE_STEP);
        app.handle_key(&Key::Character("y".into()), true)
            .expect("throttle down alias");
        assert_eq!(command.lock().unwrap().throttle, 0.0);
        assert!(app
            .handle_key(&Key::Character("t".into()), false)
            .is_none());
    }

    #[test]
    fn logical_and_physical_paths_map_to_the_same_bindings() {
        let pairs = [
            ("w", KeyCode::KeyW),
            ("s", KeyCode::KeyS),
            ("a", KeyCode::KeyA),
            ("d", KeyCode::KeyD),
            ("q", KeyCode::KeyQ),
            ("e", KeyCode::KeyE),
            ("r", KeyCode::KeyR),
            ("t", KeyCode::KeyT),
            ("f", KeyCode::KeyF),
            ("y", KeyCode::KeyY),
            ("b", KeyCode::KeyB),
            ("g", KeyCode::KeyG),
            ("p", KeyCode::KeyP),
        ];
        for (ch, code) in pairs {
            let logical = ControlBinding::from_str(ch).expect("logical binding");
            assert_eq!(Some(logical), ControlBinding::from_code(code), "{ch}");
            let upper = ch.to_uppercase();
            assert_eq!(ControlBinding::from_str(&upper), Some(logical), "{ch} upper");
        }
    }

    #[test]
    fn input_survives_command_lock_poisoning() {
        let command = Arc::new(Mutex::new(ControlCommand::default()));
        let input = WindowInput {
            command: Arc::clone(&command),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(
            input,
            &core,
            DVec3::new(0.0, 0.5, 0.8),
            None,
            None,
            Ecs::new(),
        );

        // Poison the shared mutex: a panicked thread while holding it.
        let poison = Arc::clone(&command);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = poison.lock().unwrap();
            panic!("poison the command mutex");
        }));
        assert!(Mutex::is_poisoned(&command), "mutex must be poisoned");

        // A poisoned lock must not brick the keyboard: the binding is
        // applied and written through so the physics thread sees it.
        let elev = crate::sim::MAX_ELEVATOR_RAD;
        let cmd = app
            .handle_key(&Key::Character("w".into()), true)
            .expect("handled despite poison");
        assert!((cmd.elevator_rad + elev).abs() < 1e-9);
        assert!(
            (command.lock().unwrap_or_else(|e| e.into_inner()).elevator_rad + elev).abs() < 1e-9
        );
    }
}
