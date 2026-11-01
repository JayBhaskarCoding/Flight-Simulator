//! Windowed mode: winit 0.30 event loop + wgpu surface presentation.
//!
//! The windowed path shares [`crate::sim::SimCore`] with headless mode
//! — the physics thread is identical; only presentation differs.

use crate::ecs::Ecs;
use crate::sim::{
    build_frame_input, camera_ecef_pose, spawn_aircraft_and_camera, system_build_draws_range,
    system_camera_follow, system_sync_aircraft, ControlCommand, SimCore, Snapshot,
};
use glam::{DQuat, DVec3, Mat4, Vec3};
use sim_avionics::ApMode;
use sim_geo::quadtree::tile_for_direction;
use sim_geo::{Ecef, EnuFrame, FloatingOrigin, LatLonAlt, WGS84};
use sim_render::gltf::{GltfInstance, GltfMaterial, GltfMesh, GltfModel};
use sim_render::types::GpuVertex;
use sim_render::Renderer;
use sim_world::{DemTile, FileTileSource, TerrainMesh, TileKind};
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use std::{collections::HashSet, iter};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Preferred present formats (must be sRGB — the tonemap shader emits
/// linear ACES and relies on the target's transfer function).
const TARGET_FORMAT_PREFERENCE: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgba8UnormSrgb,
];
const TERRAIN_PATCH_RADIUS: i32 = 7;

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
    terrain_instances: Option<Range<usize>>,
    terrain_root: Option<(DVec3, DQuat)>,
    ecs: Ecs,
    resources: Option<FrameResources>,
    start: Instant,
    pending_resize: Option<(u32, u32)>,
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
    let local = |lat: f64, lon: f64| {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let direction = DVec3::new(
            lat_rad.cos() * lon_rad.cos(),
            lat_rad.cos() * lon_rad.sin(),
            lat_rad.sin(),
        );
        let tile_id = tile_for_direction(direction, terrain.id.level);
        let ground = terrain_tiles
            .iter()
            .find(|tile| tile.id == tile_id)
            .unwrap_or(&terrain)
            .elevation_near(lat_rad, lon_rad);
        let p = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(lat, lon, 0.0));
        let mut enu = frame.ecef_to_enu(&p).as_dvec3();
        enu.z += ground;
        enu.as_vec3()
    };

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

    let mut overlays = Vec::new();
    if !building_indices.is_empty() {
        overlays.push((
            GltfMesh {
                vertices: building_vertices,
                indices: building_indices,
            },
            GltfMaterial {
                name: "OpenStreetMap building footprints".into(),
                base_color_factor: [0.46, 0.39, 0.32, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 0.9,
                double_sided: true,
                ..GltfMaterial::default()
            },
        ));
    }
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
        let (terrain_instances, terrain_root) = if let Some(terrain) = terrain {
            let root = (terrain.mesh.origin_ecef.as_dvec3(), terrain.enu_to_ecef);
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
                        uv: v.uv,
                    }
                }));
                indices.extend(tile.indices.iter().map(|index| first_vertex + index));
            }
            model.meshes.push(GltfMesh { vertices, indices });
            model.materials.push(GltfMaterial {
                name: "Elevation terrain".into(),
                base_color_factor: [0.29, 0.34, 0.22, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                double_sided: true,
                ..GltfMaterial::default()
            });
            model.instances.push(GltfInstance {
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
            terrain_instances,
            terrain_root,
            ecs,
            resources: None,
            start: Instant::now(),
            pending_resize: None,
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

        // ECS systems for this frame
        system_sync_aircraft(&mut self.ecs, &snap);
        system_camera_follow(&mut self.ecs);

        let cam_entity = self.ecs.camera_entities().first().copied();
        let (cam_pos_enu, cam_quat_enu) = cam_entity
            .and_then(|c| {
                self.ecs
                    .transform
                    .get(c)
                    .map(|t| (t.position, t.orientation))
            })
            .unwrap_or((snap.position_enu, snap.orientation_enu));
        let (camera_ecef, camera_quat) = camera_ecef_pose(&self.bridge, cam_pos_enu, cam_quat_enu);

        let model = self
            .model
            .as_ref()
            .ok_or("aircraft model not initialised")?;
        let mut draws = system_build_draws_range(
            &self.ecs,
            &self.bridge,
            camera_ecef,
            camera_quat,
            model,
            self.aircraft_instances.clone(),
        );
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

    fn handle_key(&self, key: &Key, pressed: bool) -> Option<ControlCommand> {
        // Returns the *new* command state for a handled key: held keys
        // deflect, released keys spring back to neutral.
        let mut cmd = *self.input.command.lock().ok()?;
        let max = crate::sim::MAX_ELEVATOR_RAD;
        let ail = crate::sim::MAX_AILERON_RAD;
        let rud = crate::sim::MAX_RUDDER_RAD;
        let ch = match key {
            Key::Character(c) => c.as_str(),
            _ => return None,
        };
        match ch {
            "r" | "R" if pressed => cmd.throttle = (cmd.throttle + 0.05).min(1.0),
            "f" | "F" if pressed => cmd.throttle = (cmd.throttle - 0.05).max(0.0),
            "w" | "W" => cmd.elevator_rad = if pressed { -max } else { 0.0 },
            "s" | "S" => cmd.elevator_rad = if pressed { max } else { 0.0 },
            "a" | "A" => cmd.aileron_rad = if pressed { -ail } else { 0.0 },
            "d" | "D" => cmd.aileron_rad = if pressed { ail } else { 0.0 },
            "q" | "Q" => cmd.rudder_rad = if pressed { -rud } else { 0.0 },
            "e" | "E" => cmd.rudder_rad = if pressed { rud } else { 0.0 },
            "b" | "B" => cmd.brake = if pressed { 1.0 } else { 0.0 },
            "g" | "G" if pressed => cmd.flap_step = (cmd.flap_step + 1).min(3),
            "v" | "V" if pressed => cmd.flap_step = cmd.flap_step.saturating_sub(1),
            "[" | "t" | "T" if pressed => {
                cmd.elevator_trim_rad = (cmd.elevator_trim_rad - 0.5f64.to_radians())
                    .clamp(-15.0f64.to_radians(), 15.0f64.to_radians());
            }
            "]" | "y" | "Y" if pressed => {
                cmd.elevator_trim_rad = (cmd.elevator_trim_rad + 0.5f64.to_radians())
                    .clamp(-15.0f64.to_radians(), 15.0f64.to_radians());
            }
            "p" | "P" if pressed => {
                if cmd.autopilot.is_some() {
                    cmd.autopilot = None;
                    cmd.ap_altitude_ft = None;
                    cmd.ap_heading_deg = None;
                } else {
                    let snap = self.input.snapshot.read().ok()?;
                    cmd.autopilot = Some(ApMode::AltitudeHold);
                    cmd.ap_altitude_ft = Some(snap.altitude_m * 3.280_839_895);
                    cmd.ap_heading_deg = None;
                }
            }
            _ => return None,
        }
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
            name: "blue trim".into(),
            base_color_factor: [0.04, 0.19, 0.43, 1.0],
            metallic_factor: 0.15,
            roughness_factor: 0.4,
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
    ];

    let parts = [
        // fuselage, high wing, tailplane, fin, cabin, propeller, three
        // simple landing gear struts/wheels; all original geometry.
        ([-3.2, -0.38, -0.34], [3.2, 0.38, 0.34], 0usize),
        ([-0.65, -5.5, -0.10], [0.65, 5.5, 0.10], 0),
        ([-2.9, -1.65, -0.10], [-2.0, 1.65, 0.10], 0),
        ([-2.95, -0.10, -1.35], [-2.15, 0.10, 0.02], 0),
        ([0.45, -0.31, -0.39], [1.65, 0.31, -0.02], 1),
        ([3.15, -0.07, -1.0], [3.27, 0.07, 1.0], 2),
        ([-0.8, -0.09, 0.15], [-0.66, 0.09, 1.15], 2),
        ([-0.95, -0.09, 0.78], [-0.55, 0.09, 1.18], 2),
        ([-0.8, 0.74, 0.15], [-0.66, 0.88, 1.15], 2),
        ([-0.95, 0.74, 0.78], [-0.55, 0.88, 1.18], 2),
    ];
    let meshes: Vec<GltfMesh> = parts
        .iter()
        .map(|(min, max, _)| box_mesh(*min, *max))
        .collect();
    let instances = parts
        .iter()
        .enumerate()
        .map(|(mesh, (_, _, material))| GltfInstance {
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
            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && event.state == ElementState::Pressed
                {
                    event_loop.exit();
                    return;
                }
                if let Some(cmd) =
                    self.handle_key(&event.logical_key, event.state == ElementState::Pressed)
                {
                    if let Ok(mut guard) = self.input.command.lock() {
                        *guard = cmd;
                    }
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

impl App {
    /// Async surface/adapter/device setup (driven by the small
    /// park-waker executor from `resumed`).
    async fn init_surface(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window: Arc<Window> = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("OpenMSFS — clean-room flight simulator")
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
    fn trim_and_flaps_keys_modify_controls() {
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

        // Flaps: G extends, V retracts
        let flapped = app.handle_key(&Key::Character("g".into()), true).unwrap();
        assert_eq!(flapped.flap_step, 1);
        *command.lock().unwrap() = flapped;
        let flapped2 = app.handle_key(&Key::Character("g".into()), true).unwrap();
        assert_eq!(flapped2.flap_step, 2);
        *command.lock().unwrap() = flapped2;
        let retracted = app.handle_key(&Key::Character("v".into()), true).unwrap();
        assert_eq!(retracted.flap_step, 1);
        *command.lock().unwrap() = retracted;

        // Trim: [ / t trims nose down (-rad), ] / y trims nose up (+rad)
        let trimmed_down = app.handle_key(&Key::Character("[".into()), true).unwrap();
        assert!(trimmed_down.elevator_trim_rad < 0.0);
        *command.lock().unwrap() = trimmed_down;
        let trimmed_up = app.handle_key(&Key::Character("]".into()), true).unwrap();
        assert!((trimmed_up.elevator_trim_rad).abs() < 1e-9);
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
        assert_eq!(model.materials.len(), 3);
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
}
