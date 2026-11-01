//! Convert a decoded elevation tile into a precision-preserving local mesh.
//!
//! Vertex positions are ENU offsets from the tile centre, computed in
//! `f64`. A renderer can subtract/rebase in `f64` before narrowing to GPU
//! floats, matching the aircraft and floating-origin path.

use crate::{DemTile, WorldError};
use glam::DVec3;
use sim_geo::quadtree::{face_and_uv, TileId};
use sim_geo::{Ecef, EnuFrame, LatLonAlt, WGS84};
use std::collections::HashMap;
use std::io::Cursor;

/// One vertex in a tile-local east-north-up frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainVertex {
    /// Metres from [`TerrainMesh::origin_ecef`] in the tile's ENU frame.
    pub position_enu: DVec3,
    /// Unit normal in the same ENU frame.
    pub normal_enu: DVec3,
    /// Normalized source-grid coordinates.
    pub uv: [f32; 2],
}

/// Triangulated elevation tile, positioned by an ECEF origin.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainMesh {
    pub id: TileId,
    pub origin_ecef: Ecef,
    pub width: u32,
    pub height: u32,
    pub vertices: Vec<TerrainVertex>,
    pub indices: Vec<u32>,
    elevations_m: Vec<f32>,
}

impl TerrainMesh {
    /// Estimate terrain elevation nearest a geographic point by searching
    /// this tile's local mesh. The input is in radians; the result is metres
    /// above the WGS84 ellipsoid. This is suitable for runway initial
    /// placement and tile-local queries.
    pub fn elevation_near(&self, latitude_rad: f64, longitude_rad: f64) -> f64 {
        let direction = DVec3::new(
            latitude_rad.cos() * longitude_rad.cos(),
            latitude_rad.cos() * longitude_rad.sin(),
            latitude_rad.sin(),
        );
        let (face, u, v) = face_and_uv(direction);
        if face != self.id.face {
            return 0.0;
        }
        let (u0, u1, v0, v1) = self.id.uv_range();
        let col = ((u - u0) / (u1 - u0) * f64::from(self.width - 1))
            .clamp(0.0, f64::from(self.width - 1));
        let row = ((v - v0) / (v1 - v0) * f64::from(self.height - 1))
            .clamp(0.0, f64::from(self.height - 1));
        sample_grid(&self.elevations_m, self.width, self.height, col, row)
            .map(f64::from)
            .unwrap_or(0.0)
    }

    /// Project every DEM sample onto the WGS84 ellipsoid plus its elevation,
    /// then triangulate adjacent samples. Tile identity and geodetic
    /// projection are retained so the same pipeline works in New York,
    /// India, and at global scale.
    pub fn from_dem(dem: &DemTile) -> Result<Self, WorldError> {
        Self::from_grid(dem.id(), dem.width(), dem.height(), dem.samples())
    }

    /// Decode a 3×3 mosaic of Mapzen Terrarium XYZ PNGs and resample it
    /// onto a cube-sphere tile. The mosaic keys are `(x, y)` at `zoom`;
    /// the centre XYZ tile must contain the cube tile centre. Terrarium
    /// elevation is `R*256 + G + B/256 - 32768` metres.
    pub fn from_terrarium_xyz(
        id: TileId,
        zoom: u8,
        tiles: &HashMap<(u32, u32), Vec<u8>>,
        grid_size: u32,
    ) -> Result<Self, WorldError> {
        if !(2..=2048).contains(&grid_size) || zoom > 22 {
            return Err(WorldError::Dem(
                "invalid Terrarium grid size or zoom".into(),
            ));
        }
        let center = id.center_lat_lon();
        let n = 1u32 << zoom;
        let (center_x, center_y) =
            web_mercator_tile(center.latitude_rad(), center.longitude_rad(), n);
        let mut decoded = HashMap::new();
        let mut tile_size = None;
        for (&key, bytes) in tiles {
            let (w, h, samples) = decode_terrarium_png(bytes)?;
            if w != h || tile_size.is_some_and(|s| s != w) {
                return Err(WorldError::Dem(
                    "Terrarium tiles must be square and equally sized".into(),
                ));
            }
            tile_size = Some(w);
            decoded.insert(key, samples);
        }
        let size =
            tile_size.ok_or_else(|| WorldError::Dem("empty Terrarium tile mosaic".into()))?;
        if !decoded.contains_key(&(center_x, center_y)) {
            return Err(WorldError::Dem(format!(
                "Terrarium mosaic lacks centre tile {center_x}/{center_y}"
            )));
        }
        let mosaic_size = size * 3;
        let mut mosaic = vec![f32::NAN; (mosaic_size * mosaic_size) as usize];
        for dy in -1i64..=1 {
            let y = center_y as i64 + dy;
            if !(0..n as i64).contains(&y) {
                continue;
            }
            for dx in -1i64..=1 {
                let x = (center_x as i64 + dx).rem_euclid(n as i64) as u32;
                let Some(samples) = decoded.get(&(x, y as u32)) else {
                    continue;
                };
                for row in 0..size {
                    let dst = ((dy + 1) as u32 * size + row) * mosaic_size + (dx + 1) as u32 * size;
                    let src = row * size;
                    mosaic[dst as usize..(dst + size) as usize]
                        .copy_from_slice(&samples[src as usize..(src + size) as usize]);
                }
            }
        }

        let width = grid_size;
        let height = grid_size;
        let (u0, u1, v0, v1) = id.uv_range();
        let mut samples = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let v = v0 + (v1 - v0) * row as f64 / f64::from(height - 1);
            for col in 0..width {
                let u = u0 + (u1 - u0) * col as f64 / f64::from(width - 1);
                let direction = id.direction_at(u, v);
                let lat = direction.z.asin();
                let lon = direction.y.atan2(direction.x);
                let (global_x, global_y) = web_mercator_pixel(lat, lon, zoom, size);
                let world = f64::from(n * size);
                let mut x = global_x - f64::from(center_x * size);
                if x > world * 0.5 {
                    x -= world;
                }
                if x < -world * 0.5 {
                    x += world;
                }
                let x = x + f64::from(size) - 0.5;
                let y = global_y - f64::from(center_y.saturating_sub(1) * size) - 0.5;
                let elevation =
                    sample_grid(&mosaic, mosaic_size, mosaic_size, x, y).ok_or_else(|| {
                        WorldError::Dem(
                            "cube tile extends beyond the supplied Terrarium mosaic".into(),
                        )
                    })?;
                samples.push(elevation);
            }
        }
        Self::from_grid(id, width, height, &samples)
    }

    fn from_grid(id: TileId, width: u32, height: u32, samples: &[f32]) -> Result<Self, WorldError> {
        if samples.len() != (width * height) as usize || width < 2 || height < 2 {
            return Err(WorldError::Dem(
                "terrain grid dimensions do not match samples".into(),
            ));
        }
        let center = id.center_lat_lon();
        let frame = EnuFrame::new(center, WGS84);
        let origin_ecef = WGS84.geodetic_to_ecef(&center);
        let (u0, u1, v0, v1) = id.uv_range();
        let mut vertices = Vec::with_capacity((width * height) as usize);

        for row in 0..height {
            let fv = row as f64 / f64::from(height - 1);
            let v = v0 + (v1 - v0) * fv;
            for col in 0..width {
                let fu = col as f64 / f64::from(width - 1);
                let u = u0 + (u1 - u0) * fu;
                let direction = id.direction_at(u, v);
                let lat = direction.z.asin();
                let lon = direction.y.atan2(direction.x);
                let altitude = f64::from(samples[(row * width + col) as usize]);
                let ecef = WGS84.geodetic_to_ecef(&LatLonAlt::new(lat, lon, altitude));
                let enu = frame.ecef_to_enu(&ecef).as_dvec3();
                vertices.push(TerrainVertex {
                    position_enu: enu,
                    normal_enu: DVec3::ZERO,
                    uv: [fu as f32, fv as f32],
                });
            }
        }

        let mut indices = Vec::with_capacity(((width - 1) * (height - 1) * 6) as usize);
        for row in 0..height - 1 {
            for col in 0..width - 1 {
                let p00 = row * width + col;
                let p10 = p00 + 1;
                let p01 = p00 + width;
                let p11 = p01 + 1;
                // Increasing face-u crossed with face-v points out of the
                // globe, which maps to up in this ENU frame.
                indices.extend_from_slice(&[p00, p10, p01, p10, p11, p01]);
            }
        }

        for tri in indices.chunks_exact(3) {
            let a = vertices[tri[0] as usize].position_enu;
            let b = vertices[tri[1] as usize].position_enu;
            let c = vertices[tri[2] as usize].position_enu;
            let mut n = (b - a).cross(c - a);
            if n.z < 0.0 {
                n = -n;
            }
            if n.length_squared() > 1e-20 {
                let n = n.normalize();
                for &i in tri {
                    vertices[i as usize].normal_enu += n;
                }
            }
        }
        for vertex in &mut vertices {
            vertex.normal_enu = if vertex.normal_enu.length_squared() > 1e-20 {
                vertex.normal_enu.normalize()
            } else {
                DVec3::Z
            };
        }

        Ok(Self {
            id,
            origin_ecef,
            width,
            height,
            vertices,
            indices,
            elevations_m: samples.to_vec(),
        })
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

fn web_mercator_pixel(lat: f64, lon: f64, zoom: u8, size: u32) -> (f64, f64) {
    let world = f64::from(1u32 << zoom) * f64::from(size);
    let x = (lon.to_degrees() + 180.0) / 360.0 * world;
    let y = (1.0 - lat.tan().asinh() / std::f64::consts::PI) * 0.5 * world;
    (x, y)
}

fn sample_grid(samples: &[f32], width: u32, height: u32, x: f64, y: f64) -> Option<f32> {
    if !(0.0..=(width - 1) as f64).contains(&x) || !(0.0..=(height - 1) as f64).contains(&y) {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = (x - f64::from(x0)) as f32;
    let fy = (y - f64::from(y0)) as f32;
    let at = |x, y| samples[(y * width + x) as usize];
    let values = [at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1)];
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let top = values[0] * (1.0 - fx) + values[1] * fx;
    let bottom = values[2] * (1.0 - fx) + values[3] * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

fn decode_terrarium_png(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), WorldError> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| WorldError::Dem(format!("Terrarium PNG: {e}")))?;
    if reader.info().bit_depth != png::BitDepth::Eight {
        return Err(WorldError::Dem(
            "Terrarium PNG must decode to 8-bit channels".into(),
        ));
    }
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| WorldError::Dem(format!("Terrarium PNG: {e}")))?;
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => return Err(WorldError::Dem("Terrarium PNG must be RGB or RGBA".into())),
    };
    let mut samples = Vec::with_capacity((info.width * info.height) as usize);
    for px in buf[..info.buffer_size()].chunks_exact(channels) {
        let height =
            f32::from(px[0]) * 256.0 + f32::from(px[1]) + f32::from(px[2]) / 256.0 - 32768.0;
        samples.push(height);
    }
    Ok((info.width, info.height, samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dem_projects_into_tile_local_enu_mesh() {
        let id = TileId {
            face: 0,
            level: 4,
            x: 8,
            y: 8,
        };
        let center = id.center_lat_lon();
        let dem = DemTile::new(id, 3, 2, vec![12.0; 6]).unwrap();
        let mesh = TerrainMesh::from_dem(&dem).unwrap();

        assert_eq!(
            mesh.origin_ecef,
            WGS84.geodetic_to_ecef(&id.center_lat_lon())
        );
        assert_eq!(mesh.vertices.len(), 6);
        assert_eq!(mesh.indices, vec![0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4]);
        assert!(mesh.vertices.iter().all(|v| v.position_enu.is_finite()));
        assert!(mesh.vertices.iter().all(|v| v.normal_enu.z > 0.99));
        assert!(
            (mesh.elevation_near(center.latitude_rad(), center.longitude_rad()) - 12.0).abs()
                < 0.01
        );
        assert!(mesh
            .vertices
            .iter()
            .all(|v| v.uv[1] == 0.0 || v.uv[1] == 1.0));
    }

    #[test]
    fn terrarium_xyz_resamples_world_height_into_cube_tile() {
        let id = TileId {
            face: 0,
            level: 12,
            x: 1160,
            y: 2500,
        };
        let center = id.center_lat_lon();
        let (cx, cy) = web_mercator_tile(center.latitude_rad(), center.longitude_rad(), 1 << 13);
        let encoded_height = [128, 12, 0]; // 128*256 + 12 - 32768 = 12m
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&encoded_height.repeat(4)).unwrap();
        }
        let mut tiles = HashMap::new();
        let n = 1u32 << 13;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let x = (cx as i64 + dx).rem_euclid(n as i64) as u32;
                let y = (cy as i64 + dy) as u32;
                tiles.insert((x, y), bytes.clone());
            }
        }

        let mesh = TerrainMesh::from_terrarium_xyz(id, 13, &tiles, 3).unwrap();
        assert_eq!(mesh.vertices.len(), 9);
        assert!((mesh.vertices[4].position_enu.z - 12.0).abs() < 0.01);
        assert!(mesh.vertices.iter().all(|v| v.normal_enu.z > 0.99));
    }
}
