//! Digital elevation models: binary `DEM1` tiles with bilinear sampling
//! through the cube-face projection.
//!
//! Container layout (all integers little-endian):
//!
//! ```text
//! offset 0   4 bytes   magic  = b"DEM1"
//! offset 4   u32       width  (columns, >= 2, <= 8192)
//! offset 8   u32       height (rows,    >= 2, <= 8192)
//! offset 12  width*height f32  row-major samples
//! ```
//!
//! Grid convention: column `i` sits at `u_local = i / (width - 1)` and row
//! `j` at `v_local = j / (height - 1)`, where `(u, v)` are the face UV
//! coordinates of [`sim_geo::quadtree::TileId::uv_range`] (row 0 is the
//! low-`v` edge). Samples are `f32` on disk — GPU-friendly — but every
//! query returns `f64`, honouring the 64-bit world law.

use crate::WorldError;
use glam::DVec3;
use sim_geo::quadtree::{face_and_uv, TileId};

/// Magic bytes of the container.
pub const DEM_MAGIC: [u8; 4] = *b"DEM1";
/// Minimum grid extent per axis (bilinear needs two samples).
pub const DEM_MIN_DIM: u32 = 2;
/// Maximum grid extent per axis (guards absurd allocations).
pub const DEM_MAX_DIM: u32 = 8192;

/// One decoded elevation tile.
#[derive(Debug, Clone, PartialEq)]
pub struct DemTile {
    id: TileId,
    width: u32,
    height: u32,
    samples: Vec<f32>,
}

impl DemTile {
    /// Build a tile, validating dimensions and payload length.
    pub fn new(
        id: TileId,
        width: u32,
        height: u32,
        samples: Vec<f32>,
    ) -> Result<Self, WorldError> {
        if !(DEM_MIN_DIM..=DEM_MAX_DIM).contains(&width)
            || !(DEM_MIN_DIM..=DEM_MAX_DIM).contains(&height)
        {
            return Err(WorldError::Dem(format!(
                "grid {width}x{height} outside [{DEM_MIN_DIM}, {DEM_MAX_DIM}]"
            )));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| WorldError::Dem("grid dimensions overflow".into()))?;
        if samples.len() != expected {
            return Err(WorldError::Dem(format!(
                "payload has {} samples, grid {width}x{height} needs {expected}",
                samples.len()
            )));
        }
        if let Some((idx, v)) = samples
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(WorldError::Dem(format!(
                "sample {idx} is not finite ({v})"
            )));
        }
        Ok(DemTile {
            id,
            width,
            height,
            samples,
        })
    }

    /// Quadtree identity this tile covers.
    pub fn id(&self) -> TileId {
        self.id
    }

    /// Grid columns.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Grid rows.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Raw samples, row-major (`row * width + column`).
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Lowest and highest sample (useful for min/max pyramids).
    pub fn min_max(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &s in &self.samples {
            let v = f64::from(s);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, hi)
    }

    /// Serialise into the `DEM1` container.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.samples.len() * 4);
        out.extend_from_slice(&DEM_MAGIC);
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        for s in &self.samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    /// Parse a `DEM1` container, validating magic, dimensions, payload
    /// length and sample finiteness.
    pub fn decode(bytes: &[u8]) -> Result<Self, WorldError> {
        if bytes.len() < 12 {
            return Err(WorldError::Dem(format!(
                "payload of {} bytes is shorter than the 12-byte header",
                bytes.len()
            )));
        }
        if bytes[0..4] != DEM_MAGIC {
            return Err(WorldError::Dem(format!(
                "bad magic {:02x?} (expected DEM1)",
                &bytes[0..4]
            )));
        }
        let width = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
        let height = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
        if !(DEM_MIN_DIM..=DEM_MAX_DIM).contains(&width)
            || !(DEM_MIN_DIM..=DEM_MAX_DIM).contains(&height)
        {
            return Err(WorldError::Dem(format!(
                "grid {width}x{height} outside [{DEM_MIN_DIM}, {DEM_MAX_DIM}]"
            )));
        }
        let count = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| WorldError::Dem("grid dimensions overflow".into()))?;
        let expected_bytes = 12 + count * 4;
        if bytes.len() != expected_bytes {
            return Err(WorldError::Dem(format!(
                "payload is {} bytes, header implies {expected_bytes}",
                bytes.len()
            )));
        }
        let mut samples = Vec::with_capacity(count);
        for chunk in bytes[12..].chunks_exact(4) {
            samples.push(f32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ]));
        }
        // `new` re-validates dimensions, length and finiteness. The
        // container carries no key; `decode_as` binds the real TileId.
        DemTile::new(TileId::root(0), width, height, samples)
    }

    /// Decode and re-bind the container to the quadtree tile it was
    /// fetched as (the container itself carries no key).
    pub fn decode_as(bytes: &[u8], id: TileId) -> Result<Self, WorldError> {
        let mut tile = Self::decode(bytes)?;
        tile.id = id;
        Ok(tile)
    }

    /// Elevation at tile-local coordinates `(u, v) ∈ [0, 1]²`, bilinear.
    /// Coordinates outside the unit square are clamped to the edges.
    /// Returns `f64` (64-bit world law).
    pub fn height_at_uv(&self, u: f64, v: f64) -> f64 {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);
        let gx = f64::from(self.width - 1) * u;
        let gy = f64::from(self.height - 1) * v;
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.width as usize - 1);
        let y1 = (y0 + 1).min(self.height as usize - 1);
        let fx = gx - f64::from(x0 as u32);
        let fy = gy - f64::from(y0 as u32);
        let at = |c: usize, r: usize| f64::from(self.samples[r * self.width as usize + c]);
        let h00 = at(x0, y0);
        let h10 = at(x1, y0);
        let h01 = at(x0, y1);
        let h11 = at(x1, y1);
        let top = h00 + (h10 - h00) * fx;
        let bot = h01 + (h11 - h01) * fx;
        top + (bot - top) * fy
    }

    /// Elevation under a unit direction from the planet centre, using this
    /// tile's face projection. Errors when `dir` lies on another cube face
    /// or outside this tile's UV rectangle.
    pub fn sample_direction(&self, dir: DVec3) -> Result<f64, WorldError> {
        let dir = dir.normalize();
        let (face, u, v) = face_and_uv(dir);
        if face != self.id.face {
            return Err(WorldError::WrongTileFace {
                tile_face: self.id.face,
                dir_face: face,
            });
        }
        let (u0, u1, v0, v1) = self.id.uv_range();
        const EPS: f64 = 1e-9;
        if u < u0 - EPS || u > u1 + EPS || v < v0 - EPS || v > v1 + EPS {
            return Err(WorldError::Dem(format!(
                "direction uv ({u:.6}, {v:.6}) lies outside tile uv range \
                 ({u0:.6}..{u1:.6}, {v0:.6}..{v1:.6})"
            )));
        }
        let lu = if (u1 - u0).abs() < f64::EPSILON {
            0.0
        } else {
            (u - u0) / (u1 - u0)
        };
        let lv = if (v1 - v0).abs() < f64::EPSILON {
            0.0
        } else {
            (v - v0) / (v1 - v0)
        };
        Ok(self.height_at_uv(lu, lv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_tile(id: TileId, w: u32, h: u32) -> DemTile {
        // h(u, v) = 100 + 20u + 4v  (all values exactly representable in f32)
        let mut samples = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                let u = f64::from(i) / f64::from(w - 1);
                let v = f64::from(j) / f64::from(h - 1);
                let s = 100.0 + 20.0 * u + 4.0 * v;
                samples.push(s as f32);
            }
        }
        DemTile::new(id, w, h, samples).unwrap()
    }

    #[test]
    fn rejects_bad_dimensions_and_lengths() {
        let id = TileId::root(0);
        assert!(DemTile::new(id, 1, 4, vec![0.0; 4]).is_err());
        assert!(DemTile::new(id, 4, 1, vec![0.0; 4]).is_err());
        assert!(DemTile::new(id, 9000, 2, vec![0.0; 18_000]).is_err());
        assert!(DemTile::new(id, 2, 2, vec![0.0; 5]).is_err());
        assert!(DemTile::new(id, 2, 2, vec![0.0, 1.0, f32::NAN, 3.0]).is_err());
    }

    #[test]
    fn encode_decode_roundtrip_preserves_tile() {
        let id = TileId {
            face: 4,
            level: 6,
            x: 33,
            y: 17,
        };
        let tile = plane_tile(id, 8, 5);
        let bytes = tile.encode();
        assert_eq!(&bytes[0..4], b"DEM1");
        assert_eq!(bytes.len(), 12 + 8 * 5 * 4);

        let back = DemTile::decode_as(&bytes, id).unwrap();
        assert_eq!(back, tile);
        assert_eq!(back.id(), id);
        assert_eq!(back.width(), 8);
        assert_eq!(back.height(), 5);
    }

    #[test]
    fn decode_reports_malformed_containers() {
        assert!(matches!(
            DemTile::decode(b"DEM"),
            Err(WorldError::Dem(_))
        ));
        assert!(matches!(
            DemTile::decode(b"XXXX\x02\0\0\0\x02\0\0\0\0\0\0\0\0\0\0\0"),
            Err(WorldError::Dem(m)) if m.contains("bad magic")
        ));
        // Dimension out of range.
        assert!(matches!(
            DemTile::decode(b"DEM1\x01\0\0\0\x02\0\0\0\0\0"),
            Err(WorldError::Dem(m)) if m.contains("outside")
        ));
        // Truncated payload (header claims 2x2 = 16 bytes of samples).
        assert!(matches!(
            DemTile::decode(b"DEM1\x02\0\0\0\x02\0\0\0\0\0\0\0"),
            Err(WorldError::Dem(m)) if m.contains("header implies")
        ));
        // decode() without an id binds the placeholder root; decode_as fixes it.
        let valid = [b"DEM1\x02\0\0\0\x02\0\0\0".as_slice(), &[0u8; 16]].concat();
        let ok = DemTile::decode(&valid).unwrap();
        assert_eq!(ok.id(), TileId::root(0));
        assert_eq!(ok.width(), 2);
        assert_eq!(ok.height(), 2);
    }

    #[test]
    fn bilinear_matches_plane_interior_and_edges() {
        let tile = plane_tile(TileId::root(0), 5, 3);
        // Corners are exact grid nodes.
        assert_eq!(tile.height_at_uv(0.0, 0.0), 100.0);
        assert_eq!(tile.height_at_uv(1.0, 0.0), 120.0);
        assert_eq!(tile.height_at_uv(0.0, 1.0), 104.0);
        assert_eq!(tile.height_at_uv(1.0, 1.0), 124.0);
        // A plane is reproduced exactly at arbitrary interior points.
        for &(u, v) in &[(0.3, 0.7), (0.5, 0.5), (0.13, 0.87), (0.99, 0.01)] {
            let expect = 100.0 + 20.0 * u + 4.0 * v;
            let got = tile.height_at_uv(u, v);
            assert!(
                (got - expect).abs() < 1e-9,
                "u={u} v={v}: got {got}, want {expect}"
            );
        }
        // Out-of-range coordinates clamp to the edges.
        assert_eq!(tile.height_at_uv(-5.0, -5.0), 100.0);
        assert_eq!(tile.height_at_uv(7.0, 2.0), 124.0);
    }

    #[test]
    fn sample_direction_agrees_with_uv_query() {
        // Face 5 (-Z) centre direction; take a tile covering the centre.
        let id = TileId {
            face: 5,
            level: 2,
            x: 1,
            y: 1,
        };
        let tile = plane_tile(id, 4, 4);
        let dir = id.center();
        let from_dir = tile.sample_direction(dir).unwrap();
        let (face, u, v) = face_and_uv(dir);
        assert_eq!(face, 5);
        let (u0, u1, v0, v1) = id.uv_range();
        let expect = tile.height_at_uv((u - u0) / (u1 - u0), (v - v0) / (v1 - v0));
        assert!((from_dir - expect).abs() < 1e-12);

        // A direction on another face is rejected with the face pair.
        let wrong = DVec3::new(1.0, 0.0, 0.0);
        assert!(matches!(
            tile.sample_direction(wrong),
            Err(WorldError::WrongTileFace {
                tile_face: 5,
                dir_face: 0
            })
        ));

        // Same face but a neighbouring tile: outside this tile's UV box.
        let other = TileId {
            face: 5,
            level: 2,
            x: 0,
            y: 0,
        };
        assert!(matches!(
            tile.sample_direction(other.center()),
            Err(WorldError::Dem(_))
        ));
    }

    #[test]
    fn min_max_scans_all_samples() {
        let id = TileId::root(1);
        let tile = DemTile::new(id, 2, 2, vec![-3.5, 9.25, 0.0, 4.0]).unwrap();
        assert_eq!(tile.min_max(), (-3.5, 9.25));
    }
}
