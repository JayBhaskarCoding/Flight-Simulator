//! GPU payload preparation and CPU mirrors of the WGSL LUT samplers.
//!
//! The three atmosphere grids (`sim-atmosphere::SkyLuts`) are uploaded
//! as `Rgba32Float` textures. The `mirror_*` functions below are the
//! bit-for-bit `f32` counterparts of the `textureLoad`-based bilinear
//! samplers in the WGSL shaders; unit tests compare them against the
//! crate's own `f64` accessors, which pins the shader formulas to the
//! CPU physics (any formula drift fails `cargo test`).

use sim_atmosphere::sky::{
    SkyLuts, AERIAL_DIST, AERIAL_MAX_DIST_M, AERIAL_ZEN, SKY_VIEW_H, SKY_VIEW_W, TRANS_H, TRANS_W,
};

/// `Rgba32Float` payloads for the four LUT textures.
#[derive(Debug, Clone)]
pub struct LutPayloads {
    /// `TRANS_W × TRANS_H` — rgb transmittance, `a = 1`.
    pub transmittance: Vec<[f32; 4]>,
    /// `SKY_VIEW_W × SKY_VIEW_H` — sky-view radiance.
    pub sky_view: Vec<[f32; 4]>,
    /// `AERIAL_DIST × AERIAL_ZEN` — path transmittance, `a = 1`.
    pub aerial_trans: Vec<[f32; 4]>,
    /// `AERIAL_DIST × AERIAL_ZEN` — in-scattered radiance, `a = 1`.
    pub aerial_radiance: Vec<[f32; 4]>,
}

impl LutPayloads {
    /// Pack the CPU grids into straight RGBA upload buffers (row-major
    /// `[y * width + x]`, matching `textureLoad` indexing).
    pub fn from_sky_luts(luts: &SkyLuts) -> Self {
        let trans = luts
            .transmittance_grid()
            .iter()
            .map(|c| [c[0], c[1], c[2], 1.0])
            .collect();
        let sky_view = luts.sky_view_grid().to_vec();
        let (mut at, mut ar) = (
            Vec::with_capacity(AERIAL_DIST * AERIAL_ZEN),
            Vec::with_capacity(AERIAL_DIST * AERIAL_ZEN),
        );
        for &(t, r) in luts.aerial_grid() {
            at.push([t[0], t[1], t[2], 1.0]);
            ar.push([r[0], r[1], r[2], 1.0]);
        }
        Self {
            transmittance: trans,
            sky_view,
            aerial_trans: at,
            aerial_radiance: ar,
        }
    }

    /// `(width, height)` of each payload in texels.
    pub fn dims(&self) -> [(u32, u32); 4] {
        [
            (TRANS_W as u32, TRANS_H as u32),
            (SKY_VIEW_W as u32, SKY_VIEW_H as u32),
            (AERIAL_DIST as u32, AERIAL_ZEN as u32),
            (AERIAL_DIST as u32, AERIAL_ZEN as u32),
        ]
    }
}

fn bilinear3(
    p00: [f32; 3],
    p10: [f32; 3],
    p01: [f32; 3],
    p11: [f32; 3],
    tx: f32,
    ty: f32,
) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let a = p00[c] + (p10[c] - p00[c]) * tx;
        let b = p01[c] + (p11[c] - p01[c]) * tx;
        out[c] = a + (b - a) * ty;
    }
    out
}

/// CPU mirror of the WGSL `sample_transmittance`.
pub fn mirror_transmittance(
    grid: &[[f32; 3]],
    altitude_m: f32,
    cos_zenith: f32,
    atmosphere_height_m: f32,
) -> [f32; 3] {
    let ay = (altitude_m.max(0.0) / atmosphere_height_m).clamp(0.0, 1.0);
    let ax = ((cos_zenith + 1.0) * 0.5).clamp(0.0, 1.0);
    let fy = ay * (TRANS_H - 1) as f32;
    let fx = ax * (TRANS_W - 1) as f32;
    let y0 = fy.floor() as i32;
    let x0 = fx.floor() as i32;
    let y1 = (y0 + 1).min(TRANS_H as i32 - 1);
    let x1 = (x0 + 1).min(TRANS_W as i32 - 1);
    let ty = fy - y0 as f32;
    let tx = fx - x0 as f32;
    let at = |x: i32, y: i32| grid[(y as usize) * TRANS_W + (x as usize)];
    bilinear3(at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1), tx, ty)
}

/// CPU mirror of the WGSL `sample_sky_view`.
pub fn mirror_sky_view(grid: &[[f32; 4]], sun_dir: [f32; 3], dir_sky: [f32; 3]) -> [f32; 3] {
    let d_len = (dir_sky[0] * dir_sky[0] + dir_sky[1] * dir_sky[1] + dir_sky[2] * dir_sky[2])
        .sqrt()
        .max(1e-9);
    let d = [dir_sky[0] / d_len, dir_sky[1] / d_len, dir_sky[2] / d_len];
    let theta = d[1].clamp(-1.0, 1.0).acos();
    let v = (theta / std::f32::consts::PI).clamp(0.0, 1.0);

    let s_len = (sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2])
        .sqrt()
        .max(1e-9);
    let sun = [sun_dir[0] / s_len, sun_dir[1] / s_len, sun_dir[2] / s_len];
    let horiz = |v: [f32; 3]| -> [f32; 3] {
        let h = [v[0], 0.0, v[2]];
        let hl = (h[0] * h[0] + h[2] * h[2]).sqrt();
        if hl > 1e-9 {
            [h[0] / hl, 0.0, h[2] / hl]
        } else {
            [1.0, 0.0, 0.0]
        }
    };
    let sh = horiz(sun);
    let dh = horiz(d);
    let cos_az = (dh[0] * sh[0] + dh[2] * sh[2]).clamp(-1.0, 1.0);
    let u = (cos_az.acos() / std::f32::consts::PI).clamp(0.0, 1.0);

    let fy = v * (SKY_VIEW_H - 1) as f32;
    let fx = u * (SKY_VIEW_W - 1) as f32;
    let y0 = fy.floor() as i32;
    let x0 = fx.floor() as i32;
    let y1 = (y0 + 1).min(SKY_VIEW_H as i32 - 1);
    let x1 = (x0 + 1).min(SKY_VIEW_W as i32 - 1);
    let ty = fy - y0 as f32;
    let tx = fx - x0 as f32;
    let at = |x: i32, y: i32| -> [f32; 3] {
        let p = grid[(y as usize) * SKY_VIEW_W + (x as usize)];
        [p[0], p[1], p[2]]
    };
    bilinear3(at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1), tx, ty)
}

/// CPU mirror of the WGSL `sample_aerial` (both grids).
pub fn mirror_aerial(
    trans: &[[f32; 3]],
    radiance: &[[f32; 3]],
    distance_m: f32,
    cos_zenith: f32,
) -> ([f32; 3], [f32; 3]) {
    let max_d = AERIAL_MAX_DIST_M as f32;
    let d = distance_m.clamp(0.0, max_d);
    let v = (d / max_d * AERIAL_DIST as f32 - 0.5).clamp(0.0, (AERIAL_DIST - 1) as f32);
    let theta = cos_zenith.clamp(-1.0, 1.0).acos();
    let z = (theta / std::f32::consts::PI * AERIAL_ZEN as f32 - 0.5)
        .clamp(0.0, (AERIAL_ZEN - 1) as f32);
    let d0 = v.floor() as i32;
    let d1 = (d0 + 1).min(AERIAL_DIST as i32 - 1);
    let z0 = z.floor() as i32;
    let z1 = (z0 + 1).min(AERIAL_ZEN as i32 - 1);
    let tv = v - d0 as f32;
    let tz = z - z0 as f32;
    let at = |grid: &[[f32; 3]], x: i32, y: i32| grid[(y as usize) * AERIAL_DIST + (x as usize)];
    let t = bilinear3(
        at(trans, d0, z0),
        at(trans, d1, z0),
        at(trans, d0, z1),
        at(trans, d1, z1),
        tv,
        tz,
    );
    let r = bilinear3(
        at(radiance, d0, z0),
        at(radiance, d1, z0),
        at(radiance, d0, z1),
        at(radiance, d1, z1),
        tv,
        tz,
    );
    (t, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_atmosphere::SkyParams;

    fn luts() -> SkyLuts {
        SkyLuts::new(&SkyParams::default())
    }

    #[test]
    fn payload_dimensions_match_lut_grids() {
        let l = luts();
        let p = LutPayloads::from_sky_luts(&l);
        assert_eq!(p.transmittance.len(), TRANS_W * TRANS_H);
        assert_eq!(p.sky_view.len(), SKY_VIEW_W * SKY_VIEW_H);
        assert_eq!(p.aerial_trans.len(), AERIAL_DIST * AERIAL_ZEN);
        assert_eq!(p.aerial_radiance.len(), AERIAL_DIST * AERIAL_ZEN);
        let dims = p.dims();
        assert_eq!(dims[0], (256, 64));
        assert_eq!(dims[1], (32, 32));
        assert_eq!(dims[2], (16, 16));
    }

    #[test]
    fn mirror_transmittance_matches_f64_accessor() {
        let l = luts();
        let atm_h = l.params().atmosphere_height_m as f32;
        let grid = l.transmittance_grid();
        for (alt, mu) in [
            (0.0f32, 1.0f32),
            (0.0, 0.0),
            (1000.0, -0.5),
            (8000.0, 0.7),
            (30000.0, 0.1),
            (63000.0, -1.0),
        ] {
            let m = mirror_transmittance(grid, alt, mu, atm_h);
            let c = l.transmittance_rgb(alt as f64, mu as f64);
            for ch in 0..3 {
                assert!(
                    (m[ch] as f64 - c[ch]).abs() < 2e-4,
                    "alt={alt} mu={mu} ch={ch}: {} vs {}",
                    m[ch],
                    c[ch]
                );
            }
        }
    }

    #[test]
    fn mirror_sky_view_matches_f64_accessor() {
        let l = luts();
        let sun = l.params().sun_direction;
        let sun_a = [sun.x as f32, sun.y as f32, sun.z as f32];
        let grid = l.sky_view_grid();
        for dir in [
            glam::DVec3::new(0.0, 1.0, 0.0),
            glam::DVec3::new(0.2, 0.6, -0.4),
            glam::DVec3::new(1.0, 0.0, 0.0),
            glam::DVec3::new(-0.3, -0.1, 0.7),
        ] {
            let d = dir.normalize();
            let m = mirror_sky_view(grid, sun_a, [d.x as f32, d.y as f32, d.z as f32]);
            let c = l.sky_view_rgb(d);
            for ch in 0..3 {
                let scale = c[ch].abs().max(1e-3);
                assert!(
                    ((m[ch] as f64 - c[ch]).abs()) / scale < 2e-3,
                    "dir={d:?} ch={ch}: {} vs {}",
                    m[ch],
                    c[ch]
                );
            }
        }
    }

    #[test]
    fn mirror_aerial_matches_f64_accessor() {
        let l = luts();
        let grid_t: Vec<[f32; 3]> = l.aerial_grid().iter().map(|p| p.0).collect();
        let grid_r: Vec<[f32; 3]> = l.aerial_grid().iter().map(|p| p.1).collect();
        for (dist, cos) in [
            (0.0f32, 1.0f32),
            (5_000.0, 0.9),
            (40_000.0, 0.2),
            (79_000.0, -0.4),
        ] {
            let m = mirror_aerial(&grid_t, &grid_r, dist, cos);
            let c = l.aerial_perspective(dist as f64, cos as f64);
            for ch in 0..3 {
                assert!(
                    (m.0[ch] as f64 - c.0[ch]).abs() < 2e-4,
                    "trans dist={dist} cos={cos} ch={ch}"
                );
                let scale = c.1[ch].abs().max(1e-3);
                assert!(
                    ((m.1[ch] as f64 - c.1[ch]).abs()) / scale < 2e-3,
                    "radiance dist={dist} cos={cos} ch={ch}: {} vs {}",
                    m.1[ch],
                    c.1[ch]
                );
            }
        }
    }
}
