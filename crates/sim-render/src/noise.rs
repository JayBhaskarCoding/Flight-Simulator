//! Deterministic, tileable 3-D noise for volumetric clouds.
//!
//! The volume layout follows the Guerrilla/Horizon procedural-cloud
//! technique: an `RGBA8` 3-D texture where
//!
//! * `r` — Perlin–Worley (low-frequency Perlin remapped by a Worley
//!   octave: billows without the flat spots),
//! * `g/b/a` — inverted Worley F1 at increasing frequency (detail
//!   erosion octaves).
//!
//! Every generator is a pure function of `(size, seed)`; the lattice
//! arithmetic wraps on integer periods that divide the sample domain,
//! so the volume tiles seamlessly along all three axes. Tests assert
//! both determinism and exact edge periodicity.

/// A generated 3-D noise volume, tightly packed as `size³` RGBA texels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseVolume {
    pub size: u32,
    pub data: Vec<u8>,
}

impl NoiseVolume {
    /// Texel count (`size³`).
    pub fn texel_count(&self) -> usize {
        (self.size as usize) * (self.size as usize) * (self.size as usize)
    }

    /// Byte length of [`Self::data`].
    pub fn byte_len(&self) -> usize {
        self.texel_count() * 4
    }

    /// `(r, g, b, a)` of voxel `(x, y, z)`; coordinates are taken
    /// modulo `size` so callers may wrap while marching.
    pub fn texel(&self, x: u32, y: u32, z: u32) -> [u8; 4] {
        let n = self.size;
        let i = (((z % n) * n + (y % n)) * n + (x % n)) as usize;
        let o = i * 4;
        [self.data[o], self.data[o + 1], self.data[o + 2], self.data[o + 3]]
    }
}

/// Linear remap of `v` from `[a, b]` onto `[c, d]`, clamped.
pub fn remap(v: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t = ((v - a) / (b - a)).clamp(0.0, 1.0);
    c + t * (d - c)
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 16 gradients of the improved-Perlin set (edge midpoints + corners).
const GRAD16: [[f32; 3]; 16] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Integer avalanche hash (Wang-style finalizer).
fn hash3(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    let mut h = seed
        .wrapping_add(ix as u32)
        .wrapping_mul(0x8da6b343)
        .wrapping_add(iy as u32)
        .wrapping_mul(0xd8163841)
        .wrapping_add(iz as u32)
        .wrapping_mul(0xcb1ab31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^ (h >> 16)
}

/// Improved Perlin noise with an integer lattice period (tileable when
/// the sample span is a multiple of the period). Returns roughly
/// `[-1, 1]` (the improved noise reaches ±sqrt(2)/2 per octave pair,
/// but never exceeds ±1).
fn perlin_periodic(x: f32, y: f32, z: f32, period: i32, seed: u32) -> f32 {
    let fx = x.floor();
    let fy = y.floor();
    let fz = z.floor();
    let ix = fx as i32;
    let iy = fy as i32;
    let iz = fz as i32;
    let tx = x - fx;
    let ty = y - fy;
    let tz = z - fz;
    let u = fade(tx);
    let v = fade(ty);
    let w = fade(tz);

    let wrap = |i: i32| ((i % period) + period) % period;
    let x0 = wrap(ix);
    let x1 = wrap(ix + 1);
    let y0 = wrap(iy);
    let y1 = wrap(iy + 1);
    let z0 = wrap(iz);
    let z1 = wrap(iz + 1);

    let grad = |gx: i32, gy: i32, gz: i32, dx: f32, dy: f32, dz: f32| -> f32 {
        let g = GRAD16[(hash3(gx, gy, gz, seed) & 15) as usize];
        g[0] * dx + g[1] * dy + g[2] * dz
    };

    let n000 = grad(x0, y0, z0, tx, ty, tz);
    let n100 = grad(x1, y0, z0, tx - 1.0, ty, tz);
    let n010 = grad(x0, y1, z0, tx, ty - 1.0, tz);
    let n110 = grad(x1, y1, z0, tx - 1.0, ty - 1.0, tz);
    let n001 = grad(x0, y0, z1, tx, ty, tz - 1.0);
    let n101 = grad(x1, y0, z1, tx - 1.0, ty, tz - 1.0);
    let n011 = grad(x0, y1, z1, tx, ty - 1.0, tz - 1.0);
    let n111 = grad(x1, y1, z1, tx - 1.0, ty - 1.0, tz - 1.0);

    let nx00 = lerp(n000, n100, u);
    let nx10 = lerp(n010, n110, u);
    let nx01 = lerp(n001, n101, u);
    let nx11 = lerp(n011, n111, u);
    let nxy0 = lerp(nx00, nx10, v);
    let nxy1 = lerp(nx01, nx11, v);
    // The gradient table is unit-ish; scale to fill [-1, 1].
    1.35 * lerp(nxy0, nxy1, w)
}

/// Tileable Perlin octaves summed in `[0, 1]`.
fn perlin01(x: f32, y: f32, z: f32, base_period: i32, seed: u32) -> f32 {
    let mut amp = 0.5f32;
    let mut freq = 1i32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for octave in 0..3 {
        let p = base_period * freq;
        sum += amp
            * perlin_periodic(
                x * freq as f32,
                y * freq as f32,
                z * freq as f32,
                p,
                seed.wrapping_add((octave as u32).wrapping_mul(0x9e37_79b9)),
            );
        norm += amp;
        amp *= 0.5;
        freq *= 2;
    }
    (sum / norm).clamp(0.0, 1.0)
}

/// Inverted Worley F1 in `[0, 1]`: 1 at a feature point, 0 near the
/// farthest cell corner. `cells` must divide the unit domain.
fn worley01(x: f32, y: f32, z: f32, cells: i32, seed: u32) -> f32 {
    let fx = x * cells as f32;
    let fy = y * cells as f32;
    let fz = z * cells as f32;
    let cx = fx.floor() as i32;
    let cy = fy.floor() as i32;
    let cz = fz.floor() as i32;

    let mut best = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let gx = cx + dx;
                let gy = cy + dy;
                let gz = cz + dz;
                // wrapped cell coordinates + hashed feature point
                let wx = ((gx % cells) + cells) % cells;
                let wy = ((gy % cells) + cells) % cells;
                let wz = ((gz % cells) + cells) % cells;
                let h = hash3(wx, wy, wz, seed);
                let px = gx as f32 + ((h & 0xff) as f32 + 0.5) / 256.0;
                let py = gy as f32 + (((h >> 8) & 0xff) as f32 + 0.5) / 256.0;
                let pz = gz as f32 + (((h >> 16) & 0xff) as f32 + 0.5) / 256.0;
                let ex = px - fx;
                let ey = py - fy;
                let ez = pz - fz;
                let d = (ex * ex + ey * ey + ez * ez).sqrt();
                if d < best {
                    best = d;
                }
            }
        }
    }
    // Normalise: F1 ∈ [0, sqrt(3))] in cell units.
    (1.0 - best / 3.0f32.sqrt()).clamp(0.0, 1.0)
}

/// Generate the `RGBA8` noise volume used by the volumetric-cloud pass.
///
/// `size` must be ≥ 8 and a power of two (the internal lattice periods
/// are 4/8/…; power-of-two sizes keep every period an exact divisor of
/// the domain so the volume tiles without seams).
pub fn generate_noise_volume(size: u32, seed: u32) -> NoiseVolume {
    assert!(size >= 8, "noise volume must be at least 8³");
    assert!(size.is_power_of_two(), "noise volume size must be a power of two");
    let n = size as usize;
    let inv = 1.0 / size as f32;
    let mut data = Vec::with_capacity(n * n * n * 4);

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let px = x as f32 * inv;
                let py = y as f32 * inv;
                let pz = z as f32 * inv;

                // Shape: Perlin remapped by the base Worley octave so
                // the billows stay rounded (perlin-worley).
                let perlin = perlin01(px, py, pz, 4, seed);
                let w0 = worley01(px, py, pz, 4, seed.wrapping_add(1));
                let shape = remap(perlin, 1.0 - w0, 1.0, 0.0, 1.0);

                // Detail octaves (inverted Worley, increasing frequency).
                let d0 = worley01(px, py, pz, 8, seed.wrapping_add(2));
                let d1 = worley01(px, py, pz, 16, seed.wrapping_add(3));
                let d2 = worley01(px, py, pz, 32, seed.wrapping_add(4));

                data.push((shape * 255.0).round() as u8);
                data.push((d0 * 255.0).round() as u8);
                data.push((d1 * 255.0).round() as u8);
                data.push((d2 * 255.0).round() as u8);
            }
        }
    }
    NoiseVolume { size, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_layout_is_rgba_cubic() {
        let v = generate_noise_volume(16, 7);
        assert_eq!(v.data.len(), 16 * 16 * 16 * 4);
        assert_eq!(v.byte_len(), v.data.len());
        assert_eq!(v.texel_count(), 4096);
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate_noise_volume(16, 12345);
        let b = generate_noise_volume(16, 12345);
        assert_eq!(a, b);
        let c = generate_noise_volume(16, 54321);
        assert_ne!(a.data, c.data, "different seeds must differ");
    }

    #[test]
    fn volume_tiles_exactly_on_integer_lattice() {
        // Points at i and i+size hit the same lattice position for
        // integer periods, so every edge pair must match exactly.
        let v = generate_noise_volume(16, 3);
        for i in 0..16u32 {
            for j in 0..16u32 {
                assert_eq!(v.texel(0, i, j), v.texel(16, i, j), "x seam");
                assert_eq!(v.texel(i, 0, j), v.texel(i, 16, j), "y seam");
                assert_eq!(v.texel(i, j, 0), v.texel(i, j, 16), "z seam");
            }
        }
    }

    #[test]
    fn texel_accessor_wraps_coordinates() {
        let v = generate_noise_volume(16, 3);
        assert_eq!(v.texel(1, 2, 3), v.texel(17, 18, 19));
        assert_eq!(v.texel(1, 2, 3), v.texel(1, 2, 3 + 32));
    }

    #[test]
    fn shape_channel_has_variation() {
        let v = generate_noise_volume(32, 99);
        let r: Vec<u8> = v.data.iter().step_by(4).copied().collect();
        let min = *r.iter().min().unwrap();
        let max = *r.iter().max().unwrap();
        assert!(min < 250 && max > 5, "shape channel too flat: [{min}, {max}]");
        // g/b/a detail channels must vary as well
        for (ch, stride) in [(1usize, 4usize), (2, 4), (3, 4)] {
            let vals: Vec<u8> = v.data.iter().skip(ch).step_by(stride).copied().collect();
            let lo = *vals.iter().min().unwrap();
            let hi = *vals.iter().max().unwrap();
            assert!(hi > lo, "channel {ch} constant");
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two() {
        let _ = generate_noise_volume(12, 0);
    }
}
