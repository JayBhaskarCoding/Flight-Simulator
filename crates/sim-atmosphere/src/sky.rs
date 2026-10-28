//! Physically-based sky radiance via single-scattering (Nishita) look-up
//! tables, ready for GPU upload (all storage `f32`, all computation `f64`).
//!
//! Model (public-domain physics):
//!
//! * Rayleigh scattering `beta_R(lambda) = 13.558e-6 * (550/lambda)^4` m^-1
//!   with exponential density (scale height 8 km).
//! * Mie scattering `3.996e-6` / extinction `4.402e-6` m^-1 at 550 nm with
//!   1.2 km scale height and Henyey-Greenstein phase (`g = 0.7`).
//! * Ozone absorption with a triangular profile centred at 25 km
//!   (Chappuis-band coefficients at 680/550/440 nm).
//!
//! LUTs built on construction:
//!
//! * **transmittance** 64 (altitude) × 256 (cosine zenith), RGB
//!   (channels = 680 / 550 / 440 nm)
//! * **sky-view** 32 × 32 over `(view zenith, azimuth relative to sun)`
//! * **aerial perspective** 16 (distance) × 16 (view zenith), RGB
//!   transmittance + RGB in-scattered radiance

use glam::DVec3;

/// Transmittance LUT height (altitude rows).
pub const TRANS_W: usize = 256;
/// Transmittance LUT width — kept as the row count alias of the spec (64).
pub const TRANS_H: usize = 64;
/// Sky-view LUT dimensions.
pub const SKY_VIEW_W: usize = 32;
pub const SKY_VIEW_H: usize = 32;
/// Aerial-perspective LUT dimensions.
pub const AERIAL_DIST: usize = 16;
pub const AERIAL_ZEN: usize = 16;

/// Rayleigh scattering at 550 nm, sea level (m^-1).
pub const RAYLEIGH_BETA_550: f64 = 13.558e-6;
/// Mie scattering at 550 nm, sea level (m^-1).
pub const MIE_BETA_550: f64 = 3.996e-6;
/// Mie extinction at 550 nm, sea level (m^-1).
pub const MIE_EXT_550: f64 = 4.402e-6;

/// Ozone absorption coefficients (m^-1) at 680 / 550 / 440 nm.
pub const OZONE_BETA: [f64; 3] = [0.650e-6, 1.881e-6, 0.085e-6];
/// Wavelengths of the RGB channels (nm).
pub const CHANNEL_NM: [f64; 3] = [680.0, 550.0, 440.0];

/// Sky model parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyParams {
    /// Direction **towards** the sun (world frame, Y-up).
    pub sun_direction: DVec3,
    /// Observer altitude above the surface, m.
    pub observer_altitude_m: f64,
    /// Effective planet radius, m.
    pub planet_radius_m: f64,
    /// Top of the modelled atmosphere above the surface, m.
    pub atmosphere_height_m: f64,
    /// Rayleigh scale height, m.
    pub rayleigh_scale_height_m: f64,
    /// Mie scale height, m.
    pub mie_scale_height_m: f64,
    /// Multiplier on the Mie coefficients (turbidity).
    pub mie_turbidity: f64,
    /// Henyey-Greenstein asymmetry for Mie.
    pub mie_g: f64,
}

impl Default for SkyParams {
    fn default() -> Self {
        Self {
            sun_direction: DVec3::new(0.3, 0.75, 0.45).normalize(),
            observer_altitude_m: 0.0,
            planet_radius_m: 6_360_000.0,
            atmosphere_height_m: 64_000.0,
            rayleigh_scale_height_m: 8_000.0,
            mie_scale_height_m: 1_200.0,
            mie_turbidity: 1.0,
            mie_g: 0.7,
        }
    }
}

impl SkyParams {
    /// Parameters with an explicit sun direction.
    pub fn with_sun(sun_direction: DVec3) -> Self {
        Self { sun_direction: sun_direction.normalize(), ..Self::default() }
    }
}

/// Built sky look-up tables.
#[derive(Debug, Clone)]
pub struct SkyLuts {
    params: SkyParams,
    /// RGB transmittance, row-major `[altitude][mu][rgb]`.
    transmittance: Vec<[f32; 3]>,
    /// RGBA sky-view radiance (`a = 1`), row-major `[zenith][azimuth]`.
    sky_view: Vec<[f32; 4]>,
    /// `[dist][zenith]` each `(rgb transmittance, rgb radiance)`.
    aerial: Vec<([f32; 3], [f32; 3])>,
}

// ---------------------------------------------------------------------------
// spectral helpers
// ---------------------------------------------------------------------------

fn rayleigh_beta(lambda_nm: f64) -> f64 {
    let r = 550.0 / lambda_nm;
    RAYLEIGH_BETA_550 * r * r * r * r
}

fn mie_extinction(params: &SkyParams) -> f64 {
    MIE_EXT_550 * params.mie_turbidity
}

fn mie_scattering(params: &SkyParams) -> f64 {
    MIE_BETA_550 * params.mie_turbidity
}

/// Relative number densities `(rayleigh, mie, ozone)` at altitude `h`.
fn densities(alt_m: f64, params: &SkyParams) -> (f64, f64, f64) {
    let r = (-alt_m / params.rayleigh_scale_height_m).exp();
    let m = (-alt_m / params.mie_scale_height_m).exp();
    // triangular ozone layer: zero at 10 km and 40 km, peak at 25 km
    let o = (1.0 - (alt_m - 25_000.0).abs() / 15_000.0).max(0.0);
    (r, m, o)
}

/// Rayleigh phase function for `cos_theta`.
fn rayleigh_phase(cos_theta: f64) -> f64 {
    3.0 / (16.0 * std::f64::consts::PI) * (1.0 + cos_theta * cos_theta)
}

/// Henyey-Greenstein phase function.
fn hg_phase(cos_theta: f64, g: f64) -> f64 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    (1.0 - g2) / (4.0 * std::f64::consts::PI * denom.max(1e-9).powf(1.5))
}

// ---------------------------------------------------------------------------
// geometry: intersections with the planet and atmosphere shells
// ---------------------------------------------------------------------------

/// Distances to the ground hit (if any) and atmosphere exit along a ray
/// from radius `r` with cosine `mu` (relative to local up).
fn ray_intervals(r: f64, mu: f64, planet_r: f64, top_r: f64) -> (Option<f64>, f64) {
    let disc_g = r * r * mu * mu - (r * r - planet_r * planet_r);
    let t_ground = if disc_g >= 0.0 {
        let sq = disc_g.sqrt();
        let t = -r * mu - sq;
        if t > 0.0 {
            Some(t)
        } else {
            None
        }
    } else {
        None
    };
    let disc_t = r * r * mu * mu - (r * r - top_r * top_r);
    let t_top = if disc_t >= 0.0 {
        (-r * mu + disc_t.sqrt()).max(0.0)
    } else {
        0.0
    };
    (t_ground, t_top)
}

/// Accumulated optical depths `(rayleigh, mie, ozone)` along `t ∈ [0, t_end]`.
fn optical_depths(
    origin: DVec3,
    dir: DVec3,
    t_end: f64,
    steps: usize,
    params: &SkyParams,
) -> [f64; 3] {
    let mut depth = [0.0f64; 3];
    if t_end <= 0.0 || steps == 0 {
        return depth;
    }
    let dt = t_end / steps as f64;
    let mut prev = densities((origin.length() - params.planet_radius_m).max(0.0), params);
    for i in 1..=steps {
        let t = dt * i as f64;
        let p = origin + dir * t;
        let alt = (p.length() - params.planet_radius_m).max(0.0);
        let cur = densities(alt, params);
        depth[0] += 0.5 * (prev.0 + cur.0) * dt;
        depth[1] += 0.5 * (prev.1 + cur.1) * dt;
        depth[2] += 0.5 * (prev.2 + cur.2) * dt;
        prev = cur;
    }
    depth
}

/// Full RGB transmittance from a point at `altitude_m` along `cos_zenith`
/// (1 = straight up), by direct quadrature (reference for the LUT).
fn transmittance_quad(
    altitude_m: f64,
    cos_zenith: f64,
    params: &SkyParams,
) -> [f64; 3] {
    let r = params.planet_radius_m + altitude_m.max(0.0);
    let (t_ground, t_top) = ray_intervals(r, cos_zenith, params.planet_radius_m, r + params.atmosphere_height_m);
    if let Some(tg) = t_ground {
        if tg < t_top || t_top <= 0.0 {
            return [0.0; 3]; // blocked by the planet
        }
    }
    if t_top <= 0.0 {
        return [0.0; 3];
    }
    let dir = DVec3::new(
        (1.0 - cos_zenith * cos_zenith).max(0.0).sqrt(),
        cos_zenith,
        0.0,
    );
    let origin = DVec3::new(0.0, r, 0.0);
    let depth = optical_depths(origin, dir, t_top, 48, params);
    let ext = mie_extinction(params);
    let mut out = [0.0f64; 3];
    for (c, lam) in CHANNEL_NM.iter().enumerate() {
        let tau = rayleigh_beta(*lam) * depth[0] + ext * depth[1] + OZONE_BETA[c] * depth[2];
        out[c] = (-tau).exp();
    }
    out
}

// ---------------------------------------------------------------------------
// LUT construction
// ---------------------------------------------------------------------------

impl SkyLuts {
    /// Build every LUT for `params`.
    pub fn new(params: &SkyParams) -> Self {
        let transmittance = Self::build_transmittance(params);
        let sky_view = Self::build_sky_view(params, &transmittance);
        let aerial = Self::build_aerial(params, &transmittance);
        Self { params: *params, transmittance, sky_view, aerial }
    }

    /// The parameters these LUTs were built with.
    pub fn params(&self) -> &SkyParams {
        &self.params
    }

    fn build_transmittance(params: &SkyParams) -> Vec<[f32; 3]> {
        let mut out = Vec::with_capacity(TRANS_H * TRANS_W);
        let top_r = params.planet_radius_m + params.atmosphere_height_m;
        let ext = mie_extinction(params);
        for iy in 0..TRANS_H {
            let frac = (iy as f64 + 0.5) / TRANS_H as f64;
            let alt = frac * params.atmosphere_height_m;
            let r = params.planet_radius_m + alt;
            for ix in 0..TRANS_W {
                let mu = -1.0 + 2.0 * ((ix as f64 + 0.5) / TRANS_W as f64);
                let (t_ground, t_top) = ray_intervals(r, mu, params.planet_radius_m, top_r);
                let blocked = matches!(t_ground, Some(tg) if tg <= t_top) || t_top <= 0.0;
                let px = if blocked {
                    [0.0f64; 3]
                } else {
                    let dir = DVec3::new((1.0 - mu * mu).max(0.0).sqrt(), mu, 0.0);
                    let origin = DVec3::new(0.0, r, 0.0);
                    let depth = optical_depths(origin, dir, t_top, 32, params);
                    let mut v = [0.0f64; 3];
                    for (c, lam) in CHANNEL_NM.iter().enumerate() {
                        let tau =
                            rayleigh_beta(*lam) * depth[0] + ext * depth[1] + OZONE_BETA[c] * depth[2];
                        v[c] = (-tau).exp();
                    }
                    v
                };
                out.push([px[0] as f32, px[1] as f32, px[2] as f32]);
            }
        }
        out
    }

    /// Bilinear RGB transmittance from the built LUT.
    pub fn transmittance_rgb(&self, altitude_m: f64, cos_zenith: f64) -> [f64; 3] {
        let params = &self.params;
        let ay = (altitude_m.max(0.0) / params.atmosphere_height_m).clamp(0.0, 1.0);
        let ax = ((cos_zenith + 1.0) * 0.5).clamp(0.0, 1.0);
        let fy = ay * (TRANS_H - 1) as f64;
        let fx = ax * (TRANS_W - 1) as f64;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(TRANS_H - 1);
        let x0 = fx.floor() as usize;
        let x1 = (x0 + 1).min(TRANS_W - 1);
        let ty = fy - y0 as f64;
        let tx = fx - x0 as f64;
        let mut out = [0.0f64; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let p00 = self.transmittance[y0 * TRANS_W + x0][c] as f64;
            let p10 = self.transmittance[y0 * TRANS_W + x1][c] as f64;
            let p01 = self.transmittance[y1 * TRANS_W + x0][c] as f64;
            let p11 = self.transmittance[y1 * TRANS_W + x1][c] as f64;
            *o = (p00 * (1.0 - tx) + p10 * tx) * (1.0 - ty) + (p01 * (1.0 - tx) + p11 * tx) * ty;
        }
        out
    }

    /// Monochromatic transmittance (direct quadrature, any wavelength).
    pub fn monochromatic_transmittance(
        &self,
        altitude_m: f64,
        cos_zenith: f64,
        wavelength_nm: f64,
    ) -> f64 {
        let params = &self.params;
        let r = params.planet_radius_m + altitude_m.max(0.0);
        let top_r = r + params.atmosphere_height_m;
        let (t_ground, t_top) = ray_intervals(r, cos_zenith, params.planet_radius_m, top_r);
        if let Some(tg) = t_ground {
            if tg <= t_top {
                return 0.0;
            }
        }
        if t_top <= 0.0 {
            return 0.0;
        }
        let dir = DVec3::new((1.0 - cos_zenith * cos_zenith).max(0.0).sqrt(), cos_zenith, 0.0);
        let origin = DVec3::new(0.0, r, 0.0);
        let depth = optical_depths(origin, dir, t_top, 64, params);
        let tau = rayleigh_beta(wavelength_nm) * depth[0]
            + mie_extinction(params) * depth[1]
            + ozone_for_wavelength(wavelength_nm) * depth[2];
        (-tau).exp()
    }

    /// Direct-quadrature RGB transmittance (reference implementation,
    /// independent of the LUT discretisation).
    pub fn transmittance_quadrature_rgb(&self, altitude_m: f64, cos_zenith: f64) -> [f64; 3] {
        transmittance_quad(altitude_m, cos_zenith, &self.params)
    }

    fn build_sky_view(params: &SkyParams, trans: &[[f32; 3]]) -> Vec<[f32; 4]> {
        // reuse the same transmittance sampling by wrapping the raw buffer
        let sample_trans = |alt: f64, mu: f64| -> [f64; 3] {
            let ay = (alt.max(0.0) / params.atmosphere_height_m).clamp(0.0, 1.0);
            let ax = ((mu + 1.0) * 0.5).clamp(0.0, 1.0);
            let fy = ay * (TRANS_H - 1) as f64;
            let fx = ax * (TRANS_W - 1) as f64;
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(TRANS_H - 1);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(TRANS_W - 1);
            let ty = fy - y0 as f64;
            let tx = fx - x0 as f64;
            let mut out = [0.0f64; 3];
            for (c, o) in out.iter_mut().enumerate() {
                let p00 = trans[y0 * TRANS_W + x0][c] as f64;
                let p10 = trans[y0 * TRANS_W + x1][c] as f64;
                let p01 = trans[y1 * TRANS_W + x0][c] as f64;
                let p11 = trans[y1 * TRANS_W + x1][c] as f64;
                *o =
                    (p00 * (1.0 - tx) + p10 * tx) * (1.0 - ty) + (p01 * (1.0 - tx) + p11 * tx) * ty;
            }
            out
        };

        let sun = params.sun_direction.normalize();
        let cam = DVec3::new(0.0, params.planet_radius_m + params.observer_altitude_m, 0.0);

        let mut out = Vec::with_capacity(SKY_VIEW_H * SKY_VIEW_W);
        for iy in 0..SKY_VIEW_H {
            // v = zenith angle / pi, cell centres
            let theta = ((iy as f64 + 0.5) / SKY_VIEW_H as f64) * std::f64::consts::PI;
            let (sin_t, cos_t) = theta.sin_cos();
            for ix in 0..SKY_VIEW_W {
                let phi = ((ix as f64 + 0.5) / SKY_VIEW_W as f64) * std::f64::consts::PI;
                // direction in the plane containing the sun azimuth:
                // build from sun horizontal basis
                let sun_h = {
                    let h = DVec3::new(sun.x, 0.0, sun.z);
                    if h.length() > 1e-9 {
                        h.normalize()
                    } else {
                        DVec3::X
                    }
                };
                let perp = DVec3::Y.cross(sun_h).normalize(); // horizontal, orthogonal
                let dir = DVec3::Y * cos_t + (sun_h * phi.cos() + perp * phi.sin()) * sin_t;
                let cos_phase = dir.dot(sun);
                let p_r = rayleigh_phase(cos_phase);
                let p_m = hg_phase(cos_phase, params.mie_g);

                let (t_ground, t_top) = ray_intervals(
                    cam.length(),
                    dir.y,
                    params.planet_radius_m,
                    params.planet_radius_m + params.atmosphere_height_m,
                );
                let t_end = match t_ground {
                    Some(tg) if tg < t_top => tg,
                    _ => t_top,
                };
                let radiance = integrate_single_scatter(
                    cam,
                    dir,
                    t_end,
                    params,
                    &sample_trans,
                    p_r,
                    p_m,
                );
                out.push([radiance[0] as f32, radiance[1] as f32, radiance[2] as f32, 1.0]);
            }
        }
        out
    }

    /// Sky-view radiance (RGB) for a world direction (Y-up) by bilinear
    /// LUT lookup. Sun-below-horizon conditions return ~0.
    pub fn sky_view_rgb(&self, view_dir: DVec3) -> [f64; 3] {
        let d = view_dir.normalize();
        let theta = d.y.clamp(-1.0, 1.0).acos(); // 0 = zenith
        let v = (theta / std::f64::consts::PI).clamp(0.0, 1.0);

        let sun = self.params.sun_direction.normalize();
        let sun_h = {
            let h = DVec3::new(sun.x, 0.0, sun.z);
            if h.length() > 1e-9 {
                h.normalize()
            } else {
                DVec3::X
            }
        };
        let dir_h = {
            let h = DVec3::new(d.x, 0.0, d.z);
            if h.length() > 1e-9 {
                h.normalize()
            } else {
                DVec3::X
            }
        };
        let cos_az = dir_h.dot(sun_h).clamp(-1.0, 1.0);
        let u = (cos_az.acos() / std::f64::consts::PI).clamp(0.0, 1.0);

        let fy = v * (SKY_VIEW_H - 1) as f64;
        let fx = u * (SKY_VIEW_W - 1) as f64;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(SKY_VIEW_H - 1);
        let x0 = fx.floor() as usize;
        let x1 = (x0 + 1).min(SKY_VIEW_W - 1);
        let ty = fy - y0 as f64;
        let tx = fx - x0 as f64;
        let mut out = [0.0f64; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let p00 = self.sky_view[y0 * SKY_VIEW_W + x0][c] as f64;
            let p10 = self.sky_view[y0 * SKY_VIEW_W + x1][c] as f64;
            let p01 = self.sky_view[y1 * SKY_VIEW_W + x0][c] as f64;
            let p11 = self.sky_view[y1 * SKY_VIEW_W + x1][c] as f64;
            *o = (p00 * (1.0 - tx) + p10 * tx) * (1.0 - ty) + (p01 * (1.0 - tx) + p11 * tx) * ty;
        }
        out
    }

    fn build_aerial(params: &SkyParams, trans: &[[f32; 3]]) -> Vec<([f32; 3], [f32; 3])> {
        let sample_trans = |alt: f64, mu: f64| -> [f64; 3] {
            let ay = (alt.max(0.0) / params.atmosphere_height_m).clamp(0.0, 1.0);
            let ax = ((mu + 1.0) * 0.5).clamp(0.0, 1.0);
            let fy = ay * (TRANS_H - 1) as f64;
            let fx = ax * (TRANS_W - 1) as f64;
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(TRANS_H - 1);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(TRANS_W - 1);
            let ty = fy - y0 as f64;
            let tx = fx - x0 as f64;
            let mut out = [0.0f64; 3];
            for (c, o) in out.iter_mut().enumerate() {
                let p00 = trans[y0 * TRANS_W + x0][c] as f64;
                let p10 = trans[y0 * TRANS_W + x1][c] as f64;
                let p01 = trans[y1 * TRANS_W + x0][c] as f64;
                let p11 = trans[y1 * TRANS_W + x1][c] as f64;
                *o =
                    (p00 * (1.0 - tx) + p10 * tx) * (1.0 - ty) + (p01 * (1.0 - tx) + p11 * tx) * ty;
            }
            out
        };

        let sun = params.sun_direction.normalize();
        let cam = DVec3::new(0.0, params.planet_radius_m + params.observer_altitude_m, 0.0);
        let _ = sun;

        let max_dist = 80_000.0f64;
        let mut out = Vec::with_capacity(AERIAL_DIST * AERIAL_ZEN);
        for iz in 0..AERIAL_ZEN {
            let theta = ((iz as f64 + 0.5) / AERIAL_ZEN as f64) * std::f64::consts::PI;
            let (sin_t, cos_t) = theta.sin_cos();
            let dir = DVec3::new(sin_t, cos_t, 0.0);
            let cos_phase = dir.dot(sun);
            let p_r = rayleigh_phase(cos_phase);
            let p_m = hg_phase(cos_phase, params.mie_g);
            for id in 0..AERIAL_DIST {
                let frac = (id as f64 + 1.0) / AERIAL_DIST as f64;
                let dist = frac * max_dist;
                // march to `dist`, stopping at the ground if hit sooner
                let (t_ground, _t_top) = ray_intervals(
                    cam.length(),
                    dir.y,
                    params.planet_radius_m,
                    params.planet_radius_m + params.atmosphere_height_m,
                );
                let t_end = match t_ground {
                    Some(tg) if tg < dist => tg,
                    _ => dist,
                };
                let (trans_rgb, radiance) = integrate_aerial(
                    cam, dir, t_end, params, &sample_trans, p_r, p_m,
                );
                let t32 = [
                    trans_rgb[0] as f32,
                    trans_rgb[1] as f32,
                    trans_rgb[2] as f32,
                ];
                let r32 = [
                    radiance[0] as f32,
                    radiance[1] as f32,
                    radiance[2] as f32,
                ];
                out.push((t32, r32));
            }
        }
        out
    }

    /// Aerial perspective along a view ray: total path transmittance and
    /// in-scattered radiance for a path of `distance_m` at `cos_zenith`.
    pub fn aerial_perspective(&self, distance_m: f64, cos_zenith: f64) -> ([f64; 3], [f64; 3]) {
        let max_dist = 80_000.0f64;
        let d = distance_m.clamp(0.0, max_dist);
        let v = (d / max_dist * AERIAL_DIST as f64 - 0.5).clamp(0.0, (AERIAL_DIST - 1) as f64);
        let theta = cos_zenith.clamp(-1.0, 1.0).acos();
        let z = (theta / std::f64::consts::PI * AERIAL_ZEN as f64 - 0.5)
            .clamp(0.0, (AERIAL_ZEN - 1) as f64);
        let d0 = v.floor() as usize;
        let d1 = (d0 + 1).min(AERIAL_DIST - 1);
        let z0 = z.floor() as usize;
        let z1 = (z0 + 1).min(AERIAL_ZEN - 1);
        let tv = v - d0 as f64;
        let tz = z - z0 as f64;
        let mut t_out = [0.0f64; 3];
        let mut r_out = [0.0f64; 3];
        for c in 0..3 {
            let get = |di: usize, zi: usize| -> (f64, f64) {
                let cell = self.aerial[zi * AERIAL_DIST + di];
                (cell.0[c] as f64, cell.1[c] as f64)
            };
            let (t00, r00) = get(d0, z0);
            let (t10, r10) = get(d1, z0);
            let (t01, r01) = get(d0, z1);
            let (t11, r11) = get(d1, z1);
            let lerp = |a: f64, b: f64, c_: f64, dd: f64| {
                (a * (1.0 - tv) + b * tv) * (1.0 - tz) + (c_ * (1.0 - tv) + dd * tv) * tz
            };
            t_out[c] = lerp(t00, t10, t01, t11);
            r_out[c] = lerp(r00, r10, r01, r11);
        }
        (t_out, r_out)
    }
}

/// Ozone absorption for an arbitrary wavelength via linear interpolation of
/// the Chappuis-band reference coefficients (constant outside the table).
fn ozone_for_wavelength(nm: f64) -> f64 {
    let table = CHANNEL_NM;
    let vals = OZONE_BETA;
    if nm <= table[0] {
        // below 680 nm the Chappuis band falls off; clamp to the 680 value
        if nm >= table[1] {
            // between 680 and 550: interpolate
            let t = (nm - table[1]) / (table[0] - table[1]);
            return vals[1] + (vals[0] - vals[1]) * t;
        }
        if nm >= table[2] {
            let t = (nm - table[2]) / (table[1] - table[2]);
            return vals[2] + (vals[1] - vals[2]) * t;
        }
        return vals[2];
    }
    vals[0]
}

/// Single-scatter integration for one view segment (sky-view LUT).
fn integrate_single_scatter<F>(
    origin: DVec3,
    dir: DVec3,
    t_end: f64,
    params: &SkyParams,
    sample_trans: &F,
    p_r: f64,
    p_m: f64,
) -> [f64; 3]
where
    F: Fn(f64, f64) -> [f64; 3],
{
    let steps = 48usize;
    if t_end <= 0.0 {
        return [0.0; 3];
    }
    let dt = t_end / steps as f64;
    let sun = params.sun_direction.normalize();
    let ext = mie_extinction(params);
    let scat_m = mie_scattering(params);
    let mut acc = [0.0f64; 3];

    for i in 0..steps {
        let t0 = dt * i as f64;
        let t1 = dt * (i + 1) as f64;
        let mut local = [0.0f64; 3];
        for (t, w) in [(t0, 0.5), (t1, 0.5)] {
            let p = origin + dir * t;
            let r = p.length();
            let alt = (r - params.planet_radius_m).max(0.0);
            let (dr, dm, _do) = densities(alt, params);
            // view transmittance accumulated analytically from depths so far
            let depth = optical_depths(origin, dir, t, 24.min(steps), params);
            let t_view = {
                let mut v = [0.0f64; 3];
                for (c, lam) in CHANNEL_NM.iter().enumerate() {
                    let tau = rayleigh_beta(*lam) * depth[0] + ext * depth[1] + OZONE_BETA[c] * depth[2];
                    v[c] = (-tau).exp();
                }
                v
            };
            // sun transmittance at the sample point
            let p_hat = p / r;
            let cos_sun = p_hat.dot(sun);
            let t_sun = sample_trans(alt, cos_sun);
            for c in 0..3 {
                let scat = rayleigh_beta(CHANNEL_NM[c]) * dr * p_r + scat_m * dm * p_m;
                local[c] += w * t_view[c] * t_sun[c] * scat;
            }
        }
        for c in 0..3 {
            acc[c] += local[c] * dt;
        }
    }
    acc
}

/// In-scatter + path transmittance for aerial-perspective segments.
fn integrate_aerial<F>(
    origin: DVec3,
    dir: DVec3,
    t_end: f64,
    params: &SkyParams,
    sample_trans: &F,
    p_r: f64,
    p_m: f64,
) -> ([f64; 3], [f64; 3])
where
    F: Fn(f64, f64) -> [f64; 3],
{
    let steps = 32usize;
    if t_end <= 0.0 {
        return ([1.0; 3], [0.0; 3]);
    }
    let dt = t_end / steps as f64;
    let sun = params.sun_direction.normalize();
    let ext = mie_extinction(params);
    let scat_m = mie_scattering(params);
    let mut acc = [0.0f64; 3];

    // final path transmittance needs total depth
    let total_depth = optical_depths(origin, dir, t_end, steps, params);
    let mut trans = [0.0f64; 3];
    for (c, lam) in CHANNEL_NM.iter().enumerate() {
        let tau = rayleigh_beta(*lam) * total_depth[0] + ext * total_depth[1] + OZONE_BETA[c] * total_depth[2];
        trans[c] = (-tau).exp();
    }

    for i in 0..steps {
        let t0 = dt * i as f64;
        let t1 = dt * (i + 1) as f64;
        let mut local = [0.0f64; 3];
        for (t, w) in [(t0, 0.5), (t1, 0.5)] {
            let p = origin + dir * t;
            let r = p.length();
            let alt = (r - params.planet_radius_m).max(0.0);
            let (dr, dm, _do) = densities(alt, params);
            let depth = optical_depths(origin, dir, t, 16.min(steps), params);
            let t_view = {
                let mut v = [0.0f64; 3];
                for (c, lam) in CHANNEL_NM.iter().enumerate() {
                    let tau = rayleigh_beta(*lam) * depth[0]
                        + ext * depth[1]
                        + OZONE_BETA[c] * depth[2];
                    v[c] = (-tau).exp();
                }
                v
            };
            let p_hat = p / r;
            let cos_sun = p_hat.dot(sun);
            let t_sun = sample_trans(alt, cos_sun);
            for c in 0..3 {
                let scat = rayleigh_beta(CHANNEL_NM[c]) * dr * p_r + scat_m * dm * p_m;
                local[c] += w * t_view[c] * t_sun[c] * scat;
            }
        }
        for c in 0..3 {
            acc[c] += local[c] * dt;
        }
    }
    (trans, acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transmittance_sea_level_zenith_550nm() {
        let params = SkyParams::default();
        let luts = SkyLuts::new(&params);
        let t = luts.monochromatic_transmittance(0.0, 1.0, 550.0);
        assert!((0.85..=0.95).contains(&t), "T(550) = {t}");
    }

    #[test]
    fn transmittance_wavelength_ordering_and_altitude() {
        let params = SkyParams::default();
        let luts = SkyLuts::new(&params);
        let t440 = luts.monochromatic_transmittance(0.0, 1.0, 440.0);
        let t680 = luts.monochromatic_transmittance(0.0, 1.0, 680.0);
        assert!(t680 > t440, "t680 {t680} <= t440 {t440}");
        let t_high = luts.monochromatic_transmittance(10_000.0, 1.0, 550.0);
        let t_low = luts.monochromatic_transmittance(0.0, 1.0, 550.0);
        assert!(t_high > t_low, "alt transmittance should rise: {t_high} vs {t_low}");
    }

    #[test]
    fn transmittance_lut_matches_quadrature() {
        let params = SkyParams::default();
        let luts = SkyLuts::new(&params);
        // exact grid points must agree with direct quadrature (within
        // discretisation error of the shared 32/48-step marches)
        for (alt, mu) in [(0.0, 1.0), (2_000.0, 0.8), (0.0, 0.5)] {
            let lut = luts.transmittance_rgb(alt, mu);
            let quad = transmittance_quad(alt, mu, &params);
            for c in 0..3 {
                assert!(
                    (lut[c] - quad[c]).abs() < 0.02,
                    "alt {alt} mu {mu} ch {c}: lut {} quad {}",
                    lut[c],
                    quad[c]
                );
            }
        }
    }

    #[test]
    fn sky_view_is_blue_at_zenith() {
        // sun at 60 degrees elevation so the blue ratio is robust
        let params = SkyParams::with_sun(DVec3::new(0.0, 0.866, 0.5).normalize());
        let luts = SkyLuts::new(&params);
        let rgb = luts.sky_view_rgb(DVec3::Y);
        let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
        assert!(r > 0.0 && g > 0.0 && b > 0.0, "non-positive radiance {rgb:?}");
        let ratio = b / r;
        assert!(ratio > 2.0, "blue/red ratio {ratio}");
    }

    #[test]
    fn sun_below_horizon_is_dark() {
        let params = SkyParams::with_sun(DVec3::new(0.2, -0.4, 0.3));
        let luts = SkyLuts::new(&params);
        let rgb = luts.sky_view_rgb(DVec3::Y);
        for (i, v) in rgb.iter().enumerate() {
            assert!(v.abs() < 1e-6, "channel {i} = {v} (should be night)");
        }
    }

    #[test]
    fn all_luts_finite_and_non_negative() {
        let params = SkyParams::default();
        let luts = SkyLuts::new(&params);
        for px in &luts.transmittance {
            for v in px {
                assert!(v.is_finite() && *v >= 0.0);
            }
        }
        for px in &luts.sky_view {
            for v in px {
                assert!(v.is_finite() && *v >= 0.0);
            }
        }
        for (t, r) in &luts.aerial {
            for v in t.iter().chain(r.iter()) {
                assert!(v.is_finite() && *v >= 0.0);
            }
        }
        // spot-check queries
        let rgb = luts.sky_view_rgb(DVec3::new(0.3, 0.4, 0.5));
        assert!(rgb.iter().all(|v| v.is_finite()));
        let (t, r) = luts.aerial_perspective(25_000.0, 0.3);
        assert!(t.iter().chain(r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn aerial_transmittance_decreases_with_distance() {
        let params = SkyParams::default();
        let luts = SkyLuts::new(&params);
        let (t_near, _) = luts.aerial_perspective(5_000.0, 0.5);
        let (t_far, _) = luts.aerial_perspective(70_000.0, 0.5);
        for c in 0..3 {
            assert!(
                t_far[c] < t_near[c],
                "ch {c}: far {} >= near {}",
                t_far[c],
                t_near[c]
            );
            assert!((0.0..=1.0).contains(&t_far[c]), "t out of range {}", t_far[c]);
        }
    }

    #[test]
    fn higher_sun_is_brighter_at_zenith() {
        let high = SkyLuts::new(&SkyParams::with_sun(DVec3::Y));
        let low = SkyLuts::new(&SkyParams::with_sun(DVec3::new(0.0, 0.2, 0.98)));
        let r_high = high.sky_view_rgb(DVec3::Y)[1];
        let r_low = low.sky_view_rgb(DVec3::Y)[1];
        assert!(r_high > r_low, "zenith green: high {r_high} low {r_low}");
    }

    #[test]
    fn default_params_are_well_formed() {
        let p = SkyParams::default();
        assert!((p.sun_direction.length() - 1.0).abs() < 1e-12);
        assert!(p.mie_g > 0.0 && p.mie_g < 1.0);
        assert!(p.atmosphere_height_m > p.mie_scale_height_m);
    }
}
