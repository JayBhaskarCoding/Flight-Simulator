//! US Standard Atmosphere 1976 (NASA SP-368) — clean-room
//! implementation of the published seven-layer model.
//!
//! Layers (geopotential altitude, lapse rate):
//!
//! | base (km) | top (km) | lapse (K/km) |
//! |-----------|----------|--------------|
//! | 0         | 11       | −6.5         |
//! | 11        | 20       | 0            |
//! | 20        | 32       | +1.0         |
//! | 32        | 47       | +2.8         |
//! | 47        | 51       | 0            |
//! | 51        | 71       | −2.8         |
//! | 71        | 84.852   | −2.0         |
//!
//! Geometric (aircraft) altitudes are converted to geopotential
//! altitudes exactly as the standard prescribes:
//! `h = r·H / (r + H)` with `r = 6 356 766 m`.

use std::f64::consts::SQRT_2;

/// Gas constant for dry air, J/(kg·K) (1976: 8314.32 / 28.9644).
pub const R_DRY_AIR: f64 = 287.052_87;
/// Gas constant for water vapour, J/(kg·K).
pub const R_WATER_VAPOUR: f64 = 461.495;
/// Sea-level standard temperature, K.
pub const T0: f64 = 288.15;
/// Sea-level standard pressure, Pa.
pub const P0: f64 = 101_325.0;
/// Sea-level standard density, kg/m³.
pub const RHO0: f64 = 1.225;
/// Standard gravity, m/s².
pub const G0: f64 = 9.806_65;
/// Ratio of specific heats for diatomic gas.
pub const GAMMA: f64 = 1.4;
/// Mean Earth radius used by the 1976 standard, m.
pub const EARTH_RADIUS_GEOPO_TENTIAL: f64 = 6_356_766.0;

/// Number of model layers.
const LAYERS: usize = 7;
/// Geopotential altitude of each layer BASE (m).
const BASE_H: [f64; LAYERS] = [
    0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0,
];
/// Lapse rate of each layer (K/m).
const LAPSE: [f64; LAYERS] = [
    -6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3,
];
/// Top of the model (geopotential m).
const TOP_H: f64 = 84_852.0;

/// Non-standard-day conditions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsaConditions {
    /// Temperature offset from standard at every level, K
    /// (`+15` = ISA+15, a "hot day").
    pub delta_t_k: f64,
    /// Sea-level pressure (QNH), Pa. Keep `101_325.0` for standard.
    pub sea_level_pressure_pa: f64,
}

impl Default for IsaConditions {
    fn default() -> Self {
        Self { delta_t_k: 0.0, sea_level_pressure_pa: P0 }
    }
}

/// Thermodynamic state at one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereState {
    /// Geometric altitude of the query, m.
    pub geometric_altitude_m: f64,
    /// Geopotential altitude used by the model, m.
    pub geopotential_altitude_m: f64,
    /// Static temperature, K.
    pub temperature_k: f64,
    /// Static pressure, Pa.
    pub pressure_pa: f64,
    /// Density, kg/m³.
    pub density_kg_m3: f64,
    /// Local temperature lapse rate, K/m.
    pub lapse_rate_k_per_m: f64,
    /// Speed of sound, m/s.
    pub speed_of_sound_mps: f64,
    /// Dynamic viscosity (Sutherland), Pa·s.
    pub dynamic_viscosity_pa_s: f64,
    /// Kinematic viscosity ν = μ/ρ, m²/s.
    pub kinematic_viscosity_m2_s: f64,
}

impl AtmosphereState {
    /// Mach number for a true airspeed.
    pub fn mach(&self, true_airspeed_mps: f64) -> f64 {
        true_airspeed_mps / self.speed_of_sound_mps
    }

    /// Dynamic pressure ½ρV².
    pub fn dynamic_pressure_pa(&self, true_airspeed_mps: f64) -> f64 {
        0.5 * self.density_kg_m3 * true_airspeed_mps * true_airspeed_mps
    }

    /// Saturation vapour pressure over water at this temperature (Buck
    /// 1996 expression, valid −45…+70 °C), Pa.
    pub fn saturation_vapour_pressure_pa(&self) -> f64 {
        saturation_vapour_pressure_pa(self.temperature_k)
    }

    /// Humid-air variant of this state: density includes the vapour
    /// correction `ρ = (p−e)/(R_d T) + e/(R_v T)` with
    /// `e = RH·e_sat(T)`. Speed of sound stays the dry-air value
    /// (error < 0.1 % at aviation conditions, per the 1976 standard's
    /// own treatment of small vapour fractions).
    pub fn with_relative_humidity(&self, relative_humidity: f64) -> AtmosphereState {
        let rh = relative_humidity.clamp(0.0, 1.0);
        let e = rh * self.saturation_vapour_pressure_pa();
        let rho = (self.pressure_pa - e) / (R_DRY_AIR * self.temperature_k)
            + e / (R_WATER_VAPOUR * self.temperature_k);
        let mut s = *self;
        s.density_kg_m3 = rho;
        s.kinematic_viscosity_m2_s = s.dynamic_viscosity_pa_s / rho;
        s
    }
}

/// Saturation vapour pressure over water (Buck 1996), Pa.
pub fn saturation_vapour_pressure_pa(temperature_k: f64) -> f64 {
    let t_c = temperature_k - 273.15;
    611.21 * ((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c))).exp()
}

/// Sutherland's viscosity formula (1976 standard), Pa·s.
pub fn sutherland_viscosity(temperature_k: f64) -> f64 {
    1.458e-6 * temperature_k.powf(1.5) / (temperature_k + 110.4)
}

/// Evaluate the seven-layer model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsaModel {
    /// Geopotential base altitudes (m).
    base_h: [f64; LAYERS],
    /// Base temperature of each layer (K) — shifted by ΔISA.
    base_t: [f64; LAYERS],
    /// Base pressure of each layer (Pa).
    base_p: [f64; LAYERS],
    /// Lapse rate per layer (K/m).
    lapse: [f64; LAYERS],
}

impl IsaModel {
    /// Build a model for the given conditions (base table recomputed
    /// with the temperature offset and sea-level pressure).
    pub fn new(conditions: IsaConditions) -> Self {
        let mut base_t = [0.0; LAYERS];
        let mut base_p = [0.0; LAYERS];
        base_t[0] = T0 + conditions.delta_t_k;
        base_p[0] = conditions.sea_level_pressure_pa;
        for i in 1..LAYERS {
            let l = LAPSE[i - 1];
            let dh = BASE_H[i] - BASE_H[i - 1];
            if l.abs() < 1e-15 {
                base_t[i] = base_t[i - 1];
                base_p[i] = base_p[i - 1] * (-G0 * dh / (R_DRY_AIR * base_t[i - 1])).exp();
            } else {
                base_t[i] = base_t[i - 1] + l * dh;
                // signed lapse T = Tb + L·h ⇒ p = pb (T/Tb)^(−g/(R·L))
                let exponent = -G0 / (l * R_DRY_AIR);
                base_p[i] = base_p[i - 1] * (base_t[i] / base_t[i - 1]).powf(exponent);
            }
        }
        Self { base_h: BASE_H, base_t, base_p, lapse: LAPSE }
    }

    /// Standard-day convenience model.
    pub fn standard() -> Self {
        Self::new(IsaConditions::default())
    }

    /// Conditions this model was built from (recovered).
    pub fn conditions(&self) -> IsaConditions {
        IsaConditions {
            delta_t_k: self.base_t[0] - T0,
            sea_level_pressure_pa: self.base_p[0],
        }
    }

    /// State at a **geometric** altitude (what an altimeter measures
    /// above the ellipsoid, modulo QNH setting).
    pub fn at_geometric_m(&self, geometric_altitude_m: f64) -> AtmosphereState {
        let r = EARTH_RADIUS_GEOPO_TENTIAL;
        // h = r·H/(r+H); monotone for H > −r, so valid below sea level too
        let h = r * geometric_altitude_m / (r + geometric_altitude_m);
        let mut state = self.at_geopotential_m(h);
        state.geometric_altitude_m = geometric_altitude_m;
        state
    }

    /// State at a **geopotential** altitude (model input).
    pub fn at_geopotential_m(&self, geopotential_altitude_m: f64) -> AtmosphereState {
        let h = geopotential_altitude_m.clamp(-1_000.0, TOP_H);
        let i = match h {
            h if h < self.base_h[1] => 0,
            h if h < self.base_h[2] => 1,
            h if h < self.base_h[3] => 2,
            h if h < self.base_h[4] => 3,
            h if h < self.base_h[5] => 4,
            h if h < self.base_h[6] => 5,
            _ => 6,
        };
        let l = self.lapse[i];
        let hb = self.base_h[i];
        let tb = self.base_t[i];
        let pb = self.base_p[i];
        let dh = h - hb;
        let (temperature, pressure) = if l.abs() < 1e-15 {
            (tb, pb * (-G0 * dh / (R_DRY_AIR * tb)).exp())
        } else {
            let t = tb + l * dh;
            (t, pb * (t / tb).powf(-G0 / (l * R_DRY_AIR)))
        };
        let density = pressure / (R_DRY_AIR * temperature);
        let mu = sutherland_viscosity(temperature);
        AtmosphereState {
            geometric_altitude_m: geopotential_altitude_m,
            geopotential_altitude_m: h,
            temperature_k: temperature,
            pressure_pa: pressure,
            density_kg_m3: density,
            lapse_rate_k_per_m: l,
            speed_of_sound_mps: (GAMMA * R_DRY_AIR * temperature).sqrt(),
            dynamic_viscosity_pa_s: mu,
            kinematic_viscosity_m2_s: mu / density,
        }
    }

    /// Altitude at which the standard profile has the same density as
    /// `density_kg_m3` (density altitude, metres).
    pub fn density_altitude_for(&self, density_kg_m3: f64) -> f64 {
        // Bisect over geometric altitude — the profile is strictly
        // monotone in the flight regime.
        let mut lo = -1_000.0;
        let mut hi = 25_000.0;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let rho = self.at_geometric_m(mid).density_kg_m3;
            if rho > density_kg_m3 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Density altitude of a measured state (classic closed form using
    /// the ISA lapse rate — works for any measured `p`, `T`).
    pub fn density_altitude_of(pressure_pa: f64, temperature_k: f64) -> f64 {
        let rho = pressure_pa / (R_DRY_AIR * temperature_k);
        // With the signed tropospheric lapse L0 (T = T0 + L0·h):
        // rho(h) = rho0 (T/T0)^(−g/(R·L0) − 1)
        let exponent = -G0 / (R_DRY_AIR * LAPSE[0]) - 1.0;
        let t_ratio = (rho / RHO0).powf(1.0 / exponent);
        T0 * (t_ratio - 1.0) / LAPSE[0]
    }
}

impl Default for IsaModel {
    fn default() -> Self {
        Self::standard()
    }
}

/// Wind shear (log-law) profile over rough terrain — used by the
/// Dryden turbulence parameterisation.
///
/// `u* = κ·w_ref / ln(z_ref / z0)` with von Kármán `κ = 0.409`.
pub fn friction_velocity(wind_10m_mps: f64, roughness_m: f64) -> f64 {
    const KAPPA: f64 = 0.409;
    let z0 = roughness_m.max(1e-4);
    KAPPA * wind_10m_mps.abs() / (10.0f64 / z0).ln().max(1e-6)
}

/// Gaussian tail-normalising helper used by tests and statistics.
pub fn normal_pdf(x: f64, mean: f64, sigma: f64) -> f64 {
    let z = (x - mean) / sigma;
    (-0.5 * z * z).exp() / (sigma * SQRT_2 * std::f64::consts::PI.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE_TOL_P: f64 = 2.0; // Pa against the published table

    #[test]
    fn sea_level_standard_values() {
        let s = IsaModel::standard().at_geometric_m(0.0);
        assert!((s.temperature_k - 288.15).abs() < 1e-9);
        assert!((s.pressure_pa - 101_325.0).abs() < 1e-6);
        assert!((s.density_kg_m3 - 1.225).abs() < 1e-3, "rho={}", s.density_kg_m3);
        assert!((s.speed_of_sound_mps - 340.29).abs() < 0.02, "a={}", s.speed_of_sound_mps);
        assert!((s.dynamic_viscosity_pa_s - 1.789e-5).abs() < 3e-7, "mu={}", s.dynamic_viscosity_pa_s);
    }

    #[test]
    fn matches_published_table() {
        let m = IsaModel::standard();
        // geopotential altitudes as tabulated in the 1976 standard
        let cases = [
            (11_000.0, 216.65, 22_632.1),
            (20_000.0, 216.65, 5_474.89),
            (32_000.0, 228.65, 868.02),
            (47_000.0, 270.65, 110.91),
            (51_000.0, 270.65, 66.94),
            (71_000.0, 214.65, 5.746),
        ];
        for (h, t, p) in cases {
            let s = m.at_geopotential_m(h);
            assert!((s.temperature_k - t).abs() < 0.01, "T at {h}: {}", s.temperature_k);
            assert!(
                (s.pressure_pa - p).abs() < TABLE_TOL_P.max(p * 1e-4),
                "p at {h}: {} vs {p}",
                s.pressure_pa
            );
        }
    }

    #[test]
    fn geometric_altitude_differs_from_geopotential() {
        let m = IsaModel::standard();
        let geo = m.at_geometric_m(11_000.0);
        let pot = m.at_geopotential_m(11_000.0);
        // geometric 11 km sits ~19 m below the geopotential 11 km level,
        // so it is slightly warmer than the tabulated 216.65 K
        assert!(geo.temperature_k > pot.temperature_k);
        assert!((geo.temperature_k - 216.65).abs() < 0.15);
        assert!(geo.geopotential_altitude_m < 11_000.0);
        assert!(geo.geopotential_altitude_m > 10_900.0);
    }

    #[test]
    fn layer_boundaries_are_continuous() {
        let m = IsaModel::standard();
        for &hb in &BASE_H[1..] {
            let a = m.at_geopotential_m(hb - 1e-3);
            let b = m.at_geopotential_m(hb + 1e-3);
            let pr = (b.pressure_pa - a.pressure_pa).abs() / a.pressure_pa;
            assert!(pr < 1e-5, "pressure jump at {hb}: {pr:e}");
            assert!((b.temperature_k - a.temperature_k).abs() < 0.001, "T jump at {hb}");
        }
    }

    #[test]
    fn non_standard_day_shifts_profile() {
        let hot = IsaModel::new(IsaConditions { delta_t_k: 10.0, sea_level_pressure_pa: P0 });
        let std = IsaModel::standard();
        let hs = std.at_geometric_m(0.0);
        let hh = hot.at_geometric_m(0.0);
        assert!((hh.temperature_k - hs.temperature_k - 10.0).abs() < 1e-9);
        assert!((hh.pressure_pa - hs.pressure_pa).abs() < 1e-6);
        assert!(hh.density_kg_m3 < hs.density_kg_m3, "hot air must be lighter");

        let low = IsaModel::new(IsaConditions { delta_t_k: 0.0, sea_level_pressure_pa: 99_000.0 });
        let l = low.at_geometric_m(0.0);
        assert!((l.pressure_pa - 99_000.0).abs() < 1e-6);
        assert!((l.density_kg_m3 - 99_000.0 / (R_DRY_AIR * T0)).abs() < 1e-9);
        // same temperature, lower pressure -> lower density, same a
        assert!((l.speed_of_sound_mps - std.speed_of_sound()).abs() < 1e-9);
    }

    impl IsaModel {
        fn speed_of_sound(&self) -> f64 {
            self.at_geometric_m(0.0).speed_of_sound_mps
        }
    }

    #[test]
    fn humidity_reduces_density() {
        let s = IsaModel::standard().at_geometric_m(0.0);
        let dry = s.with_relative_humidity(0.0).density_kg_m3;
        let moist = s.with_relative_humidity(1.0).density_kg_m3;
        assert!((dry - s.density_kg_m3).abs() < 1e-12);
        assert!(moist < dry, "moist air {moist} < dry {dry}");
        // at 15 °C, 100 % RH: about 1 % density loss
        assert!((dry - moist) / dry < 0.015 && (dry - moist) / dry > 0.005);
        // saturation pressure at 15 °C ≈ 1705 Pa (Buck)
        let es = saturation_vapour_pressure_pa(288.15);
        assert!((es - 1_705.0).abs() < 25.0, "e_sat={es}");
    }

    #[test]
    fn density_altitude_standard_and_hot() {
        let da0 = IsaModel::density_altitude_of(P0, T0);
        assert!(da0.abs() < 1.0, "std density altitude {da0}");

        // 30 °C at sea level (ISA+15 K): density altitude ≈ 526 m
        // (rule of thumb ~120 ft/°C → ~1800 ft ≈ 550 m; closed form gives
        // the exact value below)
        let hot_rho = 101_325.0 / (R_DRY_AIR * 303.15);
        let da = IsaModel::density_altitude_of(101_325.0, 303.15);
        assert!(da > 450.0 && da < 700.0, "hot density altitude {da}");

        // model-based inversion must agree with the closed form
        let m = IsaModel::standard();
        let da_m = m.density_altitude_for(hot_rho);
        assert!((da_m - da).abs() < 25.0, "closed {da} vs bisection {da_m}");

        // colder air -> density altitude below true altitude
        let cold = m.density_altitude_for(m.at_geometric_m(3_000.0).density_kg_m3 * 1.05);
        assert!(cold < 3_000.0, "{cold}");
    }

    #[test]
    fn state_helpers() {
        let s = IsaModel::standard().at_geometric_m(5_000.0);
        assert!((s.mach(150.0) - 150.0 / s.speed_of_sound_mps).abs() < 1e-12);
        let q = s.dynamic_pressure_pa(100.0);
        assert!((q - 0.5 * s.density_kg_m3 * 10_000.0).abs() < 1e-9);
        assert!(s.kinematic_viscosity_m2_s > 0.0);
        assert!(s.pressure_pa < 101_325.0 && s.density_kg_m3 < 1.225);
    }

    #[test]
    fn monotone_with_altitude_and_clamped_at_top() {
        let m = IsaModel::standard();
        let mut prev_rho = f64::INFINITY;
        // strictly inside the modelled range (the model clamps geopotential
        // at 84 852 m, reached geometrically around 86 km)
        for h in (0..85).map(|i| i as f64 * 1_000.0) {
            let rho = m.at_geometric_m(h).density_kg_m3;
            assert!(rho < prev_rho, "density not decreasing at {h}");
            assert!(rho > 0.0);
            prev_rho = rho;
        }
        // clamped beyond the model top: no NaNs, positive values
        let top = m.at_geometric_m(200_000.0);
        assert!(top.pressure_pa.is_finite() && top.pressure_pa > 0.0);
        assert!(top.temperature_k.is_finite() && top.temperature_k > 0.0);
    }

    #[test]
    fn friction_velocity_sane() {
        // 5 m/s over grass (z0 = 3 cm): u* ≈ 0.21 m/s
        let u = friction_velocity(5.0, 0.03);
        assert!((u - 0.409 * 5.0 / (10.0f64 / 0.03).ln()).abs() < 1e-9);
        assert!(u > 0.1 && u < 0.5, "{u}");
    }
}
