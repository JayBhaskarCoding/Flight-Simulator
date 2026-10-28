//! `[WEIGHT_AND_BALANCE]`, `[FUEL]`, `[AIRPLANE_FLIGHT_MODEL]`,
//! `[PERFORMANCE]` typed loader.

use crate::error::{CfgError, CfgResult};
use crate::geometry::{
    AirplaneGeometry, GAL_TO_M3, LB_TO_KG, FT_TO_M,
};
use crate::parser::{CfgFile, Section};

/// Average gasoline density (6.0 lb/US gal) in kg/m³.
pub const AVGAS_DENSITY_KG_M3: f64 = 6.0 * LB_TO_KG / GAL_TO_M3;

/// Mass / centre-of-gravity envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightAndBalance {
    /// Operating empty weight, kg.
    pub empty_weight_kg: f64,
    /// Maximum take-off weight, kg.
    pub max_gross_weight_kg: f64,
    /// Longitudinal station of the empty CG from the reference datum, m.
    pub empty_cg_arm_m: f64,
    /// Forward CG limit as a fraction of MAC (0 = LE of MAC).
    pub cg_forward_mac_frac: f64,
    /// Aft CG limit as a fraction of MAC.
    pub cg_aft_mac_frac: f64,
}

impl WeightAndBalance {
    /// Validate the envelope against itself.
    pub fn validate(&self) -> CfgResult<()> {
        if self.empty_weight_kg <= 0.0 || self.max_gross_weight_kg <= 0.0 {
            return Err(CfgError::Range {
                file: "flight_model.cfg".into(),
                line: 1,
                key: "empty_weight_lb".into(),
                message: "weights must be positive".into(),
            });
        }
        if self.empty_weight_kg >= self.max_gross_weight_kg {
            return Err(CfgError::Range {
                file: "flight_model.cfg".into(),
                line: 1,
                key: "empty_weight_lb".into(),
                message: "empty weight must be below max gross weight".into(),
            });
        }
        if !(0.0..1.0).contains(&self.cg_forward_mac_frac)
            || !(0.0..1.0).contains(&self.cg_aft_mac_frac)
            || self.cg_forward_mac_frac >= self.cg_aft_mac_frac
        {
            return Err(CfgError::Range {
                file: "flight_model.cfg".into(),
                line: 1,
                key: "cg_forward_mac_pct".into(),
                message: "CG limits must satisfy 0 <= fwd < aft <= 100 (%MAC)".into(),
            });
        }
        Ok(())
    }
}

/// One fuel tank.
#[derive(Debug, Clone, PartialEq)]
pub struct FuelTank {
    /// Tank name as authored.
    pub name: String,
    /// Total volume, m³.
    pub capacity_m3: f64,
    /// Usable fraction (0..1] of the capacity.
    pub usable_frac: f64,
    /// Body-frame position of the tank centroid (+X fwd, +Y right, +Z down), m.
    pub position_m: glam::DVec3,
}

impl FuelTank {
    /// Maximum usable mass of Avgas in this tank, kg.
    pub fn usable_mass_kg(&self) -> f64 {
        self.capacity_m3 * self.usable_frac * AVGAS_DENSITY_KG_M3
    }
    /// Mass currently carried given a fill fraction of usable capacity.
    pub fn mass_kg(&self, fill_fraction: f64) -> f64 {
        self.usable_mass_kg() * fill_fraction.clamp(0.0, 1.0)
    }
}

/// V-speeds and altitude limits (loaded as authored units, SI accessors
/// provided).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceLimits {
    /// Stall speed, flaps/gear extended, indicated knots.
    pub vs0_kias: f64,
    /// Stall speed, clean, indicated knots.
    pub vs1_kias: f64,
    /// Max structural cruising speed, knots.
    pub vno_kias: f64,
    /// Never-exceed speed, knots.
    pub vne_kias: f64,
    /// Maneuvering speed, knots.
    pub va_kias: f64,
    /// Absolute service ceiling, feet.
    pub service_ceiling_ft: f64,
    /// Best rate of climb, feet per minute.
    pub max_climb_rate_fpm: f64,
    /// Economical cruise speed, true knots.
    pub cruise_speed_ktas: f64,
}

impl PerformanceLimits {
    /// Stall speed clean in m/s (converted from KCAS ≈ KIAS at these
    /// regimes).
    pub fn vs1_mps(&self) -> f64 {
        self.vs1_kias * 0.514_444
    }
    /// Never-exceed speed in m/s.
    pub fn vne_mps(&self) -> f64 {
        self.vne_kias * 0.514_444
    }
    /// Service ceiling in metres.
    pub fn service_ceiling_m(&self) -> f64 {
        self.service_ceiling_ft * FT_TO_M
    }
}

/// Dimensionless aerodynamic coefficients for the 6-DOF solver.
///
/// Signs follow the body axes (+X fwd, +Y right, +Z down):
/// positive `Cm` pitches nose up, positive `Cn` yaws nose right,
/// positive `Cl` rolls right wing down.
#[derive(Debug, Clone, PartialEq)]
pub struct AeroCoefficients {
    /// Lift at zero angle of attack.
    pub cl0: f64,
    /// Lift-curve slope, per radian (theory ≈ 2π·AR/(AR+2)).
    pub cl_alpha: f64,
    /// Zero-lift drag coefficient.
    pub cd0: f64,
    /// Oswald span efficiency.
    pub oswald: f64,
    /// Pitch moment at zero α.
    pub cm0: f64,
    /// Pitch stiffness ∂Cm/∂α (negative = stable).
    pub cm_alpha: f64,
    /// Pitch damping ∂Cm/∂(q·c/2V).
    pub cm_q: f64,
    /// Elevator pitch authority ∂Cm/∂δe (per radian of elevator).
    pub cm_delta_e: f64,
    /// Side-force slope ∂Cy/∂β.
    pub cy_beta: f64,
    /// Roll damping ∂Cl/∂(p·b/2V).
    pub cl_p: f64,
    /// Roll due to yaw rate ∂Cl/∂(r·b/2V).
    pub cl_r: f64,
    /// Aileron roll power ∂Cl/∂δa (per radian).
    pub cl_delta_a: f64,
    /// Dihedral effect ∂Cl/∂β (negative = weathercock stable roll).
    pub cl_beta: f64,
    /// Yaw damping ∂Cn/∂(r·b/2V).
    pub cn_r: f64,
    /// Yaw due to roll rate ∂Cn/∂(p·b/2V).
    pub cn_p: f64,
    /// Rudder yaw authority ∂Cn/∂δr (per radian).
    pub cn_delta_r: f64,
    /// Directional stiffness ∂Cn/∂β (positive = stable).
    pub cn_beta: f64,
    /// Absolute stall angle of attack, radians.
    pub alpha_stall_rad: f64,
    /// Fraction of the linear lift retained immediately beyond stall.
    pub post_stall_lift_factor: f64,
    /// Incremental lift per flap deflection step (indexed `flap_cl[i]`),
    /// applied for deployed step `i` (already scaled per step).
    pub flap_cl_steps: Vec<f64>,
    /// Incremental drag per flap deflection step.
    pub flap_cd_steps: Vec<f64>,
}

impl AeroCoefficients {
    /// Linear-region lift coefficient at angle of attack `alpha` (rad),
    /// clamped to the stall envelope with the post-stall blend.
    /// `flap_step` is the NUMBER of deployed notches (0 = clean).
    pub fn lift_coefficient(&self, alpha: f64, flap_step: usize) -> f64 {
        let flap_cl: f64 = self.flap_cl_steps.iter().take(flap_step).sum();
        let (cl_lin, sign) = if alpha >= 0.0 {
            (self.cl0 + self.cl_alpha * alpha, 1.0)
        } else {
            // symmetric treatment about zero
            (self.cl0 - self.cl_alpha * alpha, -1.0)
        };
        let alpha_eff = alpha.abs();
        if alpha_eff <= self.alpha_stall_rad {
            return cl_lin + flap_cl;
        }
        // beyond stall: blend toward a flat-plate-like post-stall value
        let cl_stall = (self.cl0 + self.cl_alpha * self.alpha_stall_rad).abs()
            * self.post_stall_lift_factor;
        let over = alpha_eff - self.alpha_stall_rad;
        let blend = (-over / self.alpha_stall_rad).exp();
        let mag = cl_stall * blend + (cl_lin.abs() * (1.0 - blend)) * 0.35;
        sign * (mag + flap_cl)
    }

    /// Parasite + induced drag at lift `cl` with `flap_step` deployed.
    pub fn drag_coefficient(&self, cl: f64, aspect_ratio: f64, flap_step: usize) -> f64 {
        let flap_cd: f64 = self.flap_cd_steps.iter().take(flap_step).sum();
        let induced = if aspect_ratio > 0.0 {
            cl * cl / (std::f64::consts::PI * self.oswald * aspect_ratio)
        } else {
            0.0
        };
        self.cd0 + induced + flap_cd
    }
}

/// Complete `flight_model.cfg` document.
#[derive(Debug, Clone, PartialEq)]
pub struct FlightModelDocument {
    /// Geometry (already converted to SI).
    pub geometry: AirplaneGeometry,
    /// Mass & CG envelope.
    pub weight: WeightAndBalance,
    /// Fuel tanks.
    pub fuel: Vec<FuelTank>,
    /// V-speeds and limits.
    pub limits: PerformanceLimits,
    /// Aerodynamic coefficient set.
    pub aero: AeroCoefficients,
    /// Number of discrete flap steps defined by `flap_cl[i]`.
    pub flap_steps: usize,
}

fn require_positive(sec: &Section, key: &str) -> CfgResult<f64> {
    let v = sec.f64(key)?;
    if v <= 0.0 {
        let e = sec.find(key).expect("checked above");
        return Err(CfgError::Range {
            file: sec.file.clone(),
            line: e.line,
            key: e.key.clone(),
            message: "value must be positive".into(),
        });
    }
    Ok(v)
}

impl FlightModelDocument {
    /// Load from the parsed `flight_model.cfg`.
    pub fn from_cfg(cfg: &CfgFile) -> CfgResult<Self> {
        let geometry = AirplaneGeometry::from_cfg(cfg)?;

        let wb_sec = cfg.require_section("WEIGHT_AND_BALANCE")?;
        let empty = require_positive(wb_sec, "empty_weight_lb")? * LB_TO_KG;
        let gross = require_positive(wb_sec, "max_gross_weight_lb")? * LB_TO_KG;
        let weight = WeightAndBalance {
            empty_weight_kg: empty,
            max_gross_weight_kg: gross,
            empty_cg_arm_m: wb_sec.f64("empty_cg_arm_ft")? * FT_TO_M,
            cg_forward_mac_frac: wb_sec.f64("cg_forward_mac_pct")? / 100.0,
            cg_aft_mac_frac: wb_sec.f64("cg_aft_mac_pct")? / 100.0,
        };
        weight.validate()?;

        let fuel_sec = cfg.require_section("FUEL")?;
        let n_tanks = fuel_sec.u64("num_tanks")? as usize;
        if n_tanks == 0 || n_tanks > 8 {
            return Err(CfgError::Range {
                file: fuel_sec.file.clone(),
                line: fuel_sec.find("num_tanks").map(|e| e.line).unwrap_or(1),
                key: "num_tanks".into(),
                message: "num_tanks must be in 1..=8".into(),
            });
        }
        let caps = fuel_sec.indexed_f64s_required("tank_capacity_gal")?;
        let xs = fuel_sec.indexed_f64s_required("tank_x_ft")?;
        let ys = fuel_sec.indexed_f64s_required("tank_y_ft")?;
        let zs = fuel_sec.indexed_f64s_required("tank_z_ft")?;
        let usable = fuel_sec.indexed_f64s("tank_usable_frac")?;
        let mut fuel = Vec::with_capacity(n_tanks);
        for i in 0..n_tanks {
            let cap = *caps.get(&i).ok_or_else(|| CfgError::IndexGap {
                file: fuel_sec.file.clone(),
                section: "FUEL".into(),
                key: "tank_capacity_gal".into(),
                expected: format!("an entry for tank {i}"),
            })?;
            if cap <= 0.0 {
                return Err(CfgError::Range {
                    file: fuel_sec.file.clone(),
                    line: 1,
                    key: "tank_capacity_gal".into(),
                    message: format!("tank {i} capacity must be positive"),
                });
            }
            let name = fuel_sec
                .find_all("tank_name")
                .into_iter()
                .find(|e| e.index == Some(i))
                .and_then(|e| e.value.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("tank_{i}"));
            let uf = usable.get(&i).copied().unwrap_or(1.0);
            if !(0.0..=1.0).contains(&uf) {
                return Err(CfgError::Range {
                    file: fuel_sec.file.clone(),
                    line: 1,
                    key: "tank_usable_frac".into(),
                    message: format!("tank {i} usable fraction must be within 0..1"),
                });
            }
            fuel.push(FuelTank {
                name,
                capacity_m3: cap * GAL_TO_M3,
                usable_frac: uf,
                position_m: glam::DVec3::new(
                    *xs.get(&i).unwrap_or(&0.0) * FT_TO_M,
                    *ys.get(&i).unwrap_or(&0.0) * FT_TO_M,
                    *zs.get(&i).unwrap_or(&0.0) * FT_TO_M,
                ),
            });
        }

        let perf = cfg.require_section("PERFORMANCE")?;
        let limits = PerformanceLimits {
            vs0_kias: require_positive(perf, "vs0_kias")?,
            vs1_kias: require_positive(perf, "vs1_kias")?,
            vno_kias: require_positive(perf, "vno_kias")?,
            vne_kias: require_positive(perf, "vne_kias")?,
            va_kias: require_positive(perf, "va_kias")?,
            service_ceiling_ft: require_positive(perf, "service_ceiling_ft")?,
            max_climb_rate_fpm: require_positive(perf, "max_climb_rate_fpm")?,
            cruise_speed_ktas: require_positive(perf, "cruise_speed_ktas")?,
        };
        if !(limits.vs1_kias < limits.vno_kias
            && limits.vno_kias < limits.vne_kias
            && limits.vs1_kias < limits.va_kias)
        {
            return Err(CfgError::Range {
                file: perf.file.clone(),
                line: 1,
                key: "vne_kias".into(),
                message: "expected vs1 < vno < vne and vs1 < va".into(),
            });
        }

        let afm = cfg.require_section("AIRPLANE_FLIGHT_MODEL")?;
        let flap_cl = afm.indexed_f64s_required("flap_cl")?;
        let flap_cd = afm.indexed_f64s_required("flap_cd")?;
        if flap_cl.len() != flap_cd.len() {
            return Err(CfgError::IndexGap {
                file: afm.file.clone(),
                section: "AIRPLANE_FLIGHT_MODEL".into(),
                key: "flap_cl".into(),
                expected: "the same number of steps as flap_cd".into(),
            });
        }
        let alpha_stall = require_positive(afm, "alpha_stall_deg")?.to_radians();
        let post = afm.f64_or("post_stall_lift_factor", 0.4)?;
        if !(0.0..=1.0).contains(&post) {
            return Err(CfgError::Range {
                file: afm.file.clone(),
                line: 1,
                key: "post_stall_lift_factor".into(),
                message: "must be within 0..1".into(),
            });
        }
        let oswald = afm.f64_in("oswald_efficiency", 0.5, 1.0)?;

        let aero = AeroCoefficients {
            cl0: afm.f64("zero_alpha_lift_coef")?,
            cl_alpha: require_positive(afm, "lift_slope_per_rad")?,
            cd0: afm.f64_in("drag_coef_zero_lift", 0.0, 0.2)?,
            oswald,
            cm0: afm.f64("pitch_moment_zero_alpha")?,
            cm_alpha: afm.f64("pitch_moment_slope")?,
            cm_q: afm.f64("pitch_damping")?,
            cm_delta_e: afm.f64("elevator_pitch_power")?,
            cy_beta: afm.f64("side_force_slope")?,
            cl_p: afm.f64("roll_damping")?,
            cl_r: afm.f64("roll_yaw_coupling")?,
            cl_delta_a: afm.f64("aileron_roll_power")?,
            cl_beta: afm.f64("dihedral_effect")?,
            cn_r: afm.f64("yaw_damping")?,
            cn_p: afm.f64("yaw_roll_coupling")?,
            cn_delta_r: afm.f64("rudder_yaw_power")?,
            cn_beta: afm.f64("directional_stability")?,
            alpha_stall_rad: alpha_stall,
            post_stall_lift_factor: post,
            flap_cl_steps: flap_cl.values().copied().collect(),
            flap_cd_steps: flap_cd.values().copied().collect(),
        };

        // stability sign checks (public flight-mechanics requirements)
        if aero.cm_alpha >= 0.0 {
            return Err(CfgError::Range {
                file: afm.file.clone(),
                line: 1,
                key: "pitch_moment_slope".into(),
                message: "must be negative for longitudinal static stability".into(),
            });
        }
        if aero.cn_beta <= 0.0 {
            return Err(CfgError::Range {
                file: afm.file.clone(),
                line: 1,
                key: "directional_stability".into(),
                message: "must be positive for directional (weathercock) stability".into(),
            });
        }
        if aero.cl_p >= 0.0 || aero.cn_r >= 0.0 {
            return Err(CfgError::Range {
                file: afm.file.clone(),
                line: 1,
                key: "roll_damping".into(),
                message: "roll and yaw damping must be negative".into(),
            });
        }

        Ok(Self {
            geometry,
            weight,
            fuel,
            limits,
            aero,
            flap_steps: flap_cl.len(),
        })
    }

    /// Total usable fuel mass at full fill, kg.
    pub fn total_usable_fuel_kg(&self) -> f64 {
        self.fuel.iter().map(|t| t.usable_mass_kg()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_cfg() -> String {
        let geom = crate::tests_support::GEOMETRY_SECTION;
        let rest = r#"
        [WEIGHT_AND_BALANCE]
        empty_weight_lb = 1685
        max_gross_weight_lb = 2550
        empty_cg_arm_ft = 38.5
        cg_forward_mac_pct = 16
        cg_aft_mac_pct = 47

        [FUEL]
        num_tanks = 2
        tank_name[0] = left_wing
        tank_name[1] = right_wing
        tank_capacity_gal[0] = 28.5
        tank_capacity_gal[1] = 28.5
        tank_x_ft[0] = -1.5
        tank_x_ft[1] = -1.5
        tank_y_ft[0] = -6.5
        tank_y_ft[1] = 6.5
        tank_z_ft[0] = -0.4
        tank_z_ft[1] = -0.4
        tank_usable_frac[0] = 0.98
        tank_usable_frac[1] = 0.98

        [PERFORMANCE]
        vs0_kias = 40
        vs1_kias = 48
        vno_kias = 129
        vne_kias = 163
        va_kias = 85
        service_ceiling_ft = 14000
        max_climb_rate_fpm = 730
        cruise_speed_ktas = 124

        [AIRPLANE_FLIGHT_MODEL]
        zero_alpha_lift_coef = 0.15
        lift_slope_per_rad = 5.1
        drag_coef_zero_lift = 0.032
        oswald_efficiency = 0.78
        pitch_moment_zero_alpha = 0.03
        pitch_moment_slope = -0.85
        pitch_damping = -12.5
        elevator_pitch_power = -1.25
        side_force_slope = -0.31
        roll_damping = -0.48
        roll_yaw_coupling = 0.12
        aileron_roll_power = 0.17
        dihedral_effect = -0.075
        yaw_damping = -0.14
        yaw_roll_coupling = 0.10
        rudder_yaw_power = 0.09
        directional_stability = 0.11
        alpha_stall_deg = 16
        post_stall_lift_factor = 0.45
        flap_cl[0] = 0.35
        flap_cl[1] = 0.65
        flap_cl[2] = 0.95
        flap_cd[0] = 0.010
        flap_cd[1] = 0.032
        flap_cd[2] = 0.070
        "#;
        format!("{geom}\n{rest}")
    }

    fn doc() -> FlightModelDocument {
        let cfg = CfgFile::parse(&full_cfg(), "fm.cfg").unwrap();
        FlightModelDocument::from_cfg(&cfg).unwrap()
    }

    #[test]
    fn loads_weights_and_validates() {
        let d = doc();
        assert!((d.weight.empty_weight_kg - 1685.0 * LB_TO_KG).abs() < 1e-6);
        assert!((d.weight.max_gross_weight_kg - 2550.0 * LB_TO_KG).abs() < 1e-6);
        assert!((d.weight.cg_forward_mac_frac - 0.16).abs() < 1e-12);
        assert!(d.weight.cg_forward_mac_frac < d.weight.cg_aft_mac_frac);
    }

    #[test]
    fn fuel_tanks_sum_to_published_capacity() {
        let d = doc();
        assert_eq!(d.fuel.len(), 2);
        assert_eq!(d.fuel[0].name, "left_wing");
        // 2 × 28.5 usable gal (98 %) ≈ 55.9 gal ≈ 211 L of Avgas
        let total_gal: f64 = d.fuel.iter().map(|t| t.capacity_m3 / GAL_TO_M3).sum();
        assert!((total_gal - 57.0).abs() < 1e-6, "total {total_gal} gal");
        let usable_gal: f64 = d
            .fuel
            .iter()
            .map(|t| t.capacity_m3 * t.usable_frac / GAL_TO_M3)
            .sum();
        assert!((usable_gal - 55.86).abs() < 1e-6, "usable {usable_gal} gal");
        // usable mass: 55.86 gal ≈ 0.2114 m³ × 719 kg/m³ ≈ 152 kg
        let kg = d.total_usable_fuel_kg();
        assert!((kg - 152.1).abs() < 1.0, "{kg}");
    }

    #[test]
    fn limits_order_enforced() {
        let d = doc();
        assert!(d.limits.vs1_mps() < d.limits.vne_mps());
        assert!((d.limits.service_ceiling_m() - 14000.0 * FT_TO_M).abs() < 1e-6);

        let bad = full_cfg().replace("vne_kias = 163", "vne_kias = 40");
        let cfg = CfgFile::parse(&bad, "fm.cfg").unwrap();
        let e = FlightModelDocument::from_cfg(&cfg).unwrap_err();
        assert!(e.to_string().contains("vs1 < vno < vne"), "{e}");
    }

    #[test]
    fn unstable_coefficients_rejected() {
        let bad = full_cfg().replace("pitch_moment_slope = -0.85", "pitch_moment_slope = 0.5");
        let cfg = CfgFile::parse(&bad, "fm.cfg").unwrap();
        let e = FlightModelDocument::from_cfg(&cfg).unwrap_err();
        assert!(e.to_string().contains("static stability"), "{e}");

        let bad = full_cfg().replace("directional_stability = 0.11", "directional_stability = -0.2");
        let cfg = CfgFile::parse(&bad, "fm.cfg").unwrap();
        let e = FlightModelDocument::from_cfg(&cfg).unwrap_err();
        assert!(e.to_string().contains("weathercock"), "{e}");
    }

    #[test]
    fn lift_curve_and_stall_behaviour() {
        let d = doc();
        let a = &d.aero;
        // linear region
        let cl = a.lift_coefficient(0.1, 0);
        assert!((cl - (0.15 + 5.1 * 0.1)).abs() < 1e-12);
        // flaps add incremental lift
        let clf = a.lift_coefficient(0.1, 3);
        assert!((clf - cl - (0.35 + 0.65 + 0.95)).abs() < 1e-12);
        let cl1 = a.lift_coefficient(0.1, 1);
        assert!((cl1 - cl - 0.35).abs() < 1e-12, "one notch adds flap_cl[0]");
        // past stall the lift drops below the linear extrapolation
        let cl_lin = 0.15 + 5.1 * 0.35;
        let cl_stalled = a.lift_coefficient(0.35, 0);
        assert!(
            cl_stalled < cl_lin,
            "stalled {cl_stalled} should drop below linear {cl_lin}"
        );
        // drag grows quadratically with lift
        let d0 = a.drag_coefficient(0.2, 7.5, 0);
        let d1 = a.drag_coefficient(0.8, 7.5, 0);
        assert!(d1 > d0 + 0.01);
    }

    #[test]
    fn flap_steps_consistent() {
        let d = doc();
        assert_eq!(d.flap_steps, 3);
        assert_eq!(d.aero.flap_cd_steps.len(), 3);
    }
}
