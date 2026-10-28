//! `engines.cfg` typed loader: piston engine + propeller.

use crate::error::{CfgError, CfgResult};
use crate::parser::CfgFile;
use crate::geometry::{LB_TO_KG, SLUG_FT2_TO_KG_M2};

/// Horsepower → watts.
pub const HP_TO_W: f64 = 745.699_871_582_270_2;
/// `lb/(hp·h)` → `kg/J` fuel mass flow per unit work.
pub const LB_PER_HP_H_TO_KG_PER_J: f64 = LB_TO_KG / (HP_TO_W * 3600.0);
/// Inches → metres.
pub const IN_TO_M: f64 = 0.0254;

/// A piston engine, loaded to SI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PistonEngine {
    /// Displacement, litres (converted from in³).
    pub displacement_l: f64,
    /// Maximum continuous shaft power, W.
    pub max_power_w: f64,
    /// Rated maximum crankshaft speed, rad/s.
    pub max_angular_velocity: f64,
    /// Idle speed, rad/s.
    pub idle_angular_velocity: f64,
    /// Brake-specific fuel consumption, kg/J.
    pub bsfc_kg_per_j: f64,
    /// Constant accessory power draw at full throttle, W.
    pub accessory_load_w: f64,
    /// Engine internal friction torque coefficient, N·m per (rad/s)
    /// (linear viscous model used by the torque balance).
    pub friction_torque_coeff: f64,
}

impl PistonEngine {
    /// Available shaft power at throttle `throttle ∈ [0, 1]`
    /// (linear throttle model with friction offset at idle).
    pub fn power_at(&self, throttle: f64) -> f64 {
        let t = throttle.clamp(0.0, 1.0);
        let frac = t.max(0.0);
        (self.max_power_w * frac - self.accessory_load_w).max(0.0)
    }

    /// Fuel mass flow (kg/s) producing `shaft_power` watts.
    pub fn fuel_flow_kg_s(&self, shaft_power: f64) -> f64 {
        shaft_power.max(0.0) * self.bsfc_kg_per_j
    }
}

/// A propeller (fixed or constant speed), loaded to SI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Propeller {
    /// Diameter, metres (converted from inches).
    pub diameter_m: f64,
    /// Moment of inertia about the spin axis, kg·m².
    pub inertia_kg_m2: f64,
    /// Number of blades.
    pub blades: u32,
    /// Maximum design RPM (prop/m governor limit), rad/s.
    pub max_angular_velocity: f64,
    /// Whether a constant-speed governor is fitted.
    pub constant_speed: bool,
    /// Optimum advance ratio `J = V / (n·D)` at peak efficiency.
    pub design_advance_ratio: f64,
    /// Peak aerodynamic efficiency at the design advance ratio.
    pub max_efficiency: f64,
    /// Geometric pitch (distance advanced per revolution), metres.
    pub geometric_pitch_m: f64,
}

impl Propeller {
    /// Solidity `σ = N·c̄·R / (π·R²)`-style blade-area proxy used by the
    /// momentum-theory thrust model; we expose the blade count factor.
    pub fn blade_count_factor(&self) -> f64 {
        self.blades as f64
    }

    /// Advance ratio for flight speed `v_mps` at shaft speed `omega`.
    pub fn advance_ratio(&self, v_mps: f64, omega: f64) -> f64 {
        if omega <= 1e-6 || self.diameter_m <= 0.0 {
            0.0
        } else {
            v_mps / (omega / (2.0 * std::f64::consts::PI) * self.diameter_m)
        }
    }

    /// Efficiency model: parabolic curve peaking at
    /// [`Self::design_advance_ratio`], zero at standstill and at the
    /// zero-thrust advance `J_max = pitch / D · blade-angle factor`.
    pub fn efficiency(&self, advance_ratio: f64) -> f64 {
        let j = advance_ratio.max(0.0);
        let j_opt = self.design_advance_ratio;
        if j_opt <= 0.0 {
            return 0.0;
        }
        // zero efficiency at J = 0 (static) rises to max at j_opt, falls
        // to zero again at 2·j_opt (windmilling region beyond).
        let x = (j - j_opt) / j_opt; // -1..+1 in the working band
        if j >= 2.0 * j_opt {
            return 0.0;
        }
        self.max_efficiency * (1.0 - x * x).max(0.0)
    }
}

/// Complete `engines.cfg` document.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineDefinition {
    /// Fuel grade/name as authored (e.g. `100LL`).
    pub fuel_type: String,
    /// The piston engine.
    pub piston: PistonEngine,
    /// The driven propeller.
    pub propeller: Propeller,
}

impl EngineDefinition {
    /// Load from a parsed `engines.cfg`.
    pub fn from_cfg(cfg: &CfgFile) -> CfgResult<Self> {
        let eng = cfg.require_section("PISTON_ENGINE")?;
        let displacement_in3 = eng.f64_in("displacement_in3", 10.0, 2000.0)?;
        let max_hp = eng.f64_in("max_power_hp", 10.0, 5000.0)?;
        let max_rpm = require_positive_in(eng, "max_rpm", 500.0, 5000.0)?;
        let idle_rpm = eng.f64_in("idle_rpm", 200.0, 4000.0)?;
        if idle_rpm >= max_rpm {
            return Err(CfgError::Range {
                file: eng.file.clone(),
                line: eng.find("idle_rpm").map(|e| e.line).unwrap_or(1),
                key: "idle_rpm".into(),
                message: "idle RPM must be below max RPM".into(),
            });
        }
        let bsfc = eng.f64_in("bsfc_lb_per_hp_hr", 0.2, 1.2)?;
        let piston = PistonEngine {
            displacement_l: displacement_in3 * 0.016387064,
            max_power_w: max_hp * HP_TO_W,
            max_angular_velocity: max_rpm * std::f64::consts::TAU / 60.0,
            idle_angular_velocity: idle_rpm * std::f64::consts::TAU / 60.0,
            bsfc_kg_per_j: bsfc * LB_PER_HP_H_TO_KG_PER_J,
            accessory_load_w: eng.f64_or("accessory_load_hp", 5.0)? * HP_TO_W,
            friction_torque_coeff: eng.f64_in("friction_torque_coeff", 0.0, 100.0)?,
        };

        let prop = cfg.require_section("PROPELLER")?;
        let diameter_in = prop.f64_in("diameter_in", 24.0, 150.0)?;
        let inertia = prop.f64_in("inertia_slug_ft2", 0.05, 100.0)?;
        let blades = prop.u64("num_blades")?;
        if !(2..=6).contains(&blades) {
            return Err(CfgError::Range {
                file: prop.file.clone(),
                line: 1,
                key: "num_blades".into(),
                message: "num_blades must be in 2..=6".into(),
            });
        }
        let prop_rpm = prop.f64_in("max_rpm", 500.0, 6000.0)?;
        let propeller = Propeller {
            diameter_m: diameter_in * IN_TO_M,
            inertia_kg_m2: inertia * SLUG_FT2_TO_KG_M2,
            blades: blades as u32,
            max_angular_velocity: prop_rpm * std::f64::consts::TAU / 60.0,
            constant_speed: prop.bool_or("constant_speed", false)?,
            design_advance_ratio: prop.f64_in("design_advance_ratio", 0.1, 3.0)?,
            max_efficiency: prop.f64_in("max_efficiency", 0.1, 0.95)?,
            geometric_pitch_m: prop.f64_in("geometric_pitch_in", 4.0, 200.0)? * IN_TO_M,
        };

        let fuel_type = eng.text_or("fuel_type", "100LL")?;

        let doc = Self { fuel_type, piston, propeller };
        doc.validate()?;
        Ok(doc)
    }

    /// Cross-consistency checks between engine and propeller.
    pub fn validate(&self) -> CfgResult<()> {
        let eng_rpm = self.piston.max_angular_velocity * 60.0 / std::f64::consts::TAU;
        let prop_rpm = self.propeller.max_angular_velocity * 60.0 / std::f64::consts::TAU;
        if prop_rpm - eng_rpm > 1.0 {
            return Err(CfgError::Range {
                file: "engines.cfg".into(),
                line: 1,
                key: "PROPELLER.max_rpm".into(),
                message: format!(
                    "propeller max RPM ({prop_rpm:.0}) exceeds engine max RPM ({eng_rpm:.0})"
                ),
            });
        }
        // static pitch should not imply a absurd tip speed
        let tip_speed = self.propeller.max_angular_velocity * self.propeller.diameter_m * 0.5;
        if tip_speed > 300.0 {
            return Err(CfgError::Range {
                file: "engines.cfg".into(),
                line: 1,
                key: "diameter_in".into(),
                message: format!("propeller tip speed {tip_speed:.0} m/s exceeds 300 m/s"),
            });
        }
        Ok(())
    }
}

fn require_positive_in(
    sec: &crate::parser::Section,
    key: &str,
    lo: f64,
    hi: f64,
) -> CfgResult<f64> {
    sec.f64_in(key, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENG: &str = r#"
    [PISTON_ENGINE]
    fuel_type = 100LL
    displacement_in3 = 360
    max_power_hp = 180
    max_rpm = 2700
    idle_rpm = 700
    bsfc_lb_per_hp_hr = 0.45
    accessory_load_hp = 5
    friction_torque_coeff = 1.8

    [PROPELLER]
    diameter_in = 76
    inertia_slug_ft2 = 1.2
    num_blades = 3
    max_rpm = 2700
    constant_speed = 0
    design_advance_ratio = 0.85
    max_efficiency = 0.86
    geometric_pitch_in = 62
    "#;

    fn engine() -> EngineDefinition {
        EngineDefinition::from_cfg(&CfgFile::parse(ENG, "e.cfg").unwrap()).unwrap()
    }

    #[test]
    fn loads_and_converts() {
        let e = engine();
        assert!((e.piston.displacement_l - 360.0 * 0.016387064).abs() < 1e-9);
        assert!((e.piston.max_power_w - 180.0 * HP_TO_W).abs() < 1e-6);
        // 2700 rpm = 282.74 rad/s
        assert!((e.piston.max_angular_velocity - 2700.0 * std::f64::consts::TAU / 60.0).abs() < 1e-9);
        assert!((e.propeller.diameter_m - 76.0 * IN_TO_M).abs() < 1e-12);
        assert_eq!(e.propeller.blades, 3);
        assert!(!e.propeller.constant_speed);
    }

    #[test]
    fn bsfc_produces_realistic_fuel_flow() {
        let e = engine();
        // full power 180 hp ≈ 134 kW, bsfc 0.45 lb/hp/h ≈ 56.7 kg/h
        let flow = e.piston.fuel_flow_kg_s(e.piston.max_power_w);
        let kg_per_h = flow * 3600.0;
        assert!((kg_per_h - 180.0 * 0.45 / 2.2046226).abs() < 0.5, "{kg_per_h} kg/h");
        // idle power: throttle 0 → accessory-limited, still non-negative
        assert!(e.piston.power_at(0.0) >= 0.0);
        assert!(e.piston.power_at(1.0) > 130_000.0);
    }

    #[test]
    fn prop_efficiency_curve_peaks_at_design() {
        let e = engine();
        let p = &e.propeller;
        let peak = p.efficiency(p.design_advance_ratio);
        assert!((peak - p.max_efficiency).abs() < 1e-12);
        // static (J=0) has no propulsive efficiency in this model
        assert_eq!(p.efficiency(0.0), 0.0);
        // beyond 2·J* the prop windmills
        assert_eq!(p.efficiency(3.0 * p.design_advance_ratio), 0.0);
        // below the peak the curve is monotonic
        let a = p.efficiency(0.3 * p.design_advance_ratio);
        let b = p.efficiency(0.6 * p.design_advance_ratio);
        assert!(b > a);
    }

    #[test]
    fn advance_ratio_matches_definition() {
        let e = engine();
        let p = &e.propeller;
        // V = J·n·D: n = 2700/60 = 45 rev/s, D = 1.9304 m
        let n = 45.0;
        let v = 0.85 * n * p.diameter_m;
        assert!((p.advance_ratio(v, p.max_angular_velocity) - 0.85).abs() < 1e-9);
        assert_eq!(p.advance_ratio(10.0, 0.0), 0.0);
    }

    #[test]
    fn invalid_configs_rejected() {
        let bad = ENG.replace("idle_rpm = 700", "idle_rpm = 3000");
        let cfg = CfgFile::parse(&bad, "e.cfg").unwrap();
        let err = EngineDefinition::from_cfg(&cfg).unwrap_err();
        assert!(err.to_string().contains("idle RPM"), "{err}");

        let bad = ENG.replace("num_blades = 3", "num_blades = 1");
        let cfg = CfgFile::parse(&bad, "e.cfg").unwrap();
        let err = EngineDefinition::from_cfg(&cfg).unwrap_err();
        assert!(err.to_string().contains("2..=6"), "{err}");
    }
}
