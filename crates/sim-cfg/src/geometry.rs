//! `[AIRPLANE_GEOMETRY]` / `[REF_GEOMETRY]` typed loader with imperial →
//! SI conversion.

use crate::error::{CfgError, CfgResult};
use crate::parser::CfgFile;

/// Feet → metres.
pub const FT_TO_M: f64 = 0.3048;
/// Square feet → square metres.
pub const FT2_TO_M2: f64 = 0.09290304;
/// Slug·ft² → kg·m² (rotational inertia).
pub const SLUG_FT2_TO_KG_M2: f64 = 1.355817947;
/// Pounds → kilograms.
pub const LB_TO_KG: f64 = 0.45359237;
/// US gallons → cubic metres (fuel volume).
pub const GAL_TO_M3: f64 = 0.003785411784;

/// Planform geometry of one lifting surface as authored in the cfg
/// (all lengths converted to SI on load).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WingGeometry {
    /// Stable identifier, e.g. `wing`, `htail`, `vtail`.
    pub name: &'static str,
    /// Full span (for a wing: tip to tip), metres.
    pub span_m: f64,
    /// Root chord, metres.
    pub root_chord_m: f64,
    /// Tip chord, metres.
    pub tip_chord_m: f64,
    /// Quarter-chord sweep angle, radians (positive = tip aft).
    pub sweep_rad: f64,
    /// Dihedral angle, radians (positive = tip above root).
    pub dihedral_rad: f64,
    /// Geometric incidence at the root, radians (positive = LE up).
    pub incidence_rad: f64,
    /// Washout from root to tip, radians (negative = tip LE down).
    pub washout_rad: f64,
    /// Area as authored (one side for wings, total for tails), m².
    /// When the cfg provides `*_area`, it wins over the trapezoid
    /// integration; otherwise the trapezoid value is used.
    pub area_m2: f64,
    /// Body-frame position of the root quarter-chord
    /// (+X forward, +Y right, +Z down), metres.
    pub position_m: glam::DVec3,
}

impl WingGeometry {
    /// Linearly tapered chord at span fraction `t ∈ [0, 1]` (0 = root).
    pub fn chord_at(&self, t: f64) -> f64 {
        self.root_chord_m + (self.tip_chord_m - self.root_chord_m) * t.clamp(0.0, 1.0)
    }

    /// Mean aerodynamic chord of the trapezoid:
    /// `MAC = (2/3)·c_root·(1+λ+λ²)/(1+λ)`.
    pub fn mean_aero_chord(&self) -> f64 {
        let cr = self.root_chord_m;
        let lambda = if cr > 0.0 { self.tip_chord_m / cr } else { 0.0 };
        (2.0 / 3.0) * cr * (1.0 + lambda + lambda * lambda) / (1.0 + lambda)
    }

    /// Aspect ratio `b² / S`.
    pub fn aspect_ratio(&self) -> f64 {
        if self.area_m2 > 0.0 {
            self.span_m * self.span_m / self.area_m2
        } else {
            0.0
        }
    }

    /// Trapezoid-integrated area of ONE side (root→tip).
    pub fn trapezoid_side_area(&self) -> f64 {
        (self.root_chord_m + self.tip_chord_m) * 0.5 * (self.span_m * 0.5)
    }
}

/// A tail is just a [`WingGeometry`] with its own name; kept as a
/// distinct type for API clarity.
pub type TailSurface = WingGeometry;

/// Full `[AIRPLANE_GEOMETRY]` document, converted to SI.
#[derive(Debug, Clone, PartialEq)]
pub struct AirplaneGeometry {
    /// Main wing planform (area/span are FULL wing values).
    pub wing: WingGeometry,
    /// Horizontal stabilizer.
    pub htail: TailSurface,
    /// Vertical stabilizer (span = fin height).
    pub vtail: TailSurface,
    /// Empty inertia about the body axes, kg·m² (+X roll, +Y pitch, +Z yaw).
    pub inertia: [f64; 3],
    /// Maximum pitch control authority indicators (areas), m², per surface.
    pub elevator_areas_m2: Vec<f64>,
    /// Aileron areas, m² (index 0 = right, 1 = left as authored).
    pub aileron_areas_m2: Vec<f64>,
    /// Rudder area, m².
    pub rudder_area_m2: f64,
    /// Wheelbase (nose gear to main gear), metres.
    pub wheelbase_m: f64,
    /// Track width (main gear), metres.
    pub track_m: f64,
}

fn deg_key(sec: &crate::parser::Section, key: &str, default_deg: f64) -> CfgResult<f64> {
    Ok(sec.f64_or(key, default_deg)?.to_radians())
}

fn pos3(sec: &crate::parser::Section, key: &str) -> CfgResult<glam::DVec3> {
    let v = sec.vec_n(key, 3)?;
    // cfg body axes: x forward, y right, z down (already SI-agnostic
    // in feet; converted here).
    Ok(glam::DVec3::new(v[0] * FT_TO_M, v[1] * FT_TO_M, v[2] * FT_TO_M))
}

impl AirplaneGeometry {
    /// Load from a parsed cfg document.
    pub fn from_cfg(cfg: &CfgFile) -> CfgResult<Self> {
        let sec = cfg.require_section("AIRPLANE_GEOMETRY")?;
        let file = &sec.file;

        let wing_area_ft2 = sec.f64("wing_area")?;
        let span_ft = sec.f64("wing_span")?;
        let root_ft = sec.f64("wing_root_chord")?;
        let tip_ft = sec.f64_or("wing_tip_chord", root_ft)?;
        if wing_area_ft2 <= 0.0 {
            let e = sec.find("wing_area").expect("checked");
            return Err(CfgError::Range {
                file: file.clone(),
                line: e.line,
                key: "wing_area".into(),
                message: "wing_area must be positive".into(),
            });
        }
        if span_ft <= 0.0 || root_ft <= 0.0 {
            let e = sec.find("wing_span").expect("checked");
            return Err(CfgError::Range {
                file: file.clone(),
                line: e.line,
                key: "wing_span".into(),
                message: "wing span and chords must be positive".into(),
            });
        }

        let wing_pos = pos3(sec, "wing_position")?;
        let wing = WingGeometry {
            name: "wing",
            span_m: span_ft * FT_TO_M,
            root_chord_m: root_ft * FT_TO_M,
            tip_chord_m: tip_ft * FT_TO_M,
            sweep_rad: deg_key(sec, "wing_quarter_chord_sweep", 0.0)?,
            dihedral_rad: deg_key(sec, "wing_dihedral", 0.0)?,
            incidence_rad: deg_key(sec, "wing_incidence", 0.0)?,
            washout_rad: deg_key(sec, "wing_twist", 0.0)?,
            area_m2: wing_area_ft2 * FT2_TO_M2,
            position_m: wing_pos,
        };

        let htail = Self::load_tail(sec, "htail", "horizontal_tail")?;
        let vtail = Self::load_tail(sec, "vtail", "vertical_tail")?;

        let ixx = sec.f64("empty_Ixx")? * SLUG_FT2_TO_KG_M2;
        let iyy = sec.f64("empty_Iyy")? * SLUG_FT2_TO_KG_M2;
        let izz = sec.f64("empty_Izz")? * SLUG_FT2_TO_KG_M2;
        if ixx <= 0.0 || iyy <= 0.0 || izz <= 0.0 {
            return Err(CfgError::Range {
                file: file.clone(),
                line: sec.find("empty_Ixx").map(|e| e.line).unwrap_or(1),
                key: "empty_Ixx".into(),
                message: "inertia tensor entries must be positive".into(),
            });
        }
        // triangle inequality of an inertia tensor: Ixx + Iyy >= Izz etc.
        for (i, msg) in [
            (ixx + iyy - izz, "Ixx + Iyy < Izz"),
            (ixx + izz - iyy, "Ixx + Izz < Iyy"),
            (iyy + izz - ixx, "Iyy + Izz < Ixx"),
        ] {
            if i < 0.0 {
                return Err(CfgError::Range {
                    file: file.clone(),
                    line: 1,
                    key: "empty_Ixx".into(),
                    message: format!("inertia triangle inequality violated ({msg})"),
                });
            }
        }

        let elevator_areas = sec
            .indexed_f64s("elevator_area")?
            .into_values()
            .map(|a| a * FT2_TO_M2)
            .collect();
        let aileron_areas = sec
            .indexed_f64s("aileron_area")?
            .into_values()
            .map(|a| a * FT2_TO_M2)
            .collect();
        let rudder_area_m2 = sec.f64("rudder_area")? * FT2_TO_M2;

        Ok(Self {
            wing,
            htail,
            vtail,
            inertia: [ixx, iyy, izz],
            elevator_areas_m2: elevator_areas,
            aileron_areas_m2: aileron_areas,
            rudder_area_m2,
            wheelbase_m: sec.f64("wheelbase")? * FT_TO_M,
            track_m: sec.f64("wheel_track")? * FT_TO_M,
        })
    }

    fn load_tail(sec: &crate::parser::Section, short: &str, long: &str) -> CfgResult<WingGeometry> {
        let p = |suffix: &str| -> String { format!("{short}_{suffix}") };
        let pl = |suffix: &str| -> String { format!("{long}_{suffix}") };
        let get = |a: &str, b: &str| -> CfgResult<f64> {
            if sec.find(a).is_some() {
                sec.f64(a)
            } else {
                sec.f64(b)
            }
        };
        let area = get(&p("area"), &pl("area"))?;
        let span = get(&p("span"), &pl("span"))?;
        let root = get(&p("root_chord"), &pl("root_chord"))?;
        let tip = get(&p("tip_chord"), &pl("tip_chord")).unwrap_or(root);
        let pos = match sec.find(&p("position")) {
            Some(_) => pos3(sec, &p("position"))?,
            None => pos3(sec, &pl("position"))?,
        };
        if area <= 0.0 || span <= 0.0 || root <= 0.0 {
            return Err(CfgError::Range {
                file: sec.file.clone(),
                line: sec.find(&p("area")).or(sec.find(&pl("area"))).map(|e| e.line).unwrap_or(1),
                key: p("area"),
                message: "tail area/span/chord must be positive".into(),
            });
        }
        Ok(WingGeometry {
            name: if short == "htail" { "htail" } else { "vtail" },
            span_m: span * FT_TO_M,
            root_chord_m: root * FT_TO_M,
            tip_chord_m: tip * FT_TO_M,
            sweep_rad: deg_key(sec, &p("quarter_chord_sweep"), deg_key(sec, &pl("sweep"), 0.0)?)?,
            dihedral_rad: deg_key(sec, &p("dihedral"), 0.0)?,
            incidence_rad: deg_key(sec, &p("incidence"), 0.0)?,
            washout_rad: deg_key(sec, &p("twist"), 0.0)?,
            area_m2: area * FT2_TO_M2,
            position_m: pos,
        })
    }

    /// Total wing area (both sides), m².
    pub fn wing_area_m2(&self) -> f64 {
        self.wing.area_m2
    }

    /// MAC of the main wing, metres.
    pub fn mac_m(&self) -> f64 {
        self.wing.mean_aero_chord()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const GEOM: &str = r#"
    [AIRPLANE_GEOMETRY]
    wing_area = 174
    wing_span = 36.1
    wing_root_chord = 5.5
    wing_tip_chord = 4.15
    wing_quarter_chord_sweep = 0
    wing_dihedral = 1.73
    wing_incidence = 1.5
    wing_twist = -3
    wing_position = -1.5, 0, -0.5
    htail_area = 21.4
    htail_span = 11.0
    htail_root_chord = 2.4
    htail_tip_chord = 1.6
    htail_position = -15.0, 0, 0.8
    vtail_area = 11.6
    vtail_span = 5.5
    vtail_root_chord = 3.0
    vtail_tip_chord = 1.4
    vtail_position = -15.5, 0, -3.5
    elevator_area[0] = 7.1
    elevator_area[1] = 7.1
    aileron_area[0] = 5.9
    aileron_area[1] = 5.9
    rudder_area = 5.0
    empty_Ixx = 1269
    empty_Iyy = 1813
    empty_Izz = 2940
    wheelbase = 9.2
    wheel_track = 5.3
    "#;

    fn geom() -> AirplaneGeometry {
        let cfg = CfgFile::parse(GEOM, "g.cfg").unwrap();
        AirplaneGeometry::from_cfg(&cfg).unwrap()
    }

    #[test]
    fn converts_imperial_to_si() {
        let g = geom();
        assert!((g.wing.area_m2 - 174.0 * FT2_TO_M2).abs() < 1e-9);
        assert!((g.wing.span_m - 36.1 * FT_TO_M).abs() < 1e-9);
        assert!((g.inertia[0] - 1269.0 * SLUG_FT2_TO_KG_M2).abs() < 1e-6);
        assert_eq!(g.elevator_areas_m2.len(), 2);
        assert!((g.elevator_areas_m2[1] - 7.1 * FT2_TO_M2).abs() < 1e-9);
        assert!((g.rudder_area_m2 - 5.0 * FT2_TO_M2).abs() < 1e-9);
        assert!((g.wheelbase_m - 9.2 * FT_TO_M).abs() < 1e-9);
    }

    #[test]
    fn angles_are_radians() {
        let g = geom();
        assert!((g.wing.dihedral_rad - 1.73f64.to_radians()).abs() < 1e-12);
        assert!((g.wing.incidence_rad - 1.5f64.to_radians()).abs() < 1e-12);
        assert!((g.wing.washout_rad - (-3.0f64).to_radians()).abs() < 1e-12);
    }

    #[test]
    fn mac_and_aspect_ratio() {
        let g = geom();
        let mac = g.mac_m();
        // trapezoid MAC must sit between tip and root chord
        assert!(mac > g.wing.tip_chord_m && mac < g.wing.root_chord_m, "mac={mac}");
        // rectangular MAC = chord
        let rect = WingGeometry {
            name: "t",
            span_m: 10.0,
            root_chord_m: 2.0,
            tip_chord_m: 2.0,
            sweep_rad: 0.0,
            dihedral_rad: 0.0,
            incidence_rad: 0.0,
            washout_rad: 0.0,
            area_m2: 20.0,
            position_m: glam::DVec3::ZERO,
        };
        assert!((rect.mean_aero_chord() - 2.0).abs() < 1e-12);
        assert!((rect.aspect_ratio() - 5.0).abs() < 1e-12);
        // λ = 0.5: MAC = (2/3)(1+0.5+0.25)/1.5 · cr = 0.7778 cr
        let trap = WingGeometry { root_chord_m: 3.0, tip_chord_m: 1.5, ..rect };
        assert!((trap.mean_aero_chord() - 3.0 * (2.0 / 3.0) * 1.75 / 1.5).abs() < 1e-12);
    }

    #[test]
    fn missing_required_key_errors() {
        let cfg = CfgFile::parse("[AIRPLANE_GEOMETRY]\nwing_span = 36\n", "g.cfg").unwrap();
        let e = AirplaneGeometry::from_cfg(&cfg).unwrap_err();
        assert!(e.to_string().contains("'wing_area'"), "{e}");
    }

    #[test]
    fn zero_wing_area_rejected() {
        let cfg = CfgFile::parse(
            "[AIRPLANE_GEOMETRY]\nwing_area = 0\nwing_span = 30\nwing_root_chord = 5\n",
            "g.cfg",
        )
        .unwrap();
        let e = AirplaneGeometry::from_cfg(&cfg).unwrap_err();
        assert!(e.to_string().contains("must be positive"), "{e}");
    }
}
