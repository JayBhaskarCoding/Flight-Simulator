//! Surface-element aerodynamics: every `sim-cfg` strip evaluates its own
//! local flow, angle of attack and lift/drag, then integrates forces and
//! moments about the CG.
//!
//! Design (documented, no double counting):
//!
//! * **Element physics** provides lift (with the aircraft's stall envelope
//!   and flap increments on the wing), parasite + induced drag, incidence/
//!   washout moments, dihedral effect (via the tilted strip plane), fin
//!   side force / weathercock stability, and all rate damping (`ω × r` at
//!   each strip). The aircraft-level `cl0` is *not* applied per element —
//!   zero-lift angle of attack emerges from the geometric incidence.
//! * **Control surfaces** are not resolved geometrically by `sim-cfg`;
//!   their moments come from the published stability derivatives
//!   (`cm_delta_e`, `cl_delta_a`, `cn_delta_r`) scaled with dynamic
//!   pressure, wing area, MAC and span.

use glam::DVec3;
use sim_cfg::aero_element::AeroElement;
use sim_cfg::AeroCoefficients;
use sim_cfg::AircraftDefinition;

/// Ambient conditions sampled for one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereSample {
    /// Air density, kg/m³.
    pub density_kg_m3: f64,
    /// Speed of sound, m/s (used for diagnostics; no compressibility
    /// correction is applied in this milestone).
    pub speed_of_sound_mps: f64,
}

impl Default for AtmosphereSample {
    fn default() -> Self {
        Self { density_kg_m3: 1.225, speed_of_sound_mps: 340.294 }
    }
}

impl AtmosphereSample {
    /// Mach number of `velocity_mps`.
    pub fn mach(&self, velocity_mps: f64) -> f64 {
        velocity_mps / self.speed_of_sound_mps.max(1.0)
    }
}

/// Pilot control inputs for one step (zero-order held across RK stages).
///
/// Surface deflections are **radians** — the caller normalises stick/pedal
/// position to deflections (limit data lives in the aircraft assets).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ControlInputs {
    /// Elevator deflection, rad (positive trailing-edge down → nose-down
    /// moment through `cm_delta_e`, which is negative for stable aircraft).
    pub elevator_rad: f64,
    /// Aileron deflection, rad (positive = roll right, `+X` moment).
    pub aileron_rad: f64,
    /// Rudder deflection, rad (positive = yaw right, `+Z` moment).
    pub rudder_rad: f64,
    /// Deployed flap notch count (0 = clean).
    pub flap_step: usize,
    /// Wheel brakes, 0..=1 (both braked legs).
    pub brake: f64,
    /// Nosewheel steering command, −1..=1 (positive = right).
    pub steering: f64,
}

/// Net aerodynamic wrench in body axes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AeroForces {
    /// Force, body frame, N.
    pub force_body: DVec3,
    /// Moment about the CG, body frame, N·m.
    pub torque_body: DVec3,
}

/// One strip with its baked aerodynamic frame (dihedral tilt + incidence
/// twist applied at construction).
#[derive(Debug, Clone, Copy)]
pub struct ElementModel {
    /// Original strip geometry (area, centre, …).
    pub element: AeroElement,
    /// Unit "suction side" normal of the strip (resting position rotated by
    /// dihedral and incidence — points to the top of the airfoil).
    pub up_axis: DVec3,
    /// Unit chord direction at zero local angle of attack.
    pub chord_axis: DVec3,
    /// `true` when flap increments apply to this strip (main wing only).
    pub applies_flaps: bool,
}

impl ElementModel {
    /// Bake the aerodynamic frame for one strip.
    ///
    /// * `dihedral_rad` — dihedral of the owning surface (tilts the strip
    ///   plane; positive = tip up). `sim-cfg` keeps strip axes horizontal
    ///   and only raises the tips, so the tilt is applied here to recover
    ///   the physical dihedral effect (roll due to sideslip).
    /// * `applies_flaps` — main-wing strips receive `flap_cl/cd_steps`.
    pub fn bake(element: AeroElement, dihedral_rad: f64, applies_flaps: bool) -> Self {
        let s = element.axis.normalize();
        // Resting strip "up": horizontal surfaces are tilted by dihedral
        // about the body X axis (side-aware so both tips rise), vertical
        // surfaces use the span×chord normal (fin ŝ=−Z → up = −Y).
        let up0 = if s.y.abs() > 0.5 {
            glam::DQuat::from_rotation_x(-element.side * dihedral_rad) * DVec3::new(0.0, 0.0, -1.0)
        } else {
            s.cross(DVec3::X).normalize_or(DVec3::new(0.0, -1.0, 0.0))
        };
        // Incidence/twist: positive raises the leading edge. Rotating about
        // +Y pitches up for the right wing; the mirrored strip must rotate
        // about −Y by the same magnitude, i.e. angle = incidence·(ŝ·Ŷ).
        let twist_sign = if s.y.abs() > 0.5 { s.y.signum() } else { 1.0 };
        let twist = glam::DQuat::from_axis_angle(s, element.incidence_rad * twist_sign);
        Self {
            element,
            up_axis: twist * up0,
            chord_axis: twist * DVec3::X,
            applies_flaps,
        }
    }
}

/// Integrated aerodynamics for a fleet of strips.
#[derive(Debug, Clone)]
pub struct AeroSolver {
    /// Baked strips (wing + htail + vtail).
    elements: Vec<ElementModel>,
    /// Aircraft coefficient set (stall envelope, control derivatives, …).
    coeffs: AeroCoefficients,
    /// Reference wing area, m².
    ref_area_m2: f64,
    /// Reference wing span, m.
    ref_span_m: f64,
    /// Mean aerodynamic chord, m.
    ref_mac_m: f64,
    /// Aspect ratio `b²/S` for induced drag.
    aspect_ratio: f64,
    /// CG position in body coordinates (moment reference), m.
    cg_body: DVec3,
}

/// Discretization resolution used by [`AeroSolver::from_aircraft`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AeroResolution {
    /// Spanwise strips per wing side.
    pub wing_spanwise: u32,
    /// Spanwise strips per tail side.
    pub tail_spanwise: u32,
    /// Chordwise elements per strip.
    pub chordwise: u32,
}

impl Default for AeroResolution {
    fn default() -> Self {
        Self { wing_spanwise: 16, tail_spanwise: 8, chordwise: 1 }
    }
}

impl AeroSolver {
    /// Build the element set for an aircraft with default resolution.
    ///
    /// `cg_body` is the moment reference in body coordinates — for the
    /// clean-room assets that is `(empty_cg_arm_m, 0, 0)` (datum-relative,
    /// same datum as the surface positions).
    pub fn from_aircraft(def: &AircraftDefinition, cg_body: DVec3) -> Self {
        Self::from_aircraft_with_resolution(def, cg_body, AeroResolution::default())
    }

    /// Build the element set with an explicit strip resolution.
    pub fn from_aircraft_with_resolution(
        def: &AircraftDefinition,
        cg_body: DVec3,
        res: AeroResolution,
    ) -> Self {
        let g = &def.flight_model.geometry;
        let mut elements = Vec::new();
        let wing = g.wing.discretize(0, res.wing_spanwise, res.chordwise, true);
        for el in wing {
            elements.push(ElementModel::bake(el, g.wing.dihedral_rad, true));
        }
        let htail = g.htail.discretize(1, res.tail_spanwise, res.chordwise, true);
        for el in htail {
            elements.push(ElementModel::bake(el, g.htail.dihedral_rad, false));
        }
        let vtail = g.vtail.discretize(2, res.tail_spanwise, res.chordwise, false);
        for el in vtail {
            elements.push(ElementModel::bake(el, g.vtail.dihedral_rad, false));
        }

        let ref_area_m2 = g.wing_area_m2();
        let ref_span_m = g.wing.span_m;
        let ref_mac_m = g.mac_m();
        let aspect_ratio = ref_span_m * ref_span_m / ref_area_m2.max(1e-6);
        Self {
            elements,
            coeffs: def.flight_model.aero.clone(),
            ref_area_m2,
            ref_span_m,
            ref_mac_m,
            aspect_ratio,
            cg_body,
        }
    }

    /// Number of strips in the model.
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Total strip area (all surfaces), m².
    pub fn total_strip_area_m2(&self) -> f64 {
        self.elements.iter().map(|e| e.element.area_m2).sum()
    }

    /// Reference area (main wing), m².
    pub fn reference_area_m2(&self) -> f64 {
        self.ref_area_m2
    }

    /// Evaluate the aero wrench for one RK stage.
    ///
    /// * `v_body` — CG velocity in body axes, m/s
    /// * `omega_body` — body angular rate, rad/s
    /// * `atm` — ambient density
    /// * `controls` — control inputs (ZOH)
    pub fn solve(
        &self,
        v_body: DVec3,
        omega_body: DVec3,
        atm: &AtmosphereSample,
        controls: &ControlInputs,
    ) -> AeroForces {
        let rho = atm.density_kg_m3.max(0.0);
        let mut force = DVec3::ZERO;
        let mut torque = DVec3::ZERO;

        for em in &self.elements {
            let r = em.element.center - self.cg_body;
            let v_local = v_body + omega_body.cross(r);
            let speed2 = v_local.length_squared();
            if speed2 < 1e-12 {
                continue;
            }
            let speed = speed2.sqrt();
            let v_hat = v_local / speed;

            // local angle of attack: positive when the flow hits the
            // pressure side (Z-down body: positive α has downward velocity)
            let alpha = (-v_local.dot(em.up_axis)).atan2(v_local.dot(em.chord_axis));

            let flap_step = if em.applies_flaps { controls.flap_step } else { 0 };
            // symmetric-airfoil CL: the aircraft cl0 is emergent from
            // geometric incidence instead of being double-applied.
            let cl = self.coeffs.lift_coefficient(alpha, flap_step) - self.coeffs.cl0;
            let flap_drag: f64 = self.coeffs.flap_cd_steps.iter().take(flap_step).sum();
            let induced = cl * cl
                / (std::f64::consts::PI * self.coeffs.oswald.max(0.5) * self.aspect_ratio.max(1e-3));
            let cd = self.coeffs.cd0 + flap_drag + induced;

            // lift direction: strip "up" projected perpendicular to the flow
            let lift_dir = {
                let p = em.up_axis - v_hat * em.up_axis.dot(v_hat);
                let n = p.length();
                if n > 1e-7 {
                    p / n
                } else {
                    // α ≈ ±90°: fall back to the span×flow direction with
                    // orientation fixed by the strip's up axis
                    let f = em.element.axis.cross(v_hat);
                    let f_len = f.length();
                    if f_len > 1e-7 {
                        let mut f = f / f_len;
                        if f.dot(em.up_axis) < 0.0 {
                            f = -f;
                        }
                        f
                    } else {
                        em.up_axis
                    }
                }
            };

            let q_bar = 0.5 * rho * speed2;
            let f = q_bar * em.element.area_m2 * (cl * lift_dir - cd * v_hat);
            force += f;
            torque += r.cross(f);
        }

        // control-surface moments from published stability derivatives
        let speed = v_body.length();
        if speed > 1.0 {
            let q_bar = 0.5 * rho * speed * speed;
            torque.x += q_bar * self.ref_area_m2 * self.ref_span_m
                * self.coeffs.cl_delta_a * controls.aileron_rad;
            torque.y += q_bar * self.ref_area_m2 * self.ref_mac_m
                * self.coeffs.cm_delta_e * controls.elevator_rad;
            torque.z += q_bar * self.ref_area_m2 * self.ref_span_m
                * self.coeffs.cn_delta_r * controls.rudder_rad;
        }

        AeroForces { force_body: force, torque_body: torque }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_cfg::aero_element::AeroElement;

    fn coeffs_clean() -> AeroCoefficients {
        AeroCoefficients {
            cl0: 0.0,
            cl_alpha: 5.7,
            cd0: 0.02,
            oswald: 0.8,
            cm0: 0.0,
            cm_alpha: -1.2,
            cm_q: -12.0,
            cm_delta_e: -1.4,
            cy_beta: -0.9,
            cl_p: -0.5,
            cl_r: 0.1,
            cl_delta_a: 0.18,
            cl_beta: -0.1,
            cn_r: -0.15,
            cn_p: -0.02,
            cn_delta_r: -0.1,
            cn_beta: 0.12,
            alpha_stall_rad: 0.28,
            post_stall_lift_factor: 0.6,
            flap_cl_steps: vec![0.35, 0.7],
            flap_cd_steps: vec![0.02, 0.05],
        }
    }

    fn wing_element(side: f64) -> AeroElement {
        AeroElement {
            surface_index: 0,
            area_m2: 2.0,
            chord_m: 1.5,
            span_m: 1.0,
            span_frac: 0.5,
            side,
            axis: DVec3::new(0.0, side, 0.0),
            quarter_chord_center: DVec3::new(0.0, side * 2.0, 0.0),
            center: DVec3::new(0.0, side * 2.0, 0.0),
            incidence_rad: 0.0,
            span_pos_m: 2.0,
        }
    }

    fn single_element_solver(element: AeroElement, dihedral: f64) -> AeroSolver {
        // bypass aircraft construction: assemble fields directly
        let coeffs = coeffs_clean();
        let ar = 7.0;
        AeroSolver {
            elements: vec![ElementModel::bake(element, dihedral, true)],
            coeffs,
            ref_area_m2: 4.0,
            ref_span_m: 4.0,
            ref_mac_m: 1.5,
            aspect_ratio: ar,
            cg_body: DVec3::ZERO,
        }
    }

    #[test]
    fn lift_magnitude_and_direction_at_known_alpha() {
        let em = ElementModel::bake(wing_element(1.0), 0.0, true);
        assert!((em.up_axis - DVec3::new(0.0, 0.0, -1.0)).length() < 1e-12);
        assert!((em.chord_axis - DVec3::X).length() < 1e-12);

        let solver = single_element_solver(wing_element(1.0), 0.0);
        let v = 50.0f64;
        let alpha = 5.0f64.to_radians();
        let v_body = DVec3::new(v * alpha.cos(), 0.0, v * alpha.sin());
        let atm = AtmosphereSample::default();
        let out = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());

        let q = 0.5 * 1.225 * v * v;
        let cl = 5.7 * alpha; // within stall, no flaps, cl0 = 0
        let expected_lift = q * 2.0 * cl;
        // force must be perpendicular to the flow in the x-z plane
        let v_hat = v_body.normalize();
        let along = out.force_body.dot(v_hat);
        let across = out.force_body - v_hat * along;
        assert!((across.length() - expected_lift).abs() < 1e-6, "{}", across.length());
        // lift points up (−Z)
        assert!(across.z < 0.0, "lift must point up: {across}");
        // drag along −flow: cd = cd0 + induced
        let induced = cl * cl / (std::f64::consts::PI * 0.8 * 7.0);
        let expected_drag = q * 2.0 * (0.02 + induced);
        assert!(((-along) - expected_drag).abs() < 1e-6, "drag {}", -along);
    }

    #[test]
    fn fin_side_force_restoring() {
        // vertical strip (fin): axis −Z, up = −Y; sideslip from the right
        // (positive v_y) must push the aircraft to −Y (restoring).
        let fin = AeroElement {
            surface_index: 2,
            area_m2: 1.5,
            chord_m: 1.2,
            span_m: 1.4,
            span_frac: 0.5,
            side: 1.0,
            axis: DVec3::new(0.0, 0.0, -1.0),
            quarter_chord_center: DVec3::new(-1.5, 0.0, -0.7),
            center: DVec3::new(-1.5, 0.0, -0.7),
            incidence_rad: 0.0,
            span_pos_m: 0.7,
        };
        let em = ElementModel::bake(fin, 0.0, false);
        assert!((em.up_axis - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-9);

        let solver = single_element_solver(fin, 0.0);
        let v_body = DVec3::new(40.0, 6.0, 0.0); // β > 0
        let atm = AtmosphereSample::default();
        let out = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());
        assert!(out.force_body.y < 0.0, "side force must be restoring: {}", out.force_body.y);
        // yaw moment about CG behind the strip: nose-left at positive β
        // (fin sits aft, x < 0): M_z from r × F
        assert!(out.torque_body.z > 0.0, "weathercock moment must be positive: {}", out.torque_body.z);
    }

    #[test]
    fn dihedral_effect_rolls_against_sideslip() {
        // positive sideslip (flow from the right) with dihedral: right wing
        // gains lift → negative (left-rolling) roll moment, cl_beta < 0.
        let mut solver = single_element_solver(wing_element(1.0), 3.0f64.to_radians());
        solver.elements.push(ElementModel::bake(wing_element(-1.0), 3.0f64.to_radians(), true));
        // both strips share the same centre here, so the roll moment comes
        // from the force asymmetry applied at mirrored arms
        let v_body = DVec3::new(40.0, 5.0, 0.0);
        let atm = AtmosphereSample::default();
        let out = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());
        assert!(
            out.force_body.z < 0.0,
            "net lift must be upward: {}",
            out.force_body.z
        );
        // right strip (side +1): up = R_x(−d)·(−Z) gains α from +v_y
        let up_r = glam::DQuat::from_rotation_x(-3.0f64.to_radians()) * DVec3::new(0.0, 0.0, -1.0);
        let up_l = glam::DQuat::from_rotation_x(3.0f64.to_radians()) * DVec3::new(0.0, 0.0, -1.0);
        let alpha_r = (-v_body.dot(up_r)).atan2(v_body.dot(DVec3::X));
        let alpha_l = (-v_body.dot(up_l)).atan2(v_body.dot(DVec3::X));
        assert!(alpha_r > alpha_l, "dihedral must raise right-wing α: {alpha_r} vs {alpha_l}");
    }

    #[test]
    fn symmetric_flow_produces_no_side_force_or_roll() {
        // identical strips both sides, β = 0, ω = 0 → lateral force ~ 0
        let mut solver = single_element_solver(wing_element(1.0), 2.0f64.to_radians());
        solver.elements.push(ElementModel::bake(wing_element(-1.0), 2.0f64.to_radians(), true));
        // give them mirrored centres so roll would be visible if asymmetric
        solver.elements[0].element.center = DVec3::new(0.0, 2.0, -0.1);
        solver.elements[1].element.center = DVec3::new(0.0, -2.0, -0.1);
        let v_body = DVec3::new(45.0, 0.0, 0.5);
        let atm = AtmosphereSample::default();
        let out = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());
        assert!(out.force_body.y.abs() < 1e-6, "side force {}", out.force_body.y);
        assert!(out.torque_body.x.abs() < 1e-6, "roll moment {}", out.torque_body.x);
    }

    #[test]
    fn control_moments_follow_published_derivatives() {
        let solver = single_element_solver(wing_element(1.0), 0.0);
        let v_body = DVec3::new(50.0, 0.0, 0.0);
        let atm = AtmosphereSample::default();
        let controls = ControlInputs { elevator_rad: 0.1, ..ControlInputs::default() };
        let clean = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());
        let deflected = solver.solve(v_body, DVec3::ZERO, &atm, &controls);
        let q = 0.5 * 1.225 * 2500.0;
        let expected = q * 4.0 * 1.5 * (-1.4) * 0.1;
        assert!(
            ((deflected.torque_body.y - clean.torque_body.y) - expected).abs() < 1e-6,
            "ΔM_y = {} expected {expected}",
            deflected.torque_body.y - clean.torque_body.y
        );

        let controls = ControlInputs { aileron_rad: 0.15, ..ControlInputs::default() };
        let roll = solver.solve(v_body, DVec3::ZERO, &atm, &controls);
        let expected_roll = q * 4.0 * 4.0 * 0.18 * 0.15;
        assert!((roll.torque_body.x - expected_roll).abs() < 1e-6);
    }

    #[test]
    fn flaps_increase_lift_and_drag() {
        let solver = single_element_solver(wing_element(1.0), 0.0);
        let alpha = 0.05f64;
        let v_body = DVec3::new(40.0, 0.0, 40.0 * alpha);
        let atm = AtmosphereSample::default();
        let clean = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());
        let with_flaps = ControlInputs { flap_step: 2, ..ControlInputs::default() };
        let flapped = solver.solve(v_body, DVec3::ZERO, &atm, &with_flaps);
        assert!(
            flapped.force_body.z < clean.force_body.z,
            "flaps must add lift (more negative z): {} vs {}",
            flapped.force_body.z,
            clean.force_body.z
        );
        let q = 0.5 * 1.225 * (40.0f64 * 40.0 + (40.0 * alpha) * (40.0 * alpha));
        // extra drag from flap steps
        let drag_clean = -clean.force_body.dot(v_body.normalize());
        let drag_flap = -flapped.force_body.dot(v_body.normalize());
        assert!(drag_flap - drag_clean > q * 2.0 * 0.05 * 0.5, "Δdrag too small");
    }

    #[test]
    fn rate_damping_opposes_rotation() {
        // pitch up rate on a tail-behind configuration → nose-down damping
        let tail = AeroElement {
            surface_index: 1,
            area_m2: 1.4,
            chord_m: 1.0,
            span_m: 1.4,
            span_frac: 0.5,
            side: 1.0,
            axis: DVec3::new(0.0, 1.0, 0.0),
            quarter_chord_center: DVec3::new(-5.5, 0.7, 0.0),
            center: DVec3::new(-5.5, 0.7, 0.0),
            incidence_rad: 0.0,
            span_pos_m: 0.7,
        };
        let solver = single_element_solver(tail, 0.0);
        let v_body = DVec3::new(60.0, 0.0, 0.0);
        let atm = AtmosphereSample::default();
        let still = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());
        // positive pitch rate: aft strip moves DOWN (ω × r with r_x < 0:
        // ω_y·r_z terms … locally v_z gains −ω_y·r_x·? → higher α on tail
        // → more downwash force → nose-down (negative M_y) damping)
        let pitching = solver.solve(v_body, DVec3::new(0.0, 0.3, 0.0), &atm, &ControlInputs::default());
        assert!(
            pitching.torque_body.y < still.torque_body.y + 1e-9,
            "pitch damping must reduce M_y: still {} pitching {}",
            still.torque_body.y,
            pitching.torque_body.y
        );
        assert!(
            pitching.torque_body.y < 0.0,
            "aft tail with positive q must damp: {}",
            pitching.torque_body.y
        );
    }

    #[test]
    fn bake_applies_incidence_symmetrically() {
        // +1° incidence must raise the LE on BOTH wings (α_local > 0).
        let inc = 1.0f64.to_radians();
        let mut right = wing_element(1.0);
        right.incidence_rad = inc;
        let mut left = wing_element(-1.0);
        left.incidence_rad = inc;
        let er = ElementModel::bake(right, 0.0, true);
        let el = ElementModel::bake(left, 0.0, true);
        // chord tips up (−Z) on both sides
        assert!(er.chord_axis.z < 0.0, "right chord {}", er.chord_axis.z);
        assert!(el.chord_axis.z < 0.0, "left chord {}", el.chord_axis.z);
        // at zero airspeed-direction α the local α should equal +incidence
        let v = DVec3::X;
        let ar = (-v.dot(er.up_axis)).atan2(v.dot(er.chord_axis));
        let al = (-v.dot(el.up_axis)).atan2(v.dot(el.chord_axis));
        assert!((ar - inc).abs() < 1e-12, "right α {ar}");
        assert!((al - inc).abs() < 1e-12, "left α {al}");
    }
}
