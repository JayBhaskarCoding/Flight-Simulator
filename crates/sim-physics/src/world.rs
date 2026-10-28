//! The fixed-rate (1000 Hz) simulation world tying rigid body, aero and
//! gear together with gravity and externally injected forces.

use glam::DVec3;
use sim_cfg::AircraftDefinition;

use crate::aero::{AeroSolver, AtmosphereSample, ControlInputs};
use crate::gear::{solve_leg, GearAxle, GearContact, GearLeg, GearLegConfig, GroundProvider, PlaneGround};
use crate::rigid_body::{ForceTorque, RigidBody, RigidBodyState};
use crate::G0;

/// Physics integration rate, Hz.
pub const SIM_RATE_HZ: u32 = 1000;
/// Fixed integration step, s.
pub const DT: f64 = 1.0 / SIM_RATE_HZ as f64;

/// Per-step diagnostics returned by [`PhysicsWorld::step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepDiagnostics {
    /// Simulated time after the step, s.
    pub sim_time_s: f64,
    /// Number of completed steps.
    pub steps: u64,
    /// Wheels currently touching the ground.
    pub wheels_in_contact: usize,
    /// True when any leg exceeded its design travel.
    pub bottomed_out: bool,
    /// Translational + rotational kinetic energy, J.
    pub kinetic_energy_j: f64,
    /// True when the state became non-finite and the step was reverted.
    pub diverged: bool,
}

/// One aircraft's complete physics simulation on a flat ENU field.
pub struct PhysicsWorld {
    /// Mass properties.
    pub body: RigidBody,
    /// Dynamic state.
    pub state: RigidBodyState,
    /// Surface-element aerodynamics.
    pub aero: AeroSolver,
    /// Landing-gear legs (runtime contact bookkeeping).
    pub gear: Vec<GearLeg>,
    /// Terrain height model.
    pub ground: Box<dyn GroundProvider>,
    /// Current pilot/autopilot inputs (held for the whole step).
    pub controls: ControlInputs,
    /// Gravity vector, world frame (default `(0, 0, -G0)`).
    pub gravity_mps2: DVec3,
    /// Externally applied force (thrust from later crates), world frame.
    pub external_force_world: DVec3,
    /// Externally applied moment about the CG, body frame.
    pub external_torque_body: DVec3,
    sim_time_s: f64,
    steps: u64,
}

impl PhysicsWorld {
    /// Build a world from an aircraft definition, gear layout, initial
    /// state and ground model.
    ///
    /// `cg_body` is the moment reference in body coordinates (for the
    /// clean-room assets: `(empty_cg_arm_m, 0, 0)`).
    pub fn new(
        def: &AircraftDefinition,
        mass_kg: f64,
        cg_body: DVec3,
        gear: Vec<GearLegConfig>,
        ground: Box<dyn GroundProvider>,
        state: RigidBodyState,
    ) -> Self {
        let inertia = def.flight_model.geometry.inertia;
        let body = RigidBody::from_diagonal(mass_kg, DVec3::from_array(inertia));
        Self {
            body,
            state,
            aero: AeroSolver::from_aircraft(def, cg_body),
            gear: gear.into_iter().map(GearLeg::new).collect(),
            ground,
            controls: ControlInputs::default(),
            gravity_mps2: DVec3::new(0.0, 0.0, -G0),
            external_force_world: DVec3::ZERO,
            external_torque_body: DVec3::ZERO,
            sim_time_s: 0.0,
            steps: 0,
        }
    }

    /// Convenience constructor on a flat field with a standard tricycle
    /// gear layout derived from the geometry (wheelbase/track); the
    /// longitudinal positions are clean-room estimates pinned by the
    /// integration tests.
    pub fn new_on_flat_field(
        def: &AircraftDefinition,
        mass_kg: f64,
        cg_body: DVec3,
        state: RigidBodyState,
        field_altitude_m: f64,
    ) -> Self {
        let gear = standard_tricycle_gear(def);
        Self::new(
            def,
            mass_kg,
            cg_body,
            gear,
            Box::new(PlaneGround::new(field_altitude_m)),
            state,
        )
    }

    /// Simulated time, s.
    pub fn sim_time_s(&self) -> f64 {
        self.sim_time_s
    }

    /// Completed steps.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// One fixed 1 ms RK4 step with zero-order-held controls/environment.
    ///
    /// All four Runge–Kutta stages evaluate gravity + aero + gear with a
    /// scratch copy of each leg (pure evaluation); the real leg bookkeeping
    /// is refreshed once at the end of the step on the final state. If the
    /// resulting state is non-finite the step is reverted and
    /// [`StepDiagnostics::diverged`] is set.
    pub fn step(&mut self, atm: &AtmosphereSample) -> StepDiagnostics {
        let before = self.state;
        let controls = self.controls;
        let gravity = self.gravity_mps2;
        let external = ForceTorque {
            force_world: self.external_force_world,
            torque_body: self.external_torque_body,
        };
        let mass_kg = self.body.mass_kg;

        let next = {
            let stage = |s: &RigidBodyState| -> ForceTorque {
                let v_body = s.world_to_body(s.velocity_mps);
                let aero_out = self.aero.solve(v_body, s.omega_body_radps, atm, &controls);
                let mut force_world = s.body_to_world(aero_out.force_body);
                let mut torque_body = aero_out.torque_body;
                for leg in &self.gear {
                    let mut scratch = *leg;
                    let contact =
                        solve_leg(&mut scratch, s, self.ground.as_ref(), &controls);
                    force_world += contact.force_world;
                    torque_body += contact.torque_body;
                }
                force_world += gravity * mass_kg;
                force_world += external.force_world;
                torque_body += external.torque_body;
                ForceTorque { force_world, torque_body }
            };
            self.body.rk4_step(&self.state, DT, stage)
        };

        if !next.is_finite() {
            self.state = before;
            self.steps += 1;
            self.sim_time_s += DT;
            return self.diagnostics(true);
        }
        self.state = next;
        self.steps += 1;
        self.sim_time_s += DT;

        // refresh leg runtime state on the final configuration
        for leg in &mut self.gear {
            let _ = solve_leg(leg, &self.state, self.ground.as_ref(), &controls);
        }
        self.diagnostics(false)
    }

    /// Run `n` steps, invoking `per_step` (control updates, thrust hooks,
    /// atmosphere sampling) before every step.
    pub fn step_n(
        &mut self,
        n: u64,
        mut per_step: impl FnMut(&mut Self),
        atm: &AtmosphereSample,
    ) -> StepDiagnostics {
        let mut last = self.diagnostics(false);
        for _ in 0..n {
            per_step(self);
            last = self.step(atm);
        }
        last
    }

    fn diagnostics(&self, diverged: bool) -> StepDiagnostics {
        let mut wheels_in_contact = 0;
        let mut bottomed_out = false;
        for leg in &self.gear {
            if leg.in_contact {
                wheels_in_contact += 1;
            }
            if leg.compression_m > leg.config.max_compression_m {
                bottomed_out = true;
            }
        }
        StepDiagnostics {
            sim_time_s: self.sim_time_s,
            steps: self.steps,
            wheels_in_contact,
            bottomed_out,
            kinetic_energy_j: self
                .state
                .kinetic_energy(self.body.mass_kg, self.body.inertia_body),
            diverged,
        }
    }

    /// Solve gear contacts for the *current* state without integrating
    /// (HUD readouts and tests); updates the runtime leg bookkeeping.
    pub fn gear_contacts(&mut self) -> Vec<GearContact> {
        let controls = self.controls;
        let mut out = Vec::with_capacity(self.gear.len());
        for leg in &mut self.gear {
            out.push(solve_leg(leg, &self.state, self.ground.as_ref(), &controls));
        }
        out
    }
}

/// Standard clean-room tricycle gear layout derived from the geometry.
///
/// The assets publish wheelbase and track but not per-gear XYZ stations,
/// so the longitudinal split is a documented estimate: the nose wheel sits
/// `wheelbase·0.35` forward of the CG reference and the mains
/// `wheelbase·0.65` aft, spaced `track` apart; struts point along body
/// `+Z` (down). Spring rates scale with empty weight so the static
/// compression stays within the design travel for the reference aircraft.
pub fn standard_tricycle_gear(def: &AircraftDefinition) -> Vec<GearLegConfig> {
    let g = &def.flight_model.geometry;
    let mass = def.flight_model.weight.empty_weight_kg;
    let weight = mass * G0;
    let wheelbase = g.wheelbase_m.max(1.0);
    let track = g.track_m.max(1.0);

    let nose_x = wheelbase * 0.35;
    let main_x = -wheelbase * 0.65;
    // target static compression ≈ 40 % of the 0.30 m design travel for a
    // full-gear corner load of weight/3 → k = N / (0.4·travel)
    let spring = (weight / 3.0) / (0.4_f64 * 0.30_f64);
    let damper = 2.0 * (spring * 600.0).sqrt(); // ζ ≈ 1 over ~600 kg corners
    let strut = 0.5f64;
    let max_travel = 0.30f64;
    let wheel_r = 0.30f64;

    let mk = |x: f64, y: f64, axle: GearAxle| {
        GearLegConfig::tricycle(
            DVec3::new(x, y, 0.4),
            strut,
            spring,
            damper,
            max_travel,
            wheel_r,
            axle,
        )
    };
    vec![
        mk(nose_x, 0.0, GearAxle::Nose),
        mk(main_x, -track / 2.0, GearAxle::Main),
        mk(main_x, track / 2.0, GearAxle::Main),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_constants() {
        assert_eq!(SIM_RATE_HZ, 1000);
        assert!((DT - 0.001).abs() < 1e-15);
        assert!((DT * 1000.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn static_equilibrium_step_has_no_acceleration() {
        let def = AircraftDefinition::load_reference_c172().expect("reference cfg");
        let mass = def.flight_model.weight.empty_weight_kg;
        let k = 200_000.0f64;
        // leg directly under the CG: the contact force produces no moment,
        // so a perfect force balance must leave the body at rest.
        let cfg = GearLegConfig::tricycle(
            DVec3::new(0.0, 0.0, 0.5),
            0.5,
            k,
            6_000.0,
            0.3,
            0.3,
            GearAxle::Main,
        );
        // patch_body.z = 1.0; equilibrium compression = mg/k
        let depth = mass * G0 / k;
        let state = RigidBodyState {
            position_m: DVec3::new(0.0, 0.0, -depth - 1.0),
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(
            &def,
            mass,
            DVec3::ZERO,
            vec![cfg],
            Box::new(PlaneGround::new(0.0)),
            state,
        );
        let atm = AtmosphereSample::default();
        let diag = world.step(&atm);
        assert!(!diag.diverged);
        assert_eq!(diag.wheels_in_contact, 1);
        // net force zero at equilibrium → velocity stays ~0 and the body
        // does not rotate (no moment arm)
        assert!(
            world.state.velocity_mps.length() < 1e-9,
            "v = {}",
            world.state.velocity_mps
        );
        assert!(world.state.omega_body_radps.length() < 1e-9);
        assert!((world.sim_time_s() - DT).abs() < 1e-15);
        assert_eq!(world.steps(), 1);
    }

    #[test]
    fn external_force_accelerates_mass() {
        let def = AircraftDefinition::load_reference_c172().expect("reference cfg");
        let mass = def.flight_model.weight.empty_weight_kg;
        let state = RigidBodyState::level_flight(5_000.0, 0.0, 50.0);
        let mut world = PhysicsWorld::new(
            &def,
            mass,
            DVec3::ZERO,
            vec![],
            Box::new(PlaneGround::new(0.0)),
            state,
        );
        // cancel gravity so only thrust remains
        world.gravity_mps2 = DVec3::ZERO;
        world.external_force_world = DVec3::new(mass * 2.0, 0.0, 0.0); // 2 m/s²
        let atm = AtmosphereSample::default();
        let v0 = world.state.velocity_mps;
        for _ in 0..1000 {
            world.step(&atm);
        }
        // aero drag also acts; with 50 m/s initial the drag is significant,
        // so assert only that the speed INCREASES vs pure-deceleration case
        let dv = world.state.velocity_mps - v0;
        assert!(dv.x > 0.5, "thrust must accelerate: Δv = {dv}");
    }

    #[test]
    fn standard_gear_layout_is_symmetric_and_sane() {
        let def = AircraftDefinition::load_reference_c172().expect("reference cfg");
        let gear = standard_tricycle_gear(&def);
        assert_eq!(gear.len(), 3);
        assert!(gear[0].steerable && !gear[0].braked);
        assert!(!gear[1].steerable && gear[1].braked);
        assert_eq!(gear[1].attach_body.y, -gear[2].attach_body.y);
        assert!(gear[0].attach_body.x > 0.0 && gear[1].attach_body.x < 0.0);
        // static compression target: weight/3 per leg within travel
        let mass = def.flight_model.weight.empty_weight_kg;
        let static_c = (mass * G0 / 3.0) / gear[1].spring_n_per_m;
        assert!(static_c < gear[1].max_compression_m, "{static_c}");
    }
}
