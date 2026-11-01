//! End-to-end integration tests against the clean-room C172 assets:
//! parked stability, drop settling, airborne dynamics and the 1000 Hz
//! time base.

use glam::DVec3;
use sim_cfg::AircraftDefinition;
use sim_physics::{
    AeroSolver, AtmosphereSample, ControlInputs, PhysicsWorld, RigidBodyState, StepDiagnostics, DT,
    SIM_RATE_HZ,
};

fn reference() -> AircraftDefinition {
    AircraftDefinition::load_reference_c172().expect("clean-room reference cfg must load")
}

fn parked_world(def: &AircraftDefinition, drop_height_m: f64) -> PhysicsWorld {
    let mass = def.flight_model.weight.empty_weight_kg;
    let cg = DVec3::new(def.flight_model.weight.empty_cg_arm_m, 0.0, 0.0);
    // upright attitude: body +Z (down) maps to world-down, so the strut
    // patch (body z = attach 0.4 + strut 0.5 = 0.9) sits 0.9 m BELOW the
    // origin; start `drop_height_m` above the field.
    let state = RigidBodyState {
        orientation: RigidBodyState::upright_attitude(0.0),
        position_m: DVec3::new(0.0, 0.0, 0.9 + drop_height_m),
        ..Default::default()
    };
    PhysicsWorld::new_on_flat_field(def, mass, cg, state, 0.0)
}

#[test]
fn reference_aircraft_loads() {
    let def = reference();
    let mass = def.flight_model.weight.empty_weight_kg;
    assert!(mass > 700.0 && mass < 900.0, "empty mass {mass}");
    let g = &def.flight_model.geometry;
    assert!(g.inertia[0] > 0.0 && g.inertia[1] > 0.0 && g.inertia[2] > 0.0);
    let solver = AeroSolver::from_aircraft(&def, DVec3::ZERO);
    // default resolution: 16×2 wing + 8×2 htail + 8×1 vtail = 56 strips
    assert!(
        solver.element_count() >= 50,
        "strips {}",
        solver.element_count()
    );
    // strip area must cover wing + both tails (authored areas ± rounding)
    let area = solver.total_strip_area_m2();
    let planform = g.wing_area_m2() + g.htail.area_m2 + g.vtail.area_m2;
    assert!(
        (area - planform).abs() / planform < 0.02,
        "strip area {area} vs planform {planform}"
    );
}

#[test]
fn parked_aircraft_settles_on_all_three_gear() {
    let def = reference();
    let mut world = parked_world(&def, 0.05);
    let atm = AtmosphereSample::default();

    let mut last = StepDiagnostics {
        sim_time_s: 0.0,
        steps: 0,
        wheels_in_contact: 0,
        bottomed_out: false,
        kinetic_energy_j: 0.0,
        diverged: false,
    };
    for _ in 0..5000 {
        last = world.step(&atm);
        assert!(!last.diverged, "state diverged");
    }

    assert_eq!(last.steps, 5000);
    assert!((last.sim_time_s - 5000_f64 * DT).abs() < 1e-9);
    assert_eq!(last.wheels_in_contact, 3, "all gear must touch");
    assert!(!last.bottomed_out, "settle must not bottom the struts");
    assert!(
        world.state.velocity_mps.length() < 5e-3,
        "residual velocity {}",
        world.state.velocity_mps
    );
    assert!(
        world.state.omega_body_radps.length() < 1e-2,
        "residual rotation {}",
        world.state.omega_body_radps
    );
    // roughly level on the gear (static pitch from CG/wheelbase geometry)
    let fwd = world.state.body_to_world(DVec3::X);
    assert!(
        fwd.z.abs() < 0.12,
        "pitch attitude too far: fwd.z {}",
        fwd.z
    );
    // resting height: origin ≈ 0.9 m above the patch minus static travel
    assert!(
        (0.5..=0.95).contains(&world.state.position_m.z),
        "rest height {}",
        world.state.position_m.z
    );
}

#[test]
fn gentle_drop_settles_without_bottoming() {
    let def = reference();
    let mut world = parked_world(&def, 0.15);
    let atm = AtmosphereSample::default();
    let mut ever_bottomed = false;
    for _ in 0..8000 {
        let d = world.step(&atm);
        assert!(!d.diverged);
        ever_bottomed |= d.bottomed_out;
    }
    assert!(!ever_bottomed, "0.15 m drop must stay in travel");
    assert_eq!(world.gear.len(), 3);
    for leg in &world.gear {
        assert!(leg.in_contact);
        assert!(leg.compression_m > 0.0);
        assert!(leg.compression_m <= leg.config.max_compression_m);
    }
    assert!(world.state.velocity_mps.length() < 1e-2);
}

#[test]
fn thousand_steps_make_exactly_one_second() {
    let def = reference();
    let mut world = parked_world(&def, 0.0);
    let atm = AtmosphereSample::default();
    for _ in 0..1000 {
        world.step(&atm);
    }
    assert_eq!(SIM_RATE_HZ, 1000);
    assert!(
        (world.sim_time_s() - 1.0).abs() < 1e-9,
        "t = {}",
        world.sim_time_s()
    );
    assert_eq!(world.steps(), 1000);
}

#[test]
fn airborne_cruise_drag_brakes_without_thrust() {
    let def = reference();
    let mass = def.flight_model.weight.empty_weight_kg;
    let cg = DVec3::new(def.flight_model.weight.empty_cg_arm_m, 0.0, 0.0);
    let state = RigidBodyState::level_flight(3_000.0, 0.0, 62.0);
    let mut world = PhysicsWorld::new_on_flat_field(&def, mass, cg, state, 0.0);
    let atm = AtmosphereSample {
        density_kg_m3: 0.909, // ≈ ISA 3 km
        speed_of_sound_mps: 330.0,
    };
    let v0 = world.state.velocity_mps.length();
    let mut contacts = 0;
    for _ in 0..1000 {
        let d = world.step(&atm);
        assert!(!d.diverged);
        contacts += d.wheels_in_contact;
    }
    assert_eq!(contacts, 0, "cruise must stay airborne");
    let v1 = world.state.velocity_mps.length();
    assert!(v1 < v0, "drag must slow the aircraft: {v0} → {v1}");
    // unpowered at ~63 kt: parasite drag plus the trajectory change from
    // a partially trimmed lift vector bleeds a few m/s per second
    assert!(v1 > v0 - 6.0, "one second shouldn't bleed {v0}→{v1}");
    // altitude must not collapse instantly
    assert!(
        world.state.position_m.z > 2_900.0,
        "z = {}",
        world.state.position_m.z
    );
}

#[test]
fn elevator_command_pitch_down_from_level_flight() {
    let def = reference();
    let mass = def.flight_model.weight.empty_weight_kg;
    let cg = DVec3::new(def.flight_model.weight.empty_cg_arm_m, 0.0, 0.0);
    let state = RigidBodyState::level_flight(2_000.0, 0.0, 60.0);
    let mut world = PhysicsWorld::new_on_flat_field(&def, mass, cg, state, 0.0);
    world.controls = ControlInputs {
        elevator_rad: 0.05,
        ..Default::default()
    };
    let atm = AtmosphereSample {
        density_kg_m3: 1.0,
        speed_of_sound_mps: 335.0,
    };
    for _ in 0..500 {
        world.step(&atm);
    }
    // positive elevator (TE down) with negative cm_delta_e pitches down
    assert!(
        world.state.omega_body_radps.y < -0.01,
        "pitch rate {} must be nose-down",
        world.state.omega_body_radps.y
    );
}

#[test]
fn flaps_change_the_aero_wrench_significantly() {
    let def = reference();
    let solver = AeroSolver::from_aircraft(&def, DVec3::ZERO);
    let v_body = DVec3::new(45.0, 0.0, 45.0 * 0.06);
    let atm = AtmosphereSample::default();
    let clean = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());
    let flap2 = ControlInputs {
        flap_step: 2,
        ..ControlInputs::default()
    };
    let flapped = solver.solve(v_body, DVec3::ZERO, &atm, &flap2);
    assert!(
        flapped.force_body.z < clean.force_body.z - 50.0,
        "two flap notches must add substantial lift: {} vs {}",
        flapped.force_body.z,
        clean.force_body.z
    );
    assert!(flapped.torque_body.y.is_finite());
}

#[test]
fn ground_effect_reduces_drag_near_ground() {
    let def = reference();
    let mut solver = AeroSolver::from_aircraft(&def, DVec3::ZERO);
    let v_body = DVec3::new(40.0, 0.0, 4.0); // angle of attack ~0.1 rad
    let atm = AtmosphereSample::default();

    // High altitude (out of ground effect)
    solver.set_height_agl(50.0);
    let out_high = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());

    // Near ground (2 metres above runway)
    solver.set_height_agl(2.0);
    let out_low = solver.solve(v_body, DVec3::ZERO, &atm, &ControlInputs::default());

    let v_hat = v_body.normalize();
    let drag_high = -out_high.force_body.dot(v_hat);
    let drag_low = -out_low.force_body.dot(v_hat);

    assert!(
        drag_low < drag_high,
        "ground effect must reduce total drag near the ground: {drag_low} vs {drag_high}"
    );
}

#[test]
fn propeller_generates_realistic_left_turning_tendencies() {
    let def = reference();
    let mass = def.flight_model.weight.empty_weight_kg;
    let cg = DVec3::new(def.flight_model.weight.empty_cg_arm_m, 0.0, 0.0);
    let state = RigidBodyState::level_flight(1_000.0, 0.0, 35.0);
    let mut world = PhysicsWorld::new_on_flat_field(&def, mass, cg, state, 0.0);

    // Set full throttle prop parameters: 2700 RPM (~283 rad/s), 2200 N thrust, 1.63 kg*m^2 inertia
    world.prop_omega_radps = 282.74;
    world.prop_inertia_kgm2 = 1.63;
    world.prop_thrust_n = 2200.0;
    world.prop_torque_nm = 350.0; // engine torque

    // 1. In level flight with positive thrust:
    let torques = world.propeller_torques();

    // Torque reaction: must roll to the left (-X moment)
    assert!(
        torques.x < 0.0,
        "torque reaction must roll left: {}",
        torques.x
    );

    // Spiraling slipstream: must yaw to the left (-Z moment)
    assert!(
        torques.z < 0.0,
        "spiraling slipstream must yaw left: {}",
        torques.z
    );

    // 2. High angle of attack climb: positive alpha increases P-factor yaw to the left
    let v_climb = DVec3::new(30.0, 0.0, 4.0); // positive alpha (downward relative velocity in body frame)
    world.state.velocity_mps = world.state.body_to_world(v_climb);
    let torques_climb = world.propeller_torques();
    assert!(
        torques_climb.z < torques.z,
        "P-factor under climb angle of attack must increase left yaw: {} vs {}",
        torques_climb.z,
        torques.z
    );

    // 3. Gyroscopic precession: pitching up (q > 0) with CW propeller precesses yaw right (+Z)
    world.state.omega_body_radps = DVec3::new(0.0, 0.5, 0.0);
    let torques_pitching = world.propeller_torques();
    assert!(
        torques_pitching.z > torques_climb.z,
        "pitch-up precession must produce positive Z yaw moment"
    );
}

#[test]
fn propwash_boosts_elevator_authority_at_low_airspeed() {
    let def = reference();
    let solver = AeroSolver::from_aircraft(&def, DVec3::ZERO);
    let v_body = DVec3::new(10.0, 0.0, 0.0); // low speed (e.g. takeoff roll)
    let atm = AtmosphereSample::default();

    let cmd_clean = ControlInputs {
        elevator_rad: 0.1,
        prop_thrust_n: 0.0,
        ..ControlInputs::default()
    };
    let out_clean = solver.solve(v_body, DVec3::ZERO, &atm, &cmd_clean);

    let cmd_thrust = ControlInputs {
        elevator_rad: 0.1,
        prop_thrust_n: 2200.0, // full takeoff thrust
        ..ControlInputs::default()
    };
    let out_thrust = solver.solve(v_body, DVec3::ZERO, &atm, &cmd_thrust);

    // Elevator moment magnitude must be significantly greater with propwash
    assert!(
        out_thrust.torque_body.y.abs() > out_clean.torque_body.y.abs() * 1.5,
        "propwash must augment elevator authority at low speed: {} vs {}",
        out_thrust.torque_body.y,
        out_clean.torque_body.y
    );
}

#[test]
fn adverse_yaw_opposes_roll_command() {
    let def = reference();
    let solver = AeroSolver::from_aircraft(&def, DVec3::ZERO);
    let v_body = DVec3::new(50.0, 0.0, 0.0);
    let atm = AtmosphereSample::default();

    // Right roll command (positive aileron): adverse yaw should create negative Z moment (yaw nose left)
    let cmd = ControlInputs {
        aileron_rad: 0.15,
        ..ControlInputs::default()
    };
    let out = solver.solve(v_body, DVec3::ZERO, &atm, &cmd);

    assert!(out.torque_body.x > 0.0, "aileron must roll right (+X)");
    assert!(
        out.torque_body.z < 0.0,
        "aileron deflection must create adverse yaw to the left (-Z): {}",
        out.torque_body.z
    );
}

#[test]
fn elevator_trim_acts_identically_to_stick_trim() {
    let def = reference();
    let solver = AeroSolver::from_aircraft(&def, DVec3::ZERO);
    let v_body = DVec3::new(50.0, 0.0, 0.0);
    let atm = AtmosphereSample::default();

    let cmd_stick = ControlInputs {
        elevator_rad: 0.05,
        elevator_trim_rad: 0.0,
        ..ControlInputs::default()
    };
    let out_stick = solver.solve(v_body, DVec3::ZERO, &atm, &cmd_stick);

    let cmd_trim = ControlInputs {
        elevator_rad: 0.0,
        elevator_trim_rad: 0.05,
        ..ControlInputs::default()
    };
    let out_trim = solver.solve(v_body, DVec3::ZERO, &atm, &cmd_trim);

    assert!(
        (out_stick.torque_body.y - out_trim.torque_body.y).abs() < 1e-6,
        "trim and stick deflections must produce identical moments"
    );
}
