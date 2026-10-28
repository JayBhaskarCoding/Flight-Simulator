//! End-to-end integration tests against the clean-room C172 assets:
//! parked stability, drop settling, airborne dynamics and the 1000 Hz
//! time base.

use glam::DVec3;
use sim_cfg::AircraftDefinition;
use sim_physics::{
    AtmosphereSample, AeroSolver, ControlInputs, PhysicsWorld, RigidBodyState, StepDiagnostics,
    DT, SIM_RATE_HZ,
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
    assert!(solver.element_count() >= 50, "strips {}", solver.element_count());
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
    assert!(fwd.z.abs() < 0.12, "pitch attitude too far: fwd.z {}", fwd.z);
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
    assert!((world.sim_time_s() - 1.0).abs() < 1e-9, "t = {}", world.sim_time_s());
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
    assert!(world.state.position_m.z > 2_900.0, "z = {}", world.state.position_m.z);
}

#[test]
fn elevator_command_pitch_down_from_level_flight() {
    let def = reference();
    let mass = def.flight_model.weight.empty_weight_kg;
    let cg = DVec3::new(def.flight_model.weight.empty_cg_arm_m, 0.0, 0.0);
    let state = RigidBodyState::level_flight(2_000.0, 0.0, 60.0);
    let mut world = PhysicsWorld::new_on_flat_field(&def, mass, cg, state, 0.0);
    world.controls = ControlInputs { elevator_rad: 0.05, ..Default::default() };
    let atm = AtmosphereSample { density_kg_m3: 1.0, speed_of_sound_mps: 335.0 };
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
    let flap2 = ControlInputs { flap_step: 2, ..ControlInputs::default() };
    let flapped = solver.solve(v_body, DVec3::ZERO, &atm, &flap2);
    assert!(
        flapped.force_body.z < clean.force_body.z - 50.0,
        "two flap notches must add substantial lift: {} vs {}",
        flapped.force_body.z,
        clean.force_body.z
    );
    assert!(flapped.torque_body.y.is_finite());
}

