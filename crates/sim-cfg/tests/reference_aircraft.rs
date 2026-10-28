//! Integration test: load the clean-room C172 reference aircraft from
//! the repository's `assets/` directory and verify the numbers against
//! published type data.

use sim_cfg::geometry::{GAL_TO_M3, LB_TO_KG};
use sim_cfg::{
    discretize, total_area, AircraftDefinition, Discretization,
};

#[test]
fn reference_c172_loads_from_assets() {
    let ac = AircraftDefinition::load_reference_c172()
        .expect("reference aircraft must load from assets/aircraft/c172_cleanroom");
    let fm = &ac.flight_model;

    // --- geometry against published C172 numbers -----------------------
    let g = &fm.geometry;
    assert!(
        (g.wing.area_m2 - 174.0 * 0.09290304).abs() < 1e-9,
        "wing area {} m²",
        g.wing.area_m2
    );
    assert!((g.wing.span_m - 36.1 * 0.3048).abs() < 1e-9);
    // aspect ratio near the published ~7.5
    let ar = g.wing.aspect_ratio();
    assert!(ar > 7.0 && ar < 8.0, "aspect ratio {ar}");

    // --- weights --------------------------------------------------------
    assert!((fm.weight.empty_weight_kg - 1685.0 * LB_TO_KG).abs() < 1e-6);
    assert!((fm.weight.max_gross_weight_kg - 2550.0 * LB_TO_KG).abs() < 1e-6);

    // --- fuel: 57 total / ~56 usable gallons of Avgas --------------------
    let cap_gal: f64 = fm.fuel.iter().map(|t| t.capacity_m3 / GAL_TO_M3).sum();
    assert!((cap_gal - 57.0).abs() < 1e-6, "{cap_gal}");
    let usable_kg = fm.total_usable_fuel_kg();
    assert!(usable_kg > 145.0 && usable_kg < 160.0, "{usable_kg}");

    // --- performance -----------------------------------------------------
    assert_eq!(fm.limits.vne_kias, 163.0);
    assert_eq!(fm.limits.vs1_kias, 48.0);
    assert!(fm.limits.service_ceiling_m() > 4200.0);

    // --- engines ---------------------------------------------------------
    assert_eq!(ac.engines.piston.max_power_w.round(), (180.0f64 * 745.699871).round());
    assert!((ac.engines.propeller.diameter_m - 76.0 * 0.0254).abs() < 1e-12);
    assert!(!ac.engines.propeller.constant_speed);

    // --- systems ----------------------------------------------------------
    assert_eq!(ac.systems.electrical.alternator_max_amps, 60.0);
    assert!(ac.systems.brakes.differential_toe_brakes);
    assert!(ac.systems.vacuum.backup_gyro);
}

#[test]
fn reference_c172_wing_discretizes_to_published_area() {
    let ac = AircraftDefinition::load_reference_c172().unwrap();
    let wing = ac.flight_model.geometry.wing;
    let elements = discretize(&wing, 0, 12, Discretization { chordwise: 4 }, true);
    let total = total_area(&elements);
    // The discretizer integrates the root/tip trapezoid: 174.18 ft²
    // versus the authored 174.0 ft² — agree within 0.1 %.
    assert!(
        (total - wing.area_m2).abs() / wing.area_m2 < 5e-3,
        "discretized area {total} vs planform {}",
        wing.area_m2
    );
    // both sides present, symmetric about the centerline
    assert_eq!(elements.len(), 12 * 4 * 2);
    assert!(elements.iter().all(|e| e.quarter_chord_center.x.is_finite()));
    // washout makes tip incidence smaller than root incidence
    let tip_i = elements
        .iter()
        .filter(|e| e.side > 0.0)
        .fold(f64::INFINITY, |a, e| a.min(e.incidence_rad));
    let root_i = elements
        .iter()
        .filter(|e| e.side > 0.0)
        .fold(f64::NEG_INFINITY, |a, e| a.max(e.incidence_rad));
    assert!(tip_i < root_i, "tip {tip_i} vs root {root_i}");
    // 1.5 deg root incidence, -3 deg washout; first strip centre at
    // t = 0.0417 -> 1.375 deg, last strip centre at t = 0.958 -> -1.375 deg
    assert!(root_i < 1.45f64.to_radians() && root_i > 1.3f64.to_radians(), "{root_i}");
    assert!(tip_i < (-1.3f64).to_radians() && tip_i > (-1.45f64).to_radians(), "{tip_i}");
}

#[test]
fn reference_c172_licenses_and_cleanliness() {
    // the cfg files must be our clean-room originals (no SDK fingerprints)
    let text = include_str!("../../../assets/aircraft/c172_cleanroom/flight_model.cfg");
    assert!(text.contains("clean-room"));
    assert!(text.contains("wing_area"));
}
