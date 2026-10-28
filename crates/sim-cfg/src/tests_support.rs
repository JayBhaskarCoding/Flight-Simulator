//! Shared cfg fixtures for in-crate tests.

#[allow(dead_code)]
pub const GEOMETRY_SECTION: &str = r#"
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
htail_quarter_chord_sweep = 0
htail_dihedral = 0
htail_incidence = 0
htail_position = -15.0, 0, 0.8
vtail_area = 11.6
vtail_span = 5.5
vtail_root_chord = 3.0
vtail_tip_chord = 1.4
vtail_quarter_chord_sweep = 20
vtail_dihedral = 0
vtail_incidence = 0
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
