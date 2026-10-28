//! `systems.cfg` typed loader: electrical, brakes, flight controls,
//! vacuum.

use crate::error::{CfgError, CfgResult};
use crate::parser::CfgFile;

/// Electrical power system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectricalSystem {
    /// Battery open-circuit voltage, V.
    pub battery_voltage_v: f64,
    /// Battery capacity, A·h.
    pub battery_capacity_ah: f64,
    /// Alternator regulated voltage, V.
    pub alternator_voltage_v: f64,
    /// Alternator maximum continuous current, A.
    pub alternator_max_amps: f64,
    /// Essential-bus load at cruise (avionics + lights), A.
    pub cruise_bus_load_amps: f64,
}

impl ElectricalSystem {
    /// Battery stored energy, joules.
    pub fn battery_energy_j(&self) -> f64 {
        self.battery_voltage_v * self.battery_capacity_ah * 3600.0
    }

    /// Seconds until a dead battery is depleted by the cruise bus load.
    pub fn battery_endurance_s(&self, alternator_failed: bool) -> f64 {
        if !alternator_failed {
            return f64::INFINITY;
        }
        let amps = self.cruise_bus_load_amps.max(0.1);
        self.battery_capacity_ah * 3600.0 / amps
    }

    /// Bus voltage model: alternator when running, else battery under
    /// load (simple sag proportional to load current).
    pub fn bus_voltage(&self, alternator_running: bool, load_amps: f64) -> f64 {
        if alternator_running {
            let rated = self.alternator_max_amps.max(1.0);
            self.alternator_voltage_v - (load_amps / rated).clamp(0.0, 2.0) * 1.5
        } else {
            let sag = load_amps.max(0.0) * 0.05;
            (self.battery_voltage_v - sag).max(0.0)
        }
    }
}

/// Brake system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brakes {
    /// Maximum hydraulic line pressure, psi.
    pub max_pressure_psi: f64,
    /// Whether toe brakes are differential (independent left/right).
    pub differential_toe_brakes: bool,
    /// Parking brake available.
    pub parking_brake: bool,
}

/// Manual flight-control system limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightControls {
    /// Maximum elevator deflection, degrees.
    pub elevator_max_deflection_deg: f64,
    /// Maximum aileron deflection, degrees.
    pub aileron_max_deflection_deg: f64,
    /// Maximum rudder deflection, degrees.
    pub rudder_max_deflection_deg: f64,
    /// Control surface full-deflection travel time, seconds
    /// (first-order actuation rate used by the input mixer).
    pub control_lag_s: f64,
    /// Speed at which control authority is considered fully effective
    /// (below it authority scales with dynamic pressure), knots.
    pub control_reference_kias: f64,
}

impl FlightControls {
    /// Deflection in radians.
    pub fn elevator_max_rad(&self) -> f64 {
        self.elevator_max_deflection_deg.to_radians()
    }
    /// Aileron deflection in radians.
    pub fn aileron_max_rad(&self) -> f64 {
        self.aileron_max_deflection_deg.to_radians()
    }
    /// Rudder deflection in radians.
    pub fn rudder_max_rad(&self) -> f64 {
        self.rudder_max_deflection_deg.to_radians()
    }
}

/// Engine-driven vacuum system for the pitot-static/gyro instruments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VacuumSystem {
    /// Regulator set point, inHg.
    pub regulated_inhg: f64,
    /// Minimum acceptable instrument vacuum, inHg.
    pub minimum_inhg: f64,
    /// Whether a backup electric attitude indicator exists.
    pub backup_gyro: bool,
}

/// Complete `systems.cfg` document.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemsDefinition {
    /// Electrical system.
    pub electrical: ElectricalSystem,
    /// Brakes.
    pub brakes: Brakes,
    /// Flight controls.
    pub controls: FlightControls,
    /// Vacuum system.
    pub vacuum: VacuumSystem,
}

impl SystemsDefinition {
    /// Load from a parsed `systems.cfg`.
    pub fn from_cfg(cfg: &CfgFile) -> CfgResult<Self> {
        let el = cfg.require_section("ELECTRICAL")?;
        let electrical = ElectricalSystem {
            battery_voltage_v: el.f64_in("battery_voltage_v", 6.0, 32.0)?,
            battery_capacity_ah: el.f64_in("battery_capacity_ah", 1.0, 400.0)?,
            alternator_voltage_v: el.f64_in("alternator_voltage_v", 6.0, 36.0)?,
            alternator_max_amps: el.f64_in("alternator_max_amps", 1.0, 600.0)?,
            cruise_bus_load_amps: el.f64_in("cruise_bus_load_amps", 0.0, 300.0)?,
        };

        let br = cfg.require_section("BRAKES")?;
        let brakes = Brakes {
            max_pressure_psi: br.f64_in("max_pressure_psi", 100.0, 5000.0)?,
            differential_toe_brakes: br.bool_or("differential_toe_brakes", true)?,
            parking_brake: br.bool_or("parking_brake", true)?,
        };

        let fc = cfg.require_section("FLIGHT_CONTROLS")?;
        let controls = FlightControls {
            elevator_max_deflection_deg: fc.f64_in("elevator_max_deflection_deg", 5.0, 45.0)?,
            aileron_max_deflection_deg: fc.f64_in("aileron_max_deflection_deg", 5.0, 45.0)?,
            rudder_max_deflection_deg: fc.f64_in("rudder_max_deflection_deg", 5.0, 45.0)?,
            control_lag_s: fc.f64_in("control_lag_s", 0.0, 3.0)?,
            control_reference_kias: fc.f64_in("control_reference_kias", 20.0, 600.0)?,
        };

        let va = cfg.require_section("VACUUM")?;
        let vacuum = VacuumSystem {
            regulated_inhg: va.f64_in("regulated_inhg", 2.0, 8.0)?,
            minimum_inhg: va.f64_in("minimum_inhg", 2.0, 8.0)?,
            backup_gyro: va.bool_or("backup_gyro", false)?,
        };
        if vacuum.minimum_inhg > vacuum.regulated_inhg {
            return Err(CfgError::Range {
                file: va.file.clone(),
                line: va.find("minimum_inhg").map(|e| e.line).unwrap_or(1),
                key: "minimum_inhg".into(),
                message: "minimum vacuum cannot exceed the regulator set point".into(),
            });
        }

        Ok(Self { electrical, brakes, controls, vacuum })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYS: &str = r#"
    [ELECTRICAL]
    battery_voltage_v = 24
    battery_capacity_ah = 13.5
    alternator_voltage_v = 28
    alternator_max_amps = 60
    cruise_bus_load_amps = 18

    [BRAKES]
    max_pressure_psi = 900
    differential_toe_brakes = 1
    parking_brake = true

    [FLIGHT_CONTROLS]
    elevator_max_deflection_deg = 22
    aileron_max_deflection_deg = 18
    rudder_max_deflection_deg = 24
    control_lag_s = 0.25
    control_reference_kias = 75

    [VACUUM]
    regulated_inhg = 5.2
    minimum_inhg = 3.8
    backup_gyro = 1
    "#;

    fn systems() -> SystemsDefinition {
        SystemsDefinition::from_cfg(&CfgFile::parse(SYS, "s.cfg").unwrap()).unwrap()
    }

    #[test]
    fn loads_all_subsystems() {
        let s = systems();
        assert_eq!(s.electrical.battery_voltage_v, 24.0);
        assert!(s.brakes.differential_toe_brakes);
        assert!(s.brakes.parking_brake);
        assert!((s.controls.elevator_max_rad() - 22.0f64.to_radians()).abs() < 1e-12);
        assert!(s.vacuum.backup_gyro);
    }

    #[test]
    fn battery_math_is_physical() {
        let s = systems();
        let e = s.electrical.battery_energy_j();
        assert!((e - 24.0 * 13.5 * 3600.0).abs() < 1e-9);
        // alternator running: no battery drain
        assert_eq!(s.electrical.battery_endurance_s(false), f64::INFINITY);
        // failed alternator: 13.5 Ah / 18 A ≈ 2700 s
        let t = s.electrical.battery_endurance_s(true);
        assert!((t - 13.5 * 3600.0 / 18.0).abs() < 1.0, "{t}");
        // bus voltages
        let v_run = s.electrical.bus_voltage(true, 18.0);
        assert!(v_run > 26.0 && v_run <= 28.0, "{v_run}");
        let v_off = s.electrical.bus_voltage(false, 18.0);
        assert!(v_off < 24.0 && v_off > 20.0, "{v_off}");
    }

    #[test]
    fn vacuum_sanity_enforced() {
        let bad = SYS.replace("minimum_inhg = 3.8", "minimum_inhg = 6.5");
        let cfg = CfgFile::parse(&bad, "s.cfg").unwrap();
        let e = SystemsDefinition::from_cfg(&cfg).unwrap_err();
        assert!(e.to_string().contains("vacuum"), "{e}");
    }

    #[test]
    fn missing_section_reported() {
        let cfg = CfgFile::parse(
            "[ELECTRICAL]\nbattery_voltage_v = 24\nbattery_capacity_ah = 10\n\
             alternator_voltage_v = 28\nalternator_max_amps = 60\ncruise_bus_load_amps = 10\n",
            "s.cfg",
        )
        .unwrap();
        let e = SystemsDefinition::from_cfg(&cfg).unwrap_err();
        assert!(e.to_string().contains("[BRAKES]"), "{e}");
    }
}
