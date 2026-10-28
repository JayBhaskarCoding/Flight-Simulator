//! The SimVar bus: typed, unit-aware flight-data variables.
//!
//! Values are stored in SI base units (`f64`, the 64-bit world law) and
//! converted at the API boundary. Variable names are case-insensitive.
//! The [`SimVarBus::with_defaults`] constructor pre-declares a core
//! aviation catalog so external clients can request variables that have
//! not been written yet.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use tokio::sync::broadcast;

/// Physical dimension of a [`Unit`], used for mismatch detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dim {
    Length,
    Velocity,
    Acceleration,
    Angle,
    AngularVelocity,
    Mass,
    Force,
    Pressure,
    Temperature,
    Volume,
    Time,
    Frequency,
    Ratio,
    Dimensionless,
}

impl Dim {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Dim::Length => "length",
            Dim::Velocity => "velocity",
            Dim::Acceleration => "acceleration",
            Dim::Angle => "angle",
            Dim::AngularVelocity => "angular velocity",
            Dim::Mass => "mass",
            Dim::Force => "force",
            Dim::Pressure => "pressure",
            Dim::Temperature => "temperature",
            Dim::Volume => "volume",
            Dim::Time => "time",
            Dim::Frequency => "frequency",
            Dim::Ratio => "ratio",
            Dim::Dimensionless => "dimensionless",
        }
    }
}

/// Units understood at the bus boundary. Conversions target SI bases:
/// metre, metre/second, radian, kilogram, newton, pascal, kelvin,
/// cubic metre, second, hertz, and unitless ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    // length
    Meter,
    Foot,
    NauticalMile,
    // velocity
    MetersPerSecond,
    Knot,
    MilePerHour,
    FeetPerMinute,
    FeetPerSecond,
    // acceleration
    MetersPerSecondSquared,
    GForce,
    // angle
    Radian,
    Degree,
    // angular velocity
    RadiansPerSecond,
    DegreesPerSecond,
    RevolutionsPerMinute,
    // mass / force
    Kilogram,
    Pound,
    Newton,
    PoundForce,
    // pressure
    Pascal,
    Hectopascal,
    InchHg,
    // temperature
    Kelvin,
    Celsius,
    Fahrenheit,
    // volume
    CubicMeter,
    Liter,
    GallonUS,
    // time / frequency
    Second,
    Minute,
    Hour,
    Hertz,
    // ratios
    Ratio,
    Percent,
    /// Pure number (mach, g-load factor counted as ratio …).
    Unitless,
}

const TAU: f64 = std::f64::consts::TAU;
const G0: f64 = 9.806_65;
const LB_TO_KG: f64 = 0.453_592_37;
const LBF_TO_N: f64 = 4.448_221_615_260_5;
const INHG_TO_PA: f64 = 3_386.389;
const GAL_TO_M3: f64 = 0.003_785_411_784;

impl Unit {
    pub(crate) fn dim(self) -> Dim {
        use Dim::*;
        use Unit::*;
        match self {
            Meter | Foot | NauticalMile => Length,
            MetersPerSecond | Knot | MilePerHour | FeetPerMinute | FeetPerSecond => Velocity,
            MetersPerSecondSquared | GForce => Acceleration,
            Radian | Degree => Angle,
            RadiansPerSecond | DegreesPerSecond | RevolutionsPerMinute => AngularVelocity,
            Kilogram | Pound => Mass,
            Newton | PoundForce => Force,
            Pascal | Hectopascal | InchHg => Pressure,
            Kelvin | Celsius | Fahrenheit => Temperature,
            CubicMeter | Liter | GallonUS => Volume,
            Second | Minute | Hour => Time,
            Hertz => Frequency,
            Unit::Ratio | Unit::Percent => Dim::Ratio,
            Unitless => Dimensionless,
        }
    }

    /// Human-readable symbol for diagnostics.
    pub fn label(self) -> &'static str {
        use Unit::*;
        match self {
            Meter => "m",
            Foot => "ft",
            NauticalMile => "nmi",
            MetersPerSecond => "m/s",
            Knot => "kt",
            MilePerHour => "mph",
            FeetPerMinute => "ft/min",
            FeetPerSecond => "ft/s",
            MetersPerSecondSquared => "m/s^2",
            GForce => "g",
            Radian => "rad",
            Degree => "deg",
            RadiansPerSecond => "rad/s",
            DegreesPerSecond => "deg/s",
            RevolutionsPerMinute => "rpm",
            Kilogram => "kg",
            Pound => "lb",
            Newton => "N",
            PoundForce => "lbf",
            Pascal => "Pa",
            Hectopascal => "hPa",
            InchHg => "inHg",
            Kelvin => "K",
            Celsius => "degC",
            Fahrenheit => "degF",
            CubicMeter => "m^3",
            Liter => "L",
            GallonUS => "USgal",
            Second => "s",
            Minute => "min",
            Hour => "h",
            Hertz => "Hz",
            Ratio => "ratio",
            Percent => "%",
            Unitless => "",
        }
    }

    /// Convert `value` expressed in this unit to the SI base of its
    /// dimension.
    pub fn to_si(self, value: f64) -> f64 {
        use Unit::*;
        match self {
            Meter => value,
            Foot => value * 0.3048,
            NauticalMile => value * 1_852.0,
            MetersPerSecond => value,
            Knot => value * 1_852.0 / 3_600.0,
            MilePerHour => value * 0.447_04,
            FeetPerMinute => value * 0.3048 / 60.0,
            FeetPerSecond => value * 0.3048,
            MetersPerSecondSquared => value,
            GForce => value * G0,
            Radian => value,
            Degree => value.to_radians(),
            RadiansPerSecond => value,
            DegreesPerSecond => value.to_radians(),
            RevolutionsPerMinute => value * TAU / 60.0,
            Kilogram => value,
            Pound => value * LB_TO_KG,
            Newton => value,
            PoundForce => value * LBF_TO_N,
            Pascal => value,
            Hectopascal => value * 100.0,
            InchHg => value * INHG_TO_PA,
            Kelvin => value,
            Celsius => value + 273.15,
            Fahrenheit => (value - 32.0) * 5.0 / 9.0 + 273.15,
            CubicMeter => value,
            Liter => value * 1e-3,
            GallonUS => value * GAL_TO_M3,
            Second => value,
            Minute => value * 60.0,
            Hour => value * 3_600.0,
            Hertz => value,
            Ratio => value,
            Percent => value / 100.0,
            Unitless => value,
        }
    }

    /// Convert an SI-base value into this unit.
    pub fn from_si(self, si: f64) -> f64 {
        use Unit::*;
        match self {
            Meter => si,
            Foot => si / 0.3048,
            NauticalMile => si / 1_852.0,
            MetersPerSecond => si,
            Knot => si * 3_600.0 / 1_852.0,
            MilePerHour => si / 0.447_04,
            FeetPerMinute => si * 60.0 / 0.3048,
            FeetPerSecond => si / 0.3048,
            MetersPerSecondSquared => si,
            GForce => si / G0,
            Radian => si,
            Degree => si.to_degrees(),
            RadiansPerSecond => si,
            DegreesPerSecond => si.to_degrees(),
            RevolutionsPerMinute => si * 60.0 / TAU,
            Kilogram => si,
            Pound => si / LB_TO_KG,
            Newton => si,
            PoundForce => si / LBF_TO_N,
            Pascal => si,
            Hectopascal => si / 100.0,
            InchHg => si / INHG_TO_PA,
            Kelvin => si,
            Celsius => si - 273.15,
            Fahrenheit => (si - 273.15) * 9.0 / 5.0 + 32.0,
            CubicMeter => si,
            Liter => si * 1e3,
            GallonUS => si / GAL_TO_M3,
            Second => si,
            Minute => si / 60.0,
            Hour => si / 3_600.0,
            Hertz => si,
            Ratio => si,
            Percent => si * 100.0,
            Unitless => si,
        }
    }
}

/// Errors produced by the bus API.
#[derive(Debug, Clone, PartialEq)]
pub enum SimVarError {
    /// The variable has never been declared or written.
    UnknownVariable(String),
    /// The requested unit's dimension differs from the stored one.
    UnitMismatch {
        name: String,
        expected: &'static str,
        got: &'static str,
        got_unit: &'static str,
    },
}

impl fmt::Display for SimVarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimVarError::UnknownVariable(n) => write!(f, "unknown SimVar `{n}`"),
            SimVarError::UnitMismatch {
                name,
                expected,
                got,
                got_unit,
            } => write!(
                f,
                "SimVar `{name}` is {expected}-valued but was accessed as {got} ({got_unit})"
            ),
        }
    }
}

impl std::error::Error for SimVarError {}

/// Broadcast payload describing one bus write.
#[derive(Debug, Clone)]
pub struct SimVarUpdate {
    /// Canonical (upper-case) variable name.
    pub name: String,
    /// New value in the SI base unit of the variable's dimension.
    pub si_value: f64,
}

struct Entry {
    si: f64,
    dim: Dim,
}

/// Shared, unit-aware store of flight-data variables.
#[derive(Clone)]
pub struct SimVarBus {
    inner: Arc<RwLock<BTreeMap<String, Entry>>>,
    updates: broadcast::Sender<SimVarUpdate>,
}

/// Core catalog: name + canonical declaration unit.
pub const DEFAULT_SIMVARS: &[(&str, Unit)] = &[
    ("INDICATED AIRSPEED", Unit::Knot),
    ("TRUE AIRSPEED", Unit::Knot),
    ("GROUND SPEED", Unit::Knot),
    ("MACH", Unit::Unitless),
    ("ALTITUDE", Unit::Foot),
    ("INDICATED ALTITUDE", Unit::Foot),
    ("HEIGHT ABOVE GROUND", Unit::Foot),
    ("VERTICAL SPEED", Unit::FeetPerMinute),
    ("HEADING DEGREES", Unit::Degree),
    ("PITCH DEGREES", Unit::Degree),
    ("ROLL DEGREES", Unit::Degree),
    ("ANGLE OF ATTACK", Unit::Degree),
    ("G FORCE", Unit::GForce),
    ("POSITION LAT", Unit::Degree),
    ("POSITION LONG", Unit::Degree),
    ("FUEL QUANTITY LEFT", Unit::GallonUS),
    ("FUEL QUANTITY RIGHT", Unit::GallonUS),
    ("GROSS WEIGHT", Unit::Pound),
    ("AIRPLANE MASS", Unit::Kilogram),
    ("AMBIENT TEMPERATURE", Unit::Celsius),
    ("AMBIENT PRESSURE", Unit::Pascal),
    ("BAROMETER SETTING", Unit::InchHg),
    ("ENGINE RPM", Unit::RevolutionsPerMinute),
    ("THROTTLE POSITION", Unit::Ratio),
    ("ELEVATOR POSITION", Unit::Ratio),
    ("AILERON POSITION", Unit::Ratio),
    ("RUDDER POSITION", Unit::Ratio),
    ("FLAP POSITION", Unit::Ratio),
    ("SIMULATION TIME", Unit::Second),
];

fn canon(name: &str) -> String {
    name.trim().to_ascii_uppercase()
}

impl Default for SimVarBus {
    fn default() -> Self {
        Self::new()
    }
}

impl SimVarBus {
    /// Empty bus: only variables that are declared or written exist.
    pub fn new() -> Self {
        let (updates, _) = broadcast::channel(256);
        Self {
            inner: Arc::new(RwLock::new(BTreeMap::new())),
            updates,
        }
    }

    /// Bus pre-declared with [`DEFAULT_SIMVARS`] (all zero).
    pub fn with_defaults() -> Self {
        let bus = Self::new();
        for (name, unit) in DEFAULT_SIMVARS {
            bus.declare(name, *unit).expect("default var declare");
        }
        bus
    }

    /// Declare a variable's dimension without writing a meaningful value
    /// (stores `0.0`). Re-declaring with a different unit is an error if
    /// the variable already exists with another dimension.
    pub fn declare(&self, name: &str, unit: Unit) -> Result<(), SimVarError> {
        let key = canon(name);
        let mut map = self.inner.write().expect("simvar map poisoned");
        match map.get(&key) {
            Some(e) if e.dim != unit.dim() => Err(SimVarError::UnitMismatch {
                name: key,
                expected: e.dim.name(),
                got: unit.dim().name(),
                got_unit: unit.label(),
            }),
            Some(_) => Ok(()),
            None => {
                // "zero" means zero *in the declaration unit* — this
                // matters for scales with an offset (0 degC = 273.15 K).
                map.insert(key, Entry { si: unit.to_si(0.0), dim: unit.dim() });
                Ok(())
            }
        }
    }

    /// Write `value` in `unit`. The variable is created on first write.
    pub fn set(&self, name: &str, value: f64, unit: Unit) -> Result<(), SimVarError> {
        let key = canon(name);
        let si = unit.to_si(value);
        {
            let mut map = self.inner.write().expect("simvar map poisoned");
            match map.get_mut(&key) {
                Some(e) => {
                    if e.dim != unit.dim() {
                        return Err(SimVarError::UnitMismatch {
                            name: key,
                            expected: e.dim.name(),
                            got: unit.dim().name(),
                            got_unit: unit.label(),
                        });
                    }
                    e.si = si;
                }
                None => {
                    map.insert(key.clone(), Entry { si, dim: unit.dim() });
                }
            }
        }
        // Subscribers never observe a half-applied write; release the map
        // before broadcasting so a slow listener cannot block writers.
        let _ = self.updates.send(SimVarUpdate {
            name: key,
            si_value: si,
        });
        Ok(())
    }

    /// Read `name` converted into `unit`.
    pub fn get(&self, name: &str, unit: Unit) -> Result<f64, SimVarError> {
        let key = canon(name);
        let map = self.inner.read().expect("simvar map poisoned");
        let e = map
            .get(&key)
            .ok_or_else(|| SimVarError::UnknownVariable(key.clone()))?;
        if e.dim != unit.dim() {
            return Err(SimVarError::UnitMismatch {
                name: key,
                expected: e.dim.name(),
                got: unit.dim().name(),
                got_unit: unit.label(),
            });
        }
        Ok(unit.from_si(e.si))
    }

    /// Read the raw SI value without conversion.
    pub fn get_si(&self, name: &str) -> Result<f64, SimVarError> {
        let key = canon(name);
        let map = self.inner.read().expect("simvar map poisoned");
        map.get(&key)
            .map(|e| e.si)
            .ok_or(SimVarError::UnknownVariable(key))
    }

    /// True when the variable exists (declared or written).
    pub fn contains(&self, name: &str) -> bool {
        let key = canon(name);
        self.inner.read().expect("simvar map poisoned").contains_key(&key)
    }

    /// SI snapshot of every variable (name -> value), for server replies.
    pub fn snapshot_si(&self) -> BTreeMap<String, f64> {
        self.inner
            .read()
            .expect("simvar map poisoned")
            .iter()
            .map(|(k, e)| (k.clone(), e.si))
            .collect()
    }

    /// Subscribe to change notifications (SI values).
    pub fn subscribe(&self) -> broadcast::Receiver<SimVarUpdate> {
        self.updates.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declare_and_default_catalog() {
        let bus = SimVarBus::with_defaults();
        assert!(bus.contains("indicated airspeed")); // case-insensitive
        assert!(bus.contains("HEADING DEGREES"));
        assert_eq!(bus.get("INDICATED AIRSPEED", Unit::Knot).unwrap(), 0.0);
        // every default reads back in its own unit
        for (name, unit) in DEFAULT_SIMVARS {
            assert_eq!(bus.get(name, *unit).unwrap(), 0.0, "{name}");
        }
    }

    #[test]
    fn unit_round_trip_across_dims() {
        let bus = SimVarBus::new();
        bus.set("IAS", 120.0, Unit::Knot).unwrap();
        let mps = bus.get("IAS", Unit::MetersPerSecond).unwrap();
        assert!((mps - 120.0 * 1852.0 / 3600.0).abs() < 1e-9);
        assert!((bus.get("IAS", Unit::Knot).unwrap() - 120.0).abs() < 1e-9);

        bus.set("ALT", 10_000.0, Unit::Foot).unwrap();
        let m = bus.get("ALT", Unit::Meter).unwrap();
        assert!((m - 10_000.0 * 0.3048).abs() < 1e-6);

        bus.set("OAT", 15.0, Unit::Celsius).unwrap();
        assert!((bus.get("OAT", Unit::Kelvin).unwrap() - 288.15).abs() < 1e-9);
        assert!((bus.get("OAT", Unit::Fahrenheit).unwrap() - 59.0).abs() < 1e-9);
    }

    #[test]
    fn rpm_angular_velocity_conversion() {
        let bus = SimVarBus::new();
        bus.set("RPM", 2700.0, Unit::RevolutionsPerMinute).unwrap();
        let rad_s = bus.get("RPM", Unit::RadiansPerSecond).unwrap();
        assert!((rad_s - 2700.0 * TAU / 60.0).abs() < 1e-9);
    }

    #[test]
    fn unit_mismatch_is_rejected() {
        let bus = SimVarBus::new();
        bus.set("ALT", 1000.0, Unit::Foot).unwrap();
        let err = bus.get("ALT", Unit::Knot).unwrap_err();
        assert!(matches!(err, SimVarError::UnitMismatch { .. }));
        // writing with the wrong unit also fails and does not clobber
        assert!(bus.set("ALT", 1.0, Unit::Celsius).is_err());
        assert!((bus.get("ALT", Unit::Foot).unwrap() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn declare_conflict_rejected() {
        let bus = SimVarBus::new();
        bus.declare("SPEED", Unit::Knot).unwrap();
        assert!(bus.declare("SPEED", Unit::Foot).is_err());
        // same dimension, different unit of same family: still an error
        // because declare pins the canonical unit by dimension only —
        // velocity vs velocity is fine:
        bus.declare("SPEED2", Unit::Knot).unwrap();
        bus.declare("SPEED2", Unit::MetersPerSecond).unwrap();
    }

    #[test]
    fn unknown_variable_error() {
        let bus = SimVarBus::new();
        let err = bus.get("NOPE", Unit::Knot).unwrap_err();
        assert_eq!(err, SimVarError::UnknownVariable("NOPE".into()));
        assert_eq!(err.to_string(), "unknown SimVar `NOPE`");
    }

    #[test]
    fn change_notifications_carry_si_values() {
        let bus = SimVarBus::new();
        let mut rx = bus.subscribe();
        bus.set("IAS", 100.0, Unit::Knot).unwrap();
        let upd = rx.try_recv().unwrap();
        assert_eq!(upd.name, "IAS");
        assert!((upd.si_value - 100.0 * 1852.0 / 3600.0).abs() < 1e-9);
    }

    #[test]
    fn snapshot_contains_all_si_values() {
        let bus = SimVarBus::with_defaults();
        bus.set("ALTITUDE", 3300.0, Unit::Foot).unwrap();
        let snap = bus.snapshot_si();
        assert_eq!(snap.len(), DEFAULT_SIMVARS.len());
        assert!((snap["ALTITUDE"] - 3300.0 * 0.3048).abs() < 1e-9);
    }
}
