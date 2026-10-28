//! # sim-cfg
//!
//! Clean-room parser for Microsoft-Flight-Simulator-style aircraft
//! configuration files (`flight_model.cfg`, `engines.cfg`,
//! `systems.cfg`) plus a 3D **AeroElement** surface discretizer that
//! turns wing/tail geometry into the spanwise/chordwise element grid
//! consumed by the 6-DOF solver in `sim-physics`.
//!
//! The parser implements the publicly documented MSFS config dialect:
//!
//! * `[SECTION]` and `[SECTION.variant]` headers (case-insensitive),
//! * `key = value` entries with optional `[index]` subscripts,
//! * `//`, `;` and `#` comments (leading or trailing),
//! * scalar numbers, comma-separated numeric vectors and quoted strings,
//! * `true/false/yes/no/on/off` boolean words.
//!
//! Nothing here derives from decompiled or proprietary sources: the
//! dialect is re-implemented from the public MSFS SDK documentation and
//! the files under [`crate::assets`] are clean-room reference data built
//! from published type-certificate numbers for the C172 airframe.
//!
//! Every parse error carries the source line, so tooling can point at
//! exactly what went wrong.

pub mod aero_element;
pub mod engines;
pub mod error;
pub mod flight_model;
pub mod geometry;
pub mod parser;
pub mod systems;
#[cfg(test)]
mod tests_support;
pub mod value;

pub use aero_element::{area_centroid, discretize, total_area, AeroElement, Discretization};
pub use engines::{EngineDefinition, PistonEngine, Propeller};
pub use error::{CfgError, CfgResult};
pub use flight_model::{
    AeroCoefficients, FlightModelDocument, FuelTank, PerformanceLimits, WeightAndBalance,
};
pub use geometry::{AirplaneGeometry, TailSurface, WingGeometry};
pub use parser::{CfgFile, Entry, Section};
pub use systems::{Brakes, ElectricalSystem, FlightControls, SystemsDefinition, VacuumSystem};
pub use value::Value;

use std::path::Path;

/// The clean-room reference aircraft shipped with the repository.
pub const REFERENCE_AIRCRAFT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/aircraft/c172_cleanroom");

/// A fully loaded aircraft definition (all three cfg files).
#[derive(Debug, Clone)]
pub struct AircraftDefinition {
    /// Parsed `flight_model.cfg` (geometry, weights, fuel, flight model).
    pub flight_model: flight_model::FlightModelDocument,
    /// Parsed `engines.cfg`.
    pub engines: EngineDefinition,
    /// Parsed `systems.cfg`.
    pub systems: SystemsDefinition,
}

impl AircraftDefinition {
    /// Load the three cfg files from an aircraft directory.
    pub fn load_dir(dir: impl AsRef<Path>) -> CfgResult<Self> {
        let dir = dir.as_ref();
        let fm_text = std::fs::read_to_string(dir.join("flight_model.cfg")).map_err(|e| {
            CfgError::Io { path: dir.join("flight_model.cfg").display().to_string(), source: e }
        })?;
        let eng_text = std::fs::read_to_string(dir.join("engines.cfg")).map_err(|e| {
            CfgError::Io { path: dir.join("engines.cfg").display().to_string(), source: e }
        })?;
        let sys_text = std::fs::read_to_string(dir.join("systems.cfg")).map_err(|e| {
            CfgError::Io { path: dir.join("systems.cfg").display().to_string(), source: e }
        })?;
        Self::from_strs(&fm_text, &eng_text, &sys_text)
    }

    /// Parse the three cfg documents from strings.
    pub fn from_strs(
        flight_model: &str,
        engines: &str,
        systems: &str,
    ) -> CfgResult<Self> {
        let fm = CfgFile::parse(flight_model, "flight_model.cfg")?;
        let eg = CfgFile::parse(engines, "engines.cfg")?;
        let sy = CfgFile::parse(systems, "systems.cfg")?;
        Ok(Self {
            flight_model: flight_model::FlightModelDocument::from_cfg(&fm)?,
            engines: EngineDefinition::from_cfg(&eg)?,
            systems: SystemsDefinition::from_cfg(&sy)?,
        })
    }

    /// Load the repository's clean-room C172 reference aircraft.
    pub fn load_reference_c172() -> CfgResult<Self> {
        Self::load_dir(REFERENCE_AIRCRAFT_DIR)
    }
}
