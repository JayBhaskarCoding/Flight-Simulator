//! # sim-atmosphere
//!
//! Atmospheric modelling for the OpenMSFS engine — all implementations
//! are clean-room code from public scientific literature:
//!
//! * [`isa`] — US Standard Atmosphere 1976 (NASA SP-368) with the full
//!   seven-layer lapse-rate model, non-standard days (ΔISA), humidity
//!   correction, Sutherland viscosity and density altitude.
//! * [`metar`] — ICAO/NOAA WMO METAR decoder (public field definitions).
//! * [`dryden`] — Dryden continuous turbulence model (MIL-F-8785C /
//!   NASA SP-1977 transfer functions) discretised with exact ZOH state
//!   transitions and stationary-variance-normalised Gaussian driving
//!   noise, plus an in-crate PRNG (no external dependencies).
//! * [`sky`] — CPU-built sky LUTs: Rayleigh+Mie transmittance, sky-view
//!   and aerial-perspective textures from the public Nishita single-
//!   scattering model with wavelength-dependent coefficients.

pub mod dryden;
pub mod isa;
pub mod metar;
pub mod sky;

pub use dryden::{DrydenConfig, DrydenTurbulence, TurbulenceIntensity, Rng};
pub use isa::{AtmosphereState, IsaConditions, IsaModel};
pub use metar::{
    CloudLayer, Metar, MetarError, SkyCondition, WeatherPhenomenon, Wind,
};
pub use sky::{SkyLuts, SkyParams};
