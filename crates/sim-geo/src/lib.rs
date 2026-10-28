//! # sim-geo
//!
//! Planetary geodesy and world-reference math for the OpenMSFS engine.
//!
//! All computation in this crate is **64-bit** (`f64` / [`glam::DVec3`] /
//! [`glam::DQuat`] / [`glam::DMat3`]) in accordance with the engine's
//! *64-bit world vs. 32-bit GPU* law. The only place 32-bit values are
//! produced is [`floating_origin`], which converts earth-fixed coordinates
//! into **camera-relative** coordinates for the GPU after subtracting the
//! camera's 64-bit ECEF position on the CPU.
//!
//! Modules:
//!
//! * [`ellipsoid`] — reference ellipsoid geometry (WGS84 by default).
//! * [`types`] — latitude/longitude/altitude and ECEF/ENU vector newtypes.
//! * [`enu`] — east-north-up local tangent frames.
//! * [`floating_origin`] — camera-relative floating origin (f64 → f32).
//! * [`geodesic`] — spherical great-circle helpers.
//! * [`quadtree`] — six-face spherical quadtree used by the tile streamer.
//!
//! All formulas follow public standards: WGS84 (NGA.STD.0088), the IERS
//! Conventions (Géodésie), and the standard closed-form ecef↔geodetic
//! transformation after Zhang & Pararas (2016).

pub mod ellipsoid;
pub mod enu;
pub mod floating_origin;
pub mod geodesic;
pub mod quadtree;
pub mod types;

pub use ellipsoid::{Ellipsoid, WGS84};
pub use enu::EnuFrame;
pub use floating_origin::FloatingOrigin;
pub use types::{Ecef, Enu, LatLonAlt};
