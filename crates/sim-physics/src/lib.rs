//! 6-DOF flight dynamics at a fixed 1000 Hz integration rate.
//!
//! # Frames and conventions
//!
//! * **World frame** — a local East-North-Up (ENU) tangent plane:
//!   `+X` east, `+Y` north, `+Z` up. Gravity is `(0, 0, -G0)`. The plane is
//!   centred at the simulation origin (airfield); curvature is handled by
//!   `sim-geo` outside this crate.
//! * **Body frame** — aviation convention matching `sim-cfg`:
//!   `+X` forward, `+Y` right, `+Z` down. [`RigidBodyState::orientation`]
//!   rotates body vectors into the world frame.
//! * **Moments** — positive `+X` rolls the right wing down, positive `+Y`
//!   pitches the nose up, positive `+Z` yaws the nose right.
//!
//! All maths is `f64` (`glam::DVec3`/`DQuat`/`DMat3`); there is no GPU code
//! in this crate.
//!
//! # Pipeline
//!
//! * [`rigid_body`] — massive rigid body with an explicit RK4 integrator;
//!   forces are re-evaluated at every Runge-Kutta stage through a closure,
//!   and the quaternion is projected back to unit length after each stage
//!   (standard constraint projection, drift is ~1e-14 over hours).
//! * [`aero`] — surface-element CFD: every `sim-cfg` `AeroElement` strip
//!   evaluates its local velocity (`v + ω × r`), angle of attack (including
//!   incidence, washout and dihedral tilt), lift with the aircraft's stall
//!   envelope, and induced + parasite drag, then integrates `r × F` about
//!   the CG. Control-surface effectiveness comes from the aircraft's
//!   stability derivatives (`cm_delta_e`, `cl_delta_a`, `cn_delta_r`);
//!   rate damping, dihedral effect and fin weathercock stability all emerge
//!   physically from the element integration (they are *not* double-counted
//!   as coefficient derivatives).
//! * [`gear`] — penalty-contact landing gear with spring/damper struts,
//!   steered rolling direction, slip-angle cornering with friction-circle
//!   saturation, braking and rolling resistance, over a pluggable
//!   [`GroundProvider`] (flat plane today, DEM-backed later in `sim-world`).
//! * [`world`] — the [`PhysicsWorld`] owning one RK4 step at
//!   [`DT`] = 1 ms, with gravity, aero, gear and externally injected
//!   forces (engine/propeller hooks live in later crates).

pub mod aero;
pub mod gear;
pub mod rigid_body;
pub mod world;

pub use aero::{AeroForces, AeroSolver, AtmosphereSample, ControlInputs, ElementModel};
pub use gear::{GearAxle, GearContact, GearLeg, GearLegConfig, GroundProvider, GroundSurface, PlaneGround};
pub use rigid_body::{ForceTorque, InertiaError, RigidBody, RigidBodyState};
pub use world::{StepDiagnostics, PhysicsWorld, DT, SIM_RATE_HZ};

/// Standard gravity, m/s² (ISA / NIST SP 811).
pub const G0: f64 = 9.806_65;
