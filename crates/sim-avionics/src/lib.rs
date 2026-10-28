//! Avionics services for the simulator:
//!
//! * [`simvar`] — the typed SimVar bus: named flight-data variables with
//!   unit tracking, SI-internal storage and change broadcasting.
//! * [`pid`] — a complete PID controller with anti-windup, output limits
//!   and derivative-on-measurement filtering.
//! * [`autopilot`] — an autopilot built on PID loops that steers from bus
//!   snapshots (roll/heading/altitude/vertical-speed hold).
//! * [`gauge`] — the Wasmtime gauge host: runs instrument gauge modules
//!   written as WebAssembly with a small `gauge_*` export ABI.
//! * [`opensimconnect`] — an OpenSimConnect-style TCP server that lets
//!   external clients define, request and subscribe to SimVars over TCP.
//!
//! All state exchanged on the bus is `f64` in SI units internally
//! (the 64-bit world law); units only matter at the API boundary.

pub mod autopilot;
pub mod gauge;
pub mod opensimconnect;
pub mod pid;
pub mod simvar;

pub use autopilot::{ApMode, Autopilot, AutopilotConfig, ApOutput, ApSample};
pub use gauge::{GaugeError, GaugeHost, GaugeInstance};
pub use opensimconnect::{OpenSimConnectError, OpenSimConnectServer, DEFAULT_PORT};
pub use pid::{Pid, PidConfig, PidGains};
pub use simvar::{SimVarBus, SimVarError, SimVarUpdate, Unit};
