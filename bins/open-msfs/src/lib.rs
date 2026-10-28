//! OpenMSFS engine binary library: the ECS world, the 1000 Hz
//! physics-thread core, the ENU↔ECEF frame bridge, solar position and
//! the winit/wgpu windowed application.
//!
//! The executable ([`main.rs`](src/main.rs)) is a thin CLI around
//! these modules; keeping them in a library target makes the engine
//! API visible to integration tests and keeps `cargo check` free of
//! dead-code noise.

pub mod app;
pub mod ecs;
pub mod sim;
pub mod solar;
