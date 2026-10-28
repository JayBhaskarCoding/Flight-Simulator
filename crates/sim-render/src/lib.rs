//! Rendering for the OpenMSFS engine: a wgpu render graph with a
//! PBR glTF pipeline, the Hillaire physical-sky background and
//! volumetric clouds — all in WGSL, all fed by 64-bit world math on
//! the CPU and strict 32-bit camera-relative data on the GPU.
//!
//! Module map:
//!
//! * [`types`] — `#[repr(C)]` GPU layouts (frame/draw/material/cloud
//!   uniforms, vertices, draw requests);
//! * [`gltf`] — clean-room glTF 2.0 loader (GLB, sparse accessors,
//!   primitive modes, PNG/JPEG images);
//! * [`noise`] — deterministic tileable Perlin–Worley volumes for the
//!   cloud raymarch;
//! * [`lut`] — atmosphere LUT upload payloads + CPU mirrors of the
//!   WGSL samplers (parity-tested against `sim-atmosphere`);
//! * [`shaders`] — embedded WGSL sources, naga-validated in tests;
//! * [`graph`] — the [`graph::Renderer`]: pipelines, resource
//!   upload and frame orchestration.
//!
//! Coordinate law (64-bit world vs 32-bit GPU): this crate never
//! computes planetary coordinates itself — it consumes matrices and
//! camera-relative `f32` values produced through
//! [`sim_geo::FloatingOrigin`] (see [`graph::Renderer::draws_from_gltf`]
//! for the sanctioned `f64` → `f32` rebasing path).

pub mod gltf;
pub mod graph;
pub mod lut;
pub mod noise;
pub mod shaders;
pub mod types;

pub use gltf::{
    parse_bytes, AlphaMode, DecodedImage, GltfError, GltfInstance, GltfMaterial, GltfMesh,
    GltfModel, GltfSampler, GltfSource, WrapMode,
};
pub use graph::{DrawError, FrameInput, ModelHandle, Renderer};
pub use lut::LutPayloads;
pub use noise::{generate_noise_volume, NoiseVolume};
pub use types::{CloudUniforms, DrawItem, GpuMaterial, GpuVertex};
