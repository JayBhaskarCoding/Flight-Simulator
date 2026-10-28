//! GPU-facing data layouts.
//!
//! Everything in this module is `#[repr(C)]` + `bytemuck::Pod` and must
//! agree bit-for-bit with the WGSL declarations of the same name. The
//! layout invariants are asserted by unit tests in `tests/integration.rs`.
//!
//! Coordinate law: all *positions* that reach the GPU are 32-bit and
//! **camera-relative** (see `sim_geo::FloatingOrigin`); matrices are
//! built on the CPU from the 64-bit camera state before narrowing.

use bytemuck::{Pod, Zeroable};

/// A packable version of `glam::Mat4` is already `Pod`; the uniform
/// blocks below use plain arrays so the WGSL side can declare
/// `mat4x4<f32>` directly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    /// Camera-relative ECEF → view space (rotation only).
    pub view: [[f32; 4]; 4],
    /// Projection matrix (right-handed, view looks down −Z).
    pub proj: [[f32; 4]; 4],
    /// Inverse of `proj * view` (clip → camera-relative world).
    pub inv_view_proj: [[f32; 4]; 4],
    /// Rotation taking camera-relative ECEF directions into the
    /// planetary *sky frame* (Y-up, used by the atmosphere model).
    pub sky_from_rel: [[f32; 4]; 4],
    /// Unit direction towards the sun, in the sky frame.
    pub sun_dir_sky: [f32; 4],
    /// Sun radiance entering the atmosphere (linear HDR, RGB) and the
    /// exposure multiplier in `w`.
    pub sun_irradiance: [f32; 4],
    /// Camera-relative ECEF position of the far clip point used by the
    /// sky pass is implicit (camera = 0); this slot carries
    /// `(time_seconds, wind_x, wind_y, wind_z)`.
    pub time_wind: [f32; 4],
    /// `(exposure, flags, camera_altitude_m, atmosphere_height_m)`;
    /// flags bit0 = apply aerial perspective, bit1 = draw sun disc.
    pub params: [f32; 4],
}

/// Per-vertex mesh data (interleaved, one fixed vertex layout for the
/// whole pipeline).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Tangent (xyz) + bitangent handedness in `w` (glTF convention).
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
}

/// Per-draw parameters, addressed by the vertex stage through a
/// storage buffer (`draws[vertex.draw_id]`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DrawUniforms {
    /// Model matrix (camera-relative ECEF, f32 — narrowed on CPU).
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of `model` for normal transforms.
    pub normal: [[f32; 4]; 4],
    /// Material slot in the material storage buffer.
    pub material: u32,
    pub _pad: [u32; 3],
}

/// Physically-based material parameters (see `pbr.wgsl`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    /// `emissive_factor` (RGB) + `metallic_factor` in `w`.
    pub emissive_metallic: [f32; 4],
    /// `roughness_factor`, normal-map scale, occlusion strength,
    /// alpha cutoff in `w`.
    pub roughness: [f32; 4],
    /// Texture indices (`u32`; −1 encoded as `!0` = unused) :
    /// `[base_color, metallic_roughness, normal, emissive]`.
    pub textures: [u32; 4],
    /// `[occlusion_texture, alpha_mode, double_sided as u32, unused]`.
    pub aux: [u32; 4],
}

impl GpuMaterial {
    /// Texture-slot sentinel for "no texture bound".
    pub const NO_TEXTURE: u32 = !0;
}

/// Volumetric cloud parameters, updated every frame.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CloudUniforms {
    /// `(time_s, wind_x, wind_y, wind_z)` in metres — offsets into the
    /// procedural weather/shape fields.
    pub wind_time: [f32; 4],
    /// `(coverage, density_multiplier, anvil_percent, unused)`.
    pub coverage: [f32; 4],
    /// `(bottom_m, top_m, max_ray_len_m, step_scale)`.
    pub layer: [f32; 4],
    /// `(sun_g, ambient_boost, powder, unused)` — HG asymmetry etc.
    pub shading: [f32; 4],
    /// `(planet_radius_m, unused, unused, unused)` — kept in sync with
    /// `SkyParams::planet_radius_m` by the frame uploader.
    pub geo: [f32; 4],
}

impl Default for CloudUniforms {
    fn default() -> Self {
        Self {
            wind_time: [0.0; 4],
            coverage: [0.48, 1.0, 0.35, 0.0],
            layer: [1_500.0, 4_000.0, 60_000.0, 1.0],
            shading: [0.7, 1.0, 1.0, 0.0],
            geo: [6_360_000.0, 0.0, 0.0, 0.0],
        }
    }
}

/// A CPU-side draw request for one frame.
///
/// `model` is already camera-relative: the caller obtained it from
/// [`sim_geo::FloatingOrigin::model_matrix`] (or built the ECEF pose in
/// `f64` and rebased before narrowing).
#[derive(Debug, Clone, Copy)]
pub struct DrawItem {
    /// Index into the renderer's mesh table.
    pub mesh: u32,
    /// Index into the renderer's material table; `None` selects the
    /// model's default-material slot (glTF: primitive without
    /// `material`).
    pub material: Option<u32>,
    /// Camera-relative model matrix (f32).
    pub model: [[f32; 4]; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_c_sizes_match_wgsl_blocks() {
        // 4× mat4x4 + 4× vec4
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 4 * 64 + 4 * 16);
        assert_eq!(std::mem::size_of::<GpuVertex>(), 48);
        assert_eq!(std::mem::size_of::<DrawUniforms>(), 144);
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 80);
        assert_eq!(std::mem::size_of::<CloudUniforms>(), 80);
    }

    /// Rust's natural alignment of `[f32; N]` members is 4 (not the
    /// WGSL 16-byte vector alignment), but `#[repr(C)]` sequential
    /// layout with 16-byte-multiple field sizes/offsets produces the
    /// **identical byte layout** the WGSL declarations read. Pin the
    /// field offsets so any struct edit that would desync the shaders
    /// fails here.
    #[test]
    fn field_offsets_match_wgsl_blocks() {
        assert_eq!(std::mem::offset_of!(FrameUniforms, view), 0);
        assert_eq!(std::mem::offset_of!(FrameUniforms, proj), 64);
        assert_eq!(std::mem::offset_of!(FrameUniforms, inv_view_proj), 128);
        assert_eq!(std::mem::offset_of!(FrameUniforms, sky_from_rel), 192);
        assert_eq!(std::mem::offset_of!(FrameUniforms, sun_dir_sky), 256);
        assert_eq!(std::mem::offset_of!(FrameUniforms, sun_irradiance), 272);
        assert_eq!(std::mem::offset_of!(FrameUniforms, time_wind), 288);
        assert_eq!(std::mem::offset_of!(FrameUniforms, params), 304);

        assert_eq!(std::mem::offset_of!(DrawUniforms, model), 0);
        assert_eq!(std::mem::offset_of!(DrawUniforms, normal), 64);
        assert_eq!(std::mem::offset_of!(DrawUniforms, material), 128);
        assert_eq!(std::mem::offset_of!(DrawUniforms, _pad), 132);

        assert_eq!(std::mem::offset_of!(GpuMaterial, base_color), 0);
        assert_eq!(std::mem::offset_of!(GpuMaterial, emissive_metallic), 16);
        assert_eq!(std::mem::offset_of!(GpuMaterial, roughness), 32);
        assert_eq!(std::mem::offset_of!(GpuMaterial, textures), 48);
        assert_eq!(std::mem::offset_of!(GpuMaterial, aux), 64);

        assert_eq!(std::mem::offset_of!(GpuVertex, position), 0);
        assert_eq!(std::mem::offset_of!(GpuVertex, normal), 12);
        assert_eq!(std::mem::offset_of!(GpuVertex, tangent), 24);
        assert_eq!(std::mem::offset_of!(GpuVertex, uv), 40);

        assert_eq!(std::mem::offset_of!(CloudUniforms, geo), 64);
    }

    #[test]
    fn default_cloud_layer_is_sane() {
        let c = CloudUniforms::default();
        assert!(c.layer[0] < c.layer[1]);
        assert!(c.coverage[0] >= 0.0 && c.coverage[0] <= 1.0);
        assert!(c.shading[0] > -1.0 && c.shading[0] < 1.0);
    }
}
