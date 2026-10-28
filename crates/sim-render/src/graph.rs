//! The wgpu render graph: pipelines, resource upload and per-frame
//! orchestration.
//!
//! Frame structure:
//!
//! 1. **sky** → `hdr_a` (fullscreen, no depth) — Hillaire sky
//!    background from the CPU-built LUTs + sun disc;
//! 2. **opaque PBR** → `hdr_a` + `depth32float` — the draw list with
//!    aerial perspective; then **alpha-blend** draws (depth writes off);
//! 3. the depth attachment is **copied** to a sampling texture so the
//!    cloud pass never uses one texture as attachment and sample;
//! 4. **clouds** → `hdr_b`, sampling `hdr_a` + the depth copy,
//!    compositing volumetric in-scatter front-to-back;
//! 5. **tonemap** → the caller's sRGB target (linear ACES output; the
//!    target's transfer function encodes on write).
//!
//! Concurrency: `render` needs only `&self`; resource upload
//! (`set_model`, `set_sky_luts`, `set_noise`, `resize`) takes `&mut self`.

use crate::gltf::{AlphaMode, FilterMode, GltfMaterial, GltfModel, GltfSampler, WrapMode};
use crate::lut::LutPayloads;
use crate::noise::{generate_noise_volume, NoiseVolume};
use crate::types::{
    CloudUniforms, DrawItem, DrawUniforms, FrameUniforms, GpuMaterial, GpuVertex,
};
use bytemuck::cast_slice;
use glam::{DMat4, DQuat, DVec3, Mat4, Quat, Vec3};
use sim_atmosphere::sky::{
    SkyLuts, AERIAL_DIST, AERIAL_ZEN, SKY_VIEW_H, SKY_VIEW_W, TRANS_H, TRANS_W,
};
use sim_geo::FloatingOrigin;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Duration;
use wgpu::util::DeviceExt;

/// Draw slots in the dynamic-offset draw buffer — 256 bytes each,
/// ≥ `min_uniform_buffer_offset_alignment` on every wgpu target.
pub const MAX_DRAW_SLOTS: usize = 4096;
const DRAW_SLOT_STRIDE: u64 = 256;

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

/// Failures while building or driving the renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawError {
    /// No wgpu adapter — headless sandboxes without a software
    /// rasteriser are expected to hit this; tests treat it as "skip".
    AdapterNotFound,
    /// Device request failed.
    Device(String),
    /// Frame requested more draws than [`MAX_DRAW_SLOTS`].
    TooManyDraws { got: usize },
    /// Draws were submitted before [`Renderer::set_model`].
    NoModel,
    /// A draw references a mesh/material outside the loaded model.
    TableOutOfRange { what: &'static str, index: u32 },
    /// The tonemap target must be an sRGB format (`Bgra8UnormSrgb` or
    /// `Rgba8UnormSrgb`): the shader emits linear ACES and relies on
    /// the target's transfer function.
    UnsupportedTarget(wgpu::TextureFormat),
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrawError::AdapterNotFound => write!(f, "no suitable wgpu adapter"),
            DrawError::Device(e) => write!(f, "wgpu device request failed: {e}"),
            DrawError::TooManyDraws { got } => {
                write!(f, "frame has {got} draws, max is {MAX_DRAW_SLOTS}")
            }
            DrawError::NoModel => write!(f, "no model loaded (call set_model first)"),
            DrawError::TableOutOfRange { what, index } => {
                write!(f, "draw references {what}[{index}] which does not exist")
            }
            DrawError::UnsupportedTarget(fmt) => write!(
                f,
                "tonemap target {fmt:?} must be sRGB (Bgra8UnormSrgb or Rgba8UnormSrgb)"
            ),
        }
    }
}

impl std::error::Error for DrawError {}

// ---------------------------------------------------------------------------
// minimal block_on (no external executor dependency)
// ---------------------------------------------------------------------------

struct ParkThread {
    thread: std::thread::Thread,
}

const PARK_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const ParkThread) };
        let cloned = Arc::into_raw(arc.clone()) as *const ();
        let _ = Arc::into_raw(arc); // borrowed, not moved
        RawWaker::new(cloned, &PARK_VTABLE)
    },
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const ParkThread) };
        arc.thread.unpark();
    },
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const ParkThread) };
        arc.thread.unpark();
        let _ = Arc::into_raw(arc);
    },
    |ptr| {
        drop(unsafe { Arc::from_raw(ptr as *const ParkThread) });
    },
);

fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    let mut fut = Box::pin(fut);
    let waker = {
        let arc = Arc::new(ParkThread { thread: std::thread::current() });
        let raw = RawWaker::new(Arc::into_raw(arc) as *const (), &PARK_VTABLE);
        unsafe { Waker::from_raw(raw) }
    };
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park_timeout(Duration::from_millis(5)),
        }
    }
}

// ---------------------------------------------------------------------------
// per-frame input
// ---------------------------------------------------------------------------

/// Everything the render graph needs for one frame. All positions are
/// **camera-relative 32-bit**; double-precision poses are reduced to
/// that contract before upload (see [`Renderer::draws_from_gltf`]).
#[derive(Debug, Clone)]
pub struct FrameInput<'a> {
    /// Camera-relative ECEF → view space.
    pub view: Mat4,
    /// Projection matrix with NDC depth 0..1 (`Mat4::perspective_rh`).
    pub proj: Mat4,
    /// Rotation taking camera-relative ECEF directions into the
    /// planetary sky frame (Y-up, `sim-atmosphere` conventions).
    pub sky_from_rel: Quat,
    /// Unit direction to the sun in the sky frame.
    pub sun_dir_sky: Vec3,
    /// Solar irradiance entering the atmosphere (linear RGB).
    pub sun_irradiance: Vec3,
    /// Camera altitude above sea level (m).
    pub camera_alt_m: f32,
    /// `SkyParams::atmosphere_height_m` used to build the LUTs.
    pub atmosphere_height_m: f32,
    /// Exposure multiplier (applied by every HDR stage *and* tonemap).
    pub exposure: f32,
    /// Apply aerial perspective to opaques and clouds.
    pub aerial: bool,
    /// Draw the solar disc in the sky pass.
    pub sun_disc: bool,
    /// Simulation time (s) for cloud animation.
    pub time_s: f32,
    /// Wind vector in the sky frame (m/s).
    pub wind: Vec3,
    /// Volumetric cloud parameters (`clouds.wind_time` is *not*
    /// overwritten — keep it consistent with `time_s`/`wind`).
    pub clouds: CloudUniforms,
    /// Draw list for this frame.
    pub draws: &'a [DrawItem],
}

// ---------------------------------------------------------------------------
// resource tables
// ---------------------------------------------------------------------------

/// Index ranges of an uploaded model's GPU tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelHandle {
    pub meshes: u32,
    pub materials: u32,
    /// Slot of the default material (glTF: primitive without one).
    pub default_material: u32,
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

struct GpuImage {
    srgb: wgpu::TextureView,
    linear: wgpu::TextureView,
}

struct GpuMaterialSlot {
    bind_group: wgpu::BindGroup,
    blend: bool,
    double_sided: bool,
}

/// A texture + its default view (needed both as attachment and for
/// `write_texture`/`copy_texture_to_texture`, which only take textures).
struct Res {
    #[allow(dead_code)]
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

struct NoiseRes {
    edge: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

// ---------------------------------------------------------------------------
// layout entry helpers
// ---------------------------------------------------------------------------

fn unfilterable_lut(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn filterable_tex(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_res(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    w: u32,
    h: u32,
    extra: wgpu::TextureUsages,
) -> Res {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | extra,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Res { tex, view }
}

fn gpu_mat(m: &Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

fn narrow16(c: [f64; 16]) -> [[f32; 4]; 4] {
    Mat4::from_cols_array(&c.map(|v| v as f32)).to_cols_array_2d()
}

// ---------------------------------------------------------------------------
// renderer
// ---------------------------------------------------------------------------

/// The render graph: create once with [`Renderer::new`] (or
/// [`Renderer::with_device`] when windowing code owns the adapter),
/// feed resources, then call [`Renderer::render`] every frame.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: (u32, u32),

    frame_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,
    sky_bgl: wgpu::BindGroupLayout,
    clouds_bgl: wgpu::BindGroupLayout,
    tonemap_bgl: wgpu::BindGroupLayout,
    pbr_opaque: wgpu::RenderPipeline,
    pbr_opaque_nc: wgpu::RenderPipeline,
    pbr_blend: wgpu::RenderPipeline,
    pbr_blend_nc: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    clouds_pipeline: wgpu::RenderPipeline,
    tonemap_bgra: wgpu::RenderPipeline,
    tonemap_rgba: wgpu::RenderPipeline,

    hdr_a: Res,
    hdr_b: Res,
    depth: Res,
    depth_copy: Res,

    frame_buf: wgpu::Buffer,
    draw_buf: wgpu::Buffer,
    cloud_buf: wgpu::Buffer,
    exposure_buf: wgpu::Buffer,
    material_buf: wgpu::Buffer,
    // Invariant: `Some` after `rebuild_static_binds`, which every
    // constructor, `set_model`, `set_sky_luts`, `set_noise` and
    // `resize` runs.
    frame_bind: Option<wgpu::BindGroup>,
    sky_bind: Option<wgpu::BindGroup>,
    clouds_bind: Option<wgpu::BindGroup>,
    tonemap_bgra_bind: Option<wgpu::BindGroup>,
    tonemap_rgba_bind: Option<wgpu::BindGroup>,
    material_sampler: wgpu::Sampler,
    lut_sampler: wgpu::Sampler,
    tonemap_sampler: wgpu::Sampler,
    lut_trans: Res,
    lut_sky: Res,
    lut_atr: Res,
    lut_arr: Res,
    noise: NoiseRes,

    white_srgb: wgpu::TextureView,
    white_linear: wgpu::TextureView,
    flat_normal: wgpu::TextureView,
    black_linear: wgpu::TextureView,
    samplers: HashMap<(u8, u8, u8, u8), wgpu::Sampler>,

    meshes: Vec<GpuMesh>,
    images: Vec<GpuImage>,
    material_slots: Vec<GpuMaterialSlot>,
    materials_cpu: Vec<GpuMaterial>,
    default_material: u32,
    has_model: bool,
}

impl Renderer {
    /// Create a headless renderer drawing into offscreen HDR targets
    /// of `width × height`.
    pub fn new(width: u32, height: u32) -> Result<Self, DrawError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(DrawError::AdapterNotFound)?;

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("open-msfs-renderer"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(|e| DrawError::Device(e.to_string()))?;

        Self::with_device(device, queue, width, height)
    }

    /// Build the renderer from an already-acquired device/queue (the
    /// windowed binary owns surface/adapter selection).
    pub fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Result<Self, DrawError> {
        let width = width.max(1);
        let height = height.max(1);

        // ---- bind group layouts ---------------------------------------
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame-bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<DrawUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                unfilterable_lut(3),
                unfilterable_lut(4),
                unfilterable_lut(5),
            ],
        });

        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                filterable_tex(1),
                filterable_tex(2),
                filterable_tex(3),
                filterable_tex(4),
                filterable_tex(5),
            ],
        });

        let sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky-bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                unfilterable_lut(1),
                unfilterable_lut(2),
                unfilterable_lut(3),
                unfilterable_lut(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let clouds_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clouds-bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                filterable_tex(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                unfilterable_lut(6),
                unfilterable_lut(7),
            ],
        });

        let tonemap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap-bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                filterable_tex(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ---- shaders & pipelines ---------------------------------------
        let pbr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::PBR_WGSL.into()),
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::SKY_WGSL.into()),
        });
        let clouds_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::CLOUDS_WGSL.into()),
        });
        let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::TONEMAP_WGSL.into()),
        });

        let pbr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr-layout"),
            bind_group_layouts: &[&frame_bgl, &material_bgl],
            push_constant_ranges: &[],
        });
        let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky-layout"),
            bind_group_layouts: &[&sky_bgl],
            push_constant_ranges: &[],
        });
        let clouds_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-layout"),
            bind_group_layouts: &[&clouds_bgl],
            push_constant_ranges: &[],
        });
        let tonemap_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap-layout"),
            bind_group_layouts: &[&tonemap_bgl],
            push_constant_ranges: &[],
        });

        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2];

        let make_pbr = |label: &str, blend: bool, cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pbr_pl),
                vertex: wgpu::VertexState {
                    module: &pbr_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &pbr_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: if blend {
                            Some(wgpu::BlendState::ALPHA_BLENDING)
                        } else {
                            None
                        },
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: !blend,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let pbr_opaque = make_pbr("pbr-opaque", false, Some(wgpu::Face::Back));
        let pbr_opaque_nc = make_pbr("pbr-opaque-nc", false, None);
        let pbr_blend = make_pbr("pbr-blend", true, Some(wgpu::Face::Back));
        let pbr_blend_nc = make_pbr("pbr-blend-nc", true, None);

        let fullscreen = |label: &str,
                          layout: &wgpu::PipelineLayout,
                          shader: &wgpu::ShaderModule,
                          format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let sky_pipeline =
            fullscreen("sky", &sky_pl, &sky_shader, wgpu::TextureFormat::Rgba16Float);
        let clouds_pipeline =
            fullscreen("clouds", &clouds_pl, &clouds_shader, wgpu::TextureFormat::Rgba16Float);

        let mk_tonemap = |format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tonemap"),
                layout: Some(&tonemap_pl),
                vertex: wgpu::VertexState {
                    module: &tonemap_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &tonemap_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let tonemap_bgra = mk_tonemap(wgpu::TextureFormat::Bgra8UnormSrgb);
        let tonemap_rgba = mk_tonemap(wgpu::TextureFormat::Rgba8UnormSrgb);

        // ---- frame-sized targets ---------------------------------------
        let hdr_a = make_res(
            &device,
            "hdr-a",
            wgpu::TextureFormat::Rgba16Float,
            width,
            height,
            wgpu::TextureUsages::COPY_SRC,
        );
        let hdr_b = make_res(
            &device,
            "hdr-b",
            wgpu::TextureFormat::Rgba16Float,
            width,
            height,
            wgpu::TextureUsages::empty(),
        );
        let depth = make_res(
            &device,
            "depth",
            wgpu::TextureFormat::Depth32Float,
            width,
            height,
            wgpu::TextureUsages::COPY_SRC,
        );
        let depth_copy = make_res(
            &device,
            "depth-copy",
            wgpu::TextureFormat::Depth32Float,
            width,
            height,
            wgpu::TextureUsages::COPY_DST,
        );

        // ---- samplers ---------------------------------------------------
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let lut_sampler = device
            .create_sampler(&wgpu::SamplerDescriptor { label: Some("lut-sampler"), ..Default::default() });
        let tonemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tonemap-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ---- dynamic buffers --------------------------------------------
        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-uniform"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw-uniforms"),
            size: (MAX_DRAW_SLOTS as u64) * DRAW_SLOT_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cloud_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud-uniform"),
            size: std::mem::size_of::<CloudUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exposure_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_gpu = GltfMaterial::default().to_gpu();
        let material_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials"),
            contents: cast_slice(std::slice::from_ref(&default_gpu)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ---- LUT textures (zeroed until set_sky_luts) --------------------
        let mk_lut = |label: &str, w: u32, h: u32| -> Res {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Res { tex, view }
        };
        let lut_trans = mk_lut("lut-trans", TRANS_W as u32, TRANS_H as u32);
        let lut_sky = mk_lut("lut-sky", SKY_VIEW_W as u32, SKY_VIEW_H as u32);
        let lut_atr = mk_lut("lut-aerial-t", AERIAL_DIST as u32, AERIAL_ZEN as u32);
        let lut_arr = mk_lut("lut-aerial-r", AERIAL_DIST as u32, AERIAL_ZEN as u32);

        // ---- 1x1 default material textures ------------------------------
        let mk1 = |label: &str, format: wgpu::TextureFormat, rgba: [u8; 4]| -> wgpu::TextureView {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &t,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            t.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let white_srgb = mk1("white-srgb", wgpu::TextureFormat::Rgba8UnormSrgb, [255, 255, 255, 255]);
        let white_linear =
            mk1("white-linear", wgpu::TextureFormat::Rgba8Unorm, [255, 255, 255, 255]);
        let flat_normal = mk1("flat-normal", wgpu::TextureFormat::Rgba8Unorm, [128, 128, 255, 255]);
        let black_linear = mk1("black-linear", wgpu::TextureFormat::Rgba8Unorm, [0, 0, 0, 0]);

        // ---- default noise volume (8³, replaced by set_noise) ------------
        let default_noise = generate_noise_volume(8, 0);
        let noise_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud-noise-default"),
            size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 8 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &noise_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &default_noise.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * 4),
                rows_per_image: Some(8),
            },
            wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 8 },
        );
        let noise_view = noise_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut renderer = Self {
            device,
            queue,
            size: (width, height),
            frame_bgl,
            material_bgl,
            sky_bgl,
            clouds_bgl,
            tonemap_bgl,
            pbr_opaque,
            pbr_opaque_nc,
            pbr_blend,
            pbr_blend_nc,
            sky_pipeline,
            clouds_pipeline,
            tonemap_bgra,
            tonemap_rgba,
            hdr_a,
            hdr_b,
            depth,
            depth_copy,
            frame_buf,
            draw_buf,
            cloud_buf,
            exposure_buf,
            material_buf,
            frame_bind: None,
            sky_bind: None,
            clouds_bind: None,
            tonemap_bgra_bind: None,
            tonemap_rgba_bind: None,
            material_sampler,
            lut_sampler,
            tonemap_sampler,
            lut_trans,
            lut_sky,
            lut_atr,
            lut_arr,
            noise: NoiseRes { edge: 8, texture: noise_tex, view: noise_view },
            white_srgb,
            white_linear,
            flat_normal,
            black_linear,
            samplers: HashMap::new(),
            meshes: Vec::new(),
            images: Vec::new(),
            material_slots: Vec::new(),
            materials_cpu: Vec::new(),
            default_material: 0,
            has_model: false,
        };
        renderer.rebuild_static_binds();
        Ok(renderer)
    }

    // -- resource management ---------------------------------------------

    /// Current pixel size of the offscreen HDR/depth targets.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Create an external render target texture + view (used by tests
    /// and by callers that want to composite the tonemap output into
    /// their own surface-equivalent texture). The texture is
    /// `RENDER_ATTACHMENT` and retains the extra usages in `extra`.
    pub fn create_target(
        &self,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("external-target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn rebuild_static_binds(&mut self) {
        let noise_view = &self.noise.view;
        self.frame_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame-bind"),
            layout: &self.frame_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.draw_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<DrawUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 2, resource: self.material_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.lut_trans.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.lut_atr.view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.lut_arr.view) },
            ],
        }));
        self.sky_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky-bind"),
            layout: &self.sky_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.lut_trans.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.lut_sky.view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.lut_atr.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.lut_arr.view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.lut_sampler) },
            ],
        }));
        self.clouds_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds-bind"),
            layout: &self.clouds_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.cloud_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(noise_view),
                },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.hdr_a.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.depth_copy.view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.lut_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.lut_atr.view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.lut_arr.view) },
            ],
        }));
        let mk_tm = |device: &wgpu::Device,
                     layout: &wgpu::BindGroupLayout,
                     exposure: &wgpu::Buffer,
                     hdr: &wgpu::TextureView,
                     sampler: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tonemap-bind"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: exposure.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(hdr),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        self.tonemap_bgra_bind = Some(mk_tm(
            &self.device,
            &self.tonemap_bgl,
            &self.exposure_buf,
            &self.hdr_b.view,
            &self.tonemap_sampler,
        ));
        self.tonemap_rgba_bind = Some(mk_tm(
            &self.device,
            &self.tonemap_bgl,
            &self.exposure_buf,
            &self.hdr_b.view,
            &self.tonemap_sampler,
        ));
    }

    /// Resize the offscreen HDR/depth targets (no-op when unchanged).
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if (width, height) == self.size {
            return;
        }
        self.size = (width, height);
        self.hdr_a = make_res(
            &self.device,
            "hdr-a",
            wgpu::TextureFormat::Rgba16Float,
            width,
            height,
            wgpu::TextureUsages::COPY_SRC,
        );
        self.hdr_b = make_res(
            &self.device,
            "hdr-b",
            wgpu::TextureFormat::Rgba16Float,
            width,
            height,
            wgpu::TextureUsages::empty(),
        );
        self.depth = make_res(
            &self.device,
            "depth",
            wgpu::TextureFormat::Depth32Float,
            width,
            height,
            wgpu::TextureUsages::COPY_SRC,
        );
        self.depth_copy = make_res(
            &self.device,
            "depth-copy",
            wgpu::TextureFormat::Depth32Float,
            width,
            height,
            wgpu::TextureUsages::COPY_DST,
        );
        self.rebuild_static_binds();
    }

    /// Upload CPU-built atmosphere LUTs (`SkyLuts`).
    pub fn set_sky_luts(&mut self, luts: &SkyLuts) {
        let p = LutPayloads::from_sky_luts(luts);
        let dims = p.dims();
        let write = |queue: &wgpu::Queue, tex: &wgpu::Texture, data: &[[f32; 4]], (w, h): (u32, u32)| {
            debug_assert_eq!(data.len(), (w * h) as usize);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                cast_slice(data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 16),
                    rows_per_image: None,
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        };
        write(&self.queue, &self.lut_trans.tex, &p.transmittance, dims[0]);
        write(&self.queue, &self.lut_sky.tex, &p.sky_view, dims[1]);
        write(&self.queue, &self.lut_atr.tex, &p.aerial_trans, dims[2]);
        write(&self.queue, &self.lut_arr.tex, &p.aerial_radiance, dims[3]);
    }

    /// Upload (or replace) the procedural cloud-noise volume.
    pub fn set_noise(&mut self, volume: &NoiseVolume) {
        let extent = wgpu::Extent3d {
            width: volume.size,
            height: volume.size,
            depth_or_array_layers: volume.size,
        };
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(volume.size * 4),
            rows_per_image: Some(volume.size),
        };
        if self.noise.edge == volume.size {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.noise.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &volume.data,
                layout,
                extent,
            );
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud-noise"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &volume.data,
            layout,
            extent,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.noise = NoiseRes { edge: volume.size, texture: tex, view };
        self.rebuild_static_binds();
    }

    /// Sampler for a glTF sampler description (cached by wrap/filter).
    fn gltf_sampler(&mut self, s: GltfSampler) -> wgpu::Sampler {
        let key = (wrap_code(s.wrap_s), wrap_code(s.wrap_t), filter_code(s.mag), filter_code(s.min));
        if let Some(sm) = self.samplers.get(&key) {
            return sm.clone();
        }
        let addr = |c: u8| -> wgpu::AddressMode {
            match c {
                1 => wgpu::AddressMode::ClampToEdge,
                2 => wgpu::AddressMode::MirrorRepeat,
                _ => wgpu::AddressMode::Repeat,
            }
        };
        let filt = |c: u8| -> wgpu::FilterMode {
            if c == 0 {
                wgpu::FilterMode::Nearest
            } else {
                wgpu::FilterMode::Linear
            }
        };
        let sm = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gltf-sampler"),
            address_mode_u: addr(key.0),
            address_mode_v: addr(key.1),
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: filt(key.2),
            min_filter: filt(key.3),
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        self.samplers.insert(key, sm.clone());
        sm
    }

    /// Load a glTF model's meshes/materials/images into GPU tables.
    /// Replaces any previously loaded model.
    pub fn set_model(&mut self, model: &GltfModel) -> Result<ModelHandle, DrawError> {
        let mut meshes = Vec::with_capacity(model.meshes.len());
        for m in &model.meshes {
            let vertices = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh-vertices"),
                contents: cast_slice(&m.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let indices = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh-indices"),
                contents: cast_slice(&m.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            meshes.push(GpuMesh { vertices, indices, index_count: m.indices.len() as u32 });
        }

        let mut images = Vec::with_capacity(model.images.len());
        for img in &model.images {
            let extent = wgpu::Extent3d {
                width: img.width.max(1),
                height: img.height.max(1),
                depth_or_array_layers: 1,
            };
            let mk = |device: &wgpu::Device, format: wgpu::TextureFormat| -> wgpu::Texture {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gltf-image"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            };
            let srgb_tex = mk(&self.device, wgpu::TextureFormat::Rgba8UnormSrgb);
            let linear_tex = mk(&self.device, wgpu::TextureFormat::Rgba8Unorm);
            for tex in [&srgb_tex, &linear_tex] {
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &img.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(img.width.max(1) * 4),
                        rows_per_image: None,
                    },
                    extent,
                );
            }
            images.push(GpuImage {
                srgb: srgb_tex.create_view(&wgpu::TextureViewDescriptor::default()),
                linear: linear_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            });
        }

        // GPU material table + per-material bind groups; the default
        // material is appended as the final slot.
        let mut materials_cpu: Vec<GpuMaterial> =
            model.materials.iter().map(GltfMaterial::to_gpu).collect();
        materials_cpu.push(GltfMaterial::default().to_gpu());
        let default_material = (materials_cpu.len() - 1) as u32;

        let material_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials"),
            contents: cast_slice(&materials_cpu),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let white_srgb = self.white_srgb.clone();
        let white_linear = self.white_linear.clone();
        let flat_normal = self.flat_normal.clone();
        let black_linear = self.black_linear.clone();

        let default_mat = GltfMaterial::default();
        let mut slots = Vec::with_capacity(materials_cpu.len());
        for mat in model.materials.iter().chain(std::iter::once(&default_mat)) {
            // Sampler: glTF samplers are per texture; a bind group holds
            // one sampler, so the base-colour texture's sampler wins
            // (materials that share images but want different samplers
            // are not representable in one group — documented
            // simplification; secondary samplers still clamp/repeat
            // through their own textures' default when bound alone).
            let sampler = match mat
                .base_color_texture
                .or(mat.metallic_roughness_texture)
                .or(mat.normal_texture)
                .or(mat.occlusion_texture)
                .or(mat.emissive_texture)
            {
                Some(img) => {
                    let smp = model.samplers.get(img).copied().unwrap_or_default();
                    self.gltf_sampler(smp)
                }
                None => self.material_sampler.clone(),
            };
            let pick = |slot: Option<usize>, srgb: bool, fb: &wgpu::TextureView| -> wgpu::TextureView {
                match slot {
                    Some(idx) => {
                        let img = &images[idx];
                        if srgb {
                            img.srgb.clone()
                        } else {
                            img.linear.clone()
                        }
                    }
                    None => fb.clone(),
                }
            };
            let base = pick(mat.base_color_texture, true, &white_srgb);
            let mr = pick(mat.metallic_roughness_texture, false, &white_linear);
            let nrm = pick(mat.normal_texture, false, &flat_normal);
            let occ = pick(mat.occlusion_texture, false, &white_linear);
            let emi = pick(mat.emissive_texture, true, &black_linear);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material-bind"),
                layout: &self.material_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&base),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&mr),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&nrm),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&occ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&emi),
                    },
                ],
            });
            let blend = mat.alpha_mode == AlphaMode::Blend;
            let double_sided = mat.double_sided;
            slots.push(GpuMaterialSlot { bind_group, blend, double_sided });
        }

        self.meshes = meshes;
        self.images = images;
        self.material_slots = slots;
        self.materials_cpu = materials_cpu;
        self.material_buf = material_buf;
        self.default_material = default_material;
        self.has_model = true;
        self.rebuild_static_binds();

        Ok(ModelHandle {
            meshes: self.meshes.len() as u32,
            materials: self.material_slots.len() as u32,
            default_material,
        })
    }

    /// The default-material slot (for [`DrawItem::material`] = `None`).
    pub fn default_material(&self) -> u32 {
        self.default_material
    }

    /// Build GPU draw items from a glTF model's instances, an ECEF root
    /// pose and the camera's [`FloatingOrigin`].
    ///
    /// This is the **sanctioned 64→32 path**: the composed model matrix
    /// lives entirely in `f64` (root pose ∘ scene node matrix), the
    /// camera's `f64` ECEF position is subtracted from the translation
    /// column *in `f64`*, and only then is the matrix narrowed to `f32`.
    /// The per-draw normal matrix is the upper-left 3×3
    /// inverse-transpose derived from `model` at upload time (the
    /// translation of an affine matrix does not enter it).
    pub fn draws_from_gltf(
        model: &GltfModel,
        origin: &FloatingOrigin,
        root_position_ecef: DVec3,
        root_rotation: Quat,
        root_scale: Vec3,
    ) -> Vec<DrawItem> {
        let cam = origin.camera_ecef().as_dvec3();
        let root = DMat4::from_scale_rotation_translation(
            DVec3::new(root_scale.x as f64, root_scale.y as f64, root_scale.z as f64),
            DQuat::from_xyzw(
                root_rotation.x as f64,
                root_rotation.y as f64,
                root_rotation.z as f64,
                root_rotation.w as f64,
            ),
            root_position_ecef,
        );
        model
            .instances
            .iter()
            .map(|inst| {
                let local_cols = inst.world.to_cols_array();
                let mut local64 = [0.0f64; 16];
                for (d, s) in local64.iter_mut().zip(local_cols) {
                    *d = s as f64;
                }
                let mut cols = (root * DMat4::from_cols_array(&local64)).to_cols_array();
                cols[12] -= cam.x;
                cols[13] -= cam.y;
                cols[14] -= cam.z;
                DrawItem {
                    mesh: inst.mesh as u32,
                    material: inst.material_override.map(|m| m as u32),
                    model: narrow16(cols),
                }
            })
            .collect()
    }

    // -- frame submission --------------------------------------------------

    /// Encode and submit one frame into `target`.
    ///
    /// `target_format` must be an sRGB format (`Bgra8UnormSrgb` or
    /// `Rgba8UnormSrgb`): the tonemap shader emits linear ACES and the
    /// target's transfer function encodes on write.
    ///
    /// Draw ordering: non-blend draws in the order given, then
    /// `AlphaMode::Blend` draws (stable partition — sort blended items
    /// back-to-front on the CPU side when they overlap).
    pub fn render(
        &self,
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        frame: &FrameInput,
    ) -> Result<(), DrawError> {
        let tonemap_pipeline = match target_format {
            wgpu::TextureFormat::Bgra8UnormSrgb => &self.tonemap_bgra,
            wgpu::TextureFormat::Rgba8UnormSrgb => &self.tonemap_rgba,
            other => return Err(DrawError::UnsupportedTarget(other)),
        };
        let tonemap_bind = match target_format {
            wgpu::TextureFormat::Bgra8UnormSrgb => self
                .tonemap_bgra_bind
                .as_ref()
                .expect("static binds initialized"),
            _ => self
                .tonemap_rgba_bind
                .as_ref()
                .expect("static binds initialized"),
        };

        if frame.draws.len() > MAX_DRAW_SLOTS {
            return Err(DrawError::TooManyDraws { got: frame.draws.len() });
        }
        if !frame.draws.is_empty() && !self.has_model {
            return Err(DrawError::NoModel);
        }
        for d in frame.draws {
            if d.mesh as usize >= self.meshes.len() {
                return Err(DrawError::TableOutOfRange { what: "mesh", index: d.mesh });
            }
            let mat = d.material.unwrap_or(self.default_material);
            if mat as usize >= self.material_slots.len() {
                return Err(DrawError::TableOutOfRange { what: "material", index: mat });
            }
        }

        // ---- uniforms ---------------------------------------------------
        let inv_view_proj = (frame.proj * frame.view).inverse();
        let flags = (if frame.aerial { 1.0f32 } else { 0.0 })
            + (if frame.sun_disc { 2.0 } else { 0.0 });
        let fu = FrameUniforms {
            view: gpu_mat(&frame.view),
            proj: gpu_mat(&frame.proj),
            inv_view_proj: gpu_mat(&inv_view_proj),
            sky_from_rel: gpu_mat(&Mat4::from_quat(frame.sky_from_rel)),
            sun_dir_sky: [frame.sun_dir_sky.x, frame.sun_dir_sky.y, frame.sun_dir_sky.z, 0.0],
            sun_irradiance: [
                frame.sun_irradiance.x,
                frame.sun_irradiance.y,
                frame.sun_irradiance.z,
                0.0,
            ],
            time_wind: [frame.time_s, frame.wind.x, frame.wind.y, frame.wind.z],
            params: [frame.exposure, flags, frame.camera_alt_m, frame.atmosphere_height_m],
        };
        self.queue.write_buffer(&self.frame_buf, 0, cast_slice(&[fu]));

        if !frame.draws.is_empty() {
            let mut slots: Vec<u8> = vec![0u8; frame.draws.len() * DRAW_SLOT_STRIDE as usize];
            for (i, d) in frame.draws.iter().enumerate() {
                let m = Mat4::from_cols_array_2d(&d.model);
                let normal = m.inverse().transpose();
                let du = DrawUniforms {
                    model: d.model,
                    normal: gpu_mat(&normal),
                    material: d.material.unwrap_or(self.default_material),
                    _pad: [0; 3],
                };
                let bytes = cast_slice(std::slice::from_ref(&du));
                let off = i * DRAW_SLOT_STRIDE as usize;
                slots[off..off + bytes.len()].copy_from_slice(bytes);
            }
            self.queue.write_buffer(&self.draw_buf, 0, &slots);
        }
        self.queue.write_buffer(&self.cloud_buf, 0, cast_slice(&[frame.clouds]));
        self.queue.write_buffer(
            &self.exposure_buf,
            0,
            cast_slice(&[frame.exposure, 0.0f32, 0.0, 0.0]),
        );

        // ---- stable partition into opaque / blend (with slot indices) ---
        let mut opaque: Vec<(usize, &DrawItem)> = Vec::new();
        let mut blend: Vec<(usize, &DrawItem)> = Vec::new();
        for (i, d) in frame.draws.iter().enumerate() {
            let mat = d.material.unwrap_or(self.default_material);
            if self.material_slots[mat as usize].blend {
                blend.push((i, d));
            } else {
                opaque.push((i, d));
            }
        }

        let frame_bind = self.frame_bind.as_ref().expect("static binds initialized");
        let sky_bind = self.sky_bind.as_ref().expect("static binds initialized");
        let clouds_bind = self.clouds_bind.as_ref().expect("static binds initialized");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // ---- pass 1: sky → opaques → blend ------------------------------
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_a.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, sky_bind, &[]);
            pass.draw(0..3, 0..1);

            // state machine to avoid redundant rebinds
            // (pipeline key: blend × double_sided)
            let mut bound: Option<(bool, bool)> = None;
            let mut bound_mat: Option<u32> = None;
            let mut bound_offset: Option<u32> = None;

            let mut draw_group = |pass: &mut wgpu::RenderPass<'_>,
                                  list: &[(usize, &DrawItem)],
                                  blend: bool| {
                for &(slot, d) in list {
                    let mat_idx = d.material.unwrap_or(self.default_material);
                    let mslot = &self.material_slots[mat_idx as usize];
                    let key = (blend, mslot.double_sided);
                    if bound != Some(key) {
                        pass.set_pipeline(if blend {
                            if key.1 {
                                &self.pbr_blend_nc
                            } else {
                                &self.pbr_blend
                            }
                        } else if key.1 {
                            &self.pbr_opaque_nc
                        } else {
                            &self.pbr_opaque
                        });
                        pass.set_bind_group(0, frame_bind, &[0]);
                        bound = Some(key);
                        bound_offset = Some(0);
                        bound_mat = None;
                    }
                    if bound_mat != Some(mat_idx) {
                        pass.set_bind_group(1, &mslot.bind_group, &[]);
                        bound_mat = Some(mat_idx);
                    }
                    let offset = (slot as u64 * DRAW_SLOT_STRIDE) as u32;
                    if bound_offset != Some(offset) {
                        pass.set_bind_group(0, frame_bind, &[offset]);
                        bound_offset = Some(offset);
                    }
                    let mesh = &self.meshes[d.mesh as usize];
                    pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            };

            draw_group(&mut pass, &opaque, false);
            draw_group(&mut pass, &blend, true);
        }

        // ---- depth copy (texture used as attachment is done) ------------
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.depth.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.depth_copy.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
        );

        // ---- pass 2: volumetric clouds ----------------------------------
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clouds"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hdr_b.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.clouds_pipeline);
            pass.set_bind_group(0, clouds_bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // ---- pass 3: tonemap → sRGB target -------------------------------
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(tonemap_pipeline);
            pass.set_bind_group(0, tonemap_bind, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        Ok(())
    }
}

fn wrap_code(w: WrapMode) -> u8 {
    match w {
        WrapMode::Clamp => 1,
        WrapMode::Mirror => 2,
        WrapMode::Repeat => 0,
    }
}

fn filter_code(f: FilterMode) -> u8 {
    match f {
        FilterMode::Nearest => 0,
        FilterMode::Linear => 1,
    }
}
