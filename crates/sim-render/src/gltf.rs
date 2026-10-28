//! Clean-room glTF 2.0 loader.
//!
//! Scope (per the glTF 2.0 specification + Khronos JSON schema):
//!
//! * `.gltf` (JSON, external / data-URI buffers) and `.glb` containers;
//! * every accessor component type, `SCALAR`…`MAT4` element types,
//!   `byteStride`, `normalized` integer conversions and **sparse**
//!   accessors;
//! * primitive modes `TRIANGLES`, `TRIANGLE_STRIP` and
//!   `TRIANGLE_FAN` (the latter two are indexed into triangle lists);
//! * node hierarchies with `matrix` **or** `TRS` composition;
//! * the PBR metallic-roughness material model (factors, textures,
//!   alpha modes, double-sidedness);
//! * PNG **and** JPEG image decoding (glTF core requirements),
//!   `data:` URIs with base64 payloads, buffer views and file URIs;
//! * required-extension checking: unknown entries in
//!   `extensionsRequired` fail loudly instead of silently mis-rendering.
//!
//! Positions are mesh-local `f32`; world placement happens through
//! [`crate::types::DrawItem::model`] (camera-relative, f32) so the
//! 64/32-bit law is never violated by this module.

use crate::types::{GpuMaterial, GpuVertex};
use glam::{Mat4, Quat, Vec2, Vec3};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// error type
// ---------------------------------------------------------------------------

/// Any structural or data error found while loading a glTF asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GltfError {
    /// Malformed container / JSON / unexpected end of data.
    Malformed(&'static str),
    /// A referenced index is out of range.
    BadIndex { what: &'static str, index: usize },
    /// Unsupported accessor `componentType` / `type` / `mode`.
    Unsupported(&'static str, i64),
    /// A required extension this clean-room renderer does not implement.
    UnsupportedExtension(String),
    /// Buffer `uri` could not be fetched (bad base64 or I/O).
    BufferUri(String),
    /// Image payload could not be decoded (png/jpeg crate error).
    ImageDecode(String),
    /// A primitive carries no position attribute (spec violation).
    MissingPosition,
    /// `TRIANGLE_FAN`/`TRIANGLE_STRIP` geometry too short to form a
    /// triangle.
    DegeneratePrimitive,
}

impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GltfError::Malformed(m) => write!(f, "malformed glTF: {m}"),
            GltfError::BadIndex { what, index } => write!(f, "glTF index out of range: {what}[{index}]"),
            GltfError::Unsupported(what, v) => write!(f, "unsupported glTF {what}: {v}"),
            GltfError::UnsupportedExtension(e) => write!(f, "unsupported required glTF extension: {e}"),
            GltfError::BufferUri(u) => write!(f, "cannot fetch glTF buffer uri: {u}"),
            GltfError::ImageDecode(m) => write!(f, "glTF image decode failed: {m}"),
            GltfError::MissingPosition => write!(f, "glTF primitive has no POSITION attribute"),
            GltfError::DegeneratePrimitive => write!(f, "glTF primitive too small for its mode"),
        }
    }
}

impl std::error::Error for GltfError {}

// ---------------------------------------------------------------------------
// JSON schema (subset actually consumed)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Doc {
    #[serde(default)]
    asset: Asset,
    #[serde(default)]
    scenes: Vec<Scene>,
    #[serde(default)]
    scene: Option<usize>,
    #[serde(default)]
    nodes: Vec<Node>,
    #[serde(default)]
    meshes: Vec<Mesh>,
    #[serde(default)]
    materials: Vec<Material>,
    #[serde(default)]
    textures: Vec<Texture>,
    #[serde(default)]
    images: Vec<Image>,
    #[serde(default)]
    samplers: Vec<Sampler>,
    #[serde(default)]
    accessors: Vec<Accessor>,
    #[serde(rename = "bufferViews", default)]
    buffer_views: Vec<BufferView>,
    #[serde(default)]
    buffers: Vec<Buffer>,
    #[serde(rename = "extensionsRequired", default)]
    extensions_required: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Asset {
    #[serde(default)]
    version: String,
}

#[derive(Debug, Default, Deserialize)]
struct Scene {
    #[serde(default)]
    nodes: Vec<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Node {
    #[serde(default)]
    mesh: Option<usize>,
    #[serde(default)]
    children: Vec<usize>,
    #[serde(default)]
    matrix: Option<[f32; 16]>,
    #[serde(default)]
    translation: Option<[f32; 3]>,
    #[serde(default)]
    rotation: Option<[f32; 4]>,
    #[serde(default)]
    scale: Option<[f32; 3]>,
}

#[derive(Debug, Default, Deserialize)]
struct Mesh {
    #[serde(default)]
    primitives: Vec<Primitive>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Primitive {
    #[serde(rename = "attributes", default)]
    attributes: BTreeMap<String, usize>,
    #[serde(default)]
    indices: Option<usize>,
    #[serde(default)]
    material: Option<usize>,
    /// glTF primitive mode; default (absent) = 4 = `TRIANGLES`.
    #[serde(default)]
    mode: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Material {
    #[serde(default)]
    name: Option<String>,
    #[serde(rename = "pbrMetallicRoughness", default)]
    pbr: Pbr,
    #[serde(rename = "normalTexture", default)]
    normal_texture: Option<NormalTexture>,
    #[serde(rename = "occlusionTexture", default)]
    occlusion_texture: Option<OcclusionTexture>,
    #[serde(rename = "emissiveTexture", default)]
    emissive_texture: Option<TextureInfo>,
    #[serde(rename = "emissiveFactor", default)]
    emissive_factor: Option<[f32; 3]>,
    #[serde(rename = "alphaMode", default)]
    alpha_mode: Option<String>,
    #[serde(rename = "alphaCutoff", default)]
    alpha_cutoff: Option<f32>,
    #[serde(rename = "doubleSided", default)]
    double_sided: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Pbr {
    #[serde(rename = "baseColorFactor", default)]
    base_color_factor: Option<[f32; 4]>,
    #[serde(rename = "baseColorTexture", default)]
    base_color_texture: Option<TextureInfo>,
    #[serde(rename = "metallicFactor", default)]
    metallic_factor: Option<f32>,
    #[serde(rename = "roughnessFactor", default)]
    roughness_factor: Option<f32>,
    #[serde(rename = "metallicRoughnessTexture", default)]
    metallic_roughness_texture: Option<TextureInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextureInfo {
    index: usize,
    #[serde(default)]
    tex_coord: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NormalTexture {
    index: usize,
    #[serde(default)]
    tex_coord: u32,
    #[serde(default)]
    scale: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OcclusionTexture {
    index: usize,
    #[serde(default)]
    tex_coord: u32,
    #[serde(default)]
    strength: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Texture {
    #[serde(default)]
    sampler: Option<usize>,
    #[serde(default)]
    source: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Image {
    #[serde(default)]
    uri: Option<String>,
    #[serde(rename = "mimeType", default)]
    mime_type: Option<String>,
    #[serde(rename = "bufferView", default)]
    buffer_view: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sampler {
    #[serde(rename = "magFilter", default)]
    mag_filter: Option<u32>,
    #[serde(rename = "minFilter", default)]
    min_filter: Option<u32>,
    #[serde(rename = "wrapS", default)]
    wrap_s: Option<u32>,
    #[serde(rename = "wrapT", default)]
    wrap_t: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Accessor {
    #[serde(rename = "bufferView", default)]
    buffer_view: Option<usize>,
    #[serde(default)]
    byte_offset: Option<usize>,
    #[serde(rename = "componentType", default)]
    component_type: Option<i64>,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    count: Option<usize>,
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    sparse: Option<Sparse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sparse {
    count: usize,
    indices: SparseIndices,
    values: SparseValues,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SparseIndices {
    #[serde(rename = "bufferView")]
    buffer_view: usize,
    #[serde(default, rename = "byteOffset")]
    byte_offset: usize,
    #[serde(rename = "componentType")]
    component_type: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SparseValues {
    #[serde(rename = "bufferView")]
    buffer_view: usize,
    #[serde(default, rename = "byteOffset")]
    byte_offset: usize,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BufferView {
    #[serde(default)]
    buffer: usize,
    #[serde(default)]
    byte_offset: usize,
    #[serde(default)]
    byte_length: usize,
    #[serde(default)]
    byte_stride: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Buffer {
    #[serde(default)]
    uri: Option<String>,
    byte_length: Option<usize>,
}

// ---------------------------------------------------------------------------
// base64 (RFC 4648, alphabet + padding; no external crate needed)
// ---------------------------------------------------------------------------

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(input.len() * 3 / 4 + 3);
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    for &c in input.as_bytes() {
        if c == b'=' || c == b'\n' || c == b'\r' {
            continue;
        }
        let v = val(c)?;
        acc = (acc << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
            acc &= (1 << nbits) - 1;
        }
    }
    // leftover bits must be zero for canonical encodings
    if acc != 0 {
        return None;
    }
    Some(out)
}

/// Split a `data:` URI into `(media_type, decoded_bytes)`.
fn decode_data_uri(uri: &str) -> Option<(String, Vec<u8>)> {
    let rest = uri.strip_prefix("data:")?;
    let (header, payload) = rest.split_once(',')?;
    let parts: Vec<&str> = header.split(';').collect();
    let mime = parts.first().copied().unwrap_or("application/octet-stream").to_string();
    let is_b64 = parts.contains(&"base64");
    if is_b64 {
        Some((mime, base64_decode(payload)?))
    } else {
        // percent-encoded payload
        let bytes = payload.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
                i += 3;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        Some((mime, out))
    }
}

// ---------------------------------------------------------------------------
// decoded assets
// ---------------------------------------------------------------------------

/// An RGBA8 image decoded from a glTF `image`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// CPU-side mesh geometry ready for GPU upload.
#[derive(Debug, Clone, Default)]
pub struct GltfMesh {
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
}

/// CPU-side material (factors + texture slots into
/// [`GltfModel::images`]).
#[derive(Debug, Clone, PartialEq)]
pub struct GltfMaterial {
    pub name: String,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub base_color_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
}

impl Default for GltfMaterial {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0, 0.0, 0.0],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
        }
    }
}

impl GltfMaterial {
    /// Pack into the GPU layout (texture slots become `u32`, `!0` for
    /// unused; `aux.w` is reserved and set to 0).
    pub fn to_gpu(&self) -> GpuMaterial {
        let slot = |t: Option<usize>| -> u32 {
            match t {
                Some(i) => i as u32,
                None => GpuMaterial::NO_TEXTURE,
            }
        };
        GpuMaterial {
            base_color: self.base_color_factor,
            emissive_metallic: [
                self.emissive_factor[0],
                self.emissive_factor[1],
                self.emissive_factor[2],
                self.metallic_factor,
            ],
            roughness: [
                self.roughness_factor,
                self.normal_scale,
                self.occlusion_strength,
                self.alpha_cutoff,
            ],
            textures: [
                slot(self.base_color_texture),
                slot(self.metallic_roughness_texture),
                slot(self.normal_texture),
                slot(self.emissive_texture),
            ],
            aux: [
                slot(self.occlusion_texture),
                self.alpha_mode as u32,
                self.double_sided as u32,
                0,
            ],
        }
    }
}

/// Alpha handling mode (numeric values match the GPU `aux[1]` slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AlphaMode {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
}

/// One node instance of a mesh, with its world matrix (mesh-local →
/// scene space; for this renderer the caller re-bases it into
/// camera-relative f32 before drawing).
#[derive(Debug, Clone, Copy)]
pub struct GltfInstance {
    pub mesh: usize,
    pub material_override: Option<usize>,
    pub world: Mat4,
}

/// Texture sampler settings requested by the asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    Clamp,
    Repeat,
    Mirror,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GltfSampler {
    pub mag: FilterMode,
    pub min: FilterMode,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
}

impl Default for GltfSampler {
    fn default() -> Self {
        Self {
            mag: FilterMode::Linear,
            min: FilterMode::Linear,
            wrap_s: WrapMode::Repeat,
            wrap_t: WrapMode::Repeat,
        }
    }
}

/// A fully decoded glTF document.
#[derive(Debug, Clone)]
pub struct GltfModel {
    pub meshes: Vec<GltfMesh>,
    pub materials: Vec<GltfMaterial>,
    pub images: Vec<DecodedImage>,
    /// Per-image sampler (`images.len()` entries; defaults applied).
    pub samplers: Vec<GltfSampler>,
    pub instances: Vec<GltfInstance>,
    /// `scene = 0` semantics: which scene index is active.
    pub default_scene: usize,
}

// ---------------------------------------------------------------------------
// loader
// ---------------------------------------------------------------------------

/// Size of a single component in bytes.
fn component_size(component_type: i64) -> Result<usize, GltfError> {
    match component_type {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        other => Err(GltfError::Unsupported("componentType", other)),
    }
}

/// Number of scalar elements in an accessor `type`.
fn element_components(kind: &str) -> Result<usize, GltfError> {
    match kind {
        "SCALAR" => Ok(1),
        "VEC2" => Ok(2),
        "VEC3" => Ok(3),
        "VEC4" => Ok(4),
        "MAT2" => Ok(4),
        "MAT3" => Ok(9),
        "MAT4" => Ok(16),
        _ => Err(GltfError::Unsupported("accessor type", 0)),
    }
}

/// Byte offset of a component inside one accessor element.
///
/// Implements the glTF matrix packing rule: columns of `MAT2`/`MAT3`
/// with 1- or 2-byte components are padded to 4-byte boundaries (the
/// same rule as `std140` matrix layout).
fn component_offset_in_element(kind: &str, component_size: usize, comp: usize) -> usize {
    match kind {
        "MAT2" | "MAT3" if component_size < 4 => {
            let rows = if kind == "MAT2" { 2 } else { 3 };
            let col = comp / rows;
            let row = comp % rows;
            let col_stride = (rows * component_size + 3) & !3;
            col * col_stride + row * component_size
        }
        _ => comp * component_size,
    }
}

/// Total byte size of one accessor element (with matrix padding).
fn element_stride(kind: &str, component_size: usize, comps: usize) -> usize {
    match kind {
        "MAT2" | "MAT3" if component_size < 4 => {
            let rows = if kind == "MAT2" { 2 } else { 3 };
            let cols = comps / rows;
            let col_stride = (rows * component_size + 3) & !3;
            cols * col_stride
        }
        _ => comps * component_size,
    }
}

/// Convert one raw component to `f32`, applying the `normalized`
/// rules of the glTF spec for integer types.
fn component_to_f32(raw: [u8; 4], component_type: i64, normalized: bool) -> f32 {
    match component_type {
        5126 => f32::from_le_bytes(raw),
        5121 => {
            let v = raw[0];
            if normalized {
                v as f32 / 255.0
            } else {
                v as f32
            }
        }
        5120 => {
            let v = i8::from_le_bytes([raw[0]]) as f32;
            if normalized {
                (v / 127.0).max(-1.0)
            } else {
                v
            }
        }
        5123 => {
            let v = u16::from_le_bytes([raw[0], raw[1]]);
            if normalized {
                v as f32 / 65535.0
            } else {
                v as f32
            }
        }
        5122 => {
            let v = i16::from_le_bytes([raw[0], raw[1]]) as f32;
            if normalized {
                (v / 32767.0).max(-1.0)
            } else {
                v
            }
        }
        5125 => u32::from_le_bytes(raw) as f32,
        _ => 0.0,
    }
}

fn read_at(bytes: &[u8], off: usize, n: usize) -> Option<&[u8]> {
    bytes.get(off..off + n)
}

/// Decode one accessor into `f32` scalars (row-major, `count ×
/// components` values). Applies sparse overrides and `normalized`
/// conversions.
fn decode_accessor_f32(
    doc: &Doc,
    data: &[Vec<u8>],
    acc_idx: usize,
) -> Result<Vec<f32>, GltfError> {
    let acc = doc
        .accessors
        .get(acc_idx)
        .ok_or(GltfError::BadIndex { what: "accessors", index: acc_idx })?;
    let component_type = acc.component_type.ok_or(GltfError::Malformed("accessor without componentType"))?;
    let count = acc.count.ok_or(GltfError::Malformed("accessor without count"))?;
    let kind = acc.kind.as_deref().ok_or(GltfError::Malformed("accessor without type"))?;
    let comps = element_components(kind)?;
    let csize = component_size(component_type)?;
    let stride_elem = element_stride(kind, csize, comps);

    let mut out = vec![0.0f32; count * comps];

    // base data (bufferView is optional per spec: zeros when absent)
    if let Some(bv_idx) = acc.buffer_view {
        let bv = doc
            .buffer_views
            .get(bv_idx)
            .ok_or(GltfError::BadIndex { what: "bufferViews", index: bv_idx })?;
        let buf = data
            .get(bv.buffer)
            .ok_or(GltfError::BadIndex { what: "buffers", index: bv.buffer })?;
        let stride = bv.byte_stride.filter(|s| *s > 0).unwrap_or(stride_elem);
        if stride < stride_elem {
            return Err(GltfError::Malformed("byteStride smaller than element"));
        }
        let base = bv.byte_offset + acc.byte_offset.unwrap_or(0);
        for i in 0..count {
            let off = base + i * stride;
            for c in 0..comps {
                let eoff = off + component_offset_in_element(kind, csize, c);
                let raw_bytes = read_at(buf, eoff, csize)
                    .ok_or(GltfError::Malformed("accessor reads past buffer end"))?;
                let mut raw = [0u8; 4];
                raw[..csize].copy_from_slice(raw_bytes);
                out[i * comps + c] = component_to_f32(raw, component_type, acc.normalized);
            }
        }
    }

    // sparse overrides
    if let Some(sp) = &acc.sparse {
        let idx_type = sp.indices.component_type;
        let isize_ = component_size(idx_type)?;
        if isize_ != 1 && isize_ != 2 && isize_ != 4 {
            return Err(GltfError::Unsupported("sparse index componentType", idx_type));
        }
        let ibv = doc
            .buffer_views
            .get(sp.indices.buffer_view)
            .ok_or(GltfError::BadIndex { what: "bufferViews", index: sp.indices.buffer_view })?;
        let vbv = doc
            .buffer_views
            .get(sp.values.buffer_view)
            .ok_or(GltfError::BadIndex { what: "bufferViews", index: sp.values.buffer_view })?;
        let ibuf = data
            .get(ibv.buffer)
            .ok_or(GltfError::BadIndex { what: "buffers", index: ibv.buffer })?;
        let vbuf = data
            .get(vbv.buffer)
            .ok_or(GltfError::BadIndex { what: "buffers", index: vbv.buffer })?;
        if sp.count > count {
            return Err(GltfError::Malformed("sparse count exceeds accessor count"));
        }
        for s in 0..sp.count {
            let ioff = ibv.byte_offset + sp.indices.byte_offset + s * isize_;
            let ib = read_at(ibuf, ioff, isize_).ok_or(GltfError::Malformed("sparse index read"))?;
            let mut raw = [0u8; 4];
            raw[..isize_].copy_from_slice(ib);
            let target = match isize_ {
                1 => raw[0] as usize,
                2 => u16::from_le_bytes([raw[0], raw[1]]) as usize,
                _ => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize,
            };
            if target >= count {
                return Err(GltfError::Malformed("sparse index out of range"));
            }
            let voff = vbv.byte_offset + sp.values.byte_offset + s * stride_elem;
            for c in 0..comps {
                let voff_c = voff + component_offset_in_element(kind, csize, c);
                let vb = read_at(vbuf, voff_c, csize)
                    .ok_or(GltfError::Malformed("sparse value read"))?;
                let mut raw = [0u8; 4];
                raw[..csize].copy_from_slice(vb);
                out[target * comps + c] = component_to_f32(raw, component_type, acc.normalized);
            }
        }
    }

    Ok(out)
}

/// Decode a `SCALAR` index accessor as `u32` indices.
///
/// Integers are read raw (never round-tripped through `f32`): `u32`
/// indices beyond 2²⁴ would not survive the float conversion.
fn decode_indices(doc: &Doc, data: &[Vec<u8>], acc_idx: usize) -> Result<Vec<u32>, GltfError> {
    let acc = doc
        .accessors
        .get(acc_idx)
        .ok_or(GltfError::BadIndex { what: "accessors", index: acc_idx })?;
    let ct = acc.component_type.ok_or(GltfError::Malformed("accessor without componentType"))?;
    if ct == 5126 {
        return Err(GltfError::Unsupported("index componentType", ct));
    }
    let count = acc.count.ok_or(GltfError::Malformed("accessor without count"))?;
    let kind = acc.kind.as_deref().ok_or(GltfError::Malformed("accessor without type"))?;
    if element_components(kind)? != 1 {
        return Err(GltfError::Malformed("index accessor must be SCALAR"));
    }
    let csize = component_size(ct)?;
    let mut out = Vec::with_capacity(count);

    if let Some(bv_idx) = acc.buffer_view {
        let bv = doc
            .buffer_views
            .get(bv_idx)
            .ok_or(GltfError::BadIndex { what: "bufferViews", index: bv_idx })?;
        let buf = data
            .get(bv.buffer)
            .ok_or(GltfError::BadIndex { what: "buffers", index: bv.buffer })?;
        let stride = bv.byte_stride.filter(|s| *s > 0).unwrap_or(csize);
        let base = bv.byte_offset + acc.byte_offset.unwrap_or(0);
        for i in 0..count {
            let off = base + i * stride;
            let raw = read_at(buf, off, csize)
                .ok_or(GltfError::Malformed("index accessor read"))?;
            let v = match csize {
                1 => raw[0] as u32,
                2 => u16::from_le_bytes([raw[0], raw[1]]) as u32,
                _ => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            };
            out.push(v);
        }
    } else {
        out.resize(count, 0);
    }

    if let Some(sp) = &acc.sparse {
        let isize_ = component_size(sp.indices.component_type)?;
        let ibv = doc
            .buffer_views
            .get(sp.indices.buffer_view)
            .ok_or(GltfError::BadIndex { what: "bufferViews", index: sp.indices.buffer_view })?;
        let vbv = doc
            .buffer_views
            .get(sp.values.buffer_view)
            .ok_or(GltfError::BadIndex { what: "bufferViews", index: sp.values.buffer_view })?;
        let ibuf = data
            .get(ibv.buffer)
            .ok_or(GltfError::BadIndex { what: "buffers", index: ibv.buffer })?;
        let vbuf = data
            .get(vbv.buffer)
            .ok_or(GltfError::BadIndex { what: "buffers", index: vbv.buffer })?;
        if sp.count > count {
            return Err(GltfError::Malformed("sparse count exceeds accessor count"));
        }
        for s in 0..sp.count {
            let ioff = ibv.byte_offset + sp.indices.byte_offset + s * isize_;
            let ib = read_at(ibuf, ioff, isize_).ok_or(GltfError::Malformed("sparse index read"))?;
            let target = match isize_ {
                1 => ib[0] as usize,
                2 => u16::from_le_bytes([ib[0], ib[1]]) as usize,
                _ => u32::from_le_bytes([ib[0], ib[1], ib[2], ib[3]]) as usize,
            };
            if target >= count {
                return Err(GltfError::Malformed("sparse index out of range"));
            }
            let voff = vbv.byte_offset + sp.values.byte_offset + s * csize;
            let vb = read_at(vbuf, voff, csize).ok_or(GltfError::Malformed("sparse value read"))?;
            out[target] = match csize {
                1 => vb[0] as u32,
                2 => u16::from_le_bytes([vb[0], vb[1]]) as u32,
                _ => u32::from_le_bytes([vb[0], vb[1], vb[2], vb[3]]),
            };
        }
    }

    Ok(out)
}

/// Convert a triangle strip to a triangle list (spec winding rule:
/// odd triangles reverse).
fn strip_to_list(strip: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(strip.len().saturating_sub(2) * 3);
    for i in 0..strip.len().saturating_sub(2) {
        let a = strip[i];
        let b = strip[i + 1];
        let c = strip[i + 2];
        if a == b || b == c || a == c {
            continue; // degenerate — skip per spec recommendation
        }
        if i % 2 == 0 {
            out.extend_from_slice(&[a, b, c]);
        } else {
            out.extend_from_slice(&[a, c, b]);
        }
    }
    out
}

/// Convert a triangle fan to a triangle list.
fn fan_to_list(fan: &[u32]) -> Result<Vec<u32>, GltfError> {
    if fan.len() < 3 {
        return Err(GltfError::DegeneratePrimitive);
    }
    let mut out = Vec::with_capacity((fan.len() - 2) * 3);
    for i in 1..fan.len() - 1 {
        out.extend_from_slice(&[fan[0], fan[i], fan[i + 1]]);
    }
    Ok(out)
}

/// Compute smooth per-vertex normals from an indexed triangle list.
fn smooth_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        if a >= positions.len() || b >= positions.len() || c >= positions.len() {
            continue;
        }
        let n = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normals[a] += n;
        normals[b] += n;
        normals[c] += n;
    }
    normals.iter().map(|n| n.normalize_or_zero()).collect()
}

/// Build one mesh from one primitive.
fn build_primitive(
    doc: &Doc,
    data: &[Vec<u8>],
    prim: &Primitive,
) -> Result<GltfMesh, GltfError> {
    let pos_acc = *prim.attributes.get("POSITION").ok_or(GltfError::MissingPosition)?;
    // Skinned geometry (JOINTS_0/WEIGHTS_0) would silently render in
    // bind pose without a skinning pipeline — reject loudly instead.
    if prim.attributes.contains_key("JOINTS_0") || prim.attributes.contains_key("WEIGHTS_0") {
        return Err(GltfError::Unsupported("skinned primitive", 0));
    }
    let pos_f = decode_accessor_f32(doc, data, pos_acc)?;
    if pos_f.len() % 3 != 0 {
        return Err(GltfError::Malformed("POSITION is not VEC3"));
    }
    let vertex_count = pos_f.len() / 3;
    let positions: Vec<Vec3> = pos_f
        .chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect();

    let normals: Vec<Vec3> = match prim.attributes.get("NORMAL") {
        Some(&a) => {
            let n = decode_accessor_f32(doc, data, a)?;
            if n.len() / 3 != vertex_count {
                return Err(GltfError::Malformed("NORMAL count != POSITION count"));
            }
            n.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
        }
        None => Vec::new(), // filled after indices are known
    };

    let uvs: Vec<Vec2> = match prim.attributes.get("TEXCOORD_0") {
        Some(&a) => {
            let t = decode_accessor_f32(doc, data, a)?;
            if t.len() / 2 != vertex_count {
                return Err(GltfError::Malformed("TEXCOORD_0 count != POSITION count"));
            }
            t.chunks_exact(2).map(|c| Vec2::new(c[0], c[1])).collect()
        }
        None => vec![Vec2::ZERO; vertex_count],
    };

    let tangents: Vec<[f32; 4]> = match prim.attributes.get("TANGENT") {
        Some(&a) => {
            let t = decode_accessor_f32(doc, data, a)?;
            if t.len() / 4 != vertex_count {
                return Err(GltfError::Malformed("TANGENT count != POSITION count"));
            }
            t.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
        }
        None => vec![[0.0; 4]; vertex_count], // w = 0 ⇒ shader derives TBN
    };

    // indices + mode
    let mode = prim.mode.unwrap_or(4);
    let mut indices: Vec<u32> = match prim.indices {
        Some(i) => decode_indices(doc, data, i)?,
        None => (0..vertex_count as u32).collect(),
    };
    match mode {
        4 => {}
        5 => indices = strip_to_list(&indices),
        6 => indices = fan_to_list(&indices)?,
        other => return Err(GltfError::Unsupported("primitive mode", other as i64)),
    }
    if indices.len() < 3 {
        return Err(GltfError::DegeneratePrimitive);
    }
    if let Some(&mx) = indices.iter().max() {
        if mx as usize >= vertex_count {
            return Err(GltfError::Malformed("index references a vertex past POSITION"));
        }
    }

    let normals = if normals.is_empty() {
        smooth_normals(&positions, &indices)
    } else {
        normals
    };

    let vertices = (0..vertex_count)
        .map(|i| GpuVertex {
            position: positions[i].to_array(),
            normal: normals[i].to_array(),
            tangent: tangents[i],
            uv: uvs[i].to_array(),
        })
        .collect();

    Ok(GltfMesh { vertices, indices })
}

fn node_local_matrix(node: &Node) -> Mat4 {
    if let Some(m) = &node.matrix {
        // glTF matrices are column-major arrays of 16.
        Mat4::from_cols_array(m)
    } else {
        let t = node.translation.unwrap_or([0.0, 0.0, 0.0]);
        let r = node.rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let s = node.scale.unwrap_or([1.0, 1.0, 1.0]);
        Mat4::from_scale_rotation_translation(
            Vec3::from(s),
            Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize(),
            Vec3::from(t),
        )
    }
}

fn wrap_from_code(code: u32) -> WrapMode {
    match code {
        33071 => WrapMode::Clamp,
        33648 => WrapMode::Mirror,
        _ => WrapMode::Repeat,
    }
}

fn filter_from_code(code: u32) -> FilterMode {
    match code {
        9728 | 9984 | 9986 => FilterMode::Nearest,
        _ => FilterMode::Linear,
    }
}

fn alpha_mode_from_str(s: Option<&str>) -> Result<AlphaMode, GltfError> {
    match s.unwrap_or("OPAQUE") {
        "OPAQUE" => Ok(AlphaMode::Opaque),
        "MASK" => Ok(AlphaMode::Mask),
        "BLEND" => Ok(AlphaMode::Blend),
        _ => Err(GltfError::Unsupported("alphaMode", 0)),
    }
}

fn decode_image_bytes(bytes: &[u8], hint: Option<&str>) -> Result<DecodedImage, GltfError> {
    let is_png = bytes.len() >= 8 && bytes.starts_with(&[0x89, b'P', b'N', b'G']);
    let is_jpeg = bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF;
    let mime_ok = |m: Option<&str>| -> bool {
        matches!(m, Some("image/png") | Some("image/jpeg") | Some("image/jpg"))
    };
    if is_png || hint == Some("image/png") {
        decode_png(bytes)
    } else if is_jpeg || mime_ok(hint) {
        decode_jpeg(bytes)
    } else if hint.is_some() {
        Err(GltfError::Unsupported("image mime", 0))
    } else {
        // no sniff, no hint: try png then jpeg
        decode_png(bytes).or_else(|_| decode_jpeg(bytes))
    }
}

fn decode_png(bytes: &[u8]) -> Result<DecodedImage, GltfError> {
    use png::Transformations;
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(Transformations::EXPAND | Transformations::STRIP_16);
    let mut reader =
        decoder.read_info().map_err(|e| GltfError::ImageDecode(e.to_string()))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| GltfError::ImageDecode(e.to_string()))?;
    let w = info.width;
    let h = info.height;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for px in buf.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            out
        }
        png::ColorType::Grayscale => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for &v in &buf {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for px in buf.chunks_exact(2) {
                out.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
            }
            out
        }
        png::ColorType::Indexed => {
            return Err(GltfError::ImageDecode(
                "palette image survived EXPAND transformation".into(),
            ))
        }
    };
    if rgba.len() != w as usize * h as usize * 4 {
        return Err(GltfError::ImageDecode("unexpected png frame size".into()));
    }
    Ok(DecodedImage { width: w, height: h, rgba })
}

fn decode_jpeg(bytes: &[u8]) -> Result<DecodedImage, GltfError> {
    let mut decoder =
        jpeg_decoder::Decoder::new(Cursor::new(bytes));
    let pixels = decoder.decode().map_err(|e| GltfError::ImageDecode(e.to_string()))?;
    let info = decoder.info().ok_or_else(|| GltfError::ImageDecode("jpeg without info".into()))?;
    let (w, h) = (info.width as u32, info.height as u32);
    match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for &v in &pixels {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            Ok(DecodedImage { width: w, height: h, rgba: out })
        }
        jpeg_decoder::PixelFormat::L16 => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for px in pixels.chunks_exact(2) {
                let v = (u16::from_le_bytes([px[0], px[1]]) >> 8) as u8;
                out.extend_from_slice(&[v, v, v, 255]);
            }
            Ok(DecodedImage { width: w, height: h, rgba: out })
        }
        jpeg_decoder::PixelFormat::RGB24 => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for px in pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            Ok(DecodedImage { width: w, height: h, rgba: out })
        }
        jpeg_decoder::PixelFormat::CMYK32 => {
            // Adobe JPEG CMYK is stored inverted; un-invert then convert.
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for px in pixels.chunks_exact(4) {
                let (c, m, y, k) = (px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32);
                let r = (c * k) / 255;
                let g = (m * k) / 255;
                let b = (y * k) / 255;
                out.extend_from_slice(&[r as u8, g as u8, b as u8, 255]);
            }
            Ok(DecodedImage { width: w, height: h, rgba: out })
        }
    }
}

/// Where the loader may look for non-data-URI buffers and images.
#[derive(Debug, Clone)]
pub struct GltfSource<'a> {
    /// Directory for relative URIs (GLTF file's parent).
    pub base_dir: Option<&'a std::path::Path>,
}

impl<'a> GltfSource<'a> {
    /// In-memory only: relative file URIs are rejected.
    pub fn memory() -> Self {
        Self { base_dir: None }
    }

    /// Load from a filesystem path (`.gltf` or `.glb`); buffers and
    /// images may still be data-URIs or relative to the file.
    pub fn from_file(path: &'a std::path::Path) -> Result<GltfModel, GltfError> {
        let bytes = std::fs::read(path).map_err(|e| GltfError::BufferUri(e.to_string()))?;
        let src = Self { base_dir: path.parent() };
        if bytes.len() >= 12 && &bytes[0..4] == b"glTF" {
            src.parse_glb(&bytes)
        } else {
            let text = std::str::from_utf8(&bytes).map_err(|_| GltfError::Malformed("utf-8"))?;
            src.parse_json(text, None)
        }
    }

    fn fetch_uri(&self, uri: &str) -> Result<Vec<u8>, GltfError> {
        if let Some((_, bytes)) = decode_data_uri(uri) {
            return Ok(bytes);
        }
        match self.base_dir {
            Some(dir) => {
                let p = dir.join(uri);
                std::fs::read(&p).map_err(|e| GltfError::BufferUri(format!("{p:?}: {e}")))
            }
            None => Err(GltfError::BufferUri("relative uri without base dir".into())),
        }
    }

    /// Parse a GLB container: 12-byte header + JSON/BIN chunks.
    fn parse_glb(&self, bytes: &[u8]) -> Result<GltfModel, GltfError> {
        let u32le = |b: &[u8], o: usize| -> Result<u32, GltfError> {
            let s = b.get(o..o + 4).ok_or(GltfError::Malformed("glb truncated"))?;
            Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        };
        let magic = u32le(bytes, 0)?;
        if magic != 0x4654_6C67 {
            return Err(GltfError::Malformed("bad GLB magic"));
        }
        let version = u32le(bytes, 4)?;
        if version != 2 {
            return Err(GltfError::Unsupported("glb version", version as i64));
        }
        let total = u32le(bytes, 8)? as usize;
        if total > bytes.len() {
            return Err(GltfError::Malformed("glb length exceeds file"));
        }
        let mut off = 12usize;
        let mut json: Option<Vec<u8>> = None;
        let mut bin: Option<Vec<u8>> = None;
        while off + 8 <= total {
            let len = u32le(bytes, off)? as usize;
            let kind = u32le(bytes, off + 4)?;
            let body = off + 8;
            if body + len > bytes.len() {
                return Err(GltfError::Malformed("glb chunk exceeds file"));
            }
            match kind {
                0x4E4F_534A => json = Some(bytes[body..body + len].to_vec()),
                0x004E_4942 => bin = Some(bytes[body..body + len].to_vec()),
                _ => {} // unknown chunks are skipped per spec
            }
            off = body + len;
        }
        let json = json.ok_or(GltfError::Malformed("glb without JSON chunk"))?;
        let text = std::str::from_utf8(&json).map_err(|_| GltfError::Malformed("glb json utf-8"))?;
        self.parse_json(text, bin)
    }

    /// Parse the JSON document (with optional GLB BIN chunk) into a
    /// [`GltfModel`].
    fn parse_json(&self, text: &str, bin_chunk: Option<Vec<u8>>) -> Result<GltfModel, GltfError> {
        let doc: Doc =
            serde_json::from_str(text).map_err(|_| GltfError::Malformed("json parse"))?;
        if !doc.asset.version.starts_with("2.") {
            return Err(GltfError::Unsupported("asset version", 0));
        }
        // No required extension is implemented — fail on the first
        // one (the list may hold many; reporting the first is enough).
        if let Some(ext) = doc.extensions_required.first() {
            return Err(GltfError::UnsupportedExtension(ext.clone()));
        }

        // buffers
        let mut data: Vec<Vec<u8>> = Vec::with_capacity(doc.buffers.len());
        for (i, buf) in doc.buffers.iter().enumerate() {
            let bytes = match &buf.uri {
                Some(u) => self.fetch_uri(u)?,
                None => bin_chunk.clone().ok_or_else(|| {
                    GltfError::BufferUri(format!("buffer {i} has no uri and no GLB BIN"))
                })?,
            };
            if let Some(len) = buf.byte_length {
                if bytes.len() < len {
                    return Err(GltfError::Malformed("buffer shorter than byteLength"));
                }
            }
            data.push(bytes);
        }

        // meshes (one GltfMesh per primitive; failures carry the mesh
        // index for diagnostics)
        let mut meshes = Vec::with_capacity(doc.meshes.len());
        for mesh in doc.meshes.iter() {
            for prim in &mesh.primitives {
                match build_primitive(&doc, &data, prim) {
                    Ok(m) => meshes.push(m),
                    Err(e) => return Err(e),
                }
            }
        }
        // primitive index → mesh index mapping (meshes were flattened
        // per-primitive above)
        let mut prim_to_mesh: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        {
            let mut running = 0usize;
            for (mi, mesh) in doc.meshes.iter().enumerate() {
                for (pi, _) in mesh.primitives.iter().enumerate() {
                    prim_to_mesh.insert((mi, pi), running);
                    running += 1;
                }
            }
        }

        // materials: kept verbatim (possibly empty); the renderer appends
        // a default-material slot and maps `material_override == None`
        // to it, which is what glTF specifies when a primitive omits
        // `material`.
        let materials: Vec<GltfMaterial> = doc
            .materials
            .iter()
            .map(|m| -> Result<GltfMaterial, GltfError> {
                let tex = |t: &Option<TextureInfo>| -> Result<Option<usize>, GltfError> {
                    match t {
                        Some(t) if t.tex_coord != 0 => {
                            Err(GltfError::Unsupported("texCoord set", t.tex_coord as i64))
                        }
                        Some(t) => Ok(Some(t.index)),
                        None => Ok(None),
                    }
                };
                let extra = |info: &Option<(usize, u32)>| -> Result<Option<usize>, GltfError> {
                    match info {
                        Some((_, tc)) if *tc != 0 => {
                            Err(GltfError::Unsupported("texCoord set", *tc as i64))
                        }
                        Some((idx, _)) => Ok(Some(*idx)),
                        None => Ok(None),
                    }
                };
                Ok(GltfMaterial {
                    name: m.name.clone().unwrap_or_default(),
                    base_color_factor: m.pbr.base_color_factor.unwrap_or([1.0; 4]),
                    metallic_factor: m.pbr.metallic_factor.unwrap_or(1.0),
                    roughness_factor: m.pbr.roughness_factor.unwrap_or(1.0),
                    emissive_factor: m.emissive_factor.unwrap_or([0.0; 3]),
                    normal_scale: m.normal_texture.as_ref().and_then(|t| t.scale).unwrap_or(1.0),
                    occlusion_strength: m
                        .occlusion_texture
                        .as_ref()
                        .and_then(|t| t.strength)
                        .unwrap_or(1.0),
                    alpha_mode: alpha_mode_from_str(m.alpha_mode.as_deref())?,
                    alpha_cutoff: m.alpha_cutoff.unwrap_or(0.5),
                    double_sided: m.double_sided.unwrap_or(false),
                    base_color_texture: tex(&m.pbr.base_color_texture)?,
                    metallic_roughness_texture: tex(&m.pbr.metallic_roughness_texture)?,
                    normal_texture: extra(
                        &m.normal_texture.as_ref().map(|t| (t.index, t.tex_coord)),
                    )?,
                    occlusion_texture: extra(
                        &m.occlusion_texture.as_ref().map(|t| (t.index, t.tex_coord)),
                    )?,
                    emissive_texture: extra(&m.emissive_texture.as_ref().map(|t| (t.index, t.tex_coord)))?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // images: `images[i]` is referenced by `textures[].source`
        let mut images = Vec::with_capacity(doc.images.len());
        for img in &doc.images {
            let bytes = if let Some(bv) = img.buffer_view {
                let v = doc
                    .buffer_views
                    .get(bv)
                    .ok_or(GltfError::BadIndex { what: "bufferViews", index: bv })?;
                let buf = data
                    .get(v.buffer)
                    .ok_or(GltfError::BadIndex { what: "buffers", index: v.buffer })?;
                buf.get(v.byte_offset..v.byte_offset + v.byte_length)
                    .ok_or(GltfError::Malformed("image bufferView out of range"))?
                    .to_vec()
            } else if let Some(uri) = &img.uri {
                if uri.starts_with("data:") {
                    decode_data_uri(uri)
                        .ok_or_else(|| GltfError::BufferUri("bad image data uri".into()))?
                        .1
                } else {
                    self.fetch_uri(uri)?
                }
            } else {
                return Err(GltfError::Malformed("image without uri or bufferView"));
            };
            images.push(decode_image_bytes(&bytes, img.mime_type.as_deref())?);
        }

        // samplers (one per image; default when the texture omits one)
        let mut samplers = vec![GltfSampler::default(); images.len()];
        for tex in &doc.textures {
            let Some(src) = tex.source else { continue };
            let s = tex.sampler.unwrap_or(0);
            if let Some(smp) = doc.samplers.get(s) {
                if src < samplers.len() {
                    samplers[src] = GltfSampler {
                        mag: smp.mag_filter.map(filter_from_code).unwrap_or(FilterMode::Linear),
                        min: smp.min_filter.map(filter_from_code).unwrap_or(FilterMode::Linear),
                        wrap_s: smp.wrap_s.map(wrap_from_code).unwrap_or(WrapMode::Repeat),
                        wrap_t: smp.wrap_t.map(wrap_from_code).unwrap_or(WrapMode::Repeat),
                    };
                }
            }
        }

        // node instances of the active scene
        let active_scene = doc.scene.unwrap_or(0);
        let mut instances = Vec::new();
        if let Some(scene) = doc.scenes.get(active_scene) {
            let mut stack: Vec<(usize, Mat4)> =
                scene.nodes.iter().map(|&n| (n, Mat4::IDENTITY)).collect();
            while let Some((ni, parent)) = stack.pop() {
                let node = doc
                    .nodes
                    .get(ni)
                    .ok_or(GltfError::BadIndex { what: "nodes", index: ni })?;
                let world = parent * node_local_matrix(node);
                if let Some(mesh_idx) = node.mesh {
                    // expand every primitive of the mesh as an instance
                    let mut pi = 0usize;
                    while doc.meshes.get(mesh_idx).and_then(|m| m.primitives.get(pi)).is_some() {
                        let mesh_id = *prim_to_mesh
                            .get(&(mesh_idx, pi))
                            .ok_or(GltfError::BadIndex { what: "primitives", index: pi })?;
                        let material_override =
                            doc.meshes[mesh_idx].primitives[pi].material;
                        instances.push(GltfInstance { mesh: mesh_id, material_override, world });
                        pi += 1;
                    }
                }
                for &c in &node.children {
                    stack.push((c, world));
                }
            }
        } else if !doc.nodes.is_empty() {
            return Err(GltfError::BadIndex { what: "scenes", index: active_scene });
        }

        Ok(GltfModel {
            meshes,
            materials,
            images,
            samplers,
            instances,
            default_scene: active_scene,
        })
    }
}

/// Convenience: load a glTF/GLB document from a byte buffer.
pub fn parse_bytes(bytes: &[u8]) -> Result<GltfModel, GltfError> {
    let src = GltfSource::memory();
    if bytes.len() >= 12 && &bytes[0..4] == b"glTF" {
        src.parse_glb(bytes)
    } else {
        let text = std::str::from_utf8(bytes).map_err(|_| GltfError::Malformed("utf-8"))?;
        src.parse_json(text, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_decode("TQ==").as_deref(), Some(&[b'M'][..]));
        assert_eq!(base64_decode("TWE=").as_deref(), Some(&[b'M', b'a'][..]));
        assert_eq!(base64_decode("TWFu").as_deref(), Some(&[b'M', b'a', b'n'][..]));
        assert_eq!(
            base64_decode("aGVsbG8gd29ybGQ=").as_deref(),
            Some(b"hello world".as_slice())
        );
        assert_eq!(base64_decode("!!"), None, "invalid alphabet");
    }

    #[test]
    fn strip_winding_follows_spec() {
        // strip 0,1,2 → (0,1,2); strip 0,1,2,3 → second triangle flips
        let l = strip_to_list(&[0, 1, 2, 3]);
        // triangle 1 keeps (1,2,3) vertex order but reverses winding
        assert_eq!(l, vec![0, 1, 2, 1, 3, 2]);
        // degenerate triangles are dropped: only the first survives
        let l = strip_to_list(&[0, 1, 2, 2, 3]);
        assert_eq!(l, vec![0, 1, 2]);
        // strips shorter than a triangle produce nothing
        assert!(strip_to_list(&[0, 1]).is_empty());
    }

    #[test]
    fn fan_converts_to_list() {
        let l = fan_to_list(&[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(l, vec![0, 1, 2, 0, 2, 3, 0, 3, 4]);
        assert!(fan_to_list(&[0, 1]).is_err());
    }

    #[test]
    fn normalized_short_decoding_matches_spec() {
        let mut doc = Doc::default();
        doc.buffer_views.push(BufferView {
            buffer: 0,
            byte_offset: 0,
            byte_length: 6,
            byte_stride: None,
        });
        doc.accessors.push(Accessor {
            buffer_view: Some(0),
            byte_offset: None,
            component_type: Some(5122), // SHORT
            normalized: true,
            count: Some(3),
            kind: Some("SCALAR".into()),
            sparse: None,
        });
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i16.to_le_bytes());
        bytes.extend_from_slice(&32767i16.to_le_bytes());
        bytes.extend_from_slice(&(-32768i16).to_le_bytes());
        let out = decode_accessor_f32(&doc, &[bytes], 0).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - (-1.0)).abs() < 1e-6);
    }
}
