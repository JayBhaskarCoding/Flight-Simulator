//! sim-render integration tests: full glTF/GLB roundtrips, the 64→32
//! floating-origin upload path, and a GPU smoke test that *skips*
//! cleanly when the sandbox has no adapter.

use glam::{DQuat, DVec3, Mat4, Quat, Vec3};
use sim_render::gltf::{parse_bytes, AlphaMode};
use sim_render::graph::{FrameInput, ModelHandle, Renderer};
use sim_render::shaders;
use sim_render::types::{CloudUniforms, DrawItem};
use sim_render::{generate_noise_volume, LutPayloads};
use sim_atmosphere::sky::{SkyLuts, SkyParams};
use sim_geo::{Ecef, FloatingOrigin, LatLonAlt, WGS84};

// ---------------------------------------------------------------------------
// helpers: build a complete GLB in memory
// ---------------------------------------------------------------------------

fn base64_encode(data: &[u8]) -> String {
    const ALPH: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPH[(n >> 18) as usize & 63] as char);
        out.push(ALPH[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ALPH[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPH[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Encode a tiny RGBA PNG (1×1 red pixel) with the `png` crate.
fn tiny_png() -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, 1, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().expect("png header");
        w.write_image_data(&[255, 0, 0, 255]).expect("png data");
    }
    buf
}

/// A minimal GLB: one triangle, non-indexed, with POSITION/NORMAL/
/// TEXCOORD_0, a PBR material with factors (no textures), and a
/// two-node hierarchy (parent translate + child rotate/scale).
fn build_glb() -> Vec<u8> {
    // --- binary chunk ------------------------------------------------
    // 0..36   positions (3 × vec3 f32)
    // 36..72  normals
    // 72..96  uvs
    let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let normals: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let uvs: [f32; 6] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let mut bin: Vec<u8> = Vec::new();
    for v in positions {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    for v in normals {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    for v in uvs {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    // --- JSON ---------------------------------------------------------
    let json = format!(
        r#"{{
        "asset": {{"version": "2.0", "generator": "sim-render-test"}},
        "scene": 0,
        "scenes": [{{"nodes": [0]}}],
        "nodes": [
            {{"children": [1], "translation": [0.0, 2.0, 0.0], "mesh": 0}},
            {{"translation": [0.5, 0.0, 0.0], "scale": [2.0, 2.0, 2.0]}}
        ],
        "meshes": [{{"primitives": [{{"attributes": {{
            "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2
        }}, "material": 0}}]}}],
        "materials": [{{
            "name": "paint",
            "pbrMetallicRoughness": {{
                "baseColorFactor": [0.8, 0.1, 0.1, 1.0],
                "metallicFactor": 0.25,
                "roughnessFactor": 0.75
            }},
            "emissiveFactor": [0.01, 0.01, 0.0],
            "alphaMode": "MASK",
            "alphaCutoff": 0.4,
            "doubleSided": true
        }}],
        "accessors": [
            {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]}},
            {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}},
            {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}}
        ],
        "bufferViews": [
            {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
            {{"buffer": 0, "byteOffset": 36, "byteLength": 36}},
            {{"buffer": 0, "byteOffset": 72, "byteLength": 24}}
        ],
        "buffers": [{{"byteLength": {bin_len}}}]
        }}"#,
        bin_len = bin.len()
    );

    // --- GLB container -------------------------------------------------
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let mut out = Vec::new();
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes()); // magic glTF
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // JSON
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes()); // BIN
    out.extend_from_slice(&bin);
    out
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn glb_roundtrip_builds_hierarchy_material_and_geometry() {
    let glb = build_glb();
    let model = parse_bytes(&glb).expect("glb parses");

    assert_eq!(model.meshes.len(), 1, "one primitive");
    let mesh = &model.meshes[0];
    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.indices, vec![0, 1, 2]);
    assert_eq!(mesh.vertices[1].position, [1.0, 0.0, 0.0]);
    assert_eq!(mesh.vertices[0].uv, [0.0, 0.0]);

    // material factors
    assert_eq!(model.materials.len(), 1);
    let m = &model.materials[0];
    assert_eq!(m.name, "paint");
    assert_eq!(m.base_color_factor, [0.8, 0.1, 0.1, 1.0]);
    assert!((m.metallic_factor - 0.25).abs() < 1e-6);
    assert!((m.roughness_factor - 0.75).abs() < 1e-6);
    assert_eq!(m.alpha_mode, AlphaMode::Mask);
    assert!((m.alpha_cutoff - 0.4).abs() < 1e-6);
    assert!(m.double_sided);
    assert_eq!(m.emissive_factor, [0.01, 0.01, 0.0]);

    // hierarchy: scene root node0 (translate y+2, has mesh) with child
    // node1 (translate x+0.5, scale 2). Node1 has no mesh.
    assert_eq!(model.instances.len(), 1);
    let inst = &model.instances[0];
    assert_eq!(inst.mesh, 0);
    let t = inst.world.w_axis.truncate();
    assert_eq!(t, Vec3::new(0.0, 2.0, 0.0), "child transform not applied to mesh-less node");
}

#[test]
fn gltf_external_json_with_data_uri_png_image() {
    // JSON .gltf with a base64 buffer AND a data-uri PNG image wired to
    // a material texture.
    let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let bin: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
    let png = base64_encode(&tiny_png());
    let json = format!(
        r#"{{
        "asset": {{"version": "2.0"}},
        "scenes": [{{"nodes": [0]}}],
        "nodes": [{{"mesh": 0}}],
        "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}],
        "materials": [{{"pbrMetallicRoughness": {{
            "baseColorTexture": {{"index": 0}}, "baseColorFactor": [1.0, 1.0, 1.0, 1.0]
        }}, "alphaMode": "BLEND"}}],
        "textures": [{{"source": 0, "sampler": 0}}],
        "images": [{{"uri": "data:image/png;base64,{png}"}}],
        "samplers": [{{"magFilter": 9729, "minFilter": 9987,
                       "wrapS": 33071, "wrapT": 33648}}],
        "accessors": [{{"bufferView": 0, "componentType": 5126, "count": 3,
                        "type": "VEC3"}}],
        "bufferViews": [{{"buffer": 0, "byteOffset": 0, "byteLength": 36}}],
        "buffers": [{{"uri": "data:application/octet-stream;base64,{buf}",
                      "byteLength": 36}}]
        }}"#,
        png = png,
        buf = base64_encode(&bin)
    );
    let model = parse_bytes(json.as_bytes()).expect("gltf parses");

    assert_eq!(model.meshes.len(), 1);
    assert_eq!(model.meshes[0].vertices.len(), 3);
    assert_eq!(model.images.len(), 1, "png decoded");
    assert_eq!(model.images[0].width, 1);
    assert_eq!(model.images[0].height, 1);
    assert_eq!(model.images[0].rgba, [255, 0, 0, 255]);
    // sampler plumbing: clamp/mirror honored
    assert_eq!(model.samplers[0].wrap_s, sim_render::WrapMode::Clamp);
    assert_eq!(model.samplers[0].wrap_t, sim_render::WrapMode::Mirror);
    // texture bound on the material + BLEND mode
    assert_eq!(model.materials[0].base_color_texture, Some(0));
    assert_eq!(model.materials[0].alpha_mode, AlphaMode::Blend);
}

#[test]
fn sparse_accessor_overrides_are_applied() {
    // 14 vertices (POSITION, f32) + a SCALAR u16 strip index accessor
    // whose base values are all 7, with sparse overrides at [1,3]
    // replacing them by 11 and 13 → effective indices [7,11,7,13].
    let mut pos_bytes = Vec::new();
    for v in 0..14u32 {
        pos_bytes.extend_from_slice(&(v as f32).to_le_bytes());
        pos_bytes.extend_from_slice(&0.0f32.to_le_bytes());
        pos_bytes.extend_from_slice(&0.0f32.to_le_bytes());
    }
    let mut base = Vec::new();
    for _ in 0..4 {
        base.extend_from_slice(&7u16.to_le_bytes());
    }
    let mut idx = Vec::new();
    idx.extend_from_slice(&1u16.to_le_bytes());
    idx.extend_from_slice(&3u16.to_le_bytes());
    let mut vals = Vec::new();
    vals.extend_from_slice(&11u16.to_le_bytes());
    vals.extend_from_slice(&13u16.to_le_bytes());

    let mut bin = pos_bytes;
    let base_off = bin.len();
    bin.extend_from_slice(&base);
    let idx_off = bin.len();
    bin.extend_from_slice(&idx);
    let val_off = bin.len();
    bin.extend_from_slice(&vals);

    let json = format!(
        r#"{{
        "asset": {{"version": "2.0"}},
        "scenes": [{{"nodes": [0]}}],
        "nodes": [{{"mesh": 0}}],
        "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}},
            "indices": 1, "mode": 5}}]}}],
        "accessors": [
            {{"bufferView": 0, "componentType": 5126, "count": 14, "type": "VEC3"}},
            {{"bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR",
              "sparse": {{"count": 2,
                  "indices": {{"bufferView": 2, "componentType": 5123}},
                  "values": {{"bufferView": 3}}}}}}
        ],
        "bufferViews": [
            {{"buffer": 0, "byteOffset": 0, "byteLength": {pos_len}}},
            {{"buffer": 0, "byteOffset": {base_off}, "byteLength": 8}},
            {{"buffer": 0, "byteOffset": {idx_off}, "byteLength": 4}},
            {{"buffer": 0, "byteOffset": {val_off}, "byteLength": 4}}
        ],
        "buffers": [{{"uri": "data:application/octet-stream;base64,{buf}",
                      "byteLength": {len}}}]
        }}"#,
        pos_len = 14 * 12,
        base_off = base_off,
        idx_off = idx_off,
        val_off = val_off,
        buf = base64_encode(&bin),
        len = bin.len()
    );
    let model = parse_bytes(json.as_bytes()).expect("sparse doc parses");
    assert_eq!(model.instances.len(), 1);
    // effective strip [7,11,7,13] → triangle 0 degenerate (7,11,7 skipped
    // as a==c? no: a=7,b=11,c=7 → a==c → degenerate → skipped);
    // triangle 1 (i=1): (11,7,13), odd → reversed (11,13,7)
    assert_eq!(model.meshes[0].indices, vec![11, 13, 7]);
    assert_eq!(model.meshes[0].vertices.len(), 14);
}

#[test]
fn unsupported_required_extension_fails_loudly() {
    let json = r#"{
        "asset": {"version": "2.0"},
        "extensionsRequired": ["KHR_materials_transmission"]
    }"#;
    let err = parse_bytes(json.as_bytes()).unwrap_err();
    assert_eq!(
        err,
        sim_render::GltfError::UnsupportedExtension("KHR_materials_transmission".into())
    );
}

#[test]
fn triangle_fan_and_strip_modes_convert() {
    // fan of 4 vertices → 2 triangles; strip handled by unit tests but
    // exercise the full primitive pipeline here with a fan.
    let pos: [f32; 12] = [
        0.0, 0.0, 0.0, // 0 center
        1.0, 0.0, 0.0, // 1
        1.0, 1.0, 0.0, // 2
        0.0, 1.0, 0.0, // 3
    ];
    let mut bin: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx: [u16; 4] = [0, 1, 2, 3];
    let idx_off = bin.len();
    for i in idx {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    let json = format!(
        r#"{{
        "asset": {{"version": "2.0"}},
        "scenes": [{{"nodes": [0]}}],
        "nodes": [{{"mesh": 0}}],
        "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}},
            "indices": 1, "mode": 6}}]}}],
        "accessors": [
            {{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"}},
            {{"bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR"}}
        ],
        "bufferViews": [
            {{"buffer": 0, "byteOffset": 0, "byteLength": 48}},
            {{"buffer": 0, "byteOffset": {idx_off}, "byteLength": 8}}
        ],
        "buffers": [{{"uri": "data:application/octet-stream;base64,{buf}",
                      "byteLength": {len}}}]
        }}"#,
        idx_off = idx_off,
        buf = base64_encode(&bin),
        len = bin.len()
    );
    let model = parse_bytes(json.as_bytes()).expect("fan doc");
    assert_eq!(model.meshes[0].indices, vec![0, 1, 2, 0, 2, 3]);
}

/// The 64→32 law end-to-end: an object 100 m east of the camera, at
/// Earth-scale coordinates, must land at ~`(0, 0, 100)` relative in
/// `f32` with centimetre accuracy — even though the absolute ECEF
/// values are ~6.4e6.
#[test]
fn draws_from_gltf_rebases_in_f64_before_narrowing() {
    let glb = build_glb();
    let model = parse_bytes(&glb).expect("glb");

    let cam_llh = LatLonAlt::from_degrees(40.0, -74.0, 500.0);
    let cam_ecef = WGS84.geodetic_to_ecef(&cam_llh);
    let origin = FloatingOrigin::new(cam_ecef, DQuat::IDENTITY);

    // object 100 m east (≈ +Y at this latitude in ECEF — pick a plain
    // ECEF offset for exactness: +Z direction is close to east here;
    // use an exact offset vector instead)
    let root_pos = cam_ecef.as_dvec3() + DVec3::new(0.0, 100.0, 0.0);
    let draws = Renderer::draws_from_gltf(
        &model,
        &origin,
        root_pos,
        Quat::IDENTITY,
        Vec3::ONE,
    );
    assert_eq!(draws.len(), 1);
    // translation column = root − camera + node translation (0,2,0)
    // = (0, 102, 0), narrowed last
    let tx = draws[0].model[3][0];
    let ty = draws[0].model[3][1];
    let tz = draws[0].model[3][2];
    assert!(tx.abs() < 1e-3, "tx = {tx}");
    assert!((ty - 102.0).abs() < 1e-3, "ty = {ty}");
    assert!(tz.abs() < 1e-3, "tz = {tz}");

    // moving the camera 10 m further along the same offset must shrink
    // the reported translation to 90 m
    let origin2 = FloatingOrigin::new(
        Ecef::from(cam_ecef.as_dvec3() + DVec3::new(0.0, 10.0, 0.0)),
        DQuat::IDENTITY,
    );
    let draws2 = Renderer::draws_from_gltf(&model, &origin2, root_pos, Quat::IDENTITY, Vec3::ONE);
    assert!((draws2[0].model[3][1] - 92.0).abs() < 1e-3);
}

#[test]
fn lut_payloads_and_shaders_agree() {
    // pack payloads from real LUTs and re-sample through the CPU
    // mirrors of the WGSL formulas (parities against sim-atmosphere are
    // unit-tested; this asserts payload → mirror round-trip integrity).
    let luts = SkyLuts::new(&SkyParams::default());
    let payload = LutPayloads::from_sky_luts(&luts);
    let d = payload.dims();
    assert_eq!((d[0].0, d[0].1), (256, 64));

    // sample at a texel centre: payload must equal the raw grid value
    let raw = luts.transmittance_grid();
    let px = payload.transmittance[10 * 256 + 7];
    assert_eq!([px[0], px[1], px[2]], raw[10 * 256 + 7]);
    assert_eq!(px[3], 1.0);

    if let Err(e) = shaders::validate_all() {
        panic!("{e}");
    }
}

/// GPU smoke test — skips when no adapter exists (the usual case in a
/// headless sandbox). When a GPU *is* present it builds every pipeline,
/// loads a model and submits a frame.
#[test]
fn renderer_headless_frame_or_skip() {
    let mut renderer = match Renderer::new(64, 64) {
        Ok(r) => r,
        Err(sim_render::DrawError::AdapterNotFound) => {
            eprintln!("skipping GPU smoke test: no wgpu adapter");
            return;
        }
        Err(e) => panic!("unexpected renderer error: {e}"),
    };

    let glb = build_glb();
    let model = parse_bytes(&glb).expect("glb");
    let handle: ModelHandle = renderer.set_model(&model).expect("model upload");
    assert_eq!(handle.default_material, 1); // 1 doc material + default slot

    let noise = generate_noise_volume(32, 42);
    renderer.set_noise(&noise);
    let luts = SkyLuts::new(&SkyParams::default());
    renderer.set_sky_luts(&luts);

    let (target_tex, view) =
        renderer.create_target(wgpu::TextureFormat::Bgra8UnormSrgb, 64, 64);
    let frame = FrameInput {
        view: Mat4::IDENTITY,
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 1e7),
        sky_from_rel: Quat::IDENTITY,
        sun_dir_sky: Vec3::new(0.3, 0.7, 0.4).normalize(),
        sun_irradiance: Vec3::splat(1.5),
        camera_alt_m: 300.0,
        atmosphere_height_m: 64_000.0,
        exposure: 1.0,
        aerial: true,
        sun_disc: true,
        time_s: 0.0,
        wind: Vec3::new(5.0, 0.0, 2.0),
        clouds: CloudUniforms::default(),
        draws: &[DrawItem {
            mesh: 0,
            material: Some(0),
            model: Mat4::IDENTITY.to_cols_array_2d(),
        }],
    };
    renderer
        .render(&view, wgpu::TextureFormat::Bgra8UnormSrgb, &frame)
        .expect("frame submits");
    let _ = target_tex;

    // a non-sRGB target is rejected
    let (_, plain_view) =
        renderer.create_target(wgpu::TextureFormat::Rgba8Unorm, 4, 4);
    let err = renderer.render(&plain_view, wgpu::TextureFormat::Rgba8Unorm, &frame);
    assert!(matches!(err, Err(sim_render::DrawError::UnsupportedTarget(_))));
}
