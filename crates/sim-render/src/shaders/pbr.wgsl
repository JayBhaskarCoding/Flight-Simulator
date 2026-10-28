// Physically-based forward shading (glTF metallic-roughness model).
//
// CPU/GPU contract (see src/types.rs): every buffer is camera-relative
// f32. All constants that double as units (solar irradiance, exposure)
// arrive through `frame`.

struct Frame {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // rotation: camera-relative ECEF -> planetary sky frame (Y-up)
    sky_from_rel: mat4x4<f32>,
    // xyz: unit direction to the sun (sky frame), w: unused
    sun_dir_sky: vec4<f32>,
    // rgb: top-of-atmosphere solar irradiance, w: unused
    sun_irradiance: vec4<f32>,
    // x: time (s), yzw: wind (m/s, sky frame)
    time_wind: vec4<f32>,
    // x: exposure, y: flags (1 = aerial on, 2 = sun disc), z: camera
    // altitude above sea level (m), w: unused
    params: vec4<f32>,
};

struct Draw {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    material: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Material {
    base_color: vec4<f32>,
    // rgb: emissive factor, w: metallic factor
    emissive_metallic: vec4<f32>,
    // x: roughness, y: normal scale, z: occlusion strength, w: cutoff
    roughness: vec4<f32>,
    // texture slots (0xFFFFFFFF = none): base, metallic-roughness,
    // normal, emissive
    textures: vec4<u32>,
    // x: occlusion slot, y: alpha mode (0 opaque, 1 mask, 2 blend),
    // z: double-sided (0/1), w: reserved
    aux: vec4<u32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> draw: Draw;
@group(0) @binding(2) var<storage, read> materials: array<Material>;
@group(0) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(4) var aerial_trans_lut: texture_2d<f32>;
@group(0) @binding(5) var aerial_radiance_lut: texture_2d<f32>;
@group(1) @binding(0) var material_sampler: sampler;
@group(1) @binding(1) var base_color_tex: texture_2d<f32>;
@group(1) @binding(2) var mr_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var occlusion_tex: texture_2d<f32>;
@group(1) @binding(5) var emissive_tex: texture_2d<f32>;

const NO_TEX: u32 = 0xFFFFFFFFu;
const PI: f32 = 3.14159265359;
const AERIAL_MAX_DIST: f32 = 80000.0;
const AERIAL_DIST: i32 = 16;
const AERIAL_ZEN: i32 = 16;
const TRANS_W: i32 = 256;
const TRANS_H: i32 = 64;
const SKY_W: i32 = 32;
const SKY_H: i32 = 32;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_rel: vec3<f32>,
    @location(1) normal_rel: vec3<f32>,
    @location(2) tangent_rel: vec3<f32>,
    @location(3) bitangent_rel: vec3<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) @interpolate(flat) material: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world = draw.model * vec4<f32>(in.position, 1.0);
    out.world_rel = world.xyz;
    let n = normalize((draw.normal_mat * vec4<f32>(in.normal, 0.0)).xyz);
    out.normal_rel = n;
    // w == 0 marks "no tangents in the asset": the fragment stage
    // derives them from screen-space derivatives (MikkTSpace-style).
    let t4 = draw.normal_mat * vec4<f32>(in.tangent.xyz, 0.0);
    let t = normalize(t4.xyz);
    out.tangent_rel = t;
    out.bitangent_rel = cross(n, t) * select(1.0, -1.0, in.tangent.w < 0.0);
    out.uv = in.uv;
    out.material = draw.material;
    out.clip = frame.proj * (frame.view * world);
    return out;
}

// --- LUT sampling helpers (exact mirrors of sim-atmosphere) ---

fn sample_transmittance(altitude_m: f32, cos_zenith: f32) -> vec3<f32> {
    let ay = clamp(max(altitude_m, 0.0) / max(frame.params.w, 1.0), 0.0, 1.0);
    let ax = clamp((cos_zenith + 1.0) * 0.5, 0.0, 1.0);
    let fy = ay * f32(TRANS_H - 1);
    let fx = ax * f32(TRANS_W - 1);
    let y0 = i32(floor(fy));
    let x0 = i32(floor(fx));
    let y1 = min(y0 + 1, TRANS_H - 1);
    let x1 = min(x0 + 1, TRANS_W - 1);
    let ty = fy - f32(y0);
    let tx = fx - f32(x0);
    let p00 = textureLoad(transmittance_lut, vec2<i32>(x0, y0), 0).rgb;
    let p10 = textureLoad(transmittance_lut, vec2<i32>(x1, y0), 0).rgb;
    let p01 = textureLoad(transmittance_lut, vec2<i32>(x0, y1), 0).rgb;
    let p11 = textureLoad(transmittance_lut, vec2<i32>(x1, y1), 0).rgb;
    return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

// Aerial perspective result: path transmittance + in-scattered
// radiance for one view segment.
struct Aerial {
    trans: vec3<f32>,
    radiance: vec3<f32>,
};

// Aerial perspective for a straight segment of `distance_m` whose
// direction (sky frame) has vertical cosine `cos_zenith`. Formulas
// mirror `SkyLuts::aerial_perspective` exactly (texel-centre grid,
// `-0.5` offset then clamp, manual bilinear).
fn sample_aerial(distance_m: f32, cos_zenith: f32) -> Aerial {
    let d = clamp(distance_m, 0.0, AERIAL_MAX_DIST);
    let v = clamp(d / AERIAL_MAX_DIST * f32(AERIAL_DIST) - 0.5, 0.0, f32(AERIAL_DIST - 1));
    let theta = acos(clamp(cos_zenith, -1.0, 1.0));
    let z = clamp(theta / PI * f32(AERIAL_ZEN) - 0.5, 0.0, f32(AERIAL_ZEN - 1));
    let d0 = i32(floor(v));
    let d1 = min(d0 + 1, AERIAL_DIST - 1);
    let z0 = i32(floor(z));
    let z1 = min(z0 + 1, AERIAL_ZEN - 1);
    let tv = v - f32(d0);
    let tz = z - f32(z0);

    let t00 = textureLoad(aerial_trans_lut, vec2<i32>(d0, z0), 0).rgb;
    let t10 = textureLoad(aerial_trans_lut, vec2<i32>(d1, z0), 0).rgb;
    let t01 = textureLoad(aerial_trans_lut, vec2<i32>(d0, z1), 0).rgb;
    let t11 = textureLoad(aerial_trans_lut, vec2<i32>(d1, z1), 0).rgb;
    let r00 = textureLoad(aerial_radiance_lut, vec2<i32>(d0, z0), 0).rgb;
    let r10 = textureLoad(aerial_radiance_lut, vec2<i32>(d1, z0), 0).rgb;
    let r01 = textureLoad(aerial_radiance_lut, vec2<i32>(d0, z1), 0).rgb;
    let r11 = textureLoad(aerial_radiance_lut, vec2<i32>(d1, z1), 0).rgb;

    let t = mix(mix(t00, t10, tv), mix(t01, t11, tv), tz);
    let r = mix(mix(r00, r10, tv), mix(r01, r11, tv), tz);
    return Aerial(t, r);
}

// --- light transport ---

fn ggx_distribution(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let nh = max(dot(n, h), 0.0);
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn smith_geometry_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let k = (roughness + 1.0);
    let k2 = k * k / 8.0;
    let nv = max(dot(n, v), 1e-5);
    let nl = max(dot(n, l), 1e-5);
    let gv = nv / (nv * (1.0 - k2) + k2);
    let gl = nl / (nl * (1.0 - k2) + k2);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let mat = materials[in.material];

    // --- alpha ---
    var albedo = mat.base_color;
    if (mat.textures.x != NO_TEX) {
        let t = textureSample(base_color_tex, material_sampler, in.uv);
        albedo = albedo * t;
    }
    let alpha_mode = mat.aux.y;
    if (alpha_mode == 1u) {
        if (albedo.w < mat.roughness.w) {
            discard;
        }
    } else if (alpha_mode == 2u && albedo.w < 0.003) {
        discard;
    }

    // --- normals (double-sided flips; missing tangents derive from
    //     screen derivatives) ---
    var n = normalize(in.normal_rel);
    if (mat.aux.z == 1u && !front) {
        n = -n;
    }
    var t = in.tangent_rel;
    var b = in.bitangent_rel;
    if (dot(t, t) < 1e-8) {
        let dp1 = dpdx(in.world_rel);
        let dp2 = dpdy(in.world_rel);
        let duv1 = dpdx(in.uv);
        let duv2 = dpdy(in.uv);
        let det = duv1.x * duv2.y - duv2.x * duv1.y;
        if (abs(det) > 1e-12) {
            let inv = 1.0 / det;
            t = normalize((dp1 * duv2.y - dp2 * duv1.y) * inv);
            b = normalize((dp2 * duv1.x - dp1 * duv2.x) * inv);
        } else {
            // degenerate UVs: build an orthonormal basis around n
            let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
            t = normalize(cross(up, n));
            b = cross(n, t);
        }
    }
    if (mat.textures.z != NO_TEX) {
        let tn = textureSample(normal_tex, material_sampler, in.uv).xyz * 2.0 - 1.0;
        let n_t = vec3<f32>(tn.xy * mat.roughness.y, tn.z);
        n = normalize(t * n_t.x + b * n_t.y + n * n_t.z);
        if (dot(n, n) < 1e-8) {
            n = normalize(in.normal_rel);
        }
    }

    // --- material params ---
    var mr = vec2<f32>(1.0, 1.0); // (metallic, roughness)
    if (mat.textures.y != NO_TEX) {
        let t = textureSample(mr_tex, material_sampler, in.uv);
        mr = vec2<f32>(mat.emissive_metallic.w * t.b, mat.roughness.x * t.g);
    } else {
        mr = vec2<f32>(mat.emissive_metallic.w, mat.roughness.x);
    }
    let metallic = clamp(mr.x, 0.0, 1.0);
    let roughness = clamp(mr.y, 0.045, 1.0);

    // --- view / light directions (camera is the rel origin) ---
    let v_dir = normalize(-in.world_rel);
    let sun_rel = normalize(transpose(mat3x3<f32>(
        frame.sky_from_rel[0].xyz,
        frame.sky_from_rel[1].xyz,
        frame.sky_from_rel[2].xyz,
    )) * frame.sun_dir_sky.xyz);
    let l_dir = sun_rel;
    let h = normalize(v_dir + l_dir);
    let nl = max(dot(n, l_dir), 0.0);
    let nv = max(dot(n, v_dir), 1e-5);

    // --- direct sun (attenuated by aerial-perspective transmittance) ---
    let dist = length(in.world_rel);
    let dir_sky = (frame.sky_from_rel * vec4<f32>(in.world_rel / max(dist, 1e-4), 0.0)).xyz;
    var sun_att = vec3<f32>(1.0);
    if ((i32(frame.params.y) & 1) != 0) {
        sun_att = sample_aerial(dist, dir_sky.y).trans;
    }

    let albedo_lin = albedo.rgb;
    let f0 = mix(vec3<f32>(0.04), albedo_lin, metallic);
    let fres = fresnel_schlick(max(dot(h, v_dir), 0.0), f0);
    let d = ggx_distribution(n, h, roughness);
    let g = smith_geometry_ggx(n, v_dir, l_dir, roughness);
    let specular = (d * g) * fres / max(4.0 * nv * nl, 1e-4);
    let kd = (vec3<f32>(1.0) - fres) * (1.0 - metallic);
    let diffuse = kd * albedo_lin / PI;

    let radiance = frame.sun_irradiance.rgb * sun_att;
    var color = (diffuse + specular) * radiance * nl;

    // --- ambient (hemisphere approximation of the sky) ---
    var ao = 1.0;
    if (mat.aux.x != NO_TEX) {
        let s = textureSample(occlusion_tex, material_sampler, in.uv).r;
        ao = mix(1.0, s, mat.roughness.z);
    }
    let sky_amb = vec3<f32>(0.35, 0.45, 0.65) * 0.5 * (0.5 + 0.5 * n.y);
    color += sky_amb * albedo_lin * (1.0 - metallic * 0.7) * ao;

    // --- emissive ---
    var emissive = mat.emissive_metallic.xyz;
    if (mat.textures.w != NO_TEX) {
        emissive = emissive * textureSample(emissive_tex, material_sampler, in.uv).rgb;
    }
    color += emissive;

    // --- aerial in-scatter ---
    if ((i32(frame.params.y) & 1) != 0) {
        let aerial = sample_aerial(dist, dir_sky.y);
        color = color * aerial.trans + aerial.radiance;
    }

    color = color * frame.params.x;
    if (alpha_mode == 2u) {
        return vec4<f32>(color, albedo.w);
    }
    return vec4<f32>(color, 1.0);
}
