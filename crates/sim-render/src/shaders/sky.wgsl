// Hillaire-style physical sky background.
//
// The transmittance / sky-view / aerial LUTs are computed once on the
// CPU by `sim-atmosphere::SkyLuts` (single-scatter quadrature) and
// uploaded as Rgba32Float textures; this shader reproduces their
// sampling formulas bit-for-bit (see `lut.rs` tests) and adds the sun
// disc, which is physically part of the same radiance integral.

struct Frame {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sky_from_rel: mat4x4<f32>,
    sun_dir_sky: vec4<f32>,
    sun_irradiance: vec4<f32>,
    time_wind: vec4<f32>,
    // x: exposure, y: flags (1 = aerial on, 2 = sun disc),
    // z: camera altitude (m), w: unused
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(3) var aerial_trans_lut: texture_2d<f32>;
@group(0) @binding(4) var aerial_radiance_lut: texture_2d<f32>;
@group(0) @binding(5) var lut_sampler: sampler;

const PI: f32 = 3.14159265359;
const TRANS_W: i32 = 256;
const TRANS_H: i32 = 64;
const SKY_W: i32 = 32;
const SKY_H: i32 = 32;
const AERIAL_MAX_DIST: f32 = 80000.0;
const AERIAL_DIST: i32 = 16;
const AERIAL_ZEN: i32 = 16;
// Sun angular radius (rad) — half-angle of the solar disc.
const SUN_ANGULAR_RADIUS: f32 = 0.004650534;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// Fullscreen triangle from vertex indices 0..3.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    out.clip = vec4<f32>(pos[vi], 0.0, 1.0);
    out.ndc = pos[vi];
    return out;
}

// Mirror of `SkyLuts::transmittance_rgb` (cell-centre build grid,
// `(H-1)`-normalised manual bilinear).
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

// Mirror of `SkyLuts::sky_view_rgb`: `v = acos(dir.y)/π`,
// `u = acos(∠(dir_horiz, sun_horiz))/π`, manual bilinear on the
// cell-centre grid.
fn sample_sky_view(dir_sky: vec3<f32>) -> vec3<f32> {
    let d = normalize(dir_sky);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let v = clamp(theta / PI, 0.0, 1.0);

    let sun = normalize(frame.sun_dir_sky.xyz);
    var sun_h = vec3<f32>(sun.x, 0.0, sun.z);
    if (length(sun_h) > 1e-9) {
        sun_h = normalize(sun_h);
    } else {
        sun_h = vec3<f32>(1.0, 0.0, 0.0);
    }
    var dir_h = vec3<f32>(d.x, 0.0, d.z);
    if (length(dir_h) > 1e-9) {
        dir_h = normalize(dir_h);
    } else {
        dir_h = vec3<f32>(1.0, 0.0, 0.0);
    }
    let cos_az = clamp(dot(dir_h, sun_h), -1.0, 1.0);
    let u = clamp(acos(cos_az) / PI, 0.0, 1.0);

    let fy = v * f32(SKY_H - 1);
    let fx = u * f32(SKY_W - 1);
    let y0 = i32(floor(fy));
    let x0 = i32(floor(fx));
    let y1 = min(y0 + 1, SKY_H - 1);
    let x1 = min(x0 + 1, SKY_W - 1);
    let ty = fy - f32(y0);
    let tx = fx - f32(x0);
    let p00 = textureLoad(sky_view_lut, vec2<i32>(x0, y0), 0).rgb;
    let p10 = textureLoad(sky_view_lut, vec2<i32>(x1, y0), 0).rgb;
    let p01 = textureLoad(sky_view_lut, vec2<i32>(x0, y1), 0).rgb;
    let p11 = textureLoad(sky_view_lut, vec2<i32>(x1, y1), 0).rgb;
    return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

// Aerial perspective result: path transmittance + in-scattered
// radiance for one view segment.
struct Aerial {
    trans: vec3<f32>,
    radiance: vec3<f32>,
};

// Mirror of `SkyLuts::aerial_perspective` (two grids: trans + rad).
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the world (camera-relative) ray from NDC.
    let clip = vec4<f32>(in.ndc, 1.0, 1.0);
    var world = frame.inv_view_proj * clip;
    world = world / world.w;
    let dir_rel = normalize(world.xyz);
    let dir_sky = normalize((frame.sky_from_rel * vec4<f32>(dir_rel, 0.0)).xyz);

    // Radiance arrives per unit solar irradiance from the LUT build;
    // scale by the actual irradiance (same factor the CPU applies).
    var color = sample_sky_view(dir_sky) * frame.sun_irradiance.rgb;

    // Direct sun disc (per-pixel solid angle → disc radiance).
    if ((i32(frame.params.y) & 2) != 0) {
        let sun = normalize(frame.sun_dir_sky.xyz);
        let cos_a = clamp(dot(dir_sky, sun), -1.0, 1.0);
        let ang = acos(cos_a);
        if (ang < SUN_ANGULAR_RADIUS * 1.5) {
            let t_sun = sample_transmittance(frame.params.z, sun.y);
            let omega = PI * SUN_ANGULAR_RADIUS * SUN_ANGULAR_RADIUS;
            let disc = frame.sun_irradiance.rgb * t_sun / omega;
            // 0.05° soft edge to avoid aliasing on the limb.
            let edge = 1.0 - smoothstep(SUN_ANGULAR_RADIUS * 0.95, SUN_ANGULAR_RADIUS * 1.05, ang);
            color = mix(color, disc, edge);
        }
    }

    // Aerial perspective for the sky path (usually negligible beyond
    // the horizon but keeps ground-haze consistent for low rays).
    if ((i32(frame.params.y) & 1) != 0) {
        let aerial = sample_aerial(AERIAL_MAX_DIST, dir_sky.y);
        color = color * aerial.trans + aerial.radiance;
    }

    color = color * frame.params.x;
    return vec4<f32>(color, 1.0);
}
