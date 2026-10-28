// Volumetric cloud raymarch (Hillaire/Guerrilla-style march with a
// procedural weather field and a 3-D noise texture).
//
// Runs after opaques: reads the HDR scene colour and a *copy* of the
// depth buffer (copied outside the pass so there is no
// attachment/sampler usage conflict), marches the segment in front of
// each pixel, and composites front-to-back into a second HDR target.

struct Frame {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sky_from_rel: mat4x4<f32>,
    sun_dir_sky: vec4<f32>,
    sun_irradiance: vec4<f32>,
    time_wind: vec4<f32>,
    // x: exposure, y: flags, z: camera altitude (m), w: unused
    params: vec4<f32>,
};

struct CloudParams {
    // x: time (s), yzw: wind offset (m)
    wind_time: vec4<f32>,
    // x: coverage [0,1], y: density mult, z: anvil bias, w: unused
    coverage: vec4<f32>,
    // x: layer bottom (m ASL), y: top (m ASL), z: max ray length (m),
    // w: primary step scale
    layer: vec4<f32>,
    // x: HG asymmetry, y: ambient boost, z: powder strength, w: unused
    shading: vec4<f32>,
    // x: planet radius (m), yzw: unused
    geo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> clouds: CloudParams;
@group(0) @binding(2) var noise_tex: texture_3d<f32>;
@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var scene_depth: texture_depth_2d;
@group(0) @binding(5) var cloud_sampler: sampler;
@group(0) @binding(6) var aerial_trans_lut: texture_2d<f32>;
@group(0) @binding(7) var aerial_radiance_lut: texture_2d<f32>;

const PI: f32 = 3.14159265359;
const PRIMARY_STEPS: i32 = 64;
const LIGHT_STEPS: i32 = 6;
// Inverted-Worley density thresholds (Beer extinction, 1/m).
const SIGMA: f32 = 0.045;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

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

// --- analytic helpers -------------------------------------------------

// Two intersections of a ray with a sphere centred at `origin`'s
// negative... here: ray starting at `ro` (planet-centred) with radius
// `r`. Returns (t_near, t_far); x > y when there is no hit.
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) {
        return vec2<f32>(1.0, -1.0);
    }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * max(pow(max(denom, 1e-4), 1.5), 1e-4));
}

fn remap(v: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + clamp((v - a) / max(b - a, 1e-5), 0.0, 1.0) * (d - c);
}

// --- procedural fields ------------------------------------------------

// Low-frequency weather: coverage + humidity from a 2-D hash lattice.
fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let d = q + dot(q, q + 45.32);
    return fract(d.x * d.y);
}

fn value_noise_2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// returns (coverage, cloud_type) at a ground position
fn weather(xz: vec2<f32>) -> vec2<f32> {
    let wind = clouds.wind_time.yz;
    let t = clouds.wind_time.x;
    let p = (xz + wind * t) / 30000.0;
    var c = value_noise_2d(p) * 0.65 + value_noise_2d(p * 3.1 + 17.7) * 0.35;
    c = remap(c, 0.0, 1.0, clouds.coverage.x * 0.55, 1.0);
    // anvil bias: type > 0.55 → taller towers
    let type_n = value_noise_2d(p * 0.37 + 91.3);
    return vec2<f32>(clamp(c, 0.0, 1.0), type_n);
}

// Height gradient for the cloud type (stratus/stratocumulus/cumulus).
fn height_gradient(h01: f32, cloud_type: f32) -> f32 {
    if (cloud_type < 0.35) {
        // stratus: flat layer near the bottom
        return remap(h01, 0.0, 0.12, 0.0, 1.0) * remap(h01, 0.25, 0.4, 1.0, 0.0);
    } else if (cloud_type < 0.75) {
        // stratocumulus
        return remap(h01, 0.0, 0.1, 0.0, 1.0) * remap(h01, 0.5, 0.9, 1.0, 0.0);
    }
    // towering cumulus
    return remap(h01, 0.0, 0.15, 0.0, 1.0) * remap(h01, 0.65, 1.0, 1.0, 0.0);
}

// Base density: shape noise + erosion, sampled at a world (sky-frame)
// position. `lod_bias` sacrifices detail for distant/light samples.
fn cloud_density(pos: vec3<f32>, h01: f32, detail_lod: f32) -> f32 {
    let w = weather(pos.xz);
    var cov = w.x;
    if (h01 > 0.6) {
        cov = cov * (1.0 - clouds.coverage.z * remap(h01, 0.6, 1.0, 0.0, 1.0));
    }
    let grad = height_gradient(h01, w.y);
    if (grad <= 0.0 || cov <= 0.02) {
        return 0.0;
    }

    // wind offsets scale with altitude (faster at layer top)
    let wind = clouds.wind_time.yz;
    let t = clouds.wind_time.x;
    let p = (pos + vec3<f32>(wind.x, 0.0, wind.y) * t + vec3<f32>(0.0, clouds.layer.x, 0.0))
        / 4000.0;

    let base = textureSampleLevel(noise_tex, cloud_sampler, fract(p), 0.0);
    // billowy low octave (r = perlin-worley) + detail (gba worley)
    var shape = base.x;
    var detail = base.y * 0.625 + base.z * 0.25 + base.w * 0.125;
    shape = remap(shape, detail * 0.55 - 0.1, 1.0, 0.0, 1.0);

    var d = remap(shape * grad, 1.0 - cov, 1.0, 0.0, 1.0) * cov;

    // erosion (skip for distant samples)
    if (detail_lod > 0.5) {
        let p2 = p * 6.0;
        let det = textureSampleLevel(noise_tex, cloud_sampler, fract(p2), 0.0).rgb;
        let erosion = det.r * 0.35 + det.g * 0.5 + det.b * 0.15;
        d = remap(d, erosion * 0.35 * (1.0 - h01 * 0.5), 1.0, 0.0, 1.0);
    }
    return clamp(d * clouds.coverage.y, 0.0, 1.0);
}

// Beer–powder transmittance for `sigma_t * distance`.
fn beer_powder(optical_depth: f32, cos_theta: f32) -> f32 {
    let beer = exp(-optical_depth);
    let powder = 1.0 - exp(-optical_depth * 2.0);
    return beer * mix(1.0, powder, clouds.shading.z * 0.5) * 0.5 + beer * 0.5;
}

// March towards the sun; returns light transmittance.
fn light_march(pos: vec3<f32>, sun_dir: vec3<f32>, radius: f32) -> f32 {
    let step = radius / f32(LIGHT_STEPS);
    var depth = 0.0;
    for (var i = 1; i <= LIGHT_STEPS; i = i + 1) {
        let sp = pos + sun_dir * step * (f32(i) - 0.5);
        let alt = length(sp) - clouds.geo.x;
        let h01 = (alt - clouds.layer.x) / max(clouds.layer.y - clouds.layer.x, 1.0);
        if (h01 >= 0.0 && h01 <= 1.0) {
            depth = depth + cloud_density(sp, h01, 0.0) * step;
        }
    }
    return exp(-depth * SIGMA);
}

// --- main pass --------------------------------------------------------

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.clip.xy);
    let scene = textureLoad(scene_color, pixel, 0);

    // world ray
    let clip = vec4<f32>(in.ndc, 1.0, 1.0);
    var world = frame.inv_view_proj * clip;
    world = world / world.w;
    let dir_rel = normalize(world.xyz);
    let dir = normalize((frame.sky_from_rel * vec4<f32>(dir_rel, 0.0)).xyz);

    // depth → scene distance (sky pixels: no limit)
    let raw_depth = textureLoad(scene_depth, pixel, 0);
    var max_len = clouds.layer.z;
    if (raw_depth < 1.0) {
        let ndc_z = raw_depth * 2.0 - 1.0;
        let clip_s = vec4<f32>(in.ndc, ndc_z, 1.0);
        var ws = frame.inv_view_proj * clip_s;
        ws = ws / ws.w;
        max_len = min(max_len, length(ws.xyz));
    }

    // camera position in the sky frame: on the axis at the camera's
    // altitude (matches sim-atmosphere's observer convention).
    let cam_alt = frame.params.z;
    let ro = vec3<f32>(0.0, clouds.geo.x + cam_alt, 0.0);

    let bottom = clouds.geo.x + clouds.layer.x;
    let top = clouds.geo.x + clouds.layer.y;
    let outer = ray_sphere(ro, dir, top);
    if (outer.x > outer.y) {
        return scene; // ray misses the cloud shell entirely
    }
    var t0 = max(outer.x, 0.0);
    var t1 = outer.y;

    // Remove the region *below* the cloud floor (inside the bottom
    // sphere), split by where the camera sits:
    //   below the floor  → enter the shell where the ray exits the
    //                      floor sphere (inner.y);
    //   in the shell     → never inside the floor sphere going up;
    //                      going down, the floor cuts the segment at
    //                      inner.x;
    //   above the shell  → same cut when the ray dives through.
    let inner = ray_sphere(ro, dir, bottom);
    if (inner.x <= inner.y) {
        if (cam_alt < clouds.layer.x) {
            t0 = max(t0, inner.y);
        } else if (t0 < inner.x) {
            t1 = min(t1, inner.x);
        } else if (t0 < inner.y) {
            t0 = inner.y;
        }
    }
    t0 = max(t0, 0.0);
    t1 = min(t1, max_len);
    if (t1 <= t0) {
        return scene;
    }

    let sun = normalize(frame.sun_dir_sky.xyz);
    let cos_phase = dot(dir, sun);
    // dual-lobe phase: forward Mie + weak backscatter
    let phase = mix(hg_phase(cos_phase, -0.2), hg_phase(cos_phase, clouds.shading.x), 0.65);

    let seg = t1 - t0;
    let step_len = seg / f32(PRIMARY_STEPS) * max(clouds.layer.w, 0.1);
    var t = t0;
    var acc = vec3<f32>(0.0);
    var trans = 1.0;
    var jitter = fract(sin(dot(in.clip.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    t = t + step_len * jitter;

    let ambient = mix(vec3<f32>(0.55, 0.62, 0.75), vec3<f32>(0.18, 0.2, 0.25), 0.5)
        * clouds.shading.y * 0.6;

    for (var i = 0; i < PRIMARY_STEPS && t <= t1; i = i + 1) {
        let pos = ro + dir * t;
        let alt = length(pos) - clouds.geo.x;
        let h01 = (alt - clouds.layer.x) / max(clouds.layer.y - clouds.layer.x, 1.0);
        var density = 0.0;
        if (h01 >= 0.0 && h01 <= 1.0) {
            let lod = select(0.0, 1.0, t > 18000.0);
            density = cloud_density(pos, h01, lod);
        }
        if (density > 0.001) {
            let sigma_t = density * SIGMA;
            let light_t = light_march(pos, sun, 900.0);
            let lit = frame.sun_irradiance.rgb * light_t * phase * 6.0;
            var scatter = lit + ambient * mix(0.4, 1.0, h01);
            let sample_ext = exp(-sigma_t * step_len);
            // energy-conserving integration of the segment
            let integ = scatter * density * SIGMA * (1.0 - sample_ext) / max(sigma_t, 1e-6);
            acc = acc + trans * integ;
            trans = trans * sample_ext;
            if (trans < 0.01) {
                break;
            }
        }
        t = t + step_len;
    }

    // aerial perspective on the cloud path midpoint (straight ray ⇒
    // cos_zenith is simply the ray's vertical component)
    var cloud_rgb = acc;
    if ((i32(frame.params.y) & 1) != 0 && trans < 0.999) {
        let mid = (t0 + min(t1, t)) * 0.5;
        let aerial_t = sample_aerial_simple(mid, dir.y);
        cloud_rgb = cloud_rgb * aerial_t.trans + aerial_t.radiance;
    }

    let out = scene.rgb * trans + cloud_rgb;
    return vec4<f32>(out, 1.0);
}

// Simplified aerial helper for this pass: transmittance + in-scatter
// via the same grids as the other passes (duplicated for standalone
// module compilation; formulas are covered by lut.rs parity tests).
struct Aerial {
    trans: vec3<f32>,
    radiance: vec3<f32>,
};

fn sample_aerial_simple(distance_m: f32, cos_zenith: f32) -> Aerial {
    let d = clamp(distance_m, 0.0, 80000.0);
    let v = clamp(d / 80000.0 * 16.0 - 0.5, 0.0, 15.0);
    let theta = acos(clamp(cos_zenith, -1.0, 1.0));
    let z = clamp(theta / PI * 16.0 - 0.5, 0.0, 15.0);
    let d0 = i32(floor(v));
    let d1 = min(d0 + 1, 15);
    let z0 = i32(floor(z));
    let z1 = min(z0 + 1, 15);
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
