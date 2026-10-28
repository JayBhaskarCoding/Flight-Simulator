// Final display pass: exposure + ACES filmic tone-map. The output is
// LINEAR: the render target is an sRGB-format texture/surface, so the
// hardware applies the OETF on write (rendering into a non-sRGB target
// would skip the transfer curve and look washed out).

@group(0) @binding(0) var<uniform> exposure: vec4<f32>; // x = multiplier
@group(0) @binding(1) var hdr_color: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;

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

// Stephen Hill's fit of the ACES filmic curve (Narkowicz-style
// coefficients with the RGB→RGB matrix folded in for the per-channel
// variant used here).
fn aces_fitted(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let num = x * (a * x + b);
    let den = x * (c * x + d) + e;
    return clamp(num / den, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.ndc * 0.5 + vec2<f32>(0.5, 0.5);
    let hdr = textureSample(hdr_color, hdr_sampler, uv).rgb * exposure.x;
    let ldr = aces_fitted(hdr);
    return vec4<f32>(ldr, 1.0);
}
