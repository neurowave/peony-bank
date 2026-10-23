// resolve.wgsl — box-filter the supersampled ink image down to the canvas.
struct R {
    params: vec4f, // xy = 1/inkSize, z = texels per output pixel, w = unused
};

@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var tInk: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> VOut {
    var o: VOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2f(x, y);
    return o;
}

@fragment
fn fs_resolve(i: VOut) -> @location(0) vec4f {
    let f = r.params.z;
    let d = r.params.xy * f * 0.25;
    var c = textureSample(tInk, samp, i.uv + vec2f(-d.x, -d.y));
    c += textureSample(tInk, samp, i.uv + vec2f(d.x, -d.y));
    c += textureSample(tInk, samp, i.uv + vec2f(-d.x, d.y));
    c += textureSample(tInk, samp, i.uv + vec2f(d.x, d.y));
    return c * 0.25;
}
