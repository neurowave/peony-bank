// toon.wgsl — G-buffer pass for the "3D drawn as vector art" look.
// Writes three targets:
//   0: flat colour (3-tone cel ramp, or the live Rive screen texture)
//   1: view-space normal.xyz + linear depth   (for crease/silhouette lines)
//   2: edge id                                 (object/material boundaries)

struct U {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    viewM: mat4x4<f32>,
    dark: vec4f,
    mid: vec4f,
    light: vec4f,
    params: vec4f,   // x = edge id (0..1), y = isScreen, z = specular, w = time
    lightDir: vec4f, // world space, towards the light; w = band softness (0 = hard)
    camPos: vec4f,   // xyz camera; w = screen brightness
    notch: vec4f,    // x = notch width (uv), y = notch height (uv), z = bezel id, w = enabled
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var uiTex: texture_2d<f32>;
@group(0) @binding(2) var uiSamp: sampler;

struct VIn {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) uv: vec2f,
};

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) wpos: vec3f,
    @location(1) wnrm: vec3f,
    @location(2) uv: vec2f,
    @location(3) vdepth: f32,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    let wp = u.model * vec4f(v.pos, 1.0);
    o.clip = u.mvp * vec4f(v.pos, 1.0);
    o.wpos = wp.xyz;
    o.wnrm = (u.model * vec4f(v.nrm, 0.0)).xyz;
    o.uv = v.uv;
    o.vdepth = -(u.viewM * wp).z;
    return o;
}

struct GOut {
    @location(0) color: vec4f,
    @location(1) nd: vec4f,
    @location(2) id: vec4f,
};

fn sdRoundBox(p: vec2f, b: vec2f, r: f32) -> f32 {
    let q = abs(p) - b + vec2f(r);
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(i: VOut, @builtin(front_facing) ff: bool) -> GOut {
    var n = normalize(i.wnrm);
    if (!ff) { n = -n; }
    let V = normalize(u.camPos.xyz - i.wpos);
    // keep normals facing the viewer for lines/shading on open meshes
    let L = normalize(u.lightDir.xyz);
    let ndl = dot(n, L);

    var col: vec3f;
    var id = u.params.x;
    if (u.params.y > 0.5) {
        // live Rive artboard on the phone screen
        let t = textureSample(uiTex, uiSamp, i.uv);
        col = mix(vec3f(1.0), t.rgb / max(t.a, 1e-4), t.a) * u.camPos.w;
        // device notch, drawn as part of the bezel
        if (u.notch.w > 0.5) {
            let w = u.notch.x * 0.5;
            let h = u.notch.y;
            let p = vec2f((i.uv.x - 0.5), i.uv.y - h * 0.5 + 0.004);
            let d = sdRoundBox(p, vec2f(w, h * 0.5 + 0.004), 0.012);
            if (d < 0.0) {
                col = u.dark.rgb;
                id = u.notch.z;
            }
        }
    } else {
        // hard 3-tone ramp: shadow / base / highlight
        let s = u.lightDir.w;
        let a = smoothstep(-0.05 - s, -0.05 + s, ndl);
        let b = smoothstep(0.55 - s, 0.55 + s, ndl);
        col = mix(u.dark.rgb, u.mid.rgb, a);
        col = mix(col, u.light.rgb, b);
        // optional specular dot (spheres): a hard white-ish blob
        if (u.params.z > 0.0) {
            let h = normalize(L + V);
            let sp = step(0.985, dot(n, h));
            col = mix(col, mix(u.light.rgb, vec3f(1.0), 0.65), sp * u.params.z);
        }
    }

    let vn = normalize((u.viewM * vec4f(n, 0.0)).xyz);
    var o: GOut;
    o.color = vec4f(col, 1.0);
    o.nd = vec4f(vn, i.vdepth);
    o.id = vec4f(id, 0.0, 0.0, 1.0);
    return o;
}
