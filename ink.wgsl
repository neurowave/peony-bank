// ink.wgsl — turns the G-buffer into flat vector art with outlines.
// Runs at G-buffer (supersampled) resolution; resolve.wgsl filters it down.
// Lines come from: edge-id changes (silhouettes, part boundaries), normal
// creases (prism edges, cone rim) and depth steps (self-occlusion).

struct P {
    params: vec4f,  // x = line radius (texels), y = crease cos threshold, z = depth step (relative), w = ink darkness
    ink: vec4f,     // rgb = base ink colour, a = mix toward the local colour
    size: vec4f,    // xy = texture size
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var tColor: texture_2d<f32>;
@group(0) @binding(2) var tND: texture_2d<f32>;
@group(0) @binding(3) var tID: texture_2d<f32>;

struct VOut { @builtin(position) pos: vec4f, };

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> VOut {
    var o: VOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

fn edgeAt(q: vec2i, id0: f32, nd0: vec4f, depthStep: f32, creaseCos: f32) -> vec2f {
    // returns (isEdge, neighbourIsCloser)
    let sz = vec2i(p.size.xy);
    let qq = clamp(q, vec2i(0), sz - vec2i(1));
    let id1 = textureLoad(tID, qq, 0).r;
    let nd1 = textureLoad(tND, qq, 0);
    let bg0 = id0 < 0.5 / 255.0;
    let bg1 = id1 < 0.5 / 255.0;
    if (bg0 && bg1) { return vec2f(0.0, 0.0); }
    let closer = select(0.0, 1.0, (!bg1) && (bg0 || nd1.w < nd0.w));
    if (abs(id1 - id0) > 0.5 / 255.0) { return vec2f(1.0, closer); }
    if (dot(nd0.xyz, nd1.xyz) < creaseCos) { return vec2f(1.0, closer); }
    if (abs(nd1.w - nd0.w) > depthStep * min(nd0.w, nd1.w)) { return vec2f(1.0, closer); }
    return vec2f(0.0, 0.0);
}

@fragment
fn fs_ink(i: VOut) -> @location(0) vec4f {
    let pc = vec2i(i.pos.xy);
    let c0 = textureLoad(tColor, pc, 0);
    let nd0 = textureLoad(tND, pc, 0);
    let id0 = textureLoad(tID, pc, 0).r;
    let r = p.params.x;
    let creaseCos = p.params.y;
    let depthStep = p.params.z;

    var edge = 0.0;
    var nearest = c0;
    var nearestD = select(1e9, nd0.w, id0 >= 0.5 / 255.0);
    // 8 taps on a circle of radius r, plus 4 at r/2 for thin features
    for (var k = 0; k < 12; k++) {
        var a: f32;
        var rr: f32;
        if (k < 8) { a = f32(k) * 0.7853982; rr = r; } else { a = f32(k - 8) * 1.5707964 + 0.3926991; rr = r * 0.5; }
        let off = vec2f(cos(a), sin(a)) * rr;
        let q = pc + vec2i(round(off));
        let e = edgeAt(q, id0, nd0, depthStep, creaseCos);
        if (e.x > 0.5) {
            edge = 1.0;
            let qq = clamp(q, vec2i(0), vec2i(p.size.xy) - vec2i(1));
            let d1 = textureLoad(tND, qq, 0).w;
            let id1 = textureLoad(tID, qq, 0).r;
            if (id1 >= 0.5 / 255.0 && d1 < nearestD) {
                nearestD = d1;
                nearest = textureLoad(tColor, qq, 0);
            }
        }
    }
    if (edge < 0.5) {
        return c0; // premultiplied: objects alpha 1, background 0
    }
    // ink: dark, tinted by the nearer surface so green shapes get green-black
    // lines and the phone gets near-black ones (like the reference art).
    let tint = nearest.rgb / max(nearest.a, 1e-4);
    let inkc = mix(p.ink.rgb, tint * p.params.w, p.ink.a);
    return vec4f(inkc, 1.0);
}
