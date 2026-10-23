// glb.as: a small, general glTF 2.0 binary (.glb) loader for the GPU canvas.
//
// Reads the blob bytes, parses the JSON chunk, walks the node hierarchy
// (TRS or matrix), and builds every mesh primitive as an interleaved vertex
// buffer (pos.xyz, normal.xyz, uv.xy = 32 bytes) plus a uint32 index buffer.
// Materials contribute their name and baseColorFactor (converted to sRGB).
// The first camera node, if any, is returned too.
//
// Not handled (not needed for the vector look): textures, skins, morph targets,
// animations, sparse accessors, KHR extensions.

import { Mat4 } from "rive/mat4";
import { Vector } from "rive/vector";
import { Buffer } from "rive/buffer";
import { GPUBuffer } from "rive/gpu";
import { BufferUsage } from "rive/gpu_enums";

// ---------------------------------------------------------------- JSON
export const J_NULL: i32 = 0;
export const J_BOOL: i32 = 1;
export const J_NUM: i32 = 2;
export const J_STR: i32 = 3;
export const J_ARR: i32 = 4;
export const J_OBJ: i32 = 5;

// One JSON value. Arrays keep their items in `items`; objects keep their
// values in `items` and the matching names in `keys`.
export class JVal {
    kind: i32 = J_NULL;
    num: f64 = 0;
    str: string = "";
    items: Array<JVal> = [];
    keys: Array<string> = [];

    get length(): i32 {
        return this.items.length;
    }

    at(i: i32): JVal {
        return this.items[i];
    }

    get(key: string): JVal? {
        if (this.kind != J_OBJ) {
            return null;
        }
        for (let i = 0; i < this.keys.length; i++) {
            if (this.keys[i] == key) {
                return this.items[i];
            }
        }
        return null;
    }

    has(key: string): bool {
        let v = this.get(key);
        return v != null && v.kind != J_NULL;
    }

    numOr(key: string, fallback: f64): f64 {
        let v = this.get(key);
        return v != null && v.kind == J_NUM ? v.num : fallback;
    }

    strOr(key: string, fallback: string): string {
        let v = this.get(key);
        return v != null && v.kind == J_STR ? v.str : fallback;
    }

    // array of numbers under `key`, or null
    nums(key: string): Array<f64>? {
        let v = this.get(key);
        if (v == null || v.kind != J_ARR) {
            return null;
        }
        let out = new Array<f64>(v.items.length);
        for (let i = 0; i < v.items.length; i++) {
            out[i] = v.items[i].num;
        }
        return out;
    }
}

class JsonParser {
    s: string;
    pos: i32 = 0;
    len: i32;

    constructor(s: string) {
        this.s = s;
        this.len = s.length;
    }

    skip(): void {
        while (this.pos < this.len) {
            let c = this.s.charCodeAt(this.pos);
            if (c == 32 || c == 9 || c == 10 || c == 13) {
                this.pos++;
            } else {
                break;
            }
        }
    }

    parseString(): string {
        // assumes s[pos] == '"'
        let s = this.s;
        let i = this.pos + 1;
        let plain = i;
        let out = "";
        while (i < this.len) {
            let c = s.charCodeAt(i);
            if (c == 34) {
                out += s.substring(plain, i);
                this.pos = i + 1;
                return out;
            } else if (c == 92) {
                out += s.substring(plain, i);
                let e = s.charCodeAt(i + 1);
                if (e == 110) {
                    out += "\n";
                } else if (e == 116) {
                    out += "\t";
                } else if (e == 114) {
                    out += "\r";
                } else if (e == 98) {
                    out += "\b";
                } else if (e == 102) {
                    out += "\f";
                } else if (e == 117) {
                    out += String.fromCharCode(<i32>parseInt(s.substring(i + 2, i + 6), 16));
                    i += 4;
                } else {
                    out += String.fromCharCode(e);
                }
                i += 2;
                plain = i;
            } else {
                i++;
            }
        }
        this.pos = this.len;
        return out;
    }

    parseNumber(): f64 {
        let s = this.s;
        let a = this.pos;
        let i = a;
        while (i < this.len) {
            let c = s.charCodeAt(i);
            // digits . - + e E
            if ((c >= 48 && c <= 57) || c == 46 || c == 45 || c == 43 || c == 101 || c == 69) {
                i++;
            } else {
                break;
            }
        }
        this.pos = i;
        if (i == a) {
            this.pos = a + 1; // skip the bad byte so parsing always moves on
            return 0;
        }
        return parseFloat(s.substring(a, i));
    }

    parseValue(): JVal {
        this.skip();
        let v = new JVal();
        if (this.pos >= this.len) {
            return v;
        }
        let c = this.s.charCodeAt(this.pos);
        if (c == 123) { // {
            this.pos++;
            v.kind = J_OBJ;
            this.skip();
            if (this.s.charCodeAt(this.pos) == 125) {
                this.pos++;
                return v;
            }
            while (this.pos < this.len) {
                this.skip();
                let k = this.parseString();
                this.skip();
                this.pos++; // :
                v.keys.push(k);
                v.items.push(this.parseValue());
                this.skip();
                let d = this.s.charCodeAt(this.pos);
                this.pos++;
                if (d == 125) {
                    return v;
                }
            }
            return v;
        } else if (c == 91) { // [
            this.pos++;
            v.kind = J_ARR;
            this.skip();
            if (this.s.charCodeAt(this.pos) == 93) {
                this.pos++;
                return v;
            }
            while (this.pos < this.len) {
                v.items.push(this.parseValue());
                this.skip();
                let d = this.s.charCodeAt(this.pos);
                this.pos++;
                if (d == 93) {
                    return v;
                }
            }
            return v;
        } else if (c == 34) {
            v.kind = J_STR;
            v.str = this.parseString();
        } else if (c == 116) {
            this.pos += 4;
            v.kind = J_BOOL;
            v.num = 1;
        } else if (c == 102) {
            this.pos += 5;
            v.kind = J_BOOL;
        } else if (c == 110) {
            this.pos += 4;
        } else {
            v.kind = J_NUM;
            v.num = this.parseNumber();
        }
        return v;
    }
}

export function jsonDecode(s: string): JVal {
    return new JsonParser(s).parseValue();
}

// ---------------------------------------------------------------- types
// One mesh primitive's geometry, shared by every node that uses the mesh.
export class Mesh {
    vdata: Float32Array; // interleaved pos/normal/uv, 32 bytes per vertex
    idata: Uint32Array; // uint32 indices
    count: i32;
    mat: i32; // material index, -1 for none
    lmin: Vector; // local-space bounds
    lmax: Vector;
    vbuf: GPUBuffer? = null; // created by upload() once the host's GPU is ready
    ibuf: GPUBuffer? = null;

    constructor(vdata: Float32Array, idata: Uint32Array, count: i32, mat: i32, lmin: Vector, lmax: Vector) {
        this.vdata = vdata;
        this.idata = idata;
        this.count = count;
        this.mat = mat;
        this.lmin = lmin;
        this.lmax = lmax;
    }
}

export class Prim {
    name: string = ""; // node name
    meshName: string = "";
    material: string = "";
    color: StaticArray<f64> = [0.8, 0.8, 0.8, 1]; // sRGB 0..1, rgba
    gloss: bool = false; // roughnessFactor < 0.35: gets a hard specular dot
    isScreen: bool = false;
    mesh: Mesh;
    world: Mat4; // authored world matrix of the owning node
    root: i32 = 0; // index into Scene.roots
    meshIndex: i32 = 0;

    constructor(mesh: Mesh, world: Mat4) {
        this.mesh = mesh;
        this.world = world;
    }

    get count(): i32 {
        return this.mesh.count;
    }

    get lmin(): Vector {
        return this.mesh.lmin;
    }

    get lmax(): Vector {
        return this.mesh.lmax;
    }
}

export class Root {
    name: string = "Root";
    center: Vector = Vector.xyz(0, 0, 0); // world-space bounds centre (idle-motion pivot)
    radius: f64 = 0;
    // accumulated world bounds while loading
    minX: f64 = 1e9;
    minY: f64 = 1e9;
    minZ: f64 = 1e9;
    maxX: f64 = -1e9;
    maxY: f64 = -1e9;
    maxZ: f64 = -1e9;
}

export class Camera {
    world: Mat4 = Mat4.identity();
    yfov: f64 = 30 * Math.PI / 180;
    znear: f64 = 0.05;
    zfar: f64 = 100;
}

export class Scene {
    prims: Array<Prim> = [];
    roots: Array<Root> = [];
    camera: Camera? = null;
    bmin: Vector = Vector.xyz(1e9, 1e9, 1e9);
    bmax: Vector = Vector.xyz(-1e9, -1e9, -1e9);
    nodeCount: i32 = 0;
}

class Material {
    name: string = "";
    color: StaticArray<f64> = [1, 1, 1, 1];
    roughness: f64 = 1;
}

class Accessor {
    data: Float64Array;
    n: i32;
    count: i32;

    constructor(data: Float64Array, n: i32, count: i32) {
        this.data = data;
        this.n = n;
        this.count = count;
    }
}

// ---------------------------------------------------------------- helpers
function compSize(ct: i32): i32 {
    if (ct == 5120 || ct == 5121) return 1;
    if (ct == 5122 || ct == 5123) return 2;
    return 4;
}

function typeN(t: string): i32 {
    if (t == "VEC2") return 2;
    if (t == "VEC3") return 3;
    if (t == "VEC4") return 4;
    if (t == "MAT4") return 16;
    return 1;
}

function toSRGB(c: f64): f64 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

export function lower(s: string): string {
    return s.toLowerCase();
}

export function contains(s: string, part: string): bool {
    return s.indexOf(part) >= 0;
}

// Reads an accessor into a flat f64 array.
function readAccessor(j: JVal, view: DataView, binOff: i32, index: i32): Accessor {
    let acc = j.get("accessors")!.at(index);
    let n = typeN(acc.strOr("type", "SCALAR"));
    let count = <i32>acc.numOr("count", 0);
    let out = new Float64Array(count * n);
    if (!acc.has("bufferView")) {
        return new Accessor(out, n, count);
    }
    let bv = j.get("bufferViews")!.at(<i32>acc.numOr("bufferView", 0));
    let ct = <i32>acc.numOr("componentType", 5126);
    let csize = compSize(ct);
    let stride = <i32>bv.numOr("byteStride", <f64>(csize * n));
    let base = binOff + <i32>bv.numOr("byteOffset", 0) + <i32>acc.numOr("byteOffset", 0);
    let normv = acc.get("normalized");
    let norm = normv != null && normv.kind == J_BOOL && normv.num != 0;
    let k = 0;
    for (let i = 0; i < count; i++) {
        let o = base + i * stride;
        for (let c = 0; c < n; c++) {
            let p = o + c * csize;
            let v: f64;
            if (ct == 5126) {
                v = <f64>view.getFloat32(p, true);
            } else if (ct == 5125) {
                v = <f64>view.getUint32(p, true);
            } else if (ct == 5123) {
                v = <f64>view.getUint16(p, true);
                if (norm) v = v / 65535;
            } else if (ct == 5121) {
                v = <f64>view.getUint8(p);
                if (norm) v = v / 255;
            } else if (ct == 5122) {
                v = <f64>view.getInt16(p, true);
                if (norm) v = Math.max(v / 32767, -1);
            } else {
                v = <f64>view.getInt8(p);
                if (norm) v = Math.max(v / 127, -1);
            }
            out[k] = v;
            k++;
        }
    }
    return new Accessor(out, n, count);
}

function nodeLocal(nd: JVal): Mat4 {
    let m = nd.nums("matrix");
    if (m != null && m.length >= 16) {
        return Mat4.values(
            <f32>m[0], <f32>m[1], <f32>m[2], <f32>m[3],
            <f32>m[4], <f32>m[5], <f32>m[6], <f32>m[7],
            <f32>m[8], <f32>m[9], <f32>m[10], <f32>m[11],
            <f32>m[12], <f32>m[13], <f32>m[14], <f32>m[15]);
    }
    let t = nd.nums("translation");
    let r = nd.nums("rotation");
    let s = nd.nums("scale");
    let tx = t != null ? t[0] : 0, ty = t != null ? t[1] : 0, tz = t != null ? t[2] : 0;
    let x = r != null ? r[0] : 0, y = r != null ? r[1] : 0, z = r != null ? r[2] : 0, w = r != null ? r[3] : 1;
    let sx = s != null ? s[0] : 1, sy = s != null ? s[1] : 1, sz = s != null ? s[2] : 1;
    return Mat4.values(
        <f32>((1 - 2 * (y * y + z * z)) * sx), <f32>((2 * (x * y + z * w)) * sx), <f32>((2 * (x * z - y * w)) * sx), 0,
        <f32>((2 * (x * y - z * w)) * sy), <f32>((1 - 2 * (x * x + z * z)) * sy), <f32>((2 * (y * z + x * w)) * sy), 0,
        <f32>((2 * (x * z + y * w)) * sz), <f32>((2 * (y * z - x * w)) * sz), <f32>((1 - 2 * (x * x + y * y)) * sz), 0,
        <f32>tx, <f32>ty, <f32>tz, 1);
}

// ---------------------------------------------------------------- load
class Loader {
    j: JVal;
    view: DataView;
    binOff: i32;
    scene: Scene = new Scene();
    mats: Array<Material> = [];
    meshCache: Map<i32, Array<Mesh>> = new Map<i32, Array<Mesh>>();

    constructor(j: JVal, view: DataView, binOff: i32) {
        this.j = j;
        this.view = view;
        this.binOff = binOff;
    }

    loadMaterials(): void {
        let ms = this.j.get("materials");
        if (ms == null) {
            return;
        }
        for (let i = 0; i < ms.length; i++) {
            let m = ms.at(i);
            let mat = new Material();
            let f: StaticArray<f64> = [1, 1, 1, 1];
            let pbr = m.get("pbrMetallicRoughness");
            if (pbr != null) {
                let bc = pbr.nums("baseColorFactor");
                if (bc != null) {
                    for (let c = 0; c < 4 && c < bc.length; c++) {
                        f[c] = bc[c];
                    }
                }
                mat.roughness = pbr.numOr("roughnessFactor", 1);
            }
            mat.name = m.strOr("name", "Material" + (i + 1).toString());
            mat.color = [toSRGB(f[0]), toSRGB(f[1]), toSRGB(f[2]), f[3]];
            this.mats.push(mat);
        }
    }

    // build each mesh primitive once
    meshList(mi: i32): Array<Mesh> {
        if (this.meshCache.has(mi)) {
            return this.meshCache.get(mi);
        }
        let list: Array<Mesh> = [];
        let mesh = this.j.get("meshes")!.at(mi);
        let prs = mesh.get("primitives")!;
        for (let pi = 0; pi < prs.length; pi++) {
            let pr = prs.at(pi);
            let mode = <i32>pr.numOr("mode", 4);
            let attrs = pr.get("attributes")!;
            if (mode != 4 || !attrs.has("POSITION")) {
                continue;
            }
            let pa = readAccessor(this.j, this.view, this.binOff, <i32>attrs.numOr("POSITION", 0));
            let P = pa.data;
            let vcount = pa.count;
            let N: Float64Array? = null;
            if (attrs.has("NORMAL")) {
                N = readAccessor(this.j, this.view, this.binOff, <i32>attrs.numOr("NORMAL", 0)).data;
            }
            let T: Float64Array? = null;
            if (attrs.has("TEXCOORD_0")) {
                T = readAccessor(this.j, this.view, this.binOff, <i32>attrs.numOr("TEXCOORD_0", 0)).data;
            }
            let I: Float64Array;
            if (pr.has("indices")) {
                I = readAccessor(this.j, this.view, this.binOff, <i32>pr.numOr("indices", 0)).data;
            } else {
                I = new Float64Array(vcount);
                for (let k = 0; k < vcount; k++) {
                    I[k] = <f64>k;
                }
            }
            // flat normals when the file has none
            if (N == null) {
                let nn = new Float64Array(vcount * 3);
                for (let t = 0; t + 2 < I.length; t += 3) {
                    let a = <i32>I[t], b = <i32>I[t + 1], c = <i32>I[t + 2];
                    let ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
                    let ux = P[b * 3] - ax, uy = P[b * 3 + 1] - ay, uz = P[b * 3 + 2] - az;
                    let vx = P[c * 3] - ax, vy = P[c * 3 + 1] - ay, vz = P[c * 3 + 2] - az;
                    let cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
                    nn[a * 3] += cx; nn[a * 3 + 1] += cy; nn[a * 3 + 2] += cz;
                    nn[b * 3] += cx; nn[b * 3 + 1] += cy; nn[b * 3 + 2] += cz;
                    nn[c * 3] += cx; nn[c * 3 + 1] += cy; nn[c * 3 + 2] += cz;
                }
                N = nn;
            }
            let NN = N!;
            let vb = new Float32Array(vcount * 8);
            let mnx = 1e9, mny = 1e9, mnz = 1e9;
            let mxx = -1e9, mxy = -1e9, mxz = -1e9;
            for (let v = 0; v < vcount; v++) {
                let x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
                mnx = Math.min(mnx, x);
                mny = Math.min(mny, y);
                mnz = Math.min(mnz, z);
                mxx = Math.max(mxx, x);
                mxy = Math.max(mxy, y);
                mxz = Math.max(mxz, z);
                let nx = NN[v * 3], ny = NN[v * 3 + 1], nz = NN[v * 3 + 2];
                let l = Math.sqrt(nx * nx + ny * ny + nz * nz);
                if (l > 0) {
                    nx /= l;
                    ny /= l;
                    nz /= l;
                }
                let o = v * 8;
                vb[o] = <f32>x;
                vb[o + 1] = <f32>y;
                vb[o + 2] = <f32>z;
                vb[o + 3] = <f32>nx;
                vb[o + 4] = <f32>ny;
                vb[o + 5] = <f32>nz;
                if (T != null) {
                    vb[o + 6] = <f32>T[v * 2];
                    vb[o + 7] = <f32>T[v * 2 + 1];
                }
            }
            let icount = I.length;
            let ib = new Uint32Array(icount);
            for (let k = 0; k < icount; k++) {
                ib[k] = <u32>I[k];
            }
            list.push(new Mesh(vb, ib, icount, <i32>pr.numOr("material", -1),
                Vector.xyz(<f32>mnx, <f32>mny, <f32>mnz), Vector.xyz(<f32>mxx, <f32>mxy, <f32>mxz)));
        }
        this.meshCache.set(mi, list);
        return list;
    }

    visit(ni: i32, parent: Mat4, ri: i32): void {
        let j = this.j;
        let sc = this.scene;
        let nd = j.get("nodes")!.at(ni);
        sc.nodeCount++;
        let world = parent * nodeLocal(nd);
        let cams = j.get("cameras");
        if (nd.has("camera") && sc.camera == null && cams != null) {
            let cam = cams.at(<i32>nd.numOr("camera", 0));
            let c = new Camera();
            c.world = world;
            let p = cam.get("perspective");
            if (p != null) {
                c.yfov = p.numOr("yfov", 30 * Math.PI / 180);
                c.znear = p.numOr("znear", 0.05);
                c.zfar = p.numOr("zfar", 100);
            }
            sc.camera = c;
        }
        if (nd.has("mesh")) {
            let mi = <i32>nd.numOr("mesh", 0);
            let list = this.meshList(mi);
            let meshName = j.get("meshes")!.at(mi).strOr("name", "Mesh" + mi.toString());
            let root = sc.roots[ri];
            for (let g = 0; g < list.length; g++) {
                let gp = list[g];
                let p = new Prim(gp, world);
                let hasMat = gp.mat >= 0 && gp.mat < this.mats.length;
                p.name = nd.strOr("name", "Node" + ni.toString());
                p.meshName = meshName;
                p.material = hasMat ? this.mats[gp.mat].name : "Default";
                if (hasMat) {
                    let mc = this.mats[gp.mat].color;
                    p.color = [mc[0], mc[1], mc[2], mc[3]];
                }
                p.gloss = hasMat ? this.mats[gp.mat].roughness < 0.35 : false;
                p.isScreen = contains(lower(p.material), "screen");
                p.root = ri;
                p.meshIndex = mi;
                sc.prims.push(p);
                // world bounds (8 corners)
                for (let c = 0; c < 8; c++) {
                    let x = (c & 1) == 0 ? gp.lmin.x : gp.lmax.x;
                    let y = (c & 2) == 0 ? gp.lmin.y : gp.lmax.y;
                    let z = (c & 4) == 0 ? gp.lmin.z : gp.lmax.z;
                    let wp = world.transformPoint(x, y, z);
                    root.minX = Math.min(root.minX, <f64>wp.x);
                    root.minY = Math.min(root.minY, <f64>wp.y);
                    root.minZ = Math.min(root.minZ, <f64>wp.z);
                    root.maxX = Math.max(root.maxX, <f64>wp.x);
                    root.maxY = Math.max(root.maxY, <f64>wp.y);
                    root.maxZ = Math.max(root.maxZ, <f64>wp.z);
                }
            }
        }
        let kids = nd.get("children");
        if (kids != null) {
            for (let k = 0; k < kids.length; k++) {
                this.visit(<i32>kids.at(k).num, world, ri);
            }
        }
    }
}

// Why load() returned null, for the caller to log.
export let loadError: string = "";

export function load(bin: ArrayBuffer): Scene? {
    let total = bin.byteLength;
    let view = new DataView(bin);
    if (total < 20 || view.getUint32(0, true) != 0x46546C67) {
        loadError = "glb: not a binary glTF";
        return null;
    }
    let jsonLen = <i32>view.getUint32(12, true);
    let j = jsonDecode(Buffer.readstring(bin, 20, jsonLen));
    let binOff = 20 + jsonLen + 8;
    if (binOff > total) {
        binOff = total;
    }
    let ld = new Loader(j, view, binOff);
    ld.loadMaterials();
    let sc = ld.scene;

    let rootList: Array<i32> = [];
    let scenes = j.get("scenes");
    let sceneDef: JVal? = null;
    if (scenes != null) {
        let si = <i32>j.numOr("scene", 0);
        if (si < scenes.length) {
            sceneDef = scenes.at(si);
        }
    }
    let nodes = j.get("nodes");
    let sn = sceneDef != null ? sceneDef.get("nodes") : null;
    if (sn != null) {
        for (let i = 0; i < sn.length; i++) {
            rootList.push(<i32>sn.at(i).num);
        }
    } else if (nodes != null) {
        for (let i = 0; i < nodes.length; i++) {
            rootList.push(i);
        }
    }
    if (nodes == null) {
        return sc;
    }
    for (let r = 0; r < rootList.length; r++) {
        let ni = rootList[r];
        let root = new Root();
        root.name = nodes.at(ni).strOr("name", "Root");
        sc.roots.push(root);
        ld.visit(ni, Mat4.identity(), sc.roots.length - 1);
    }
    for (let r = 0; r < sc.roots.length; r++) {
        let rt = sc.roots[r];
        if (rt.minX <= rt.maxX) {
            rt.center = Vector.xyz(<f32>((rt.minX + rt.maxX) / 2), <f32>((rt.minY + rt.maxY) / 2), <f32>((rt.minZ + rt.maxZ) / 2));
            let dx = rt.maxX - rt.minX, dy = rt.maxY - rt.minY, dz = rt.maxZ - rt.minZ;
            rt.radius = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
            sc.bmin = Vector.xyz(<f32>Math.min(<f64>sc.bmin.x, rt.minX), <f32>Math.min(<f64>sc.bmin.y, rt.minY), <f32>Math.min(<f64>sc.bmin.z, rt.minZ));
            sc.bmax = Vector.xyz(<f32>Math.max(<f64>sc.bmax.x, rt.maxX), <f32>Math.max(<f64>sc.bmax.y, rt.maxY), <f32>Math.max(<f64>sc.bmax.z, rt.maxZ));
        }
    }
    return sc;
}

// Create the GPU buffers. Kept out of load(): on the web runtime the GPU
// context is not ready yet when a script's init() runs. Primitives that share
// a mesh share one pair of buffers.
export function upload(scene: Scene): void {
    for (let i = 0; i < scene.prims.length; i++) {
        let m = scene.prims[i].mesh;
        if (m.vbuf == null) {
            m.vbuf = new GPUBuffer(BufferUsage.vertex, <u32>m.vdata.byteLength, m.vdata.buffer, true);
        }
        if (m.ibuf == null) {
            m.ibuf = new GPUBuffer(BufferUsage.index, <u32>m.idata.byteLength, m.idata.buffer, true);
        }
    }
}
