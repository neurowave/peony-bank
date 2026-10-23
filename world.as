// world.as: renders a glTF scene (assets/scene.glb, a blob asset) on the GPU
// canvas and styles it as flat vector art: 3-tone cel ramp per material,
// supersampled G-buffer, crease/silhouette ink lines, box-filtered resolve.
// Any material named "Screen" is textured with a live Rive artboard (the
// Phone Screen component), and pointer events that land on it are ray-cast
// into that artboard, so the screen is fully interactive.

import { Layout, Context } from "rive/host";
import { Renderer } from "rive/renderer";
import { Artboard } from "rive/artboard";
import { ViewModel } from "rive/data";
import { Canvas } from "rive/canvas";
import { PointerEvent, KeyboardEvent } from "rive/events";
import { Image, ImageSampler, ImageWrap, ImageFilter } from "rive/image";
import { BlendMode } from "rive/paint";
import { Mat2D } from "rive/mat2d";
import { Mat4 } from "rive/mat4";
import { Vector } from "rive/vector";
import {
    GPUBindGroup, GPUBindGroupSampler, GPUBindGroupTexture, GPUBindGroupUBO, GPUBuffer, GPUCanvas,
    GPUColorAttachment, GPUColorTargetDesc, GPUDepthStencilDesc, GPUPassDesc, GPUPipeline, GPUPipelineDesc,
    GPURenderPass, GPUSampler, GPUShaderModule, GPUTexture, GPUTextureDesc, GPUTextureView, GPUTextureViewDesc,
    GPUVertexAttribute, GPUVertexBufferLayout,
} from "rive/gpu";
import {
    BufferUsage, CompareFunction, CullMode, Filter, IndexFormat, LoadOp, PrimitiveTopology, StoreOp,
    TextureAspect, TextureFormat, VertexFormat, WrapMode,
} from "rive/gpu_enums";
import { Prim, Root, Scene, load, loadError, upload, lower, contains } from "./glb";

const UBO_SIZE: u32 = 304;
const MIN_RENDER_SCALE: f64 = 2; // see resize(): hosts may under-report the device scale

// click-drag orbit
const ORBIT_PER_PX: f64 = 0.006; // radians per pixel (sideways)
const PITCH_PER_PX: f64 = 0.0018; // up/down is deliberately ~1/3 as sensitive
const PITCH_LIMIT: f64 = 0.28; // ~16 degrees either way; the orbit is mostly sideways
const DRAG_SLOP: f64 = 5; // px before a press becomes an orbit drag
const LIGHT_X: f64 = 0.80;
const LIGHT_Y: f64 = 0.42;
const LIGHT_Z: f64 = 0.30;
// the props occupy ~the middle 55% of the authored 1200x630 frame; only
// widen the lens when the viewport gets narrower than that block
const AUTHOR_ASPECT: f64 = 1.1;

// R replays the load animation
const KEY_R: f64 = 82;

function clamp(v: f64, lo: f64, hi: f64): f64 {
    return Math.min(Math.max(v, lo), hi);
}

// ---------------------------------------------------------------- motion helpers
function spring01(t: f64): f64 {
    // 0 -> 1 with ~12% overshoot, settles by t~1
    if (t <= 0) {
        return 0;
    }
    if (t >= 1.4) {
        return 1;
    }
    return 1 - Math.exp(-6.5 * t) * Math.cos(9.0 * t);
}

function easeOutCubic(t: f64): f64 {
    let x = clamp(t, 0, 1);
    let y = 1 - x;
    return 1 - y * y * y;
}

// 3x3 rotations, row-major, matching Mat4.fromRotationX/Y/Z
function rotX(out: StaticArray<f64>, a: f64): void {
    let c = Math.cos(a), s = Math.sin(a);
    out[0] = 1; out[1] = 0; out[2] = 0;
    out[3] = 0; out[4] = c; out[5] = -s;
    out[6] = 0; out[7] = s; out[8] = c;
}

function rotY(out: StaticArray<f64>, a: f64): void {
    let c = Math.cos(a), s = Math.sin(a);
    out[0] = c; out[1] = 0; out[2] = s;
    out[3] = 0; out[4] = 1; out[5] = 0;
    out[6] = -s; out[7] = 0; out[8] = c;
}

function rotZ(out: StaticArray<f64>, a: f64): void {
    let c = Math.cos(a), s = Math.sin(a);
    out[0] = c; out[1] = -s; out[2] = 0;
    out[3] = s; out[4] = c; out[5] = 0;
    out[6] = 0; out[7] = 0; out[8] = 1;
}

// out = a * b (3x3 row-major); out must not alias a or b
function mul3(out: StaticArray<f64>, a: StaticArray<f64>, b: StaticArray<f64>): void {
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
}

function colorAttachment(view: GPUTextureView?, r: f32, g: f32, b: f32, a: f32): GPUColorAttachment {
    let c = new GPUColorAttachment();
    c.view = view;
    c.loadOp = LoadOp.clear;
    c.storeOp = StoreOp.store;
    c.clearR = r;
    c.clearG = g;
    c.clearB = b;
    c.clearA = a;
    return c;
}

function colorTarget(fmt: TextureFormat): GPUColorTargetDesc {
    let c = new GPUColorTargetDesc();
    c.format = fmt;
    return c;
}

function renderTex(w: u32, h: u32, fmt: TextureFormat): GPUTexture {
    let d = new GPUTextureDesc();
    d.width = w;
    d.height = h;
    d.format = fmt;
    d.renderTarget = true;
    return new GPUTexture(d);
}

// ---------------------------------------------------------------- colour ramps
function luma(c: StaticArray<f64>): f64 {
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
}

class Obj {
    prim: Prim;
    ubo: GPUBuffer? = null;
    bg: GPUBindGroup? = null;
    id: i32;
    edgeId: i32;
    dark: StaticArray<f64> = [0, 0, 0];
    mid: StaticArray<f64> = [0, 0, 0];
    light: StaticArray<f64> = [0, 0, 0];
    spec: f64 = 0;
    isScreen: bool = false;
    model: Mat4 = Mat4.identity();

    constructor(prim: Prim, id: i32) {
        this.prim = prim;
        this.id = id;
        this.edgeId = id;
        let base = prim.color;
        let l = luma(base);
        let k = clamp((l - 0.08) / 0.4, 0, 1); // dark materials get a compressed ramp
        let hi = 0.34 * k + 0.05; // pale faces must stay clear of the blush field
        for (let c = 0; c < 3; c++) {
            this.dark[c] = Math.pow(base[c], 1.35) * 0.82;
            this.mid[c] = base[c];
            this.light[c] = base[c] + (1 - base[c]) * hi;
        }
    }
}

export class World extends Layout {
    @input screen: Artboard? = null;
    @input lineWidth: f64 = 1.3;
    @input motion: f64 = 1;
    @input intro: f64 = 1;

    context: Context? = null;
    vm: ViewModel? = null;
    scene: Scene? = null;
    objs: Array<Obj> = [];
    rootMats: Array<Mat4> = [];
    rootPhone: Array<bool> = [];
    inst: Artboard? = null;
    uiCanvas: Canvas? = null;
    uiScale: f64 = 1;
    uiW: f64 = 390;
    uiH: f64 = 844;

    gpu: GPUCanvas? = null;
    gpuImage: Image? = null;
    toonPipe: GPUPipeline? = null;
    inkPipe: GPUPipeline? = null;
    resPipe: GPUPipeline? = null;
    gColorV: GPUTextureView? = null;
    gNDV: GPUTextureView? = null;
    gIDV: GPUTextureView? = null;
    gDepthV: GPUTextureView? = null;
    inkV: GPUTextureView? = null;
    inkUbo: GPUBuffer? = null;
    resUbo: GPUBuffer? = null;
    inkBG: GPUBindGroup? = null;
    resBG: GPUBindGroup? = null;
    uiSampler: GPUSampler? = null;
    linSampler: GPUSampler? = null;
    gPassDesc: GPUPassDesc? = null;
    inkPassDesc: GPUPassDesc? = null;
    resPassDesc: GPUPassDesc? = null;
    gPass: GPURenderPass? = null;
    inkPass: GPURenderPass? = null;
    resPass: GPURenderPass? = null;
    bgDirty: bool = false;
    scratch: ArrayBuffer = new ArrayBuffer(512);
    sf: Float32Array = new Float32Array(0);
    imgSampler: ImageSampler = ImageSampler(ImageWrap.clamp, ImageWrap.clamp, ImageFilter.bilinear);

    sizeX: f64 = 0;
    sizeY: f64 = 0;
    outScale: f64 = 1;
    gScale: f64 = 2;
    gw: u32 = 1;
    gh: u32 = 1;
    ow: u32 = 1;
    oh: u32 = 1;

    time: f64 = 0;
    px: f64 = 0;
    py: f64 = 0;
    tpx: f64 = 0;
    tpy: f64 = 0;
    viewProj: Mat4 = Mat4.identity();
    view: Mat4 = Mat4.identity();
    proj: Mat4 = Mat4.identity();
    mvp: Mat4 = Mat4.identity();
    eyeX: f64 = 0;
    eyeY: f64 = 0;
    eyeZ: f64 = 1;
    screenObj: Obj? = null;
    screenInv: Mat4 = Mat4.identity();
    screenInvOk: bool = false;
    hoverScreen: bool = false;
    printed: bool = false;
    printedScale: bool = false;
    gpuReady: bool = false;
    needsResize: bool = false;
    hostScale: f64 = 1;

    // click-drag orbit
    yaw: f64 = 0;
    pitch: f64 = 0;
    yawVel: f64 = 0;
    pitchVel: f64 = 0;
    down: bool = false;
    dragging: bool = false;
    downX: f64 = 0;
    downY: f64 = 0;
    lastX: f64 = 0;
    lastY: f64 = 0;
    lastMoveT: f64 = 0;
    uiPressed: bool = false;

    // scratch for rootAnim and screenPoint (no per-frame allocations)
    r0: StaticArray<f64> = new StaticArray<f64>(9);
    r1: StaticArray<f64> = new StaticArray<f64>(9);
    r2: StaticArray<f64> = new StaticArray<f64>(9);
    r3: StaticArray<f64> = new StaticArray<f64>(9);
    r4: StaticArray<f64> = new StaticArray<f64>(9);
    tmp: Mat4 = Mat4.identity();
    hitX: f64 = 0;
    hitY: f64 = 0;
    fwd: PointerEvent = new PointerEvent(0, Vector.xy(0, 0));

    log(s: string): void {
        let c = this.context;
        if (c != null) {
            c.log(s);
        }
    }

    v4(off: i32, a: f64, b: f64, c: f64, d: f64): void {
        let i = off >> 2;
        this.sf[i] = <f32>a;
        this.sf[i + 1] = <f32>b;
        this.sf[i + 2] = <f32>c;
        this.sf[i + 3] = <f32>d;
    }

    // ------------------------------------------------------------ setup
    override init(context: Context): bool {
        this.context = context;
        this.sf = Float32Array.wrap(this.scratch);
        this.hookViewModel();
        if (this.scene == null) {
            let blob = context.blob("scene");
            if (blob == null) {
                this.log("world: blob 'scene' missing");
                return true;
            }
            let sc = load(blob);
            if (sc == null) {
                this.log(loadError);
                return true;
            }
            this.scene = sc;
            // one draw object per primitive
            let bezel = new Map<string, i32>();
            for (let i = 0; i < sc.prims.length; i++) {
                let p = sc.prims[i];
                let o = new Obj(p, i + 1);
                let ln = lower(p.meshName);
                o.spec = p.gloss || contains(ln, "sphere") ? 1 : 0;
                o.isScreen = p.isScreen;
                o.model = Mat4.identity();
                Mat4.multiply(o.model, Mat4.identity(), p.world);
                // front cover of the device: prefer "bezel", else a non-back "glass"
                let lm = lower(p.material);
                if (contains(lm, "bezel")) {
                    bezel.set(p.name, i);
                } else if (contains(lm, "glass") && !contains(lm, "back") && !bezel.has(p.name)) {
                    bezel.set(p.name, i);
                }
                this.objs.push(o);
            }
            // the screen shares an edge id with its bezel: no line between them
            for (let i = 0; i < this.objs.length; i++) {
                let o = this.objs[i];
                if (o.isScreen) {
                    if (bezel.has(o.prim.name)) {
                        let b = bezel.get(o.prim.name);
                        o.edgeId = b + 1;
                        let m = this.objs[b].mid;
                        o.dark = [m[0], m[1], m[2]];
                    }
                    this.screenObj = o;
                }
            }
            for (let r = 0; r < sc.roots.length; r++) {
                this.rootMats.push(Mat4.identity());
                this.rootPhone.push(contains(lower(sc.roots[r].name), "phone"));
            }
            this.log("world: glb loaded, " + sc.nodeCount.toString() + " nodes, " + sc.prims.length.toString() +
                " primitives, " + sc.roots.length.toString() + " roots, camera=" + (sc.camera != null ? "yes" : "auto"));
        }
        this.instanceScreen();
        return true;
    }

    // (Re)instances the Phone Screen artboard so its own Intro and chart draw-in play from the start.
    instanceScreen(): void {
        let ctx = this.context;
        let input = this.screen;
        if (ctx == null || input == null) {
            return;
        }
        let inst = input.instance(ctx.rootViewModel());
        inst.frameOrigin = true;
        this.inst = inst;
        this.uiW = inst.width;
        this.uiH = inst.height;
    }

    // All GPU objects are created here, lazily, the first time the host's GPU
    // is usable. Retried every frame until it succeeds.
    createGPU(): bool {
        let ctx = this.context!;
        let gpu = ctx.gpuCanvas();
        if (gpu == null) {
            return false;
        }
        this.gpu = gpu;
        this.gpuImage = gpu.image;
        this.uiSampler = new GPUSampler(Filter.linear, Filter.linear, Filter.linear, WrapMode.clampToEdge, WrapMode.clampToEdge);
        this.linSampler = new GPUSampler(Filter.linear, Filter.linear, Filter.linear, WrapMode.clampToEdge, WrapMode.clampToEdge);
        upload(this.scene!);
        for (let i = 0; i < this.objs.length; i++) {
            this.objs[i].ubo = new GPUBuffer(BufferUsage.uniform, UBO_SIZE);
        }
        return true;
    }

    fullPipe(module: GPUShaderModule, vs: string, fs: string, fmt: TextureFormat): GPUPipeline {
        let d = new GPUPipelineDesc();
        d.vertexModule = module;
        d.vertexEntryPoint = vs;
        d.fragmentModule = module;
        d.fragmentEntryPoint = fs;
        d.colorTargets = [colorTarget(fmt)];
        return new GPUPipeline(d);
    }

    makePipelines(): void {
        let ctx = this.context!;
        let toon = ctx.shader("toon");
        let inkS = ctx.shader("ink");
        let resS = ctx.shader("resolve");
        if (toon == null || inkS == null || resS == null) {
            this.log("world: shader missing");
            return;
        }
        let layout = new GPUVertexBufferLayout();
        layout.stride = 32;
        layout.attributes = [
            new GPUVertexAttribute(VertexFormat.float3, 0, 0),
            new GPUVertexAttribute(VertexFormat.float3, 1, 12),
            new GPUVertexAttribute(VertexFormat.float2, 2, 24),
        ];
        let d = new GPUPipelineDesc();
        d.vertexModule = toon;
        d.vertexEntryPoint = "vs_main";
        d.fragmentModule = toon;
        d.fragmentEntryPoint = "fs_main";
        d.vertexBuffers = [layout];
        d.colorTargets = [colorTarget(TextureFormat.rgba8unorm), colorTarget(TextureFormat.rgba16float), colorTarget(TextureFormat.rgba8unorm)];
        let ds = new GPUDepthStencilDesc();
        ds.format = TextureFormat.depth32float;
        ds.compare = CompareFunction.lessEqual;
        ds.write = true;
        d.depthStencil = ds;
        d.cullMode = CullMode.none;
        d.topology = PrimitiveTopology.triangleList;
        this.toonPipe = new GPUPipeline(d);
        this.inkPipe = this.fullPipe(inkS, "vs_full", "fs_ink", TextureFormat.rgba8unorm);
        this.resPipe = this.fullPipe(resS, "vs_full", "fs_resolve", <TextureFormat>this.gpu!.format);
        this.inkUbo = new GPUBuffer(BufferUsage.uniform, 48);
        this.resUbo = new GPUBuffer(BufferUsage.uniform, 16);
    }

    rebuildBindGroups(): void {
        let pipe = this.toonPipe;
        let ui = this.uiCanvas;
        let samp = this.uiSampler;
        if (pipe == null || ui == null || samp == null) {
            return;
        }
        let uiView = ui.image.view();
        if (uiView == null) {
            this.bgDirty = true; // the canvas has no backing yet: retry next frame
            return;
        }
        let layout = pipe.getBindGroupLayout(0);
        for (let i = 0; i < this.objs.length; i++) {
            let o = this.objs[i];
            o.bg = new GPUBindGroup(layout, [new GPUBindGroupUBO(0, o.ubo!)], [new GPUBindGroupTexture(1, uiView)],
                [new GPUBindGroupSampler(2, samp)]);
        }
        this.bgDirty = false;
    }

    doResize(): void {
        let gpu = this.gpu;
        if (gpu == null || this.scene == null) {
            return;
        }
        let scale = this.hostScale;
        // Output at device pixels, G-buffer supersampled on top. Some hosts (the
        // web runtime) report scale = 1 even on a Retina/HiDPI screen and apply the
        // device pixel ratio as a transform when presenting, which upscaled this
        // canvas 2x and made the 3D soft. So never render below 2x; on a true 1x
        // display that just acts as extra supersampling.
        this.outScale = clamp(Math.max(scale, MIN_RENDER_SCALE), 1, 2);
        let ss = this.outScale < 1.5 ? 2.0 : 1.25;
        this.gScale = this.outScale * ss;
        this.ow = <u32>Math.max(1, Math.floor(this.sizeX * this.outScale));
        this.oh = <u32>Math.max(1, Math.floor(this.sizeY * this.outScale));
        this.gw = <u32>Math.max(1, Math.floor(this.sizeX * this.gScale));
        this.gh = <u32>Math.max(1, Math.floor(this.sizeY * this.gScale));
        gpu.resize(this.ow, this.oh);
        if (this.toonPipe == null) {
            this.makePipelines();
        }
        let gw = this.gw, gh = this.gh;
        let gColor = renderTex(gw, gh, TextureFormat.rgba8unorm);
        let gND = renderTex(gw, gh, TextureFormat.rgba16float);
        let gID = renderTex(gw, gh, TextureFormat.rgba8unorm);
        let gDepth = renderTex(gw, gh, TextureFormat.depth32float);
        let ink = renderTex(gw, gh, TextureFormat.rgba8unorm);
        let gColorV = gColor.view();
        let gNDV = gND.view();
        let gIDV = gID.view();
        let dv = new GPUTextureViewDesc();
        dv.aspect = TextureAspect.depthOnly;
        let gDepthV = gDepth.view(dv);
        let inkV = ink.view();
        this.gColorV = gColorV;
        this.gNDV = gNDV;
        this.gIDV = gIDV;
        this.gDepthV = gDepthV;
        this.inkV = inkV;

        // pass descriptors, built once per size and reused every frame
        let gp = new GPUPassDesc();
        gp.color = [
            colorAttachment(gColorV, 0, 0, 0, 0),
            colorAttachment(gNDV, 0, 0, 1, 10000),
            colorAttachment(gIDV, 0, 0, 0, 0),
        ];
        gp.depthView = gDepthV;
        gp.depthLoadOp = LoadOp.clear;
        gp.depthStoreOp = StoreOp.discard;
        gp.depthClearValue = 1;
        this.gPassDesc = gp;
        let ip = new GPUPassDesc();
        ip.color = [colorAttachment(inkV, 0, 0, 0, 0)];
        this.inkPassDesc = ip;
        let rp = new GPUPassDesc();
        rp.color = [colorAttachment(null, 0, 0, 0, 0)];
        this.resPassDesc = rp;

        // phone screen texture: roughly match its on-screen texel density
        if (this.inst != null) {
            let s = clamp(<f64>gh * 0.8 / Math.max(this.uiH, 1), 1, 3);
            this.uiScale = s;
            let w = <u32>Math.floor(this.uiW * s);
            let h = <u32>Math.floor(this.uiH * s);
            let ui = this.uiCanvas;
            if (ui == null) {
                this.uiCanvas = this.context!.canvas(w, h);
            } else if (ui.width != w || ui.height != h) {
                ui.resize(w, h);
            }
        }
        this.rebuildBindGroups();

        let inkPipe = this.inkPipe;
        let resPipe = this.resPipe;
        if (inkPipe != null && resPipe != null) {
            this.inkBG = new GPUBindGroup(inkPipe.getBindGroupLayout(0), [new GPUBindGroupUBO(0, this.inkUbo!)], [
                new GPUBindGroupTexture(1, gColorV),
                new GPUBindGroupTexture(2, gNDV),
                new GPUBindGroupTexture(3, gIDV),
            ]);
            this.resBG = new GPUBindGroup(resPipe.getBindGroupLayout(0), [new GPUBindGroupUBO(0, this.resUbo!)],
                [new GPUBindGroupTexture(1, inkV)], [new GPUBindGroupSampler(2, this.linSampler!)]);
        }
    }

    trySetup(): void {
        if (this.scene == null || this.context == null) {
            return;
        }
        if (!this.gpuReady) {
            if (!this.createGPU()) {
                if (!this.printed) {
                    this.log("world: waiting for GPU");
                    this.printed = true;
                }
                return;
            }
            this.gpuReady = true;
            this.needsResize = true;
        }
        if (this.needsResize && this.sizeX > 0) {
            this.needsResize = false;
            this.doResize();
        }
    }

    override resize(width: f64, height: f64, scale: f64): void {
        if (scale != this.hostScale || !this.printedScale) {
            this.log("world: host scale " + scale.toString() + ", size " + (<i32>width).toString() + "x" + (<i32>height).toString());
            this.printedScale = true;
        }
        this.sizeX = width;
        this.sizeY = height;
        this.hostScale = scale;
        this.needsResize = true;
        this.trySetup();
    }

    // ------------------------------------------------------------ camera + motion
    updateCamera(): void {
        let sc = this.scene!;
        let cam = sc.camera;
        let aspect = this.sizeX / Math.max(this.sizeY, 1);
        let cx = (<f64>sc.bmin.x + <f64>sc.bmax.x) / 2;
        let cy = (<f64>sc.bmin.y + <f64>sc.bmax.y) / 2;
        let cz = (<f64>sc.bmin.z + <f64>sc.bmax.z) / 2;
        let ex: f64, ey: f64, ez: f64, tx: f64, ty: f64, tz: f64, ux: f64, uy: f64, uz: f64;
        let yfov: f64;
        let znear: f64 = 0.05, zfar: f64 = 100;
        if (cam != null) {
            let w = cam.world;
            ex = <f64>w[12]; ey = <f64>w[13]; ez = <f64>w[14];
            let fx = -<f64>w[8], fy = -<f64>w[9], fz = -<f64>w[10];
            ux = <f64>w[4]; uy = <f64>w[5]; uz = <f64>w[6];
            let dist = Math.max((cx - ex) * fx + (cy - ey) * fy + (cz - ez) * fz, 0.01);
            tx = ex + fx * dist; ty = ey + fy * dist; tz = ez + fz * dist;
            yfov = cam.yfov;
            znear = cam.znear;
            zfar = cam.zfar;
        } else {
            let dx = <f64>sc.bmax.x - <f64>sc.bmin.x, dy = <f64>sc.bmax.y - <f64>sc.bmin.y, dz = <f64>sc.bmax.z - <f64>sc.bmin.z;
            let rad = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
            yfov = 30 * Math.PI / 180;
            tx = cx; ty = cy; tz = cz;
            ex = cx; ey = cy; ez = cz + rad / Math.tan(yfov / 2);
            ux = 0; uy = 1; uz = 0;
            znear = rad * 0.05;
            zfar = rad * 20;
        }
        // keep the authored horizontal coverage on narrow viewports
        if (aspect < AUTHOR_ASPECT) {
            yfov = 2 * Math.atan(Math.tan(yfov / 2) * AUTHOR_ASPECT / aspect);
        }
        // pointer parallax plus the click-drag orbit (pitch first, then yaw about world up)
        let ox = ex - tx, oy = ey - ty, oz = ez - tz;
        let yaw = -this.px * 0.10 + this.yaw;
        let pitch = this.py * 0.06 + this.pitch;
        let cp = Math.cos(pitch), sp = Math.sin(pitch);
        let oy2 = oy * cp - oz * sp;
        let oz2 = oy * sp + oz * cp;
        let cyw = Math.cos(yaw), syw = Math.sin(yaw);
        let ox3 = ox * cyw + oz2 * syw;
        let oz3 = -ox * syw + oz2 * cyw;
        this.eyeX = tx + ox3;
        this.eyeY = ty + oy2;
        this.eyeZ = tz + oz3;
        Mat4.lookAtInto(this.view, <f32>this.eyeX, <f32>this.eyeY, <f32>this.eyeZ, <f32>tx, <f32>ty, <f32>tz, <f32>ux, <f32>uy, <f32>uz);
        Mat4.perspectiveInto(this.proj, <f32>yfov, <f32>aspect, <f32>znear, <f32>zfar);
        Mat4.multiply(this.viewProj, this.proj, this.view);
    }

    // Idle + entrance transform for a root, about its own centre, into rootMats[ri].
    rootAnim(ri: i32, root: Root): void {
        let t = this.time;
        let m = this.motion;
        let n = <f64>(ri + 1); // the motion seeds count roots from 1
        let seed = n * 1.731;
        let isPhone = this.rootPhone[ri];
        let intro = this.intro;
        let delay = isPhone ? 0.0 : 0.18 + (n % 7) * 0.07;
        let k = intro > 0.5 ? spring01((t - delay) / 0.9) : 1.0;
        let bob: f64, rx: f64, ry: f64, rz: f64;
        if (isPhone) {
            bob = Math.sin(t * 0.9) * 0.0016 * m;
            rx = Math.sin(t * 0.55 + 1.0) * 0.035 * m;
            ry = Math.sin(t * 0.7) * 0.07 * m;
            rz = Math.sin(t * 0.45 + 2.0) * 0.02 * m;
            // entrance: rise and unwind
            let e = intro > 0.5 ? easeOutCubic(t / 1.1) : 1.0;
            bob += (1 - e) * -0.06;
            ry += (1 - spring01(t / 1.1)) * 0.9;
        } else {
            let r = root.radius;
            bob = Math.sin(t * (0.8 + (n % 3) * 0.17) + seed) * r * 0.10 * m;
            rx = Math.sin(t * 0.6 + seed * 2.0) * 0.22 * m;
            ry = t * (0.10 + (n % 4) * 0.04) * m + seed;
            rz = Math.sin(t * 0.5 + seed) * 0.16 * m;
            ry -= seed; // no static offset; spin starts from authored pose
        }
        let s = Math.max(k, 0.0001);
        let R = this.r4;
        if (isPhone) {
            rotY(this.r0, ry);
            rotX(this.r1, rx);
            rotZ(this.r2, rz);
        } else {
            rotZ(this.r0, rz);
            rotX(this.r1, rx);
            rotY(this.r2, ry * 0.6);
        }
        mul3(this.r3, this.r0, this.r1);
        mul3(R, this.r3, this.r2);
        // T(c + bob) * R * S(s) * T(-c): linear part s*R, translation c + bob - s*R*c
        let c = root.center;
        let cx = <f64>c.x, cy = <f64>c.y, cz = <f64>c.z;
        let out = this.rootMats[ri];
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) {
                out[col * 4 + row] = <f32>(s * R[row * 3 + col]);
            }
            out[row + 3 * 4] = <f32>(-s * (R[row * 3] * cx + R[row * 3 + 1] * cy + R[row * 3 + 2] * cz));
        }
        out[12] = <f32>(<f64>out[12] + cx);
        out[13] = <f32>(<f64>out[13] + cy + bob);
        out[14] = <f32>(<f64>out[14] + cz);
        out[3] = 0;
        out[7] = 0;
        out[11] = 0;
        out[15] = 1;
    }

    // ------------------------------------------------------------ view model
    // The Replay button fires the view model trigger `replay`.
    hookViewModel(): void {
        let ctx = this.context;
        if (this.vm != null || ctx == null) {
            return;
        }
        let vm = ctx.rootViewModel();
        if (vm == null) {
            return;
        }
        this.vm = vm;
        let trig = vm.getTrigger("replay");
        if (trig != null) {
            trig.addListenerWith(this, (self: World): void => {
                self.replay();
            });
        }
    }

    // R (or the Replay button) replays the load animation: restarts the 3D entrance
    // (props pop in, phone rises and unwinds), resets the orbit to the hero pose, and
    // re-instances the Phone Screen artboard so its own Intro + chart draw-in play again.
    replay(): void {
        this.time = 0;
        this.yaw = 0;
        this.pitch = 0;
        this.yawVel = 0;
        this.pitchVel = 0;
        this.down = false;
        this.dragging = false;
        this.uiPressed = false;
        this.hoverScreen = false;
        this.instanceScreen();
    }

    // ------------------------------------------------------------ per frame
    override advance(dt: f64): bool {
        this.trySetup();
        let d = Math.min(dt, 1.0 / 20);
        this.hookViewModel();
        if (this.inst == null) {
            this.instanceScreen();
        }
        this.time += d;
        let k = 1 - Math.exp(-d * 4.0);
        this.px += (this.tpx - this.px) * k;
        this.py += (this.tpy - this.py) * k;
        // orbit inertia after release: sideways keeps its fling, up/down has
        // almost none and eases back to level so the shot never ends up tipped
        if (!this.dragging) {
            this.yaw += this.yawVel * d;
            this.yawVel *= Math.exp(-d * 3.2);
            this.pitchVel = 0;
            this.pitch *= Math.exp(-d * 1.6);
        }
        this.pitch = clamp(this.pitch, -PITCH_LIMIT, PITCH_LIMIT);
        let inst = this.inst;
        if (inst != null) {
            inst.advance(dt);
        }
        let sc = this.scene;
        if (sc != null) {
            this.updateCamera();
            for (let r = 0; r < sc.roots.length; r++) {
                this.rootAnim(r, sc.roots[r]);
            }
            for (let i = 0; i < this.objs.length; i++) {
                let o = this.objs[i];
                Mat4.multiply(o.model, this.rootMats[o.prim.root], o.prim.world);
            }
            let so = this.screenObj;
            if (so != null) {
                Mat4.multiply(this.tmp, this.viewProj, so.model);
                this.screenInvOk = Mat4.invert(this.screenInv, this.tmp);
            }
        }
        return true;
    }

    writeUBO(o: Obj): void {
        let b = this.scratch;
        Mat4.multiply(this.mvp, this.viewProj, o.model);
        this.mvp.writeToBuffer(b, 0);
        o.model.writeToBuffer(b, 64);
        this.view.writeToBuffer(b, 128);
        let scr = o.isScreen ? 1.0 : 0.0;
        this.v4(192, o.dark[0], o.dark[1], o.dark[2], 1);
        this.v4(208, o.mid[0], o.mid[1], o.mid[2], 1);
        this.v4(224, o.light[0], o.light[1], o.light[2], 1);
        this.v4(240, <f64>o.edgeId / 255, scr, o.spec, this.time);
        this.v4(256, LIGHT_X, LIGHT_Y, LIGHT_Z, 0.0);
        this.v4(272, this.eyeX, this.eyeY, this.eyeZ, 1.0);
        // notch: ~49% of the screen width, ~3.6% of its height (iPhone 12 proportions)
        this.v4(288, 0.49, 0.036, <f64>o.edgeId / 255, scr);
        let ubo = o.ubo;
        if (ubo != null) {
            ubo.update(0, b, UBO_SIZE, 0);
        }
    }

    override draw(renderer: Renderer): void {
        let gpu = this.gpu;
        let toonPipe = this.toonPipe;
        let gPassDesc = this.gPassDesc;
        if (!this.gpuReady || gpu == null || this.scene == null || toonPipe == null || gPassDesc == null) {
            return;
        }
        if (gpu.width <= 0) {
            return;
        }
        if (this.bgDirty) {
            this.rebuildBindGroups();
        }
        // 1. live phone screen artboard -> texture
        let ui = this.uiCanvas;
        let inst = this.inst;
        if (ui != null && inst != null) {
            let r = ui.beginFrame(0xFFFFFFFF);
            if (r != null) {
                r.save();
                r.transform(Mat2D.withUniformScale(<f32>this.uiScale));
                inst.draw(r);
                r.restore();
                ui.endFrame();
            }
        }

        // 2. G-buffer
        let pass = gpu.beginRenderPass(gPassDesc, this.gPass);
        this.gPass = pass;
        pass.setPipeline(toonPipe);
        for (let i = 0; i < this.objs.length; i++) {
            let o = this.objs[i];
            let bg = o.bg;
            let m = o.prim.mesh;
            let vb = m.vbuf;
            let ib = m.ibuf;
            if (bg != null && vb != null && ib != null) {
                this.writeUBO(o);
                pass.setBindGroup(0, bg);
                pass.setVertexBuffer(0, vb);
                pass.setIndexBuffer(ib, <u32>IndexFormat.uint32);
                pass.drawIndexed(<u32>m.count);
            }
        }
        pass.finish();

        // 3. ink lines at G-buffer resolution
        let lw = Math.max(this.lineWidth, 0.2);
        this.v4(0, lw * 0.5 * this.gScale, 0.82, 0.035, 0.32); // radius in texels, crease cos, depth step, ink = local colour * this
        this.v4(16, 0.07, 0.09, 0.06, 0.55); // ink tint, mix toward tinted ink
        this.v4(32, <f64>this.gw, <f64>this.gh, 0, 0);
        this.inkUbo!.update(0, this.scratch, 48, 0);
        let p2 = gpu.beginRenderPass(this.inkPassDesc!, this.inkPass);
        this.inkPass = p2;
        p2.setPipeline(this.inkPipe!);
        p2.setBindGroup(0, this.inkBG!);
        p2.draw(3);
        p2.finish();

        // 4. resolve to the canvas
        this.v4(0, 1 / <f64>this.gw, 1 / <f64>this.gh, this.gScale / this.outScale, 0);
        this.resUbo!.update(0, this.scratch, 16, 0);
        let p3 = gpu.beginRenderPass(this.resPassDesc!, this.resPass);
        this.resPass = p3;
        p3.setPipeline(this.resPipe!);
        p3.setBindGroup(0, this.resBG!);
        p3.draw(3);
        p3.finish();

        renderer.save();
        renderer.transform(Mat2D.withUniformScale(<f32>(1 / this.outScale)));
        renderer.drawImage(this.gpuImage!, this.imgSampler, BlendMode.srcOver, 1);
        renderer.restore();
    }

    // ------------------------------------------------------------ pointer -> phone screen
    // Ray-casts the pointer into the screen primitive's local space. On a hit,
    // stores artboard coordinates in hitX/hitY and returns true.
    screenPoint(px: f64, py: f64, unbounded: bool): bool {
        let so = this.screenObj;
        if (!this.screenInvOk || so == null || this.sizeX <= 0) {
            return false;
        }
        let inv = this.screenInv;
        let nx = <f32>(px / this.sizeX * 2 - 1);
        let ny = <f32>(1 - py / this.sizeY * 2);
        let a = inv.transformPoint(nx, ny, 0);
        let b = inv.transformPoint(nx, ny, 1);
        let lmin = so.prim.lmin;
        let lmax = so.prim.lmax;
        let zp = <f64>lmax.z; // screen is planar at its local max z
        let dz = <f64>b.z - <f64>a.z;
        if (Math.abs(dz) < 1e-12) {
            return false;
        }
        let t = (zp - <f64>a.z) / dz;
        let x = <f64>a.x + (<f64>b.x - <f64>a.x) * t;
        let y = <f64>a.y + (<f64>b.y - <f64>a.y) * t;
        let u = (x - <f64>lmin.x) / (<f64>lmax.x - <f64>lmin.x);
        let v = 1 - (y - <f64>lmin.y) / (<f64>lmax.y - <f64>lmin.y);
        if (!unbounded && (u < 0 || u > 1 || v < 0 || v > 1)) {
            return false;
        }
        this.hitX = u * this.uiW;
        this.hitY = v * this.uiH;
        return true;
    }

    // The forwarded event, reused so pointer moves allocate nothing.
    forward(id: f64, x: f64, y: f64): PointerEvent {
        let e = this.fwd;
        e.id = id;
        e.position = Vector.xy(<f32>x, <f32>y);
        return e;
    }

    override keyboardEvent(event: KeyboardEvent): bool {
        if (event.phase == "down" && event.key == KEY_R && !(event.control || event.meta || event.alt)) {
            this.replay();
            return true;
        }
        return false;
    }

    override pointerMove(event: PointerEvent): void {
        let px = <f64>event.position.x;
        let py = <f64>event.position.y;
        if (this.sizeX > 0) {
            this.tpx = clamp(px / this.sizeX * 2 - 1, -1, 1);
            this.tpy = clamp(py / this.sizeY * 2 - 1, -1, 1);
        }
        let inst = this.inst;
        // a press that started on the phone screen belongs to the app: forward the
        // whole gesture (scrolling, taps) and never orbit
        if (this.down && this.uiPressed && inst != null) {
            if (this.screenPoint(px, py, true)) {
                inst.pointerMove(this.forward(event.id, this.hitX, this.hitY));
            }
            event.hit();
            return;
        }
        if (this.down) {
            let dx = px - this.downX, dy = py - this.downY;
            if (!this.dragging && dx * dx + dy * dy > DRAG_SLOP * DRAG_SLOP) {
                this.dragging = true;
                this.yawVel = 0;
                this.pitchVel = 0;
            }
            if (this.dragging) {
                let mx = px - this.lastX, my = py - this.lastY;
                let dYaw = -mx * ORBIT_PER_PX;
                // soft limit: up/down gets stiffer as it nears PITCH_LIMIT
                let room = Math.max(0, 1 - Math.abs(this.pitch) / PITCH_LIMIT);
                let dPitch = my * PITCH_PER_PX * ((my > 0) == (this.pitch > 0) ? room : 1.0);
                this.yaw += dYaw;
                this.pitch = clamp(this.pitch + dPitch, -PITCH_LIMIT, PITCH_LIMIT);
                // velocity for the fling (per second, smoothed)
                let mdt = Math.max(this.time - this.lastMoveT, 1.0 / 120);
                this.yawVel = this.yawVel * 0.5 + (dYaw / mdt) * 0.5;
                this.pitchVel = this.pitchVel * 0.5 + (dPitch / mdt) * 0.5;
                this.lastMoveT = this.time;
                this.lastX = px;
                this.lastY = py;
                event.hit();
                return;
            }
            this.lastX = px;
            this.lastY = py;
        }
        let onScreen = this.screenPoint(px, py, false);
        if (inst != null && onScreen) {
            inst.pointerMove(this.forward(event.id, this.hitX, this.hitY));
            this.hoverScreen = true;
        } else if (inst != null && this.hoverScreen) {
            inst.pointerExit(this.forward(event.id, -100, -100));
            this.hoverScreen = false;
        }
    }

    override pointerDown(event: PointerEvent): void {
        let px = <f64>event.position.x;
        let py = <f64>event.position.y;
        this.down = true;
        this.dragging = false;
        this.downX = px;
        this.downY = py;
        this.lastX = px;
        this.lastY = py;
        this.lastMoveT = this.time;
        this.yawVel = 0; // grabbing stops a fling
        this.pitchVel = 0;
        let inst = this.inst;
        if (inst != null && this.screenPoint(px, py, false)) {
            inst.pointerDown(this.forward(event.id, this.hitX, this.hitY));
            this.uiPressed = true;
        }
        // presses off the phone screen stay translucent so listeners elsewhere
        // (the Replay button) still get them
        event.hit(!this.uiPressed);
    }

    override pointerUp(event: PointerEvent): void {
        let px = <f64>event.position.x;
        let py = <f64>event.position.y;
        let wasDragging = this.dragging;
        this.down = false;
        this.dragging = false;
        if (wasDragging) {
            // a fling only if the pointer was still moving at release
            if (this.time - this.lastMoveT > 0.08) {
                this.yawVel = 0;
                this.pitchVel = 0;
            }
            event.hit();
            return;
        }
        let inst = this.inst;
        let onScreen = this.uiPressed;
        if (inst != null && onScreen && this.screenPoint(px, py, true)) {
            inst.pointerUp(this.forward(event.id, this.hitX, this.hitY));
        }
        this.uiPressed = false;
        event.hit(!onScreen);
    }

    override pointerExit(event: PointerEvent): void {
        this.tpx = 0;
        this.tpy = 0;
        this.down = false;
        this.dragging = false;
        this.uiPressed = false;
        let inst = this.inst;
        if (inst != null && this.hoverScreen) {
            inst.pointerExit(this.forward(event.id, -100, -100));
            this.hoverScreen = false;
        }
    }
}
