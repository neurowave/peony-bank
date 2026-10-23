#!/usr/bin/env python3
"""Author assets/scene.glb — a stand-in for a glTF export from a 3D app such as C4D or Blender.

Everything the Rive file needs comes from this one binary glTF, exactly as it
would from Cinema 4D's glTF exporter: meshes with normals + UVs, named
materials (baseColorFactor, linear), a node hierarchy with TRS transforms, and
a perspective camera that frames the composition.

Swap this file for a real export and the Rive scene picks it up unchanged:
  * any material named "Screen" (case-insensitive) is textured with the live
    Rive phone-screen artboard,
  * every other material is restyled as flat vector art from its base colour,
  * top-level nodes get the idle float animation,
  * the first camera in the file is used (else an auto-fit camera).
"""
import json, math, struct, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "scene.glb")

# ---------------------------------------------------------------- helpers
def srgb_to_lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def hexcol(h):
    h = h.lstrip("#")
    return [srgb_to_lin(int(h[i:i + 2], 16)) for i in (0, 2, 4)] + [1.0]

def quat_from_euler(rx, ry, rz, order="ZXY"):
    """Rotation as quaternion (x,y,z,w); angles in degrees, applied Y then X then Z
    (i.e. R = Rz * Rx * Ry for order ZXY)."""
    def q_axis(ax, deg):
        a = math.radians(deg) / 2
        s = math.sin(a)
        v = [0, 0, 0]
        v["xyz".index(ax)] = s
        return np.array(v + [math.cos(a)])
    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz])
    q = np.array([0, 0, 0, 1.0])
    angles = {"x": rx, "y": ry, "z": rz}
    for ax in order.lower():          # left-most applied last
        q = qmul(q, q_axis(ax, angles[ax]))
    return [float(v) for v in q / np.linalg.norm(q)]


def quat_axis_y_to(d, twist=0.0):
    """Quaternion turning local +Y onto direction d, then spinning `twist` deg about it."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    ax = np.cross(y, d); sn = np.linalg.norm(ax); c = float(np.dot(y, d))
    if sn < 1e-8:
        q1 = np.array([0, 0, 0, 1.0]) if c > 0 else np.array([1.0, 0, 0, 0])
    else:
        ax /= sn; a = math.atan2(sn, c) / 2
        q1 = np.array([*(ax * math.sin(a)), math.cos(a)])
    t = math.radians(twist) / 2
    q2 = np.array([0, math.sin(t), 0, math.cos(t)])   # local spin first
    ax_, ay, az, aw = q1; bx, by, bz, bw = q2
    q = np.array([aw*bx + ax_*bw + ay*bz - az*by, aw*by - ax_*bz + ay*bw + az*bx,
                  aw*bz + ax_*by - ay*bx + az*bw, aw*bw - ax_*bx - ay*by - az*bz])
    return [float(v) for v in q / np.linalg.norm(q)]


class Mesh:
    """Indexed triangle soup: pos, normal, uv."""
    def __init__(self):
        self.p, self.n, self.t, self.i = [], [], [], []

    def vert(self, p, n, t=(0.0, 0.0)):
        self.p.append(p); self.n.append(n); self.t.append(t)
        return len(self.p) - 1

    def tri(self, a, b, c):
        self.i += [a, b, c]

    def flat_poly(self, pts, uvs=None, outward_from=None):
        """Convex planar polygon (CCW seen from its front), fan triangulated.
        With outward_from, the face is flipped if its normal points at that point."""
        pts = [np.asarray(p, float) for p in pts]
        # robust normal (Newell)
        n = np.zeros(3)
        for k in range(len(pts)):
            a, b = pts[k], pts[(k + 1) % len(pts)]
            n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
        n = n / np.linalg.norm(n)
        if outward_from is not None:
            c = sum(pts) / len(pts)
            if np.dot(n, c - np.asarray(outward_from, float)) < 0:
                n = -n
                pts = pts[::-1]
                uvs = uvs[::-1] if uvs else None
        ids = [self.vert(p, n, uvs[k] if uvs else (0, 0)) for k, p in enumerate(pts)]
        for k in range(1, len(ids) - 1):
            self.tri(ids[0], ids[k], ids[k + 1])

    def arrays(self):
        return (np.array(self.p, np.float32), np.array(self.n, np.float32),
                np.array(self.t, np.float32), np.array(self.i, np.uint32))


def rounded_rect(w, h, r, seg=10):
    """CCW outline of a rounded rectangle centred on the origin (XY plane)."""
    pts = []
    corners = [(w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
               (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)]
    for cx, cy, a0 in corners:
        for k in range(seg + 1):
            a = math.radians(a0 + 90 * k / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def extrude_sides(m, outline, z0, z1, smooth=True):
    """Side walls of an outline extruded from z0 (back) to z1 (front)."""
    n = len(outline)
    o = np.array(outline)
    # per-vertex outward normals (average of adjacent edge normals)
    en = []
    for k in range(n):
        a, b = o[k], o[(k + 1) % n]
        d = b - a
        en.append(np.array([d[1], -d[0]]) / np.linalg.norm(d))
    for k in range(n):
        a, b = o[k], o[(k + 1) % n]
        if smooth:
            na = en[k - 1] + en[k]; na /= np.linalg.norm(na)
            nb = en[k] + en[(k + 1) % n]; nb /= np.linalg.norm(nb)
        else:
            na = nb = en[k]
        i0 = m.vert((a[0], a[1], z0), (na[0], na[1], 0))
        i1 = m.vert((b[0], b[1], z0), (nb[0], nb[1], 0))
        i2 = m.vert((b[0], b[1], z1), (nb[0], nb[1], 0))
        i3 = m.vert((a[0], a[1], z1), (na[0], na[1], 0))
        m.tri(i0, i1, i2); m.tri(i0, i2, i3)


def cap(m, outline, z, front=True, uv_rect=None):
    pts = [(x, y, z) for x, y in outline]
    uvs = None
    if uv_rect:
        x0, y0, w, h = uv_rect
        uvs = [((x - x0) / w, 1.0 - (y - y0) / h) for x, y in outline]
    if not front:
        pts = pts[::-1]
        uvs = uvs[::-1] if uvs else None
    m.flat_poly(pts, uvs)


def box(m, cx, cy, cz, sx, sy, sz):
    x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2
    P = lambda x, y, z: (x, y, z)
    c = (cx, cy, cz)
    m.flat_poly([P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)], outward_from=c)
    m.flat_poly([P(x1, y0, z0), P(x0, y0, z0), P(x0, y1, z0), P(x1, y1, z0)], outward_from=c)
    m.flat_poly([P(x1, y0, z1), P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1)], outward_from=c)
    m.flat_poly([P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0)], outward_from=c)
    m.flat_poly([P(x0, y1, z1), P(x1, y1, z1), P(x1, y1, z0), P(x0, y1, z0)], outward_from=c)
    m.flat_poly([P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)], outward_from=c)


def prism(n, radius, length, twist=0.0):
    """Flat-shaded n-gon prism along Y, centred."""
    m = Mesh()
    ring = [(radius * math.cos(2 * math.pi * k / n + twist), radius * math.sin(2 * math.pi * k / n + twist)) for k in range(n)]
    y0, y1 = -length / 2, length / 2
    for k in range(n):
        a, b = ring[k], ring[(k + 1) % n]
        m.flat_poly([(a[0], y0, -a[1]), (b[0], y0, -b[1]), (b[0], y1, -b[1]), (a[0], y1, -a[1])], outward_from=(0, 0, 0))
    m.flat_poly([(x, y1, -z) for x, z in ring], outward_from=(0, 0, 0))
    m.flat_poly([(x, y0, -z) for x, z in ring], outward_from=(0, 0, 0))
    return m


def cone(radius, height, seg=40):
    """Smooth cone along +Y (apex at +h/2), flat base."""
    m = Mesh()
    y0, y1 = -height / 2, height / 2
    sl = math.atan2(radius, height)
    for k in range(seg):
        a0 = 2 * math.pi * k / seg
        a1 = 2 * math.pi * (k + 1) / seg
        am = (a0 + a1) / 2
        def nrm(a):
            return (math.cos(sl) * math.cos(a), math.sin(sl), -math.cos(sl) * math.sin(a))
        p0 = (radius * math.cos(a0), y0, -radius * math.sin(a0))
        p1 = (radius * math.cos(a1), y0, -radius * math.sin(a1))
        i0 = m.vert(p0, nrm(a0)); i1 = m.vert(p1, nrm(a1)); i2 = m.vert((0, y1, 0), nrm(am))
        m.tri(i0, i1, i2)
    base = [(radius * math.cos(2 * math.pi * k / seg), y0, -radius * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
    m.flat_poly(base, outward_from=(0, 0, 0))
    return m


def lathe(profile, seg=48):
    """Revolve a closed (r, y) profile around Y. Each profile edge is its own
    band: vertices are shared around the band (smooth), duplicated between bands
    (creased), so silhouette and crease edges stay crisp."""
    m = Mesh()
    for e in range(len(profile) - 1):
        (r0, y0), (r1, y1) = profile[e], profile[e + 1]
        dr, dy = r1 - r0, y1 - y0
        L = math.hypot(dr, dy)
        if L < 1e-9 or (abs(r0) < 1e-9 and abs(r1) < 1e-9):
            continue
        nr, ny = -dy / L, dr / L          # outward for a clockwise (r,y) profile
        base = len(m.p)
        for k in range(seg + 1):
            a = 2 * math.pi * k / seg
            ca, sa = math.cos(a), -math.sin(a)
            n = (nr * ca, ny, nr * sa)
            m.vert((r0 * ca, y0, r0 * sa), n)
            m.vert((r1 * ca, y1, r1 * sa), n)
        for k in range(seg):
            i0, i3 = base + 2 * k, base + 2 * k + 1
            i1, i2 = i0 + 2, i3 + 2
            m.tri(i0, i2, i1); m.tri(i0, i3, i2)
    return m


def sphere(radius, seg_u=40, seg_v=24):
    m = Mesh()
    for j in range(seg_v + 1):
        v = j / seg_v
        ph = math.pi * v
        for k in range(seg_u + 1):
            u = k / seg_u
            th = 2 * math.pi * u
            n = (math.sin(ph) * math.cos(th), math.cos(ph), -math.sin(ph) * math.sin(th))
            m.vert(tuple(radius * c for c in n), n, (u, v))
    row = seg_u + 1
    for j in range(seg_v):
        for k in range(seg_u):
            a = j * row + k; b = a + 1; c = a + row; d = c + 1
            m.tri(a, c, b); m.tri(b, c, d)
    return m


# ---------------------------------------------------------------- phone
W, H, D = 0.0715, 0.1467, 0.0078
R = 0.0105
BEZEL = 0.0034
SW, SH = W - 2 * BEZEL, H - 2 * BEZEL


def phone_meshes():
    outline = rounded_rect(W, H, R, 12)
    frame = Mesh()
    extrude_sides(frame, outline, -D / 2, D / 2, smooth=True)
    # side buttons (power right, volume + silent left)
    box(frame, W / 2 + 0.0004, 0.022, 0, 0.0012, 0.017, 0.0026)
    box(frame, -W / 2 - 0.0004, 0.034, 0, 0.0012, 0.006, 0.0026)
    box(frame, -W / 2 - 0.0004, 0.020, 0, 0.0012, 0.010, 0.0026)
    box(frame, -W / 2 - 0.0004, 0.006, 0, 0.0012, 0.010, 0.0026)

    bezel = Mesh()
    cap(bezel, outline, D / 2, front=True)

    back = Mesh()
    cap(back, outline, -D / 2, front=False)

    # rear camera module (top right seen from the front = top left seen from the back)
    cam = Mesh()
    lens = Mesh()
    cx, cy = W / 2 - 0.0045 - 0.0145, H / 2 - 0.0045 - 0.0145
    plate = [(cx + x, cy + y) for x, y in rounded_rect(0.029, 0.029, 0.0075, 10)]
    extrude_sides(cam, plate, -D / 2 - 0.0011, -D / 2, smooth=True)
    cap(cam, plate, -D / 2 - 0.0011, front=False)
    for lx, ly, r in [(-0.0068, 0.0068, 0.0058), (-0.0068, -0.0068, 0.0058), (0.0072, 0.0, 0.0050)]:
        ring = lathe([(0.0, 0.0018), (r * 0.62, 0.0018), (r * 0.8, 0.0012), (r, 0.0012), (r, 0.0), (0.0, 0.0)], 24)
        for k, (px, py, pz) in enumerate(ring.p):
            nx, ny, nz = ring.n[k]
            ring.p[k] = (px + cx + lx, pz + cy + ly, -py - D / 2 - 0.0011)
            ring.n[k] = (nx, nz, -ny)
        base = len(lens.p)
        lens.p += ring.p; lens.n += ring.n; lens.t += ring.t
        lens.i += [i + base for i in ring.i]

    screen = Mesh()
    s_outline = rounded_rect(SW, SH, R - BEZEL * 0.85, 12)
    cap(screen, s_outline, D / 2 + 0.00006, front=True, uv_rect=(-SW / 2, -SH / 2, SW, SH))
    return [("Frame", frame), ("Bezel", bezel), ("Back", back), ("Screen", screen), ("Camera", cam), ("Lens", lens)]



# ---------------------------------------------------------------- mesh ops
def Tm(x, y, z):
    M = np.eye(4); M[:3, 3] = (x, y, z); return M

def Sm(x, y=None, z=None):
    y = x if y is None else y; z = x if z is None else z
    return np.diag([x, y, z, 1.0])

def Rm(axis, deg):
    a = math.radians(deg); c, s_ = math.cos(a), math.sin(a)
    M = np.eye(4)
    if axis == "x": M[1:3, 1:3] = [[c, -s_], [s_, c]]
    elif axis == "y": M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s_, -s_, c
    else: M[0:2, 0:2] = [[c, -s_], [s_, c]]
    return M

def xform(m, M):
    out = Mesh()
    N = np.linalg.inv(M[:3, :3]).T
    for p, n, t in zip(m.p, m.n, m.t):
        pp = M @ np.array([p[0], p[1], p[2], 1.0])
        nn = N @ np.array(n, float); nn /= max(np.linalg.norm(nn), 1e-12)
        out.p.append(tuple(pp[:3])); out.n.append(tuple(nn)); out.t.append(t)
    out.i = list(m.i)
    return out

def merge(*ms):
    out = Mesh()
    for m in ms:
        b = len(out.p)
        out.p += m.p; out.n += m.n; out.t += m.t; out.i += [i + b for i in m.i]
    return out

def cylinder(r, h, seg=28):
    return lathe([(0.0, h / 2), (r, h / 2), (r, -h / 2), (0.0, -h / 2)], seg)

def boxm(sx, sy, sz):
    m = Mesh(); box(m, 0, 0, 0, sx, sy, sz); return m

def torus(R, r, segU=40, segV=12, arc=2 * math.pi):
    """Smooth torus around Y (ring in the XZ plane); `arc` < 2pi gives an open segment."""
    m = Mesh()
    nu = segU + (1 if arc < 2 * math.pi - 1e-6 else 0)
    for i in range(segU + 1):
        u = arc * i / segU
        for j in range(segV + 1):
            v = 2 * math.pi * j / segV
            n = (math.cos(v) * math.cos(u), math.sin(v), math.cos(v) * math.sin(u))
            p = ((R + r * math.cos(v)) * math.cos(u), r * math.sin(v), (R + r * math.cos(v)) * math.sin(u))
            m.vert(p, n)
    row = segV + 1
    for i in range(segU):
        for j in range(segV):
            a = i * row + j; b = a + 1; c = a + row; d = c + 1
            m.tri(a, b, c); m.tri(b, d, c)
    return m

def rbox(w, h, d, r, seg=8):
    """Rounded-rectangle slab: outline in XY, thickness along Z."""
    m = Mesh(); o = rounded_rect(w, h, r, seg)
    extrude_sides(m, o, -d / 2, d / 2, smooth=True)
    cap(m, o, d / 2, front=True); cap(m, o, -d / 2, front=False)
    return m

def tri_prism_xy(base, height, depth):
    """Pediment: isosceles triangle in XY (base on y=0) extruded along Z."""
    m = Mesh(); hb, z0, z1 = base / 2, -depth / 2, depth / 2
    c = (0, height / 3, 0)
    A, B, C = (-hb, 0), (hb, 0), (0, height)
    m.flat_poly([(A[0], A[1], z1), (B[0], B[1], z1), (C[0], C[1], z1)], outward_from=c)
    m.flat_poly([(A[0], A[1], z0), (B[0], B[1], z0), (C[0], C[1], z0)], outward_from=c)
    for P0, P1 in [(A, B), (B, C), (C, A)]:
        m.flat_poly([(P0[0], P0[1], z0), (P1[0], P1[1], z0), (P1[0], P1[1], z1), (P0[0], P0[1], z1)], outward_from=c)
    return m


# ---------------------------------------------------------------- banking props (each centred, +Y up)
def bank_building():
    main, light = [], []
    main.append(xform(boxm(0.058, 0.004, 0.032), Tm(0, 0.002, 0)))
    main.append(xform(boxm(0.052, 0.004, 0.028), Tm(0, 0.006, 0)))
    main.append(xform(boxm(0.046, 0.028, 0.004), Tm(0, 0.022, -0.008)))          # back wall
    for x in (-0.018, -0.006, 0.006, 0.018):
        light.append(xform(cylinder(0.0029, 0.026), Tm(x, 0.021, 0.005)))
        light.append(xform(boxm(0.0078, 0.0018, 0.0078), Tm(x, 0.0089, 0.005)))
        light.append(xform(boxm(0.0078, 0.0018, 0.0078), Tm(x, 0.0331, 0.005)))
    main.append(xform(boxm(0.054, 0.005, 0.030), Tm(0, 0.0365, 0)))              # architrave
    main.append(xform(tri_prism_xy(0.058, 0.015, 0.030), Tm(0, 0.039, 0)))        # pediment
    return xform(merge(*main), Tm(0, -0.027, 0)), xform(merge(*light), Tm(0, -0.027, 0))

def credit_cards():
    a = rbox(0.054, 0.034, 0.0018, 0.0036)
    b = xform(rbox(0.054, 0.034, 0.0018, 0.0036), Tm(0.007, 0.008, -0.006) @ Rm("z", 14))
    chip = merge(xform(rbox(0.0095, 0.0075, 0.0008, 0.0014), Tm(-0.015, 0.002, 0.0011)),
                 xform(boxm(0.022, 0.0026, 0.0006), Tm(-0.006, -0.009, 0.0010)))
    return a, b, chip

def piggy_bank():
    body = [xform(sphere(1.0, 36, 22), Sm(0.023, 0.0185, 0.0185))]
    body.append(xform(cylinder(0.0068, 0.0065), Tm(0.0235, 0.001, 0) @ Rm("z", -90)))
    for x in (-0.011, 0.011):
        for z in (-0.0085, 0.0085):
            body.append(xform(cylinder(0.0043, 0.009, 18), Tm(x, -0.0165, z)))
    for z in (-0.0075, 0.0075):
        body.append(xform(cone(0.0045, 0.0075, 18), Tm(0.012, 0.0165, z) @ Rm("z", -18)))
    body.append(xform(torus(0.0025, 0.0008, 24, 10, 1.6 * math.pi), Tm(-0.0245, 0.004, 0) @ Rm("x", -90)))
    dark = [xform(boxm(0.0105, 0.0016, 0.0022), Tm(-0.002, 0.0185, 0))]
    for z in (-0.0072, 0.0072):
        dark.append(xform(sphere(0.0016, 16, 10), Tm(0.0178, 0.0062, z)))
    for z in (-0.0022, 0.0022):
        dark.append(xform(sphere(0.0011, 12, 8), Tm(0.0268, 0.0015, z)))
    return merge(*body), merge(*dark)

COIN_PROFILE = [(0.0, 0.0024), (0.0105, 0.0024), (0.0118, 0.0034), (0.0150, 0.0034),
                (0.0150, -0.0034), (0.0118, -0.0034), (0.0105, -0.0024), (0.0, -0.0024)]

def coin_stack():
    coins = []
    for k, (dx, dz, rot) in enumerate([(0, 0, 0), (0.0012, -0.0008, 10), (-0.0010, 0.0011, 25), (0.0016, 0.0006, 40)]):
        coins.append(xform(lathe(COIN_PROFILE, 36), Tm(dx, -0.010 + k * 0.0069, dz) @ Rm("y", rot)))
    coins.append(xform(lathe(COIN_PROFILE, 36), Tm(0.021, -0.006, 0.004) @ Rm("z", 72) @ Rm("x", 8)))  # leaning coin
    return merge(*coins)

def padlock():
    body = rbox(0.027, 0.022, 0.010, 0.0042)
    shackle = merge(xform(torus(0.0084, 0.0022, 24, 10, math.pi), Tm(0, 0.0145, 0) @ Rm("x", -90)),
                    xform(cylinder(0.0022, 0.0055, 14), Tm(0.0084, 0.0118, 0)),
                    xform(cylinder(0.0022, 0.0055, 14), Tm(-0.0084, 0.0118, 0)))
    key = merge(xform(cylinder(0.0024, 0.0008, 24), Tm(0, 0.0012, 0.0052) @ Rm("x", 90)),
                xform(boxm(0.0016, 0.0050, 0.0008), Tm(0, -0.0015, 0.0052)))
    return xform(body, Tm(0, -0.004, 0)), xform(shackle, Tm(0, -0.004, 0)), xform(key, Tm(0, -0.004, 0))

def percent_sign():
    rings = merge(xform(torus(0.0046, 0.0017, 28, 10), Tm(-0.0065, 0.0068, 0) @ Rm("x", -90)),
                  xform(torus(0.0046, 0.0017, 28, 10), Tm(0.0065, -0.0068, 0) @ Rm("x", -90)))
    slash = xform(boxm(0.0032, 0.027, 0.0034), Rm("z", -33))
    return rings, slash

# ---------------------------------------------------------------- glTF writer
class GLTF:
    def __init__(self):
        self.bin = bytearray()
        self.j = {"asset": {"version": "2.0", "generator": "peony-bank make_scene_glb.py (C4D export stand-in)"},
                  "scene": 0, "scenes": [{"name": "Scene", "nodes": []}], "nodes": [], "meshes": [],
                  "materials": [], "accessors": [], "bufferViews": [], "buffers": [], "cameras": []}
        self.mat_index = {}
        self.dequant = {}

    def material(self, name, hexc, roughness=0.6):
        if name in self.mat_index:
            return self.mat_index[name]
        self.j["materials"].append({"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": hexcol(hexc), "metallicFactor": 0.0, "roughnessFactor": roughness}})
        self.mat_index[name] = len(self.j["materials"]) - 1
        return self.mat_index[name]

    def _view(self, data, target):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        self.j["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        return len(self.j["bufferViews"]) - 1

    def _acc(self, arr, typ, comp, target, minmax=False, normalized=False, stride=None, ncomp=None):
        v = self._view(arr.tobytes(), target)
        if stride:
            self.j["bufferViews"][v]["byteStride"] = stride
        a = {"bufferView": v, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if normalized:
            a["normalized"] = True
        if minmax:
            cols = arr[:, :ncomp] if ncomp else arr
            cast = int if comp in (5120, 5121, 5122, 5123, 5125) else float
            a["min"] = [cast(x) for x in cols.min(axis=0)]
            a["max"] = [cast(x) for x in cols.max(axis=0)]
        self.j["accessors"].append(a)
        return len(self.j["accessors"]) - 1

    def mesh(self, name, parts, quantize=True):
        """parts: [(material_index, Mesh)].
        KHR_mesh_quantization: positions as normalized int16 (dequantized by a
        uniform scale + offset on a child node), normals as normalized int8.
        UVs are only written for the Screen material (nothing else samples them)."""
        allp = np.concatenate([np.array(m.p, np.float64) for _, m in parts])
        c = (allp.min(0) + allp.max(0)) / 2
        e = float(np.abs(allp - c).max()) or 1.0
        prims = []
        for mi, m in parts:
            p, n, t, i = m.arrays()
            attrs = {}
            if quantize:
                q = np.zeros((p.shape[0], 4), np.int16)
                q[:, :3] = np.round((p - c) / e * 32767).clip(-32767, 32767)
                attrs["POSITION"] = self._acc(q, "VEC3", 5122, 34962, minmax=True, normalized=True, stride=8, ncomp=3)
                qn = np.zeros((n.shape[0], 4), np.int8)
                qn[:, :3] = np.round(n * 127).clip(-127, 127)
                attrs["NORMAL"] = self._acc(qn, "VEC3", 5120, 34962, normalized=True, stride=4, ncomp=3)
            else:
                attrs["POSITION"] = self._acc(p, "VEC3", 5126, 34962, True)
                attrs["NORMAL"] = self._acc(n, "VEC3", 5126, 34962)
            if self.j["materials"][mi]["name"].lower().find("screen") >= 0:
                attrs["TEXCOORD_0"] = self._acc(t, "VEC2", 5126, 34962)
            idx = i.astype(np.uint16) if p.shape[0] < 65536 else i
            prims.append({"attributes": attrs,
                          "indices": self._acc(idx, "SCALAR", 5123 if idx.dtype == np.uint16 else 5125, 34963),
                          "material": mi})
        self.j["meshes"].append({"name": name, "primitives": prims})
        mesh_i = len(self.j["meshes"]) - 1
        if quantize:
            self.dequant[mesh_i] = (c, e)
        return mesh_i

    def node(self, name, mesh=None, t=None, r=None, s=None, children=None, camera=None, root=True):
        nd = {"name": name}
        if mesh is not None and mesh in self.dequant:
            c, e = self.dequant[mesh]
            child = self.node(name + "_mesh", mesh=None, t=tuple(c), s=e, root=False)
            self.j["nodes"][child]["mesh"] = mesh
            children = (children or []) + [child]
            mesh = None
        if mesh is not None: nd["mesh"] = mesh
        if camera is not None: nd["camera"] = camera
        if t: nd["translation"] = [float(x) for x in t]
        if r: nd["rotation"] = [float(x) for x in r]
        if s: nd["scale"] = [float(x) for x in s] if hasattr(s, "__len__") else [float(s)] * 3
        if children: nd["children"] = children
        self.j["nodes"].append(nd)
        idx = len(self.j["nodes"]) - 1
        if root:
            self.j["scenes"][0]["nodes"].append(idx)
        return idx

    def write(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.j["buffers"] = [{"byteLength": len(self.bin)}]
        if not self.j["cameras"]:
            del self.j["cameras"]
        if self.dequant:
            self.j["extensionsUsed"] = ["KHR_mesh_quantization"]
            self.j["extensionsRequired"] = ["KHR_mesh_quantization"]
        js = json.dumps(self.j, separators=(",", ":")).encode()
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942)); f.write(self.bin)
        return total



# ---------------------------------------------------------------- clearance solver
PHONE_T = (0.004, -0.002, 0.0)
PHONE_Q = quat_from_euler(10, -32, -17)
EYE = (0.0, 0.004, 0.60)

def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def phone_samples():
    """Dense points on the phone's front/back faces and rim, world space."""
    R = quat_mat(PHONE_Q); t = np.array(PHONE_T)
    pts = []
    for x in np.linspace(-W / 2, W / 2, 24):
        for y in np.linspace(-H / 2, H / 2, 48):
            for z in (-D / 2 - 0.0012, D / 2):
                pts.append((x, y, z))
    for x, y in rounded_rect(W + 0.0016, H, R_CORNER, 10):
        for z in np.linspace(-D / 2, D / 2, 3):
            pts.append((x, y, z))
    return (np.array(pts) @ R.T) + t

R_CORNER = 0.0105

def swept_cylinder(verts):
    """Bounds of a prop spinning about world Y around its AABB centre, with the
    idle tilt (rx 0.22, rz 0.16 rad) and bob (10% of radius) folded in."""
    c = (verts.min(0) + verts.max(0)) / 2
    d = verts - c
    rh = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2).max()
    hh = np.abs(d[:, 1]).max()
    rad = np.sqrt((d ** 2).sum(1)).max()
    return c, rh + hh * 0.24, hh + rh * 0.24 + rad * 0.10

def resolve_clearance(props, phone_t, phone_q, eye, gap=0.0025, sway=0.004):
    eye = np.array(eye)
    phone = phone_samples()
    base = []
    for name, parts, t, q in props:
        R = quat_mat(q)
        v = np.concatenate([np.array(m.p) for _, m in parts]) @ R.T     # rotated, unscaled, about node origin
        c0, rh, hh = swept_cylinder(v)
        centre = np.array(t) + c0
        base.append(dict(name=name, rot_c=c0, rh=rh, hh=hh, centre=centre, k=1.0))

    def state(b):
        # slide the pivot along the eye ray by factor k; scale k keeps the on-screen size
        c = eye + (b["centre"] - eye) * b["k"]
        return c, b["rh"] * b["k"], b["hh"] * b["k"]

    def hits_phone(b):
        c, rh, hh = state(b)
        d = phone - c
        inside = (np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2) < rh + sway) & (np.abs(d[:, 1]) < hh + sway)
        return bool(inside.any())

    def hits(a, b):
        ca, ra, ha = state(a); cb, rb, hb = state(b)
        horiz = math.hypot(ca[0] - cb[0], ca[2] - cb[2]) < ra + rb + gap
        vert = abs(ca[1] - cb[1]) < ha + hb + gap
        return horiz and vert

    for it in range(400):
        moved = False
        for i, b in enumerate(base):
            if hits_phone(b):
                b["k"] *= 1.01; moved = True
            for j, o in enumerate(base):
                if j <= i or not hits(b, o):
                    continue
                far = b if np.linalg.norm(state(b)[0] - eye) >= np.linalg.norm(state(o)[0] - eye) else o
                far["k"] *= 1.01; moved = True
        if not moved:
            break
    out = []
    for b in base:
        c, _, _ = state(b)
        t = c - b["rot_c"] * b["k"]
        print(f"  {b['name']:10s} pushed back x{b['k']:.3f}  centre z {b['centre'][2]:+.3f} -> {c[2]:+.3f}")
        out.append(((float(t[0]), float(t[1]), float(t[2])), float(b["k"])))
    print("  clearance solved in", it + 1, "passes")
    return out

# ---------------------------------------------------------------- scene
def build():
    g = GLTF()
    GREEN = g.material("Green", "#F2559A")
    MINT = g.material("Mint", "#F78DBB")
    LIME = g.material("Lime", "#FF7FB4")
    FRAME = g.material("Frame", "#A04F74")
    BEZEL_M = g.material("Bezel", "#1E1C2E")
    SCREEN = g.material("Screen", "#FFFFFF")

    pm = {k: v for k, v in phone_meshes()}
    BACK_M = g.material("BackGlass", "#BC6F92")
    LENS_M = g.material("Lens", "#1E1C2E")
    phone_mesh = g.mesh("iPhone", [(FRAME, pm["Frame"]), (BEZEL_M, pm["Bezel"]), (BACK_M, pm["Back"]),
                                    (SCREEN, pm["Screen"]), (FRAME, pm["Camera"]), (LENS_M, pm["Lens"])])
    g.node("Phone", phone_mesh, t=PHONE_T, r=PHONE_Q)

    # banking props — sizes in metres, placed around the phone (phone centre = origin)
    PINK_LIGHT = g.material("PinkLight", "#F78DBB")
    PINK_GLOSS = g.material("PinkGloss", "#FF7FB4", roughness=0.2)   # glossy -> specular dot
    PINK_DEEP = g.material("PinkDeep", "#D8467F")
    PINK_INK = g.material("PinkInk", "#6A1F43")

    # Authored positions (close in around the phone, as in the approved layout).
    # resolve_clearance() then slides each prop back along its camera ray — scaled
    # up so it looks identical from the render camera — until its swept volume
    # (it spins about world Y, tilts and bobs) clears the swaying phone and every
    # other prop. Screen composition stays; 3D intersections go away.
    bm, bl = bank_building()
    ca, cb, chip = credit_cards()
    pl_body, pl_shackle, pl_key = padlock()
    pig, pig_dark = piggy_bank()
    rings, slash = percent_sign()
    props = [
        ("Bank", [(GREEN, bm), (PINK_LIGHT, bl)], (-0.052, 0.032, -0.052), quat_from_euler(10, 30, -5)),
        ("Cards", [(PINK_DEEP, ca), (PINK_LIGHT, cb), (PINK_GLOSS, chip)], (0.052, -0.046, -0.034), quat_from_euler(-22, -34, 28)),
        ("Padlock", [(GREEN, pl_body), (PINK_LIGHT, pl_shackle), (PINK_INK, pl_key)], (0.060, 0.058, -0.058), quat_from_euler(6, -28, 14)),
        ("PiggyBank", [(PINK_GLOSS, pig), (PINK_INK, pig_dark)], (-0.060, -0.056, -0.028), quat_from_euler(4, -125, 0)),
        ("Coins", [(PINK_LIGHT, coin_stack())], (-0.084, -0.015, -0.014), quat_from_euler(16, 10, -8)),
        ("Percent", [(GREEN, rings), (GREEN, slash)], (0.004, 0.091, -0.046), quat_from_euler(0, -22, 6)),
    ]
    placed = resolve_clearance(props, PHONE_T, PHONE_Q, EYE)
    for (name, parts, _t, q), (t, sc) in zip(props, placed):
        g.node(name, g.mesh(name, parts), t=t, r=q, s=sc)

    # camera (what a C4D render camera would export as)
    g.j["cameras"].append({"name": "RenderCam", "type": "perspective",
                           "perspective": {"yfov": math.radians(21.0), "aspectRatio": 1200 / 630,
                                           "znear": 0.05, "zfar": 3.0}})
    g.node("RenderCam", camera=0, t=EYE)
    n = g.write(OUT)
    print("wrote", OUT, n, "bytes;", len(g.j["nodes"]), "nodes,", len(g.j["meshes"]), "meshes")


if __name__ == "__main__":
    build()
