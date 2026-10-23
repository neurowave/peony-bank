#!/usr/bin/env python3
"""Generates scene.rml for the Peony (fictitious bank) vector-3D phone demo.

Two artboards:
  * "Hero"          1200x630 — lime field + one ScriptedLayout (world.as) that
                    loads assets/scene.glb on the GPU canvas and draws it as
                    flat vector art.
  * "Phone Screen"  390x844 component — a real Rive layout/text UI (flex rows,
                    data-bound text, state machine, listeners). world.as
                    instances it and maps it onto the glTF "Screen" material.

Both share the "Portfolio" view model, so editing a string in the view model
(or in the editor) changes the 3D phone's screen.
"""
import hashlib, math, os, random
from xml.sax.saxutils import quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# ---------------------------------------------------------------- ids + tree
_next = [10]
def nid():
    _next[0] += 1
    return f"0:{_next[0]}"


# A second id sequence for later additions, so adding them shifts none of the 0:N ids above.
_xnext = [0]
def xid():
    _xnext[0] += 1
    return f"2:{_xnext[0]}"


class E:
    def __init__(self, _tag, *children, **attrs):
        self.tag = _tag
        self.attrs = {}
        for k, v in attrs.items():
            if v is None:
                continue
            k = k.rstrip("_")
            if isinstance(v, bool):
                v = "true" if v else "false"
            elif isinstance(v, float):
                v = f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else str(v)
            self.attrs[k] = str(v)
        self.children = [c for c in children if c is not None]

    def add(self, *cs):
        self.children += [c for c in cs if c is not None]
        return self

    @property
    def id(self):
        return self.attrs.get("id")

    def assign_ids(self, path="", used=None):
        """Give every element a stable id derived from its tree path, so `rive push`
        has nothing to write back and regenerating keeps element identity."""
        used = used if used is not None else set()
        if self.tag != "Rive" and "id" not in self.attrs:
            key = path
            h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 900_000_000 + 100_000_000
            while f"1:{h}" in used:
                h += 1
            self.attrs["id"] = f"1:{h}"
        if "id" in self.attrs:
            used.add(self.attrs["id"])
        counts = {}
        for c in self.children:
            label = c.tag + ":" + c.attrs.get("name", "")
            counts[label] = counts.get(label, 0) + 1
            c.assign_ids(f"{path}/{label}#{counts[label]}", used)
        return used

    def render(self, ind=0):
        pad = "    " * ind
        a = " ".join(f"{k}={quoteattr(v)}" for k, v in self.attrs.items())
        head = f"{pad}<{self.tag}{(' ' + a) if a else ''}"
        if not self.children:
            return head + "/>\n"
        return head + ">\n" + "".join(c.render(ind + 1) for c in self.children) + f"{pad}</{self.tag}>\n"


def argb(hexrgb, a=255):
    return f"{a:02X}{hexrgb.lstrip('#').upper()}"


# ---------------------------------------------------------------- palette / type
INK = "#000000"
GREY = "#6F7377"
LINE = "#E8EAEC"
GREEN = "#E63E86"     # Peony pink (accent)
LIME = "#FFD7DF"     # hero field: blush, nudged warm of the props' magenta-pink
WGHT = 2003265652

FONT_ID = nid()

# ---------------------------------------------------------------- view model
VM_ID = nid()
VMI_ID = nid()
VM_PROPS = {}   # name -> (prop id, type, default)

def vm_prop(name, typ, default, pid=None):
    VM_PROPS[name] = (pid or nid(), typ, default)
    return VM_PROPS[name][0]

for n, t, d in [
    ("time", "string", "9:41"),
    ("greeting", "string", "Good morning, Maya"),
    ("balance", "string", "$12,480.32"),
    ("change", "string", "+$320.18"),
    ("period", "string", "this month"),
    ("cardNumber", "string", "\u2022\u2022\u2022\u2022  4821"),
    ("cardName", "string", "MAYA REYES"),
    ("spendPeriod", "string", "Spent this week"),
    ("spendAmount", "string", "$642.18"),
    ("range", "number", 0),
    ("lineWidth", "number", 1.25),
    ("motion", "number", 1),
    # start-up splash: the animated Rive wordmark on black, then the black lifts off the scene.
    # world.as drives showLogo (true plays the wordmark In, false plays Out) and splashFade
    # (opacity of the black). splash = 0 skips it.
    ("splash", "number", 1),
    ("showLogo", "bool", True),
    ("splashFade", "number", 1),
    # fired by a click on the Replay button; world.as listens and replays the 3D entrance
    ("replay", "trigger", 0),
]:
    vm_prop(n, t, d)

# Replay button hover/press states, and the cursor the native previewer shows
# (rive.yaml cursor.property; the web runtime ignores it).
for n, t, d in [
    ("replayHover", "bool", False),
    ("replayDown", "bool", False),
    ("cursor", "string", "arrow"),
]:
    vm_prop(n, t, d, pid=xid())

def path(name):
    return f"{VM_ID}-{VM_PROPS[name][0]}"

def bind(name, key, **kw):
    return E("DataBindContext", sourcePathIds=path(name), propertyKey=key, **kw)


# ---------------------------------------------------------------- layout helpers
def style(**kw):
    """LayoutComponentStyle with *UnitsValue twins added automatically."""
    sid = nid()
    attrs = {}
    for k, v in kw.items():
        if v is None:
            continue
        attrs[k] = v
        for unit_key in ("padding", "margin", "gap", "position"):
            if k.startswith(unit_key) and not k.endswith("UnitsValue") and k != "positionTypeValue":
                attrs[k + "UnitsValue"] = "points"
    return E("LayoutComponentStyle", id=sid, name="Style", **attrs), sid


def box(name, *children, w=None, h=None, ws="hug", hs="hug", dir="column", align="topLeft",
        pad=None, gap=None, fill=None, fill_id=None, radius=None, stroke=None, clip=None, abs_=None,
        margin=None, grow=None, bid=None, opacity=None, x=None, y=None, extra=None):
    st = dict(layoutWidthScaleType=ws, layoutHeightScaleType=hs, flexDirectionValue=dir,
              layoutAlignmentType=align)
    if pad is not None:
        t, r, b, l = pad if isinstance(pad, (tuple, list)) else (pad,) * 4
        st.update(paddingTop=t, paddingRight=r, paddingBottom=b, paddingLeft=l)
    if margin is not None:
        t, r, b, l = margin
        st.update(marginTop=t, marginRight=r, marginBottom=b, marginLeft=l)
    if gap is not None:
        st.update(gapHorizontal=gap, gapVertical=gap)
    if radius is not None:
        st.update(cornerRadiusTL=radius, linkCornerRadius=True)
    if abs_ is not None:
        st["positionTypeValue"] = "absolute"
        for k, v in abs_.items():
            st["position" + k.capitalize()] = v
    if grow is not None:
        st["flexGrow"] = grow
    if extra:
        st.update(extra)
    s, sid = style(**st)
    lc = E("LayoutComponent", s, name=name, id=bid or nid(), styleId=sid,
           width=w if w is not None else 0, height=h if h is not None else 0,
           clip=clip, opacity=opacity, x=x, y=y)
    if fill is not None:
        lc.add(E("Fill", E("SolidColor", colorValue=fill, id=fill_id or nid(), name="C"), name="Fill"))
    if stroke is not None:
        col, th = stroke
        lc.add(E("Stroke", E("SolidColor", colorValue=col, name="C"), thickness=th, name="Stroke"))
    lc.add(*children)
    return lc


def text(name, s, size, weight=400, color=INK, bindname=None, ws="hug", wrap=False, align="left",
         lh=None, ls=None, cid=None, tid=None, sid=None, rid=None):
    sid = sid or nid()
    cid = cid or nid()
    paint = E("TextStylePaint",
              E("Fill", E("SolidColor", colorValue=argb(color), id=cid, name="C"), name="Fill"),
              E("TextStyleAxis", tag=WGHT, axisValue=weight, name="Weight"),
              fontSize=size, fontAssetId=FONT_ID, lineHeight=lh, letterSpacing=ls, id=sid, name="Style")
    run = E("TextValueRun", styleId=sid, text=s, name="Run", id=rid)
    if bindname:
        run.add(bind(bindname, 268))
    part = E("LayoutParticipant", layoutWidthScaleType=ws, layoutHeightScaleType="hug", name="Layout")
    attrs = dict(sizingValue="autoHeight" if wrap else "autoWidth", wrapValue="wrap" if wrap else "noWrap",
                 alignValue=align)
    return E("Text", part, paint, run, name=name, id=tid or nid(), **attrs)


def node(name, *children, x=0, y=0, nid_=None, opacity=None, scale=None):
    return E("Node", *children, name=name, x=x, y=y, id=nid_ or nid(), opacity=opacity,
             scaleX=scale, scaleY=scale)


def slot(name, w, h, *art, align="center"):
    """Fixed box holding hand-placed vector art (Node keeps it from stretching)."""
    return box(name, node(name + " Art", *art), w=w, h=h, ws="fixed", hs="fixed", align=align)


def shape(name, *children, x=0, y=0, sid=None, opacity=None):
    return E("Shape", *children, name=name, x=x, y=y, id=sid or nid(), opacity=opacity)


def solid(col, cid=None):
    return E("Fill", E("SolidColor", colorValue=col, id=cid or nid(), name="C"), name="Fill")


def stroke(col, th, cap="round", join="round", *fx):
    return E("Stroke", E("SolidColor", colorValue=col, name="C"), *fx, thickness=th, cap=cap, join=join, name="Stroke")


def poly(pts, closed=False, name="Path"):
    return E("PointsPath", *[E("StraightVertex", x=round(px, 2), y=round(py, 2)) for px, py in pts],
             isClosed=closed, name=name)


def arc_pts(cx, cy, r, a0, a1, n=16):
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]


# ---------------------------------------------------------------- icons (vector)
def icon_signal():
    bars = []
    for k, hgt in enumerate([4, 6, 8.5, 11]):
        bars.append(E("Rectangle", x=1.5 + k * 4.5, y=12 - hgt / 2, width=3, height=hgt, cornerRadiusTL=0.8, name=f"Bar{k}"))
    return slot("Signal", 18, 12, shape("Bars", *bars, solid(argb(INK))))


def icon_wifi():
    arcs = []
    for r in (3.2, 6.6, 10):
        arcs.append(shape(f"Arc{r}", poly(arc_pts(8.5, 11.5, r, -math.pi * 0.77, -math.pi * 0.23, 10)), stroke(argb(INK), 2.0, "round")))
    return slot("Wifi", 17, 12, *arcs, shape("Dot", E("Ellipse", x=8.5, y=11, width=2.6, height=2.6), solid(argb(INK))))


def icon_battery():
    return slot("Battery", 27, 13,
                shape("Level", E("Rectangle", x=11.5, y=6.5, width=19, height=8, cornerRadiusTL=1.5), solid(argb(INK))),
                shape("Case", E("Rectangle", x=12, y=6.5, width=23, height=12, cornerRadiusTL=3.2), stroke(argb(INK, 110), 1.1)),
                shape("Tip", E("Rectangle", x=25.2, y=6.5, width=1.6, height=4, cornerRadiusTL=0.8), solid(argb(INK, 110))))


def icon_gear():
    pts = []
    teeth = 8
    for k in range(teeth * 4):
        a = 2 * math.pi * k / (teeth * 4) - math.pi / (teeth * 4) * 1.5
        r = 10.5 if (k % 4) in (0, 1) else 7.8
        pts.append((11 + r * math.cos(a), 11 + r * math.sin(a)))
    return slot("Settings", 22, 22,
                shape("Hole", E("Ellipse", x=11, y=11, width=6.4, height=6.4), solid(argb("#FFFFFF"))),
                shape("Gear", poly(pts, True), solid(argb(INK))))


def icon_search():
    return slot("Search", 22, 22,
                shape("Lens", E("Ellipse", x=9.5, y=9.5, width=14, height=14), stroke(argb(INK), 2.2)),
                shape("Handle", poly([(14.6, 14.6), (19.5, 19.5)]), stroke(argb(INK), 2.4)))


def icon_bell():
    pts = [(-8.5, 6.5)]
    for k in range(13):
        a = math.pi + math.pi * k / 12
        pts.append((6.4 * math.cos(a) * (1.0 if k in (0, 12) else 1.0), -1.5 + 7.6 * math.sin(a)))
    pts += [(6.4, -1.5), (6.8, 3.6), (8.5, 6.5)]
    pts = [(11 + x, 11 + y) for x, y in pts]
    return slot("Notifications", 22, 22,
                shape("Body", poly(pts, True), solid(argb(INK))),
                shape("Clapper", E("Ellipse", x=11, y=19.4, width=5.4, height=3.6), solid(argb(INK))))


def chevron(col=INK, w=8, h=12, th=1.8):
    return slot("Chevron", w, h, shape("Chev", poly([(w * 0.3, h * 0.12), (w * 0.75, h * 0.5), (w * 0.3, h * 0.88)]), stroke(argb(col), th)))


def icon_recurring():
    """Card illustration: a coin stack wrapped by two recurring arrows."""
    arts = []
    a1 = arc_pts(32, 32, 24, math.radians(200), math.radians(330), 18)
    a2 = arc_pts(32, 32, 24, math.radians(20), math.radians(150), 18)
    head = lambda tip, ang: [(tip[0] + 7 * math.cos(ang + 2.5), tip[1] + 7 * math.sin(ang + 2.5)), tip,
                             (tip[0] + 7 * math.cos(ang - 2.5), tip[1] + 7 * math.sin(ang - 2.5))]
    arts.append(shape("ArrowHeadA", poly(head(a1[-1], math.radians(330 + 90)), True), solid(argb("#B6E35A"))))
    arts.append(shape("ArrowHeadB", poly(head(a2[-1], math.radians(150 + 90)), True), solid(argb("#B6E35A"))))
    arts.append(shape("ArrowA", poly(a1), stroke(argb("#B6E35A"), 6.5, "butt")))
    arts.append(shape("ArrowB", poly(a2), stroke(argb("#B6E35A"), 6.5, "butt")))
    for k, (dx, dy) in enumerate([(0, -5), (-4, 1), (4, 6)]):
        arts.append(shape(f"Coin{k}",
                          E("Ellipse", x=32 + dx, y=32 + dy, width=19, height=19),
                          stroke(argb("#5E6670"), 1.6), solid(argb("#D9DEE3" if k != 1 else "#EEF1F3"))))
    return slot("Illustration", 64, 64, *arts[::-1])


# ---------------------------------------------------------------- spending data
# (label, period text, amount, 7 bar values 0..1, 7 bar labels)
RANGES = [
    ("Week", "Spent this week", "$642.18", [0.42, 0.66, 0.38, 0.92, 0.55, 0.74, 0.30], ["M", "T", "W", "T", "F", "S", "S"]),
    ("Month", "Spent this month", "$2,318.40", [0.58, 0.81, 0.47, 0.69, 0.95, 0.52, 0.36], ["1", "6", "11", "16", "21", "26", "31"]),
    ("Year", "Spent this year", "$24,906.12", [0.71, 0.63, 0.88, 0.77, 0.59, 0.97, 0.68], ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]),
]
BAR_MAX = 104
PALE = "#F9C9DE"
TRACK = "#FCE6F0"

TRANSACTIONS = [
    ("Bloom Coffee", "Food & drink", "-$4.80", "B", False),
    ("Studio North", "Paycheck", "+$3,200.00", "S", True),
    ("Metro Transit", "Transport", "-$2.75", "M", False),
    ("Corner Market", "Groceries", "-$38.12", "C", False),
]

# ---------------------------------------------------------------- Phone Screen artboard
SCROLL_PHYS = nid()
anims = []
keys = {}

def kf(value, frame, interp="cubic", kind="KeyFrameDouble", ease=(0.25, 0.1, 0.25, 1.0)):
    k = E(kind, value=value, frame=frame, interpolationType=interp)
    if interp == "cubic":
        k.add(E("CubicEaseInterpolator", x1=ease[0], y1=ease[1], x2=ease[2], y2=ease[3]))
    return k

def keyed(obj_id, *props):
    return E("KeyedObject", *props, objectId=obj_id)

def kprop(key, *frames):
    return E("KeyedProperty", *frames, propertyKey=key)


def peony_mark(size=22, col=None):
    """Five-petal flower mark (vector)."""
    col = col or GREEN
    c = size / 2
    petals = []
    for k in range(5):
        a = -math.pi / 2 + k * 2 * math.pi / 5
        r = size * 0.24
        petals.append(shape(f"Petal{k}", E("Ellipse", width=size * 0.36, height=size * 0.5),
                            solid(argb(col)), x=c + r * math.cos(a), y=c + r * math.sin(a)))
        petals[-1].attrs["rotation"] = f"{a + math.pi / 2:.4f}"
    centre = shape("Centre", E("Ellipse", width=size * 0.26, height=size * 0.26), solid(argb("#FFFFFF")), x=c, y=c)
    return slot("Peony Mark", size, size, centre, *petals)


def icon_action(kind):
    s = 22
    w = argb(INK)
    if kind == "send":
        art = [shape("A", poly([(6, 16), (16, 6)]), stroke(w, 2.2)), shape("H", poly([(8.5, 6), (16, 6), (16, 13.5)]), stroke(w, 2.2))]
    elif kind == "request":
        art = [shape("A", poly([(16, 6), (6, 16)]), stroke(w, 2.2)), shape("H", poly([(6, 8.5), (6, 16), (13.5, 16)]), stroke(w, 2.2))]
    elif kind == "topup":
        art = [shape("V", poly([(11, 4.5), (11, 17.5)]), stroke(w, 2.2)), shape("Hz", poly([(4.5, 11), (17.5, 11)]), stroke(w, 2.2))]
    else:  # freeze: six-armed snowflake
        art = []
        for k in range(3):
            a = k * math.pi / 3
            dx, dy = 7.5 * math.cos(a), 7.5 * math.sin(a)
            art.append(shape(f"Arm{k}", poly([(11 - dx, 11 - dy), (11 + dx, 11 + dy)]), stroke(w, 2.0)))
    return slot("Icon", s, s, *art)


def icon_contactless(col="#FFFFFF"):
    arcs = [shape(f"W{r}", poly(arc_pts(4, 11, r, -0.9, 0.9, 10)), stroke(argb(col), 1.8)) for r in (4, 8, 12)]
    return slot("Contactless", 18, 22, *arcs)


def build_screen():
    status = box("Status Bar",
                 box("Time Slot", text("Time", "9:41", 15, 600, bindname="time"), pad=(0, 0, 0, 30)),
                 box("Indicators", icon_signal(), icon_wifi(), icon_battery(), dir="row", align="center", gap=5,
                     pad=(0, 24, 0, 0)),
                 ws="fill", hs="fixed", h=50, dir="row", align="spaceBetweenCenter", pad=(14, 0, 0, 0))

    avatar = box("Avatar", text("Initials", "MR", 13, 650, GREEN), w=34, h=34, ws="fixed", hs="fixed",
                 align="center", radius=17, fill=argb(TRACK))
    header = box("Header",
                 box("Brand", peony_mark(22), text("Wordmark", "Peony", 20, 700, ls=-0.3), dir="row", align="center", gap=7),
                 box("Actions", icon_search(), avatar, dir="row", gap=16, align="center"),
                 ws="fill", hs="fixed", h=46, dir="row", align="spaceBetweenCenter", pad=(0, 20, 0, 20))

    tri_up = slot("Up", 10, 9, shape("Tri", E("Triangle", x=5, y=5, width=9, height=7.5), solid(argb(GREEN))))
    balance = box("Balance Block",
                  text("Greeting", "Good morning, Maya", 14, 450, GREY, bindname="greeting"),
                  text("Balance", "$12,480.32", 32, 660, bindname="balance", ls=-0.5),
                  box("Change Row", tri_up,
                      text("Change", "+$320.18", 13, 620, GREEN, bindname="change"),
                      text("Period", "this month", 13, 450, INK, bindname="period"),
                      dir="row", align="centerLeft", gap=5, pad=(3, 0, 0, 0)),
                  ws="fill", pad=(8, 20, 0, 20), bid=nid())
    keys["balance"] = balance.id

    # --- the card (hero)
    card_deco = node("Card Deco",
                     shape("Ring Big", E("Ellipse", width=260, height=260), solid(argb("#FFFFFF", 26)), x=330, y=10),
                     shape("Ring Small", E("Ellipse", width=150, height=150), solid(argb("#FFFFFF", 30)), x=300, y=170))
    card = box("Card",
               card_deco,
               box("Card Top", box("Card Brand", peony_mark(20, "#FFFFFF"), text("Card Word", "Peony", 17, 700, "#FFFFFF"),
                                   dir="row", align="center", gap=6),
                   icon_contactless(), ws="fill", dir="row", align="spaceBetweenCenter"),
               box("Chip", w=38, h=28, ws="fixed", hs="fixed", radius=6, fill=argb("#FFE3EF"),
                   stroke=(argb("#C9417F"), 1)),
               box("Card Bottom",
                   box("Card Ids", text("Card Number", "\u2022\u2022\u2022\u2022  4821", 16, 600, "#FFFFFF", bindname="cardNumber", ls=1.2),
                       text("Card Name", "MAYA REYES", 11, 600, "#FFFFFF", bindname="cardName", ls=1.0), gap=4),
                   text("Card Type", "DEBIT", 12, 750, "#FFFFFF", ls=1.5),
                   ws="fill", dir="row", align="spaceBetweenEnd"),
               ws="fill", hs="fixed", h=196, align="spaceBetweenStart", pad=(18, 18, 18, 18),
               margin=(16, 16, 0, 16), radius=18, fill=argb(GREEN), clip=True, bid=nid())
    keys["card"] = card.id

    # --- quick actions
    acts = []
    for kind, lab in [("send", "Send"), ("request", "Request"), ("topup", "Top up"), ("freeze", "Freeze")]:
        acts.append(box(f"Action {lab}",
                        box("Disc", icon_action(kind), w=50, h=50, ws="fixed", hs="fixed", align="center", radius=25,
                            fill=argb(TRACK)),
                        text("Action Label", lab, 12, 550),
                        align="topCenter", gap=6, ws="fill", extra={"flexBasis": 0, "flexBasisUnitsValue": "points"}))
    actions = box("Quick Actions", *acts, ws="fill", dir="row", align="topCenter", pad=(18, 14, 0, 14), bid=nid())
    keys["actions"] = actions.id

    # --- spending: segmented tabs + bar chart
    tabs = []
    keys["tab_box"], keys["tab_fill"], keys["tab_text"] = [], [], []
    for i, (lab, *_r) in enumerate(RANGES):
        fid, tcid = nid(), nid()
        on = i == 0
        t = box(f"Tab {lab}", text("Label", lab, 12.5, 620, INK if on else GREY, cid=tcid),
                w=58, h=26, ws="fixed", hs="fixed", align="center", radius=8,
                fill=argb("#FFFFFF") if on else argb(TRACK, 255), fill_id=fid, bid=nid())
        tabs.append(t)
        keys["tab_box"].append(t.id); keys["tab_fill"].append(fid); keys["tab_text"].append(tcid)
    seg = box("Segmented", *tabs, dir="row", align="center", gap=2, pad=(3, 3, 3, 3), radius=10, fill=argb(TRACK))
    spend_head = box("Spending Head",
                     box("Spend Text", text("Spend Period", "Spent this week", 13, 500, GREY, bindname="spendPeriod"),
                         text("Spend Amount", "$642.18", 22, 660, bindname="spendAmount", ls=-0.3), gap=1),
                     seg, ws="fill", dir="row", align="spaceBetweenCenter")
    cols = []
    keys["bars"], keys["bar_fill"], keys["bar_label"] = [], [], []
    vals0 = RANGES[0][3]
    top0 = vals0.index(max(vals0))
    for k in range(7):
        bid_, fid, lid = nid(), nid(), nid()
        bar = box(f"Bar {k}", w=24, h=BAR_MAX * vals0[k], ws="fixed", hs="fixed", radius=7,
                  fill=argb(GREEN if k == top0 else PALE), fill_id=fid, bid=bid_)
        cols.append(box(f"Col {k}", bar, text("Day", RANGES[0][4][k], 11.5, 550, GREY, rid=lid),
                        ws="fill", hs="fixed", h=BAR_MAX + 24, align="bottomCenter", gap=7,
                        extra={"flexBasis": 0, "flexBasisUnitsValue": "points"}))
        keys["bars"].append(bid_); keys["bar_fill"].append(fid); keys["bar_label"].append(lid)
    chart = box("Bars", *cols, ws="fill", dir="row", align="bottomCenter", pad=(12, 0, 0, 0))
    spending = box("Spending", spend_head, chart, ws="fill", pad=(22, 20, 0, 20), bid=nid())
    keys["spending"] = spending.id

    # --- recent transactions
    rows = []
    for name, cat, amt, ini, pos in TRANSACTIONS:
        rows.append(box(f"Txn {name}",
                        box("Txn Left",
                            box("Txn Icon", text("Txn Initial", ini, 15, 700, GREEN), w=40, h=40, ws="fixed", hs="fixed",
                                align="center", radius=12, fill=argb(TRACK)),
                            box("Txn Names", text("Txn Name", name, 15, 600), text("Txn Cat", cat, 12.5, 450, GREY), gap=2),
                            dir="row", align="centerLeft", gap=12),
                        text("Txn Amount", amt, 15, 640, GREEN if pos else INK),
                        ws="fill", hs="fixed", h=58, dir="row", align="spaceBetweenCenter"))
    recent = box("Recent",
                 box("Recent Head", text("Recent Title", "Recent", 18, 680), text("See All", "See all", 13, 600, GREEN),
                     ws="fill", dir="row", align="spaceBetweenCenter", pad=(0, 0, 4, 0)),
                 *rows, ws="fill", pad=(22, 20, 0, 20), bid=nid())
    keys["recent"] = recent.id

    # --- more below the fold, so the app has something to scroll to
    more_rows = []
    for name, cat, amt, ini, pos in [("Juniper Gym", "Health", "-$29.00", "J", False),
                                     ("Refund · Atlas Shoes", "Shopping", "+$64.99", "A", True),
                                     ("Northline Energy", "Bills", "-$71.40", "N", False)]:
        more_rows.append(box(f"Txn {name}",
                             box("Txn Left",
                                 box("Txn Icon", text("Txn Initial", ini, 15, 700, GREEN), w=40, h=40, ws="fixed", hs="fixed",
                                     align="center", radius=12, fill=argb(TRACK)),
                                 box("Txn Names", text("Txn Name", name, 15, 600), text("Txn Cat", cat, 12.5, 450, GREY), gap=2),
                                 dir="row", align="centerLeft", gap=12),
                             text("Txn Amount", amt, 15, 640, GREEN if pos else INK),
                             ws="fill", hs="fixed", h=58, dir="row", align="spaceBetweenCenter"))
    recent.add(*more_rows)

    def goal(name, saved, target, frac):
        return box(f"Goal {name}",
                   box("Goal Head", text("Goal Name", name, 15, 620), text("Goal Pct", f"{int(frac * 100)}%", 13, 650, GREEN),
                       ws="fill", dir="row", align="spaceBetweenCenter"),
                   box("Goal Track",
                       box("Goal Fill", w=round(310 * frac), h=8, ws="fixed", hs="fixed", radius=4, fill=argb(GREEN)),
                       ws="fill", hs="fixed", h=8, radius=4, fill=argb(TRACK), dir="row"),
                   text("Goal Amounts", f"{saved} of {target}", 12.5, 450, GREY),
                   ws="fill", gap=8, pad=(14, 14, 14, 14), radius=14, fill=argb("#FFFFFF"), stroke=(argb("#F1D5E2"), 1))
    goals = box("Goals",
                box("Goals Head", text("Goals Title", "Savings goals", 18, 680), text("Goals New", "+ New", 13, 600, GREEN),
                    ws="fill", dir="row", align="spaceBetweenCenter"),
                goal("Summer trip", "$3,400", "$5,000", 0.68),
                goal("New laptop", "$560", "$1,600", 0.35),
                goal("Rainy day fund", "$7,250", "$10,000", 0.725),
                ws="fill", gap=10, pad=(26, 20, 0, 20))
    insight = box("Insight",
                  box("Insight Icon", peony_mark(20, "#FFFFFF"), w=40, h=40, ws="fixed", hs="fixed", align="center",
                      radius=20, fill=argb(GREEN)),
                  box("Insight Text", text("Insight Title", "You spent 12% less on food", 15, 650),
                      box("Insight Body Wrap", text("Insight Body", "Nice work, Maya. That's $46 back in your pocket compared to last month.",
                                                    13, 450, GREY, ws="fill", wrap=True, lh=18), ws="fill"),
                      ws="fill", gap=3, grow=1),
                  ws="fill", dir="row", gap=12, align="topLeft", pad=(14, 14, 14, 14), margin=(18, 20, 0, 20),
                  radius=14, fill=argb(TRACK))

    # scrolling body: clipped viewport + hugging content with a ScrollConstraint
    scroll_body = box("Scroll Body",
                      E("ScrollConstraint", directionValue="vertical", physicsId=SCROLL_PHYS, name="Scroll", id=nid()),
                      balance, card, actions, spending, recent, insight, goals,
                      box("Bottom Spacer", ws="fill", hs="fixed", h=110),
                      ws="fill", hs="hug", bid=nid())
    keys["scroll_body"] = scroll_body.id
    viewport = box("Scroll Viewport", scroll_body, ws="fill", hs="fill", clip=True)

    # fixed bottom tab bar
    def tab_icon(kind, col):
        c = argb(col)
        if kind == "home":
            art = [shape("Roof", poly([(3, 10), (11, 3.5), (19, 10)]), stroke(c, 2.0)),
                   shape("House", poly([(5.5, 9), (5.5, 18.5), (16.5, 18.5), (16.5, 9)]), stroke(c, 2.0))]
        elif kind == "cards":
            art = [shape("Card", E("Rectangle", x=11, y=11, width=17, height=12, cornerRadiusTL=2.5), stroke(c, 2.0)),
                   shape("Stripe", poly([(3, 9), (19, 9)]), stroke(c, 2.0))]
        elif kind == "pay":
            art = [shape("Up", poly([(7, 18), (7, 4), (3.5, 7.5)]), stroke(c, 2.0)),
                   shape("Down", poly([(15, 4), (15, 18), (18.5, 14.5)]), stroke(c, 2.0))]
        else:
            art = [shape("Head", E("Ellipse", x=11, y=7.5, width=7.5, height=7.5), stroke(c, 2.0)),
                   shape("Body", poly(arc_pts(11, 20, 7, math.pi * 1.08, math.pi * 1.92, 12)), stroke(c, 2.0))]
        return slot("Tab Icon", 22, 22, *art)
    tab_items = []
    for kind, lab, on in [("home", "Home", True), ("cards", "Cards", False), ("pay", "Pay", False), ("profile", "Profile", False)]:
        col = GREEN if on else "#9A9DA1"
        tab_items.append(box(f"Nav {lab}", tab_icon(kind, col), text("Nav Label", lab, 11, 600, col),
                             align="topCenter", gap=4, ws="fill", extra={"flexBasis": 0, "flexBasisUnitsValue": "points"}))
    tab_bar = box("Tab Bar",
                  box("Tab Bar Rule", ws="fill", hs="fixed", h=1, fill=argb("#EFE3E9")),
                  box("Tab Items", *tab_items, ws="fill", dir="row", align="topCenter", pad=(10, 10, 0, 10)),
                  ws="fill", hs="fixed", h=84, fill=argb("#FFFFFF"))
    # overlay layer (stacked over the content; first sibling draws on top)
    tab_layer = box("Tab Bar Layer", tab_bar, ws="fill", hs="fill", align="bottomCenter")

    home = shape("Home Indicator", E("Rectangle", width=134, height=5, cornerRadiusTL=2.5), solid(argb(INK)),
                 x=195, y=830)

    content = box("Content", status, header, viewport, ws="fill", hs="fill", clip=True)
    screen_root = box("Screen Root", tab_layer, content, ws="fill", hs="fill", clip=True,
                      extra={"layoutTypeValue": "stack"})

    sm_id = nid()
    s, sid = style(layoutWidthScaleType="fixed", layoutHeightScaleType="fixed")
    ab = E("Artboard", s, name="Phone Screen", id=nid(), width=390, height=844, styleId=sid,
           isComponent=True, viewModelId=VM_ID, viewModelInstanceId=VMI_ID, defaultStateMachineId=sm_id,
           x=1400, y=0, clip=True)
    ab.add(E("Fill", E("SolidColor", colorValue=argb("#FFFFFF"), name="C"), name="Fill"))
    ab.add(E("ElasticScrollPhysics", friction=2.6, speedMultiplier=1.2, elasticFactor=0.7, name="Scroll Physics", id=SCROLL_PHYS))
    ab.add(home, screen_root)
    keys["screen_ab"] = ab.id

    # ------------------------------------------------ animations
    A = {}
    def anim(name, dur, loop="oneShot", *objs):
        a = E("LinearAnimation", *objs, name=name, id=nid(), duration=dur, fps=60, loopValue=loop)
        anims.append(a)
        A[name] = a.id
        return a

    intro_objs = []
    for k, bid in enumerate([keys["balance"], keys["card"], keys["actions"], keys["spending"], keys["recent"]]):
        d = 10 + k * 6
        intro_objs.append(keyed(bid,
                                kprop(18, kf(0, 0, "hold"), kf(0, d), kf(1, d + 24)),
                                kprop(14, kf(26, 0, "hold"), kf(26, d, ease=(0.2, 0.9, 0.3, 1.0)), kf(0, d + 30))))
    # the card also tips in with a little scale pop
    intro_objs.append(keyed(keys["card"],
                            kprop(16, kf(0.92, 0, "hold"), kf(0.92, 16, ease=(0.3, 1.5, 0.5, 1.0)), kf(1, 46)),
                            kprop(17, kf(0.92, 0, "hold"), kf(0.92, 16, ease=(0.3, 1.5, 0.5, 1.0)), kf(1, 46))))
    anim("Intro", 90, "oneShot", *intro_objs)

    # per range: bars regrow (staggered, overshoot), highlight the top bar, relabel the axis
    for i, (lab, per, amt, vals, labels) in enumerate(RANGES):
        top = vals.index(max(vals))
        objs = []
        for k in range(7):
            d = 4 + k * 3
            objs.append(keyed(keys["bars"][k],
                              kprop(8, kf(4, 0, "hold"), kf(4, d, ease=(0.3, 1.35, 0.5, 1.0)), kf(BAR_MAX * vals[k], d + 26))))
            objs.append(keyed(keys["bar_fill"][k], kprop(37, kf(argb(GREEN if k == top else PALE), 0, "hold", "KeyFrameColor"))))
            objs.append(keyed(keys["bar_label"][k], kprop(268, kf(labels[k], 0, "hold", "KeyFrameString"))))
        anim(f"Range {lab}", 50, "oneShot", *objs)

    for i, (lab, *_r) in enumerate(RANGES):
        anim(f"Tab {lab} On", 1, "oneShot",
             keyed(keys["tab_fill"][i], kprop(37, kf(argb("#FFFFFF"), 0, "hold", "KeyFrameColor"))),
             keyed(keys["tab_text"][i], kprop(37, kf(argb(INK), 0, "hold", "KeyFrameColor"))))
        anim(f"Tab {lab} Off", 1, "oneShot",
             keyed(keys["tab_fill"][i], kprop(37, kf(argb(TRACK), 0, "hold", "KeyFrameColor"))),
             keyed(keys["tab_text"][i], kprop(37, kf(argb(GREY), 0, "hold", "KeyFrameColor"))))

    # ------------------------------------------------ state machine
    def cond_range(op, val):
        return E("TransitionViewModelCondition",
                 E("TransitionPropertyViewModelComparator",
                   E("BindablePropertyNumber", bind("range", 636))),
                 E("TransitionValueNumberComparator", value=val), opValue=op)

    layers = []
    st_intro = nid()
    layers.append(E("StateMachineLayer",
                    E("AnyState", x=0, y=-120), E("ExitState", x=300, y=-120),
                    E("EntryState", E("StateTransition", stateToId=st_intro), x=0, y=0),
                    E("AnimationState", animationId=A["Intro"], id=st_intro, x=200, y=0),
                    name="Intro", id=nid()))
    st_r = [nid() for _ in RANGES]
    rstates = []
    for i, (lab, *_r) in enumerate(RANGES):
        trans = [E("StateTransition", cond_range("equal", j), stateToId=st_r[j], duration=0)
                 for j in range(len(RANGES)) if j != i]
        rstates.append(E("AnimationState", *trans, animationId=A[f"Range {lab}"], id=st_r[i], x=200 + 160 * i, y=120))
    entry = [E("StateTransition", cond_range("equal", j), stateToId=st_r[j]) for j in range(len(RANGES))]
    layers.append(E("StateMachineLayer", E("AnyState", x=0, y=-120), E("ExitState", x=300, y=-120),
                    E("EntryState", *entry, x=0, y=0), *rstates, name="Spending", id=nid()))
    for i, (lab, *_r) in enumerate(RANGES):
        on_id, off_id = nid(), nid()
        layers.append(E("StateMachineLayer", E("AnyState", x=0, y=-120), E("ExitState", x=300, y=-120),
                        E("EntryState", E("StateTransition", cond_range("equal", i), stateToId=on_id),
                          E("StateTransition", cond_range("notEqual", i), stateToId=off_id), x=0, y=0),
                        E("AnimationState", E("StateTransition", cond_range("notEqual", i), stateToId=off_id, duration=140),
                          animationId=A[f"Tab {lab} On"], id=on_id, x=200, y=0),
                        E("AnimationState", E("StateTransition", cond_range("equal", i), stateToId=on_id, duration=140),
                          animationId=A[f"Tab {lab} Off"], id=off_id, x=200, y=120),
                        name=f"Tab {lab}", id=nid()))
    listeners = []
    for i, (lab, per, amt, *_r) in enumerate(RANGES):
        listeners.append(E("StateMachineListenerSingle",
                           E("ListenerViewModelChange", E("BindablePropertyNumber", bind("range", 636, direction=True), propertyValue=i)),
                           E("ListenerViewModelChange", E("BindablePropertyString", bind("spendPeriod", 635, direction=True), propertyValue=per)),
                           E("ListenerViewModelChange", E("BindablePropertyString", bind("spendAmount", 635, direction=True), propertyValue=amt)),
                           targetId=keys["tab_box"][i], listenerTypeValue="click", name=f"Select {lab}"))
    sm = E("StateMachine", *layers, *listeners, name="Screen", id=sm_id)
    ab.add(sm, *anims)
    return ab


# ---------------------------------------------------------------- Hero artboard
# ids in logo.rml (the animated Rive wordmark, shared with the other Rive CLI demos)
LOGO_AB, LOGO_VM, LOGO_SHOW_PROP, LOGO_SM = "90:547", "90:470", "90:473", "90:1003"

def build_hero(screen_ab_id):
    world_script = SCRIPTS["world"]
    sl = E("ScriptedLayout",
           E("ScriptInputArtboard", artboardId=screen_ab_id, name="screen"),
           E("ScriptInputNumber", bind("lineWidth", 243), propertyValue=1.25, name="lineWidth"),
           E("ScriptInputNumber", bind("motion", 243), propertyValue=1, name="motion"),
           E("ScriptInputNumber", propertyValue=1, name="intro"),
           E("ScriptInputNumber", bind("splash", 243), propertyValue=1, name="splash"),
           E("FocusData"),   # lets the script receive keyboardEvent (R = replay)
           scriptAssetId=world_script, name="Vector 3D World", id=nid())
    # controls hint, bottom centre (stacked over the 3D; first sibling draws on top)
    hint_col = "#8A2350"
    # R key or a click on this button replays the entrance (the near-transparent fill
    # gives the click a hit area across the whole button, gaps included). It stays a
    # quiet hint at rest; hover lifts a soft pill behind it and firms up the keycap.
    key_r = box("Key R", text("R", "R", 11, 700, hint_col), w=20, h=20, ws="fixed", hs="fixed",
                align="center", radius=5, stroke=(argb(hint_col, 150), 1.2))
    replay_btn = box("Replay Button", key_r, text("Replay", "Replay", 12, 500, hint_col),
                     dir="row", align="center", gap=6, pad=(4, 5, 4, 5), radius=7, fill=argb(LIME, 1))
    btn_fill_id = next(c for c in replay_btn.children if c.tag == "Fill").children[0].id
    ring = next(c for c in key_r.children if c.tag == "Stroke").children[0]
    ring.attrs["id"] = xid()
    hint = box("Controls Hint",
               box("Hint Row",
                   replay_btn,
                   box("Spacer", w=10, h=1, ws="fixed", hs="fixed"),
                   text("Drag", "Drag", 12, 700, hint_col),
                   text("Rotate", "around the phone to rotate", 12, 500, hint_col),
                   box("Spacer 2", w=14, h=1, ws="fixed", hs="fixed"),
                   text("Tap", "Use", 12, 700, hint_col),
                   text("Tabs", "the phone to scroll and switch tabs", 12, 500, hint_col),
                   dir="row", align="center", gap=6),
               ws="fill", hs="fill", align="bottomCenter", pad=(0, 0, 14, 0), opacity=0.75)
    stage = box("Stage", sl, ws="fill", hs="fill")
    # splash: black over everything, the Rive wordmark (logo.rml, NuRiveWordmark01) in the middle.
    # Its own state machine plays In while showLogo is true and Out when it turns false.
    logo = E("NestedArtboardLeaf",
             E("ViewModelInstance",
               E("ViewModelInstanceBoolean", bind("showLogo", 593), propertyValue=True,
                 viewModelPropertyId=LOGO_SHOW_PROP),
               viewModelId=LOGO_VM, exports=True, name="Logo Instance", id=nid()),
             E("NestedStateMachine", animationId=LOGO_SM, name="Logo SM", id=nid()),
             artboardId=LOGO_AB, isStateful=True, fit="contain", name="Rive Wordmark", id=nid())
    splash = box("Splash", box("Logo Slot", logo, w=560, h=280, ws="fixed", hs="fixed"),
                 ws="fill", hs="fill", align="center", fill=argb("#000000"))
    splash.add(bind("splashFade", 18))
    root = box("Root", splash, hint, stage, ws="fill", hs="fill", fill=argb(LIME), extra={"layoutTypeValue": "stack"})
    s, sid = style()
    sm_id = nid()
    ab = E("Artboard", s, root, name="Hero", id=nid(), width=1200, height=630, styleId=sid,
           viewModelId=VM_ID, viewModelInstanceId=VMI_ID, defaultStateMachineId=sm_id)
    st = nid()
    idle = E("LinearAnimation", name="Idle", id=nid(), duration=60, loopValue="loop")
    # Keyboard focus. keyboardEvent (R = replay) only reaches the script while its
    # node holds Rive focus. The CLI viewer focuses the first FocusData for you; an
    # embedded runtime (rive.to, web) does not, and it clears focus whenever the
    # canvas blurs. So: focus the world as the machine starts, and take focus back
    # on every pointer down anywhere on the artboard.
    ab.add(E("StateMachine",
             E("StateMachineLayer", E("AnyState", x=0, y=-120), E("ExitState", x=300, y=-120),
               E("EntryState", E("StateTransition", stateToId=st), x=0, y=0),
               E("AnimationState", E("FocusActionTarget", targetId=sl.id),
                 animationId=idle.id, id=st, x=200, y=0), name="Idle", id=nid()),
             E("StateMachineListenerSingle", E("FocusActionTarget", targetId=sl.id),
               targetId=ab.id, listenerTypeValue="down", name="Refocus World"),
             E("StateMachineListenerSingle",
               E("ListenerViewModelChange", E("BindablePropertyTrigger", bind("replay", 686, direction=True), propertyValue=1)),
               targetId=replay_btn.id, listenerTypeValue="click", name="Replay"),
             *replay_feedback(replay_btn.id, btn_fill_id, ring.id, hint_col),
             name="Hero", id=sm_id), idle, *REPLAY_ANIMS)
    return ab


REPLAY_ANIMS = []

def replay_feedback(btn_id, fill_id, ring_id, hint_col):
    """Rest / hover / pressed looks for the Replay button: a layer driven by the
    replayHover and replayDown view model booleans, plus the listeners that write them."""
    def look(name, fill, ring):
        a = E("LinearAnimation",
              keyed(fill_id, kprop(37, kf(fill, 0, "hold", "KeyFrameColor"))),
              keyed(ring_id, kprop(37, kf(ring, 0, "hold", "KeyFrameColor"))),
              name=name, id=xid(), duration=1, fps=60, loopValue="oneShot")
        REPLAY_ANIMS.append(a)
        return a.id

    def when(prop, val):
        return E("TransitionViewModelCondition",
                 E("TransitionPropertyViewModelComparator", E("BindablePropertyBoolean", bind(prop, 634))),
                 E("TransitionValueBooleanComparator", value=val), opValue="equal")

    def set_bool(prop, val):
        return E("ListenerViewModelChange", E("BindablePropertyBoolean", bind(prop, 634, direction=True), propertyValue=val))

    def set_str(prop, val):
        return E("ListenerViewModelChange", E("BindablePropertyString", bind(prop, 635, direction=True), propertyValue=val))

    rest = look("Replay Rest", argb(LIME, 1), argb(hint_col, 150))
    hover = look("Replay Hover", argb("#FFFFFF", 140), argb(hint_col, 255))
    down = look("Replay Down", argb(hint_col, 34), argb(hint_col, 255))
    s_rest, s_hover, s_down = xid(), xid(), xid()
    layer = E("StateMachineLayer", E("AnyState", x=0, y=-120), E("ExitState", x=300, y=-120),
              E("EntryState", E("StateTransition", stateToId=s_rest), x=0, y=0),
              E("AnimationState", E("StateTransition", when("replayHover", True), stateToId=s_hover, duration=120),
                animationId=rest, id=s_rest, x=200, y=0),
              E("AnimationState", E("StateTransition", when("replayDown", True), stateToId=s_down, duration=60),
                E("StateTransition", when("replayHover", False), stateToId=s_rest, duration=220),
                animationId=hover, id=s_hover, x=400, y=0),
              E("AnimationState", E("StateTransition", when("replayDown", False), stateToId=s_hover, duration=180),
                animationId=down, id=s_down, x=600, y=0),
              name="Replay Button", id=xid())
    lis = lambda kind, name, *acts: E("StateMachineListenerSingle", *acts, targetId=btn_id,
                                      listenerTypeValue=kind, name=name)
    # exit clears down too, so a press dragged off the button never sticks
    return [layer,
            lis("enter", "Replay Hover", set_bool("replayHover", True), set_str("cursor", "pointer")),
            lis("exit", "Replay Leave", set_bool("replayHover", False), set_bool("replayDown", False),
                set_str("cursor", "arrow")),
            lis("down", "Replay Press", set_bool("replayDown", True)),
            lis("up", "Replay Release", set_bool("replayDown", False))]


def build_vm():
    props, vals = [], []
    for name, (pid, typ, default) in VM_PROPS.items():
        if typ == "string":
            props.append(E("ViewModelPropertyString", name=name, id=pid))
            vals.append(E("ViewModelInstanceString", propertyValue=default, viewModelPropertyId=pid))
        elif typ == "bool":
            props.append(E("ViewModelPropertyBoolean", name=name, id=pid))
            vals.append(E("ViewModelInstanceBoolean", propertyValue=bool(default), viewModelPropertyId=pid))
        elif typ == "trigger":
            props.append(E("ViewModelPropertyTrigger", name=name, id=pid))
            vals.append(E("ViewModelInstanceTrigger", viewModelPropertyId=pid))
        else:
            props.append(E("ViewModelPropertyNumber", name=name, id=pid))
            vals.append(E("ViewModelInstanceNumber", propertyValue=float(default), viewModelPropertyId=pid))
    return E("ViewModel", *props, E("ViewModelInstance", *vals, exports=True, name="Default", id=VMI_ID),
             name="Peony", id=VM_ID, defaultInstanceId=VMI_ID)


SCRIPTS = {}

def main():
    SCRIPTS["glb"] = nid()
    SCRIPTS["world"] = nid()
    screen = build_screen()
    hero = build_hero(screen.id)
    doc = E("Rive",
            hero, screen,
            E("ComponentAsset", artboardId=screen.id, name="Phone Screen"),
            build_vm(),
            # Editor/app builds: full Inter, referenced (the host supplies it), with the
            # editor's built-in "Include: Used" glyph subsetting on export.
            # EMBED_FONT=1 (self-contained web page): embed fonts/Inter-used.ttf, the same
            # used-glyph subset made by tools/subset_font.py, since CLI builds ignore
            # includeTypeValue.
            E("FontAsset", file="fonts/Inter-used.ttf" if os.environ.get("EMBED_FONT") else "Inter.ttf",
              name="Inter", id=FONT_ID, includeTypeValue="used",
              exportTypeValue=None if os.environ.get("EMBED_FONT") else "referenced"),
            E("BlobAsset", file="assets/scene.glb", name="scene", id=nid()),
            E("ScriptAsset", file="glb.as", isModule=True, name="glb", id=SCRIPTS["glb"]),
            E("ScriptAsset", file="world.as", name="world", id=SCRIPTS["world"]),
            version="1", kind="fragment")
    doc.assign_ids()
    out = doc.render()
    with open(os.path.join(ROOT, "scene.rml"), "w") as f:
        f.write("<!-- GENERATED by tools/gen.py — edit the generator, not this file -->\n" + out)
    print("scene.rml:", out.count("\n"), "lines, last id", _next[0])


if __name__ == "__main__":
    main()
