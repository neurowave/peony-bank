#!/usr/bin/env python3
"""Subset Inter to the glyphs the file actually uses (mirrors the editor's
FontAsset "Include: Used" export option, which CLI builds don't apply).

Run with the project venv (fontTools):  .venv/bin/python tools/subset_font.py
Reads every text run / string value in scene.rml, keeps the variable-font
axes (wght), writes fonts/Inter-used.ttf.
"""
import html, os, re
from fontTools import subset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rml = open(os.path.join(ROOT, "scene.rml"), encoding="utf-8").read()
chars = set(" ")
for m in re.finditer(r'(?:text|propertyValue)="([^"]*)"', rml):
    chars.update(html.unescape(m.group(1)))
chars = sorted(c for c in chars if c.isprintable())

os.makedirs(os.path.join(ROOT, "fonts"), exist_ok=True)
src = os.path.join(ROOT, "Inter.ttf")
out = os.path.join(ROOT, "fonts", "Inter-used.ttf")
opts = subset.Options()
opts.layout_features = ["kern", "liga", "calt", "tnum"]
opts.name_IDs = ["*"]
opts.notdef_outline = True
font = subset.load_font(src, opts)
s = subset.Subsetter(opts)
s.populate(unicodes=[ord(c) for c in chars])
s.subset(font)
subset.save_font(font, out, opts)
print(f"{len(chars)} chars -> {out} ({os.path.getsize(out)} bytes, was {os.path.getsize(src)})")
print("".join(chars))
