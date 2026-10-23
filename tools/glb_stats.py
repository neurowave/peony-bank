import json, struct, sys
b = open(sys.argv[1], "rb").read()
jl = struct.unpack("<I", b[12:16])[0]
j = json.loads(b[20:20 + jl])
print("total", len(b), "json", jl)
for m in j["meshes"]:
    tot = 0
    parts = []
    for p in m["primitives"]:
        vc = j["accessors"][p["attributes"]["POSITION"]]["count"]
        ic = j["accessors"][p["indices"]]["count"]
        parts.append(f"{vc}v/{ic // 3}t")
        tot += vc
    print(f"{m['name']:10s} verts {tot:6d}  {' '.join(parts)}")
