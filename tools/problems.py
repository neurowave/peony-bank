import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("problems")
print("problems:", len(p) if p is not None else None)
print(json.dumps(p, indent=1)[:3000])
