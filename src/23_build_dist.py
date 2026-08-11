"""Assemble dist/ — a static site: one HTML file plus one binary.

The colour spans are measured, not chosen: each mode's ramp ends where its own
field ends, so neither wastes half its range.

The payload ships gzipped and the page inflates it itself, so the download is the
same size whether or not the host is configured to compress .bin.
"""
import gzip
import json
import pathlib

import numpy as np

meta = json.load(open("data/map_payload.json"))
blob = open("data/map_payload.bin", "rb").read()
G, nO = meta["grid"], len(meta["origins"])

b = meta["buffers"]["grids"]
grids = (np.frombuffer(blob, np.uint16, count=b["count"], offset=b["offset"]).astype(float)
         / meta["stretchScale"]).reshape(nO, 2, G["nlat"], G["nlon"])
lon = G["lon0"] + G["d"] * np.arange(G["nlon"])
lat = G["lat0"] + G["d"] * np.arange(G["nlat"])
land = np.ix_((lat > 25) & (lat < 49), (lon > -124) & (lon < -68))

spans = []
for mode in (0, 1):
    v = grids[:, mode][:, land[0], land[1]].ravel()
    hi, lo = np.log(np.percentile(v, 99)), np.log(np.percentile(v, 1))
    spans.append([round(float(max(hi, 0.05)), 4), round(float(max(-lo, 0.05)), 4)])
    print(f"mode {mode}: stretch p1 {np.exp(lo):.2f}  p50 {np.median(v):.2f}  p99 {np.exp(hi):.2f}"
          f"  -> arms {np.exp(-spans[-1][1]):.2f}× .. {np.exp(spans[-1][0]):.2f}×")

out = pathlib.Path("dist")
out.mkdir(exist_ok=True)
html = (open("src/map_template.html").read()
        .replace("{{META}}", json.dumps(meta, separators=(",", ":")))
        .replace("{{SPAN0}}", json.dumps(spans[0]))
        .replace("{{SPAN1}}", json.dumps(spans[1])))
assert "{{" not in html
(out / "index.html").write_text(html)
packed = gzip.compress(blob, 9)
(out / "timemap.bin.gz").write_bytes(packed)

print(f"dist/index.html     {len(html) / 1024:.0f} KB")
print(f"dist/timemap.bin.gz {len(packed) / 1048576:.2f} MB "
      f"({len(blob) / 1048576:.2f} MB inflated, {len(packed) / len(blob):.0%})")
