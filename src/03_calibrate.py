"""Estimate OSRM's speed bias by comparing it with Valhalla on a diverse sample.

OSRM's default car profile is conservative; Valhalla's auto costing tracks
real-world driving times closely (see the reference table in the report).
Valhalla's public instance refuses paths over 400 km, so the sample is drawn
from short/medium hops around ten anchors spanning coast, plains, desert and
mountain terrain.
"""
import time

import numpy as np
import pandas as pd
import requests
from pyproj import Geod

GEOD = Geod(ellps="WGS84")
UA = {"User-Agent": "time-map-research/1.0 (alexisobrenovic@gmail.com)"}
ANCHORS = {
    "Washington DC": (38.8899, -77.0091), "Boston MA": (42.3601, -71.0589),
    "Atlanta GA": (33.7490, -84.3880), "Minneapolis MN": (44.9778, -93.2650),
    "Dallas TX": (32.7767, -96.7970), "Denver CO": (39.7392, -104.9903),
    "Salt Lake City UT": (40.7608, -111.8910), "Phoenix AZ": (33.4484, -112.0740),
    "Seattle WA": (47.6062, -122.3321), "Bakersfield CA": (35.3733, -119.0187),
}
MAX_GC_KM = 250
N_PER_ANCHOR = 45

pts = pd.read_csv("data/county_points.csv", dtype={"fips": str})
rows = []
for name, (alat, alon) in ANCHORS.items():
    gc = GEOD.inv(np.full(len(pts), alon), np.full(len(pts), alat),
                  pts.lon.values, pts.lat.values)[2] / 1000
    near = pts[(gc > 40) & (gc < MAX_GC_KM)]
    near = near.sample(min(N_PER_ANCHOR, len(near)), random_state=7)

    coords = f"{alon:.6f},{alat:.6f};" + ";".join(f"{r.lon:.6f},{r.lat:.6f}" for r in near.itertuples())
    o = requests.get("https://router.project-osrm.org/table/v1/driving/" + coords,
                     params={"sources": "0", "annotations": "duration,distance"}, timeout=120).json()
    osrm_s, osrm_m = o["durations"][0][1:], o["distances"][0][1:]

    v = requests.post("https://valhalla1.openstreetmap.de/sources_to_targets", headers=UA, timeout=300,
                      json={"sources": [{"lat": alat, "lon": alon}],
                            "targets": [{"lat": float(r.lat), "lon": float(r.lon)} for r in near.itertuples()],
                            "costing": "auto", "units": "km"})
    v.raise_for_status()
    val = v.json()["sources_to_targets"][0]

    for (_, c), os_, om, vt in zip(near.iterrows(), osrm_s, osrm_m, val):
        if os_ and vt.get("time"):
            rows.append(dict(anchor=name, fips=c.fips, state=c.state,
                             osrm_h=os_ / 3600, valhalla_h=vt["time"] / 3600, road_km=om / 1000))
    print(f"{name:18s} {len(near)} sampled")
    time.sleep(2)

df = pd.DataFrame(rows)
df["ratio"] = df.osrm_h / df.valhalla_h
df.to_csv("data/calibration_sample.csv", index=False)

# Distance-weighted factor: long trips should dominate, since they dominate the map.
k = df.osrm_h.sum() / df.valhalla_h.sum()
print(f"\nn = {len(df)} pairs")
print(f"OSRM/Valhalla ratio: mean {df.ratio.mean():.3f}  median {df.ratio.median():.3f}  sd {df.ratio.std():.3f}")
print(f"time-weighted factor k = {k:.4f}  ->  correction 1/k = {1 / k:.4f}")
print("\nby anchor:")
print(df.groupby("anchor").ratio.agg(["mean", "std", "count"]).round(3).to_string())
print("\nby road distance band:")
band = pd.cut(df.road_km, [0, 100, 200, 300, 1000])
print(df.groupby(band, observed=True).ratio.agg(["mean", "std", "count"]).round(3).to_string())
