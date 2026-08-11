"""Sanity-check the flight model: where does flying start to win, and by how much?"""
import numpy as np
import pandas as pd

OSRM_BIAS, R_EARTH = 1.1873, 6371.0088
DEP, AIR_FIX, CONNECT, ARR, MIN_FLY_KM, CRUISE = 1.6, 0.45, 1.4, 0.5, 300.0, 800.0
N_NEAR = 5

org = pd.read_csv("data/nodes_origins.csv", dtype={"fips": str})
dst = pd.read_csv("data/sim_dests.csv", dtype={"fips": str})
air = pd.read_csv("data/nodes_airports.csv")
routes = pd.read_csv("data/nodes_routes.csv")
t_org = np.load("data/nodes_times.npy") / 3600 / OSRM_BIAS
t_air = np.load("data/air_times.npy") / 3600 / OSRM_BIAS
t_o2a = np.load("data/org_air.npy") / 3600 / OSRM_BIAS


def gc(lon0, lat0, lon, lat):
    p1, p2 = np.radians(lat0), np.radians(lat)
    h = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


pos = {c: i for i, c in enumerate(air.iata)}
gc_ap = gc(air.lon.values[:, None], air.lat.values[:, None],
           air.lon.values[None, :], air.lat.values[None, :])
leg = AIR_FIX + gc_ap / CRUISE + CONNECT
direct = np.zeros_like(leg, bool)
for a, b in zip(routes.a, routes.b):
    if a in pos and b in pos:
        direct[pos[a], pos[b]] = direct[pos[b], pos[a]] = True
leg[direct] -= CONNECT
leg[gc_ap < MIN_FLY_KM] = np.inf

rows = []
for i, o in enumerate(org.itertuples()):
    near = np.argsort(t_o2a[i])[:N_NEAR]
    via = np.stack([t_o2a[i, a] + DEP + np.min(leg[a][:, None] + t_air + ARR, axis=0)
                    for a in near]).min(axis=0)
    rows.append(pd.DataFrame(dict(origin=o.label, km=gc(o.lon, o.lat, dst.lon.values, dst.lat.values),
                                  drive=t_org[i], fly=via)))
d = pd.concat(rows, ignore_index=True)
d["flew"] = d.fly < d.drive - 0.01
d["best"] = np.minimum(d.drive, d.fly)

print("share of trips where flying wins, by straight-line distance")
band = pd.cut(d.km, [0, 200, 400, 600, 800, 1200, 2000, 5000])
t = d.groupby(band, observed=True).agg(flown=("flew", "mean"), drive_h=("drive", "median"),
                                       best_h=("best", "median"), n=("flew", "size"))
t["saved"] = 1 - t.best_h / t.drive_h
print(t.assign(flown=(t.flown * 100).round(0), saved=(t.saved * 100).round(0)).round(1).to_string())

cross = d[d.flew].groupby("origin").km.min()
print(f"\nshortest trip where flying wins, per origin: median {cross.median():.0f} km, "
      f"p10 {cross.quantile(.1):.0f} km, p90 {cross.quantile(.9):.0f} km")
print("\nspot checks from Washington D.C. (nearest sampled county to each city):")
# rows for one origin follow dst's order, so realign on position
w = d[d.origin.str.startswith("Washington")].reset_index(drop=True)
w = pd.concat([w, dst[["seat", "state", "lat", "lon"]].reset_index(drop=True)], axis=1)
TARGETS = {"Philadelphia PA": (39.95, -75.17), "Pittsburgh PA": (40.44, -79.996),
           "Chicago IL": (41.88, -87.63), "Denver CO": (39.74, -104.99),
           "Los Angeles CA": (34.05, -118.24), "Seattle WA": (47.61, -122.33),
           "Miami FL": (25.76, -80.19)}
for name, (la, lo) in TARGETS.items():
    r = w.iloc[int(np.argmin(gc(lo, la, w.lon.values, w.lat.values)))]
    print(f"  {name:16s} via {r.seat + ', ' + r.state:22s} {r.km:5.0f} km  "
          f"drive {r.drive:5.1f} h  fly {r.fly:5.1f} h  -> {'FLY' if r.flew else 'drive'}")
