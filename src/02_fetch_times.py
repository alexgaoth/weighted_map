"""Fetch driving time + road distance from each origin to every county point.

Uses the public OSRM table service, 1 source x 99 destinations per request.
"""
import sys
import time

import pandas as pd
import requests

ORIGINS = {
    # US Capitol, Washington DC -- the political centre (analogue of Tiananmen)
    "capitol": (38.88980, -77.00902),
    # Geographic centre of the contiguous 48 states, near Lebanon, Kansas
    "center": (39.82830, -98.57950),
}
BATCH = 99
URL = "https://router.project-osrm.org/table/v1/driving/"


def fetch(origin_lat, origin_lon, dests):
    rows = []
    for i in range(0, len(dests), BATCH):
        chunk = dests.iloc[i:i + BATCH]
        coords = f"{origin_lon:.6f},{origin_lat:.6f};" + ";".join(
            f"{r.lon:.6f},{r.lat:.6f}" for r in chunk.itertuples()
        )
        for attempt in range(5):
            try:
                r = requests.get(URL + coords,
                                 params={"sources": "0", "annotations": "duration,distance"},
                                 timeout=120)
                r.raise_for_status()
                d = r.json()
                if d.get("code") != "Ok":
                    raise RuntimeError(d)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"  retry {attempt + 1}: {str(e)[:80]}", file=sys.stderr)
                time.sleep(3 * (attempt + 1))

        durs, dists = d["durations"][0][1:], d["distances"][0][1:]
        snaps = [x["distance"] for x in d["destinations"][1:]]
        for (_, c), du, di, sn in zip(chunk.iterrows(), durs, dists, snaps):
            rows.append(dict(fips=c.fips, drive_s=du, road_m=di, snap_m=sn))
        print(f"  {i + len(chunk)}/{len(dests)}", end="\r", flush=True)
        time.sleep(0.3)
    return pd.DataFrame(rows)


pts = pd.read_csv("data/county_points.csv", dtype={"fips": str})
out = pts.copy()
for name, (olat, olon) in ORIGINS.items():
    print(f"origin {name} ({olat}, {olon})")
    res = fetch(olat, olon, pts).rename(columns={
        "drive_s": f"{name}_drive_s", "road_m": f"{name}_road_m", "snap_m": f"{name}_snap_m"})
    out = out.merge(res, on="fips")
    print()

out.to_csv("data/county_times_raw.csv", index=False)

for name in ORIGINS:
    d = out[f"{name}_drive_s"]
    print(f"{name}: {d.notna().sum()}/{len(d)} routed, "
          f"max {d.max() / 3600:.1f} h, median snap {out[f'{name}_snap_m'].median():.0f} m, "
          f"p99 snap {out[f'{name}_snap_m'].quantile(0.99) / 1000:.1f} km")
