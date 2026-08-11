"""Driving times from every simulation origin to the destination sample.

64 origins x 1200 destinations. Saves after each origin, so an interrupted run
resumes instead of re-fetching.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

BATCH = 99
URL = "https://router.project-osrm.org/table/v1/driving/"
OUT = "data/sim_times.npy"

org = pd.read_csv("data/sim_origins.csv", dtype={"fips": str})
dst = pd.read_csv("data/sim_dests.csv", dtype={"fips": str})

times = np.load(OUT) if os.path.exists(OUT) else np.full((len(org), len(dst)), np.nan)
assert times.shape == (len(org), len(dst))

for oi, o in enumerate(org.itertuples()):
    if np.isfinite(times[oi]).all():
        continue
    row = np.full(len(dst), np.nan)
    for i in range(0, len(dst), BATCH):
        chunk = dst.iloc[i:i + BATCH]
        coords = f"{o.lon:.6f},{o.lat:.6f};" + ";".join(
            f"{r.lon:.6f},{r.lat:.6f}" for r in chunk.itertuples())
        for attempt in range(6):
            try:
                r = requests.get(URL + coords, params={"sources": "0", "annotations": "duration"},
                                 timeout=120)
                r.raise_for_status()
                d = r.json()
                if d.get("code") != "Ok":
                    raise RuntimeError(str(d)[:200])
                break
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"  retry {attempt + 1}: {str(e)[:70]}", file=sys.stderr, flush=True)
                time.sleep(4 * (attempt + 1))
        row[i:i + len(chunk)] = d["durations"][0][1:]
        time.sleep(0.35)
    times[oi] = row
    np.save(OUT, times)
    print(f"[{oi + 1}/{len(org)}] {o.label:38s} median {np.nanmedian(row) / 3600:5.2f} h",
          flush=True)

print(f"done: {np.isfinite(times).all()}  missing {int(np.isnan(times).sum())}")
