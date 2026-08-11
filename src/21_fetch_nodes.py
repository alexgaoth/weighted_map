"""Every drive time the multimodal map needs.

  origins  -> destinations   (the drive field, reusing anything already fetched)
  airports -> destinations   (the last leg of a flight itinerary)
  origins  -> airports       (the first leg)

Saves after each row, so an interrupted run resumes.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

BATCH = 99
URL = "https://router.project-osrm.org/table/v1/driving/"

dst = pd.read_csv("data/sim_dests.csv", dtype={"fips": str})
org = pd.read_csv("data/nodes_origins.csv", dtype={"fips": str})
air = pd.read_csv("data/nodes_airports.csv")


def table(slat, slon, targets):
    row = np.full(len(targets), np.nan)
    for i in range(0, len(targets), BATCH):
        chunk = targets.iloc[i:i + BATCH]
        coords = f"{slon:.6f},{slat:.6f};" + ";".join(
            f"{r.lon:.6f},{r.lat:.6f}" for r in chunk.itertuples())
        for attempt in range(6):
            try:
                r = requests.get(URL + coords, params={"sources": "0", "annotations": "duration"},
                                 timeout=120)
                r.raise_for_status()
                d = r.json()
                if d.get("code") != "Ok":
                    raise RuntimeError(str(d)[:160])
                break
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"    retry {attempt + 1}: {str(e)[:70]}", file=sys.stderr, flush=True)
                time.sleep(4 * (attempt + 1))
        row[i:i + len(chunk)] = d["durations"][0][1:]
        time.sleep(0.35)
    return row


def fetch(path, sources, targets, name):
    mat = np.load(path) if os.path.exists(path) else np.full((len(sources), len(targets)), np.nan)
    if mat.shape != (len(sources), len(targets)):
        mat = np.full((len(sources), len(targets)), np.nan)
    for i, s in enumerate(sources.itertuples()):
        if np.isfinite(mat[i]).all():
            continue
        mat[i] = table(s.lat, s.lon, targets)
        np.save(path, mat)
        print(f"  [{name} {i + 1}/{len(sources)}] {getattr(s, 'label', getattr(s, 'iata', ''))}"
              f"  median {np.nanmedian(mat[i]) / 3600:5.2f} h", flush=True)
    return mat


def carry(path, cols):
    """Re-index an origin-keyed matrix onto the current origin list, by fips.

    `fetch` throws away any matrix whose shape has changed, so without this a re-pick
    of the origins would refetch every row, not just the new ones.
    """
    prev = "data/nodes_origins_prev.csv"
    if not (os.path.exists(path) and os.path.exists(prev)):
        return
    old = np.load(path)
    if old.shape == (len(org), cols):
        return
    lut = {f: i for i, f in enumerate(pd.read_csv(prev, dtype={"fips": str}).fips)}
    seed = np.full((len(org), cols), np.nan)
    hits = 0
    for i, f in enumerate(org.fips):
        if f in lut and lut[f] < len(old):
            seed[i] = old[lut[f]]
            hits += 1
    np.save(path, seed)
    print(f"carried {hits}/{len(org)} rows into {os.path.basename(path)}")


carry("data/nodes_times.npy", len(dst))
carry("data/org_air.npy", len(air))

print(f"origins {len(org)} x dests {len(dst)}")
fetch("data/nodes_times.npy", org, dst, "org")
print(f"airports {len(air)} x dests {len(dst)}")
fetch("data/air_times.npy", air, dst, "air")
print(f"origins {len(org)} x airports {len(air)}")
fetch("data/org_air.npy", org, air, "o2a")
print("all complete")
