"""Choose the origins the simulation can switch between, and the destinations used
to fit each origin's field.

Origins are real county seats so every one is routable: recognisable cities first,
then k-means centroids to cover the gaps, so no part of the country is far from a
sample point.
"""
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.cluster.vq import kmeans2

N_ORIGINS = 64
N_DESTS = 1200
CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Houston", "TX"),
    ("Phoenix", "AZ"), ("Philadelphia", "PA"), ("San Antonio", "TX"), ("San Diego", "CA"),
    ("Dallas", "TX"), ("Austin", "TX"), ("Jacksonville", "FL"), ("San Jose", "CA"),
    ("Columbus", "OH"), ("Charlotte", "NC"), ("Indianapolis", "IN"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Boston", "MA"), ("Nashville", "TN"), ("Detroit", "MI"),
    ("Portland", "OR"), ("Las Vegas", "NV"), ("Memphis", "TN"), ("Louisville", "KY"),
    ("Milwaukee", "WI"), ("Albuquerque", "NM"), ("Tucson", "AZ"), ("Atlanta", "GA"),
    ("Kansas City", "MO"), ("Miami", "FL"), ("Omaha", "NE"), ("Minneapolis", "MN"),
    ("New Orleans", "LA"), ("Salt Lake City", "UT"), ("Boise", "ID"), ("Billings", "MT"),
    ("Fargo", "ND"), ("Rapid City", "SD"), ("Cheyenne", "WY"), ("Bangor", "ME"),
    ("El Paso", "TX"), ("Reno", "NV"), ("Spokane", "WA"), ("Bismarck", "ND"),
    ("St. Louis", "MO"), ("Oklahoma City", "OK"), ("Little Rock", "AR"), ("Wichita", "KS"),
]
# Counties whose seat is the place itself, so the city name never appears in the
# seat column, plus the two origins the static maps use.
NAMED = {"11001": "Washington D.C. — the Capitol", "20157": "Lebanon, Kansas — the centre",
         "36061": "New York, NY", "29510": "St. Louis, MO"}

pts = pd.read_csv("data/county_points.csv", dtype={"fips": str})
to_m = Transformer.from_crs(4326, 5070, always_xy=True).transform
xy = np.column_stack(to_m(pts.lon.values, pts.lat.values))


def pick_spread(pool_idx, k, seeded_xy):
    """k-means over the whole country, then take the nearest unused seat to each
    centroid — skipping centroids already covered by a seeded origin."""
    cent, _ = kmeans2(xy, k, minit="++", seed=3, iter=60)
    chosen = []
    used = set()
    for c in cent:
        if len(seeded_xy) and np.hypot(*(seeded_xy - c).T).min() < 180_000:
            continue
        d = np.hypot(*(xy[pool_idx] - c).T)
        for j in np.argsort(d):
            if pool_idx[j] not in used:
                chosen.append(pool_idx[j])
                used.add(pool_idx[j])
                break
    return chosen


seed_idx = []
missed = []
for name, st in CITIES:
    m = pts.index[(pts.seat == name) & (pts.state == st)]
    (seed_idx.append(int(m[0])) if len(m) else missed.append(f"{name}, {st}"))
for fips in NAMED:
    m = pts.index[pts.fips == fips]
    if len(m) and int(m[0]) not in seed_idx:
        seed_idx.append(int(m[0]))

pool = np.array([i for i in range(len(pts)) if i not in set(seed_idx)])
fill = pick_spread(pool, N_ORIGINS, xy[seed_idx])
origin_idx = sorted(set(seed_idx) | set(fill[: max(0, N_ORIGINS - len(seed_idx))]))

org = pts.loc[origin_idx, ["fips", "state", "seat", "county", "lat", "lon"]].copy()


def label(r):
    if r.fips in NAMED:
        return NAMED[r.fips]
    # Independent cities have no separate seat; name them after the county unit.
    name = r.seat if r.seat != "(internal point)" else r.county.removesuffix(" County")
    return f"{name}, {r.state}"


org["label"] = [label(r) for r in org.itertuples()]
org.to_csv("data/sim_origins.csv", index=False)

# Destinations: an even sample of the country, used only to fit each origin's field.
cent, _ = kmeans2(xy, N_DESTS, minit="++", seed=11, iter=40)
dest_idx = sorted({int(np.hypot(*(xy - c).T).argmin()) for c in cent})
pts.loc[dest_idx].to_csv("data/sim_dests.csv", index=False)

if missed:
    print("cities not matched to a county seat:", ", ".join(missed))
print(f"origins: {len(org)}   destinations: {len(dest_idx)}")
print(f"requests needed: {len(org)} x {int(np.ceil(len(dest_idx) / 99))} = "
      f"{len(org) * int(np.ceil(len(dest_idx) / 99))}")
nn = [np.sort(np.hypot(*(xy[origin_idx] - p).T))[0] / 1000 for p in xy]
print(f"distance from any county to its nearest origin: median {np.median(nn):.0f} km, "
      f"p95 {np.percentile(nn, 95):.0f} km, max {max(nn):.0f} km")
