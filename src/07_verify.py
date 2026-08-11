"""Checks on the warp itself: does it honour the data, and does it fold?"""
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

cty = pd.read_csv("data/county_warped.csv", dtype={"fips": str})
ORIGINS = {"capitol": (38.88980, -77.00902), "center": (39.82830, -98.57950)}


def signed_areas(pts, simplices):
    a, b, c = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                  (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


# Rebuild the interpolators exactly as 04_warp.py builds them, without re-running the
# whole pipeline: execute the file only up to the point where the warping starts.
ns = {}
exec(compile(open("src/04_warp.py").read().split("county_rows, geoms = [], {}")[0],
             "04_warp.py", "exec"), ns)
warpers = ns["warpers"]

print(f"{'origin':9s} {'method':8s} {'max |error| at county sites':>28s}")
for origin, (olat, olon) in ORIGINS.items():
    d = cty[cty.origin == origin]
    src = d[["x", "y"]].values
    dst = d[["wx", "wy"]].values
    for name, fn in warpers(src, dst).items():
        err = np.hypot(*(fn(src) - dst).T)
        print(f"{origin:9s} {name:8s} {err.max():>25.3f} m")
        assert err.max() < 1.0, f"{origin}/{name} does not reproduce its own data"

    tri = Delaunay(src)
    a0, a1 = signed_areas(src, tri.simplices), signed_areas(dst, tri.simplices)
    folded = (np.sign(a0) != np.sign(a1)).sum()
    print(f"{origin:9s} {'':8s} {folded} of {len(tri.simplices)} triangles invert "
          f"({folded / len(tri.simplices):.2%}) — places that swap order along a ray")
