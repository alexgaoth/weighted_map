"""Pack the simulation payload: a warpable mesh, line layers, and one stretch grid
per origin.

The mesh matters more than it looks. Position is computed per vertex in the shader,
so a triangle's interior is a linear blend of its corners; large triangles would
warp as straight facets. Refining the interior to ~20 km keeps every curve honest.
"""
import base64
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}
MESH_M = 20_000        # interior mesh spacing
# per layer: (simplify tolerance, max segment length). Lines are densified so they
# can bend under the warp, and simplified so densifying them stays affordable.
LINE_PARAMS = {"county": (5_000, 45_000), "state": (3_000, 28_000), "nation": (1_500, 18_000)}
# Coordinates ship as uint16 over this window: ~90 m in lon, ~60 m in lat, well
# under a third of a pixel at country scale.
QLON, QLAT = (-130.0, 70.0), (20.0, 35.0)
GRID = dict(lon0=-126.0, lat0=23.0, d=0.5, nlon=121, nlat=55)
KM_PER_HOUR = 100.0
OSRM_BIAS = 1.1873

to_m = Transformer.from_crs(4326, 5070, always_xy=True)
to_deg = Transformer.from_crs(5070, 4326, always_xy=True)

states = gpd.read_file("data/cb_2023_us_state_5m/cb_2023_us_state_5m.shp")
states = states[~states.STATEFP.isin(NON_CONUS)].to_crs(5070)
counties = gpd.read_file("data/cb_2023_us_county_5m/cb_2023_us_county_5m.shp")
counties = counties[~counties.STATEFP.isin(NON_CONUS)].to_crs(5070)
nation = shapely.union_all(states.geometry.values)

# ---- mesh -------------------------------------------------------------------
outline = shapely.segmentize(nation, MESH_M)
bpts = np.vstack([np.asarray(g.coords) for g in shapely.get_parts(outline.boundary)])
x0, y0, x1, y1 = nation.bounds
gx, gy = np.meshgrid(np.arange(x0, x1, MESH_M), np.arange(y0, y1, MESH_M * 0.866))
gx[1::2] += MESH_M / 2                                   # hex offset
inner = np.column_stack([gx.ravel(), gy.ravel()])
shapely.prepare(nation)
inner = inner[shapely.contains_xy(nation, inner[:, 0], inner[:, 1])]
# Drop interior points that sit on top of the outline, which makes slivers.
keep = shapely.distance(shapely.points(inner), outline.boundary) > MESH_M * 0.45
pts = np.vstack([bpts, inner[keep]])

tri = Delaunay(pts)
cen = pts[tri.simplices].mean(axis=1)
inside = shapely.contains_xy(nation, cen[:, 0], cen[:, 1])
# Delaunay is unconstrained, so it bridges narrow bays with long thin triangles
# whose centroid still lands on land. Legitimate triangles are ~MESH_M on a side.
tv = pts[tri.simplices]
longest = np.max([np.hypot(*(tv[:, i] - tv[:, (i + 1) % 3]).T) for i in range(3)], axis=0)
tris = tri.simplices[inside & (longest < MESH_M * 2.3)]
print(f"triangles: {inside.sum():,} inside, {len(tris):,} kept after edge filter")

# ---- line layers ------------------------------------------------------------
def lines_of(geom, kind):
    tol, seg = LINE_PARAMS[kind]
    g = shapely.segmentize(shapely.simplify(geom, tol), seg)
    segs = []
    for part in shapely.get_parts(shapely.line_merge(shapely.node(g))):
        c = np.asarray(part.coords)
        segs.append(np.stack([c[:-1], c[1:]], axis=1))
    return np.concatenate(segs).reshape(-1, 2)


layers = {
    "county": lines_of(shapely.union_all(counties.geometry.boundary.values), "county"),
    "state": lines_of(shapely.union_all(states.geometry.boundary.values), "state"),
    "nation": lines_of(nation.boundary, "nation"),
}

# ---- stretch grids ----------------------------------------------------------
org = pd.read_csv("data/sim_origins.csv", dtype={"fips": str})
dst = pd.read_csv("data/sim_dests.csv", dtype={"fips": str})
times = np.load("data/sim_times.npy")
ready = np.isfinite(times).all(axis=1)
print(f"origins with complete data: {ready.sum()}/{len(org)}")

R_EARTH = 6371.0088  # the sphere the shader uses; distances must be defined the same way


def great_circle(lon0, lat0, lon, lat):
    p1, p2 = np.radians(lat0), np.radians(lat)
    dp, dl = p2 - p1, np.radians(lon - lon0)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


lon_g = GRID["lon0"] + GRID["d"] * np.arange(GRID["nlon"])
lat_g = GRID["lat0"] + GRID["d"] * np.arange(GRID["nlat"])
LG, TG = np.meshgrid(lon_g, lat_g)
gxy = np.column_stack(to_m.transform(LG.ravel(), TG.ravel()))
dxy = np.column_stack(to_m.transform(dst.lon.values, dst.lat.values))
mesh_lon, mesh_lat = to_deg.transform(pts[:, 0], pts[:, 1])

grids = np.zeros((len(org), GRID["nlat"], GRID["nlon"]), np.uint16)
extents = np.zeros((len(org), 2, 4), np.float32)  # per origin: [true, time] x [x0,y0,x1,y1]
stats = []
for i, o in enumerate(org.itertuples()):
    if not ready[i]:
        stats.append(None)
        continue
    r_km = great_circle(o.lon, o.lat, dst.lon.values, dst.lat.values)
    hours = times[i] / 3600 / OSRM_BIAS
    s = np.where(r_km > 1e-6, hours * KM_PER_HOUR / np.maximum(r_km, 1e-6), 1.0)
    lin, near = LinearNDInterpolator(dxy, s), NearestNDInterpolator(dxy, s)
    v = lin(gxy)
    bad = ~np.isfinite(v)
    v[bad] = near(gxy[bad])
    v = np.clip(v, 0.5, 6.5)
    grids[i] = v.reshape(GRID["nlat"], GRID["nlon"]) * 10000

    # View extents: where the mesh lands with no warp, and with the full warp.
    mr = great_circle(o.lon, o.lat, mesh_lon, mesh_lat)
    p1, p2 = np.radians(o.lat), np.radians(mesh_lat)
    dl = np.radians(mesh_lon - o.lon)
    az = np.arctan2(np.sin(dl) * np.cos(p2),
                    np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))
    ms = np.clip(lin(np.column_stack(to_m.transform(mesh_lon, mesh_lat))), 0.5, 6.5)
    ms[~np.isfinite(ms)] = 1.0
    for j, rad in enumerate((mr, mr * ms)):
        x, y = rad * np.sin(az), rad * np.cos(az)
        extents[i, j] = [x.min(), y.min(), x.max(), y.max()]

    far = int(np.argmax(hours))
    stats.append(dict(medianH=round(float(np.median(hours)), 1),
                      maxH=round(float(hours[far]), 1),
                      farthest=f"{dst.seat.iloc[far]}, {dst.state.iloc[far]}",
                      area=round(float(((extents[i, 1, 2] - extents[i, 1, 0]) *
                                        (extents[i, 1, 3] - extents[i, 1, 1])) /
                                       ((extents[i, 0, 2] - extents[i, 0, 0]) *
                                        (extents[i, 0, 3] - extents[i, 0, 1]))), 2)))

# ---- pack -------------------------------------------------------------------
buffers, meta = [], {}
offset = 0


def put(name, arr):
    global offset
    b = arr.tobytes()
    meta[name] = dict(offset=offset, count=int(arr.size), dtype=arr.dtype.name)
    buffers.append(b)
    offset += len(b)


def quantize(xy_m):
    lon, lat = to_deg.transform(xy_m[:, 0], xy_m[:, 1])
    q = np.empty((len(lon), 2), np.uint16)
    q[:, 0] = np.round(np.clip((lon - QLON[0]) / QLON[1], 0, 1) * 65535)
    q[:, 1] = np.round(np.clip((lat - QLAT[0]) / QLAT[1], 0, 1) * 65535)
    return q


assert len(pts) < 65536, "mesh too large for uint16 indices"
put("mesh_xy", quantize(pts))
put("mesh_idx", tris.astype(np.uint16).ravel())
for k, v in layers.items():
    put(f"line_{k}", quantize(v))
put("grids", grids.ravel())

blob = b"".join(buffers)
payload = dict(
    grid=GRID, kmPerHour=KM_PER_HOUR, buffers=meta,
    qlon=QLON, qlat=QLAT, stretchScale=10000,
    origins=[dict(label=o.label, lat=round(o.lat, 5), lon=round(o.lon, 5), ready=bool(ready[i]),
                  extTrue=[round(float(v), 1) for v in extents[i, 0]],
                  extTime=[round(float(v), 1) for v in extents[i, 1]],
                  stats=stats[i])
             for i, o in enumerate(org.itertuples())],
)
open("data/sim_payload.json", "w").write(json.dumps(payload))
open("data/sim_payload.bin", "wb").write(blob)

print(f"mesh: {len(pts):,} vertices, {len(tris):,} triangles")
for k, v in layers.items():
    print(f"line_{k}: {len(v):,} points ({len(v) // 2:,} segments)")
print(f"grids: {grids.shape} = {grids.nbytes / 1024:.0f} KB")
print(f"blob: {len(blob) / 1048576:.2f} MB -> base64 {len(blob) * 1.34 / 1048576:.2f} MB")
