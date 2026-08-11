"""Pack the standalone map: a warpable mesh, line layers, and two stretch fields
per origin — one for driving, one for driving or flying, whichever is quicker.

The flight model, all of it:
    leave        1.6 h   get to the gate and through security
    in the air   0.45 h  taxi, climb and descent, plus distance / 800 km/h
    connect      1.4 h   only when the pair has no direct route in OpenFlights
    land         0.5 h   deplane and pick up a car
    minimum      300 km  below this nobody flies
Total time is the better of that itinerary and simply driving.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}
MESH_M = 20_000
LINE_PARAMS = {"county": (5_000, 45_000), "state": (3_000, 28_000), "nation": (1_500, 18_000),
               "interstate": (3_000, 25_000)}
QLON, QLAT = (-130.0, 70.0), (20.0, 35.0)
GRID = dict(lon0=-126.0, lat0=23.0, d=0.5, nlon=121, nlat=55)
KM_PER_HOUR, OSRM_BIAS, R_EARTH = 100.0, 1.1873, 6371.0088
DEP, AIR_FIX, CONNECT, ARR, MIN_FLY_KM, CRUISE = 1.6, 0.45, 1.4, 0.5, 300.0, 800.0
N_NEAR_AIRPORTS = 5
S_LO, S_HI, S_SCALE = 0.05, 6.5, 10000

to_m = Transformer.from_crs(4326, 5070, always_xy=True)
to_deg = Transformer.from_crs(5070, 4326, always_xy=True)


def great_circle(lon0, lat0, lon, lat):
    p1, p2 = np.radians(lat0), np.radians(lat)
    dp, dl = p2 - p1, np.radians(lon - lon0)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# ---- geometry ---------------------------------------------------------------
states = gpd.read_file("data/cb_2023_us_state_5m/cb_2023_us_state_5m.shp")
states = states[~states.STATEFP.isin(NON_CONUS)].to_crs(5070)
counties = gpd.read_file("data/cb_2023_us_county_5m/cb_2023_us_county_5m.shp")
counties = counties[~counties.STATEFP.isin(NON_CONUS)].to_crs(5070)
nation = shapely.union_all(states.geometry.values)

outline = shapely.segmentize(nation, MESH_M)
bpts = np.vstack([np.asarray(g.coords) for g in shapely.get_parts(outline.boundary)])
x0, y0, x1, y1 = nation.bounds
gx, gy = np.meshgrid(np.arange(x0, x1, MESH_M), np.arange(y0, y1, MESH_M * 0.866))
gx[1::2] += MESH_M / 2
inner = np.column_stack([gx.ravel(), gy.ravel()])
shapely.prepare(nation)
inner = inner[shapely.contains_xy(nation, inner[:, 0], inner[:, 1])]
inner = inner[shapely.distance(shapely.points(inner), outline.boundary) > MESH_M * 0.45]
pts = np.vstack([bpts, inner])

tri = Delaunay(pts)
cen = pts[tri.simplices].mean(axis=1)
tv = pts[tri.simplices]
longest = np.max([np.hypot(*(tv[:, i] - tv[:, (i + 1) % 3]).T) for i in range(3)], axis=0)
tris = tri.simplices[shapely.contains_xy(nation, cen[:, 0], cen[:, 1]) & (longest < MESH_M * 2.3)]


def lines_of(geom, kind):
    tol, seg = LINE_PARAMS[kind]
    g = shapely.segmentize(shapely.simplify(geom, tol), seg)
    out = []
    for part in shapely.get_parts(shapely.line_merge(shapely.node(g))):
        c = np.asarray(part.coords)
        out.append(np.stack([c[:-1], c[1:]], axis=1))
    return np.concatenate(out).reshape(-1, 2)


# The interstates are not a predictor of stretch — proximity to one says nothing about how
# far a county is in time (Spearman -0.01). They are here as the skeleton: the network every
# one of these trips is actually constrained to run along.
roads = gpd.read_file("data/tl_primaryroads/tl_2023_us_primaryroads.shp")
roads = roads[roads.RTTYP == "I"]
b = roads.bounds
roads = roads[(b.minx > -125) & (b.maxx < -66) & (b.miny > 24) & (b.maxy < 50)].to_crs(5070)


def road_lines(geoms, kind):
    """Segments for drawing only. Noding these would split at every interchange and blow
    the vertex count up twelvefold, for topology nothing here needs."""
    tol, seg = LINE_PARAMS[kind]
    out = []
    for part in shapely.segmentize(shapely.simplify(geoms.values, tol), seg):
        for p in shapely.get_parts(part):
            c = np.asarray(p.coords)
            if len(c) > 1:
                out.append(np.stack([c[:-1], c[1:]], axis=1))
    return np.concatenate(out).reshape(-1, 2)


layers = {"county": lines_of(shapely.union_all(counties.geometry.boundary.values), "county"),
          "state": lines_of(shapely.union_all(states.geometry.boundary.values), "state"),
          "nation": lines_of(nation.boundary, "nation"),
          "interstate": road_lines(roads.geometry, "interstate")}

# ---- travel times -----------------------------------------------------------
org = pd.read_csv("data/nodes_origins.csv", dtype={"fips": str})
dst = pd.read_csv("data/sim_dests.csv", dtype={"fips": str})
air = pd.read_csv("data/nodes_airports.csv")
routes = pd.read_csv("data/nodes_routes.csv")

t_org = np.load("data/nodes_times.npy") / 3600 / OSRM_BIAS      # origins x dests, hours
t_air = np.load("data/air_times.npy") / 3600 / OSRM_BIAS        # airports x dests
t_o2a = np.load("data/org_air.npy") / 3600 / OSRM_BIAS          # origins x airports
for name, m in (("org", t_org), ("air", t_air), ("o2a", t_o2a)):
    assert np.isfinite(m).all(), f"{name} matrix incomplete"

# Air leg between every pair of airports, with a penalty when there is no direct route.
iata = list(air.iata)
pos = {c: i for i, c in enumerate(iata)}
gc_ap = great_circle(air.lon.values[:, None], air.lat.values[:, None],
                     air.lon.values[None, :], air.lat.values[None, :])
leg = AIR_FIX + gc_ap / CRUISE + CONNECT
direct = np.zeros_like(leg, bool)
for a, b in zip(routes.a, routes.b):
    if a in pos and b in pos:
        direct[pos[a], pos[b]] = direct[pos[b], pos[a]] = True
leg[direct] -= CONNECT
leg[gc_ap < MIN_FLY_KM] = np.inf                                # too close to be worth flying

# ---- fields -----------------------------------------------------------------
lon_g = GRID["lon0"] + GRID["d"] * np.arange(GRID["nlon"])
lat_g = GRID["lat0"] + GRID["d"] * np.arange(GRID["nlat"])
LG, TG = np.meshgrid(lon_g, lat_g)
gxy = np.column_stack(to_m.transform(LG.ravel(), TG.ravel()))
dxy = np.column_stack(to_m.transform(dst.lon.values, dst.lat.values))

grids = np.zeros((len(org), 2, GRID["nlat"], GRID["nlon"]), np.uint16)
meta_org = []
for i, o in enumerate(org.itertuples()):
    r_km = great_circle(o.lon, o.lat, dst.lon.values, dst.lat.values)
    drive = t_org[i]

    near = np.argsort(t_o2a[i])[:N_NEAR_AIRPORTS]
    # cost of reaching each destination via each usable first airport, then the best of them
    via = np.stack([t_o2a[i, a] + DEP + np.min(leg[a][:, None] + t_air + ARR, axis=0)
                    for a in near])
    combo = np.minimum(drive, via.min(axis=0))

    row = []
    for mode, hours in enumerate((drive, combo)):
        s = np.where(r_km > 1e-6, hours * KM_PER_HOUR / np.maximum(r_km, 1e-6), 1.0)
        lin, near_i = LinearNDInterpolator(dxy, s), NearestNDInterpolator(dxy, s)
        v = lin(gxy)
        bad = ~np.isfinite(v)
        v[bad] = near_i(gxy[bad])
        grids[i, mode] = np.clip(v, S_LO, S_HI).reshape(GRID["nlat"], GRID["nlon"]) * S_SCALE
        far = int(np.argmax(hours))
        row.append(dict(med=round(float(np.median(hours)), 1),
                        max=round(float(hours[far]), 1),
                        far=f"{dst.seat.iloc[far]}, {dst.state.iloc[far]}",
                        flown=round(float((combo < drive - 0.01).mean()), 3) if mode else 0.0))
    meta_org.append(dict(label=o.label, lat=round(o.lat, 5), lon=round(o.lon, 5), modes=row))

# ---- pack -------------------------------------------------------------------
buffers, meta, offset = [], {}, 0


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
open("data/map_payload.bin", "wb").write(blob)
open("data/map_payload.json", "w").write(json.dumps(dict(
    grid=GRID, kmPerHour=KM_PER_HOUR, qlon=QLON, qlat=QLAT, stretchScale=S_SCALE,
    buffers=meta, origins=meta_org,
    airports=[dict(iata=a.iata, lat=round(a.lat, 4), lon=round(a.lon, 4)) for a in air.itertuples()],
), separators=(",", ":")))

d0 = np.array([m["modes"][0]["med"] for m in meta_org])
d1 = np.array([m["modes"][1]["med"] for m in meta_org])
fl = np.array([m["modes"][1]["flown"] for m in meta_org])
print(f"mesh {len(pts):,} vertices / {len(tris):,} triangles")
print(f"origins {len(org)}   airports {len(air)}")
print(f"median drive {d0.mean():.1f} h -> with flights {d1.mean():.1f} h "
      f"({(1 - d1.mean() / d0.mean()):.0%} faster)")
print(f"share of destinations where flying wins: {fl.mean():.0%} (range {fl.min():.0%}-{fl.max():.0%})")
s_all = grids[:, 1].astype(float) / S_SCALE
print(f"multimodal stretch: p1 {np.percentile(s_all, 1):.2f}  p50 {np.percentile(s_all, 50):.2f}  "
      f"p99 {np.percentile(s_all, 99):.2f}")
print(f"blob {len(blob) / 1048576:.2f} MB")
