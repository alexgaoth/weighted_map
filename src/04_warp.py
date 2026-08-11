"""Displace counties by driving time and warp the boundaries to match.

Everything happens in an azimuthal equidistant plane centred on the origin, where
straight-line distance from the centre is the true geodesic distance and the
polar angle is the true azimuth. The transform is then exactly the stated idea:
keep theta, replace r with (driving hours) x 100 km.

Boundaries are carried along by three interpolators fitted to the county points:
  radial - piecewise-linear interpolation of the log stretch field, applied radially
  affine - piecewise-affine (Delaunay barycentric) interpolation of displacement
  tps    - thin-plate spline on displacement, the smooth/"topological" variant
"""
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from shapely.ops import transform as shp_transform

OSRM_BIAS = 1.1873      # measured in 03_calibrate.py
KM_PER_HOUR = 100.0     # the map scale: 1 hour of driving == 100 km on paper
SEGMENT_M = 10_000      # densify boundaries so warped straight borders can bend
ORIGINS = {"capitol": (38.88980, -77.00902), "center": (39.82830, -98.57950)}
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}

times = pd.read_csv("data/county_times_raw.csv", dtype={"fips": str})
states = gpd.read_file("data/cb_2023_us_state_5m/cb_2023_us_state_5m.shp")
states = states[~states.STATEFP.isin(NON_CONUS)].to_crs(4326)
nation = gpd.GeoDataFrame(geometry=[shapely.union_all(states.geometry.values)], crs=4326)

counties = gpd.read_file("data/cb_2023_us_county_5m/cb_2023_us_county_5m.shp")
counties = counties[~counties.STATEFP.isin(NON_CONUS)].to_crs(4326)
counties["fips"] = counties.STATEFP + counties.COUNTYFP
counties = counties[["fips", "geometry"]].set_index("fips").loc[times.fips].reset_index()


def warpers(src_xy, dst_xy):
    """Return {name: fn(xy)->xy} interpolators fitted to the county displacement."""
    # A few counties share a seat (Todd/Tripp SD, Fall River/Oglala Lakota SD);
    # duplicate sites make the thin-plate spline system singular.
    _, keep = np.unique(src_xy.round(1), axis=0, return_index=True)
    src_xy, dst_xy = src_xy[keep], dst_xy[keep]
    r = np.hypot(*src_xy.T)
    log_s = np.log(np.hypot(*dst_xy.T) / np.where(r > 0, r, 1))
    lin_s = LinearNDInterpolator(src_xy, log_s)
    near_s = NearestNDInterpolator(src_xy, log_s)
    disp = dst_xy - src_xy
    lin_d = LinearNDInterpolator(src_xy, disp)
    near_d = NearestNDInterpolator(src_xy, disp)
    tps = RBFInterpolator(src_xy, disp, kernel="thin_plate_spline", neighbors=64)

    def fill(v, fallback, xy):
        bad = ~np.isfinite(v if v.ndim == 1 else v[:, 0])
        if bad.any():
            v[bad] = fallback(xy[bad])
        return v

    def radial(xy):
        s = np.exp(fill(lin_s(xy), near_s, xy))
        r_ = np.hypot(*xy.T)
        with np.errstate(invalid="ignore"):
            unit = np.where(r_[:, None] > 0, xy / np.where(r_, r_, 1)[:, None], 0)
        return unit * (r_ * s)[:, None]

    return {"radial": radial,
            "affine": lambda xy: xy + fill(lin_d(xy), near_d, xy),
            "tps": lambda xy: xy + tps(xy)}


county_rows, geoms = [], {}
for origin, (olat, olon) in ORIGINS.items():
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={olat} +lon_0={olon} +datum=WGS84 +units=m +no_defs")
    fwd = Transformer.from_crs(4326, crs, always_xy=True).transform

    x, y = fwd(times.lon.values, times.lat.values)
    src = np.column_stack([x, y])
    r_km = np.hypot(x, y) / 1000
    hours = times[f"{origin}_drive_s"].values / 3600 / OSRM_BIAS
    road_km = times[f"{origin}_road_m"].values / 1000

    r_new_km = hours * KM_PER_HOUR
    scale = r_new_km / r_km
    dst = src * scale[:, None]

    county_rows.append(pd.DataFrame(dict(
        fips=times.fips, state=times.state, county=times.county, seat=times.seat,
        lat=times.lat, lon=times.lon, origin=origin,
        x=x, y=y, wx=dst[:, 0], wy=dst[:, 1],
        gc_km=r_km, road_km=road_km, hours=hours, time_km=r_new_km, stretch=scale,
        speed_kmh=road_km / hours, detour=road_km / r_km,
        drift_km=np.hypot(*(dst - src).T) / 1000)))

    fns = warpers(src, dst)
    # County fills only need the primary method; the other two exist for the
    # method-comparison figure, which only draws outlines.
    for layer, gdf, methods in (("states", states, fns), ("nation", nation, fns),
                                ("counties", counties, {"radial": fns["radial"]})):
        proj = gdf.to_crs(crs)
        proj = proj.set_geometry(shapely.segmentize(proj.geometry.values, SEGMENT_M))
        geoms[(origin, "true", layer)] = proj
        for method, fn in methods.items():
            warped = proj.copy()
            warped["geometry"] = [
                shp_transform(lambda xs, ys, fn=fn: tuple(fn(np.column_stack([xs, ys])).T), g)
                for g in proj.geometry
            ]
            geoms[(origin, method, layer)] = warped
        print(f"  {origin}: {layer} x {len(methods)}")

cty = pd.concat(county_rows, ignore_index=True)
cty.to_csv("data/county_warped.csv", index=False)
pickle.dump({"geoms": geoms, "origins": ORIGINS}, open("data/warped_geoms.pkl", "wb"))

for origin in ORIGINS:
    d = cty[cty.origin == origin]
    print(f"\n=== {origin} ===")
    print(f"hours      : median {d.hours.median():5.2f}  max {d.hours.max():5.2f}")
    print(f"speed km/h : p1 {d.speed_kmh.quantile(.01):5.1f}  median {d.speed_kmh.median():5.1f}  p99 {d.speed_kmh.quantile(.99):5.1f}")
    print(f"stretch    : min {d.stretch.min():4.2f}  median {d.stretch.median():4.2f}  max {d.stretch.max():4.2f}")
    print(f"drift km   : median {d.drift_km.median():5.0f}  max {d.drift_km.max():5.0f}")
    assert d.speed_kmh.between(20, 130).all(), "implausible implied speed"
    assert np.isfinite(d[["wx", "wy"]].values).all()
