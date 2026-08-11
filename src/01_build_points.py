"""Build the destination point set: one point per county in the contiguous US.

Prefers the county seat (Wikidata, keyed on FIPS 6-4 code) because seats are real
towns that sit on the road network; falls back to the Census gazetteer internal
point for counties with no seat in Wikidata (mostly independent cities, which are
their own seat).
"""
import json

import geopandas as gpd
import pandas as pd
from pyproj import Geod

NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}  # AK, HI, AS, GU, MP, PR, VI
GEOD = Geod(ellps="WGS84")

counties = gpd.read_file("data/cb_2023_us_county_5m/cb_2023_us_county_5m.shp")
counties = counties[~counties.STATEFP.isin(NON_CONUS)].copy()
counties["fips"] = counties.STATEFP + counties.COUNTYFP

gaz = pd.read_csv("data/gaz_counties/2023_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str})
gaz.columns = [c.strip() for c in gaz.columns]
gaz = gaz.set_index("GEOID")[["INTPTLAT", "INTPTLONG"]]

wd = json.load(open("data/wikidata_seats.json"))["results"]["bindings"]
seats = {}
for row in wd:
    fips = row["fips"]["value"].zfill(5)
    seats[fips] = (row["seatLabel"]["value"], float(row["lat"]["value"]), float(row["lon"]["value"]))

recs = []
for _, c in counties.iterrows():
    ilat, ilon = gaz.loc[c.fips, "INTPTLAT"], gaz.loc[c.fips, "INTPTLONG"]
    if c.fips in seats:
        name, lat, lon = seats[c.fips]
        # Guard against bad Wikidata coordinates: the seat must be near its county.
        offset_km = GEOD.inv(ilon, ilat, lon, lat)[2] / 1000
        source = "seat"
        if offset_km > 200:
            name, lat, lon, source = "(internal point)", ilat, ilon, "internal_point_far_seat"
    else:
        name, lat, lon, source = "(internal point)", ilat, ilon, "internal_point"
    recs.append(
        dict(fips=c.fips, state=c.STUSPS, county=c.NAMELSAD, seat=name,
             lat=lat, lon=lon, int_lat=ilat, int_lon=ilon, point_source=source)
    )

df = pd.DataFrame(recs).sort_values("fips").reset_index(drop=True)
df.to_csv("data/county_points.csv", index=False)

print(f"counties: {len(df)}  states: {df.state.nunique()}")
print(df.point_source.value_counts().to_string())
print(f"lat range {df.lat.min():.3f}..{df.lat.max():.3f}   lon range {df.lon.min():.3f}..{df.lon.max():.3f}")
assert df.lat.between(24, 50).all() and df.lon.between(-125, -66).all(), "point outside CONUS bbox"
assert df.fips.is_unique
