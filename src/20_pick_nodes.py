"""Origins and airports for the multimodal map.

Origins are real county seats. The curated list leads with places people recognise
and can look for by name; k-means fills the gaps so no county is far from one.
"""
import pathlib
import shutil

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.cluster.vq import kmeans2

N_TOTAL = 248
N_AIRPORTS = 70
# Every origin is one layer of a 2D texture array, and WebGL2 only guarantees 256 of them.
MAX_ORIGINS = 256
NON_CONUS_STATES = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}

CITIES = """
New York|NY, Los Angeles|CA, Chicago|IL, Houston|TX, Phoenix|AZ, Philadelphia|PA,
San Antonio|TX, San Diego|CA, Dallas|TX, Austin|TX, Jacksonville|FL, San Jose|CA,
Columbus|OH, Charlotte|NC, Indianapolis|IN, Seattle|WA, Denver|CO, Boston|MA, Nashville|TN,
Detroit|MI, Portland|OR, Las Vegas|NV, Memphis|TN, Louisville|KY, Milwaukee|WI,
Albuquerque|NM, Tucson|AZ, Atlanta|GA, Kansas City|MO, Miami|FL, Omaha|NE, Minneapolis|MN,
New Orleans|LA, Salt Lake City|UT, Boise|ID, Billings|MT, Fargo|ND, Rapid City|SD,
Cheyenne|WY, Bangor|ME, El Paso|TX, Reno|NV, Spokane|WA, Bismarck|ND, Oklahoma City|OK,
Little Rock|AR, Wichita|KS, San Francisco|CA, Sacramento|CA, Fresno|CA, Bakersfield|CA,
Santa Barbara|CA, San Luis Obispo|CA, Redding|CA, Eugene|OR, Bend|OR, Medford|OR,
Yakima|WA, Missoula|MT, Great Falls|MT, Idaho Falls|ID, Twin Falls|ID, Pocatello|ID,
Grand Junction|CO, Pueblo|CO, Durango|CO, Santa Fe|NM, Flagstaff|AZ, Yuma|AZ, Amarillo|TX,
Lubbock|TX, Midland|TX, Corpus Christi|TX, Laredo|TX, Brownsville|TX, Waco|TX, Tulsa|OK,
Shreveport|LA, Jackson|MS, Mobile|AL, Birmingham|AL, Montgomery|AL, Knoxville|TN,
Lexington|KY, Asheville|NC, Raleigh|NC, Wilmington|NC, Charleston|SC, Savannah|GA,
Augusta|GA, Tallahassee|FL, Gainesville|FL, Orlando|FL, Tampa|FL, Fort Myers|FL,
Key West|FL, Norfolk|VA, Richmond|VA, Roanoke|VA, Charleston|WV, Pittsburgh|PA,
Scranton|PA, Buffalo|NY, Albany|NY, Burlington|VT, Manchester|NH, Portland|ME, Hartford|CT,
Providence|RI, Cleveland|OH, Toledo|OH, Cincinnati|OH, Grand Rapids|MI, Marquette|MI,
Duluth|MN, Sioux Falls|SD, Des Moines|IA, Springfield|MO, Dodge City|KS, North Platte|NE,
Casper|WY, Elko|NV, Ely|NV, Winnemucca|NV, Fort Worth|TX, Oakland|CA, Riverside|CA,
Santa Ana|CA, Stockton|CA, Newark|NJ, Baton Rouge|LA, Colorado Springs|CO, Tacoma|WA,
Madison|WI, Akron|OH, Dayton|OH, Fort Wayne|IN, Chattanooga|TN, Huntsville|AL, Durham|NC,
Winston-Salem|NC, Columbia|SC, Harrisburg|PA, Allentown|PA, Rochester|NY, Syracuse|NY,
Worcester|MA, Springfield|MA, Provo|UT, Lincoln|NE, Topeka|KS, Salem|OR, Vancouver|WA,
Ann Arbor|MI, Evansville|IN, Fort Collins|CO, Trenton|NJ, Wilmington|DE, Annapolis|MD,
Lafayette|LA, Green Bay|WI, Cedar Rapids|IA, Davenport|IA, Peoria|IL, Springfield|IL,
Rockford|IL, Columbia|MO, Fayetteville|AR, Abilene|TX, Beaumont|TX, Galveston|TX, Macon|GA,
Columbus|GA, Athens|GA, Tuscaloosa|AL, Gulfport|MS, Las Cruces|NM, Ogden|UT, Bozeman|MT,
Helena|MT, Coeur d'Alene|ID, Pierre|SD, South Bend|IN, Canton|OH, Youngstown|OH,
Kalamazoo|MI, Flint|MI, Erie|PA, Reading|PA, Lancaster|PA, Utica|NY, Sioux City|IA,
Eau Claire|WI, Rochester|MN, Clarksville|TN, Bowling Green|KY, Owensboro|KY,
Terre Haute|IN, Bloomington|IN, Everett|WA, Grand Forks|ND, Minot|ND, Concord|NH,
Augusta|ME, Glasgow|MT, Marfa|TX, Tonopah|NV, Roswell|NM, Sault Ste. Marie|MI, Del Rio|TX,
International Falls|MN, Elizabeth City|NC, Staunton|VA, St. Johns|AZ, Goodland|KS,
Green River|WY, Lakeview|OR, Miles City|MT, Alliance|NE, Traverse City|MI, Aberdeen|SD,
Challis|ID, Burns|OR, Altus|OK, Craig|CO, Walla Walla|WA, Williston|ND, Gillette|WY,
Alpena|MI, St. George|UT, Guymon|OK, Valentine|NE, Havre|MT, Rhinelander|WI, Houlton|ME,
Lamar|CO, Ukiah|CA, Bemidji|MN, Lander|WY, Woodward|OK, Tupelo|MS, Murphysboro|IL,
Raton|NM, Dickinson|ND, Monroe|LA, Clovis|NM, Bridgeport|CA, Eureka|CA
"""
# Counties that are their own seat, so the city name never shows up in the seat column.
BY_FIPS = {"11001": "Washington D.C.", "20157": "Lebanon, Kansas", "36061": "New York, NY",
           "29510": "St. Louis, MO", "24510": "Baltimore, MD", "51710": "Norfolk, VA",
           "51760": "Richmond, VA", "51770": "Roanoke, VA", "32510": "Carson City, NV",
           # Connecticut replaced counties with planning regions in 2022, so no seats exist;
           # the Capitol region's internal point is 11 km from Hartford.
           "09110": "Hartford, CT",
           # Hillsborough's seat came back from Wikidata without an English label.
           "12057": "Tampa, FL"}

pts = pd.read_csv("data/county_points.csv", dtype={"fips": str})
to_m = Transformer.from_crs(4326, 5070, always_xy=True).transform
xy = np.column_stack(to_m(pts.lon.values, pts.lat.values))

seed, missed = [], []
for tok in CITIES.replace("\n", " ").split(","):
    if not tok.strip():
        continue
    name, st = tok.strip().split("|")
    m = pts.index[(pts.seat == name) & (pts.state == st)]
    seed.append(int(m[0])) if len(m) else missed.append(f"{name}, {st}")
for fips in BY_FIPS:
    m = pts.index[pts.fips == fips]
    if len(m) and int(m[0]) not in seed:
        seed.append(int(m[0]))
seed = sorted(set(seed))

# Fill the empty quarters of the map so every county has a sample point nearby.
cent, _ = kmeans2(xy, N_TOTAL, minit="++", seed=5, iter=60)
seed_xy = xy[seed]
pool = np.array([i for i in range(len(pts)) if i not in set(seed)])
fill, used = [], set()
for c in cent:
    if np.hypot(*(seed_xy - c).T).min() < 200_000:
        continue
    for j in np.argsort(np.hypot(*(xy[pool] - c).T)):
        if pool[j] not in used:
            fill.append(int(pool[j])); used.add(pool[j]); break

idx = sorted(set(seed) | set(fill[: max(0, N_TOTAL - len(seed))]))
assert len(idx) <= MAX_ORIGINS, f"{len(idx)} origins is more texture layers than WebGL2 promises"
org = pts.loc[idx, ["fips", "state", "seat", "county", "lat", "lon"]].copy()


def label(r):
    if r.fips in BY_FIPS:
        return BY_FIPS[r.fips]
    name = r.seat if r.seat != "(internal point)" else r.county.removesuffix(" County")
    return f"{name}, {r.state}"


org["label"] = [label(r) for r in org.itertuples()]
# keep the outgoing list so a re-pick can carry over rows already fetched, by fips
if pathlib.Path("data/nodes_origins.csv").exists():
    shutil.copyfile("data/nodes_origins.csv", "data/nodes_origins_prev.csv")
org.to_csv("data/nodes_origins.csv", index=False)

# ---- airports ----------------------------------------------------------------
cols = "id name city country iata icao lat lon alt tz dst tzdb type source".split()
ap = pd.read_csv("data/airports.dat", header=None, names=cols, na_values=["\\N"])
rt = pd.read_csv("data/routes.dat", header=None, names="al aid src sid dst did code stops eq".split(),
                 na_values=["\\N"])
deg = pd.concat([rt.src, rt.dst]).value_counts()
ap = ap[(ap.country == "United States") & ap.iata.notna() & (ap.type == "airport")]
ap = ap[(ap.lat.between(24, 50)) & (ap.lon.between(-125, -66))]
ap = ap.assign(routes=ap.iata.map(deg).fillna(0)).nlargest(N_AIRPORTS, "routes")
ap[["iata", "name", "city", "lat", "lon", "routes"]].to_csv("data/nodes_airports.csv", index=False)

# Direct-route pairs, so an itinerary that needs a connection can be penalised.
sel = set(ap.iata)
pairs = {tuple(sorted(p)) for p in zip(rt.src, rt.dst) if p[0] in sel and p[1] in sel}
pd.DataFrame(sorted(pairs), columns=["a", "b"]).to_csv("data/nodes_routes.csv", index=False)

if missed:
    print("NOT MATCHED (fix these):", ", ".join(missed))
print(f"origins {len(org)} ({len(seed)} curated + {len(org) - len(seed)} fill)")
print(f"airports {len(ap)}  direct pairs {len(pairs)}")
nn = [np.sort(np.hypot(*(xy[idx] - p).T))[0] / 1000 for p in xy]
print(f"county to nearest origin: median {np.median(nn):.0f} km, p95 {np.percentile(nn, 95):.0f} km")
print("SF present:", bool((org.label == "San Francisco, CA").any()))
