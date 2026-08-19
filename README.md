# The United States redrawn by driving time

![](weighted-map.gif)

A US counterpart to the Amap/Gaode "China by driving time" project. Every county seat in the
contiguous 48 states keeps its true compass bearing from a fixed origin, but its distance is
replaced by **driving hours × 100 km**. The state and national boundaries are then warped to
follow the counties.

Two origins, mirroring the original's Tiananmen / Yongdeng County pair:

| origin | location | rationale |
|---|---|---|
| `capitol` | 38.8898 N, 77.0090 W | US Capitol, Washington D.C. — the political centre |
| `center`  | 39.8283 N, 98.5795 W | geographic centre of the 48 states, near Lebanon, Kansas |

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pandas matplotlib geopandas shapely pyproj requests

.venv/bin/python src/01_build_points.py   # county seats  -> data/county_points.csv
.venv/bin/python src/02_fetch_times.py    # 6,218 routes  -> data/county_times_raw.csv   (~2 min, network)
.venv/bin/python src/03_calibrate.py      # OSRM vs Valhalla -> data/calibration_sample.csv (network)
.venv/bin/python src/04_warp.py           # displacement + warped geometry
.venv/bin/python src/05_plot.py           # six figures x light/dark -> out/
.venv/bin/python src/06_stats.py          # rankings -> data/summary.json
.venv/bin/python src/07_verify.py         # checks the warp honours its own data
.venv/bin/python src/08_report.py         # report.html with figures inlined
```

Step 02 and 03 hit public servers. Both are polite (64 and 20 requests respectively); the raw
outputs are committed so the rest of the pipeline runs offline.

## Data

- **County seats** — Wikidata SPARQL keyed on the FIPS 6-4 code (`P882` → `P36` → `P625`).
  3,049 of 3,109 counties resolve to a seat; the remaining 60 (mostly Virginia independent cities,
  which are their own seat) fall back to the Census gazetteer internal point.
- **Boundaries** — Census cartographic boundary files, 1:5,000,000 (`GENZ2023`), densified to
  10 km segments so straight state lines can bend under the warp.
- **Routing** — OSRM public server on the OpenStreetMap network, `table` service, one source
  against 99 destinations per request. Free-flow car times, no traffic.

### The calibration

OSRM's default car profile is conservative. Measured against Valhalla — an independent engine on
the same OSM data — over 224 routes around ten anchors from Boston to Bakersfield, the ratio is
**1.187 ± 0.076**, flat across distance bands (1.176 / 1.191 / 1.184 for 0–100 / 100–200 /
200–300 km). Every duration used downstream is a raw OSRM time divided by that single scalar
(`OSRM_BIAS` in `src/04_warp.py`). Held out against Valhalla on the four long routes its public
server will accept, the calibrated times land within ±3%.

## Method

The work happens in an azimuthal equidistant projection centred on the origin, where straight-line
distance from the centre is true geodesic distance and polar angle is true azimuth. The transform
is then exactly the premise: keep θ, set r = hours × 100 km.

Boundaries are carried along by three interpolators fitted to the same county displacements
(`warpers()` in `src/04_warp.py`):

- `radial` — piecewise-linear interpolation of the log stretch field, applied along each point's
  own bearing. Keeps the warp purely radial.
- `affine` — Delaunay barycentric interpolation of the displacement vectors; the classic
  triangulation deformation.
- `tps` — thin-plate spline on the displacement vectors.

All three reproduce the 3,109 measured positions to 0.000 m (`src/07_verify.py`); they diverge only
where nothing constrains them — over the Great Lakes, past the coasts, beyond the border.

## The interactive version

`simulation.html` deforms the map live from any of 64 origins, in WebGL. A published page cannot
call a routing server, so the physics ships with it:

```bash
.venv/bin/python src/10_pick_origins.py   # 64 origins + a 1,200-county destination sample
.venv/bin/python src/11_fetch_grid.py     # 76,800 routes -> data/sim_times.npy   (~20 min, network)
.venv/bin/python src/12_build_payload.py  # stretch grids + warpable mesh -> data/sim_payload.*
.venv/bin/python src/13_build_sim.py      # simulation.html, everything inlined
```

Each origin's stretch field is fitted onto a 121x55 half-degree grid and shipped as a `R16UI`
texture array. The vertex shader samples the grid, computes true bearing and great-circle distance
from the origin on the sphere, and places the vertex at `r x stretch` — so the whole warp is one
GPU pass over a 42,494-vertex mesh. Switching origins morphs between two measured fields; the
slider mixes stretch between 1.0 and the sampled value, dissolving the true map into the time map.

Every resting state is measured data, not interpolation between origins. Origins are real county
seats, so the median county sits 134 km from the nearest one (p95 284 km).

Total payload 2.3 MB binary. Two details that will bite if you change them: the grid row is
`nlon x 2 = 242` bytes and `nlon` is odd, so `UNPACK_ALIGNMENT` must be set to 1 or the texture
upload is rejected outright and every stretch reads as zero; and the mesh must stay under 65,536
vertices for `uint16` indices.

## The standalone site (`dist/`)

`dist/` is a plain static site with no dependencies and no build step — drop it on GitHub Pages,
Netlify, S3, anywhere:

```bash
.venv/bin/python src/20_pick_nodes.py     # 253 origins + 70 airports + direct-route pairs
.venv/bin/python src/21_fetch_nodes.py    # ~405k routes -> data/nodes_*.npy   (~90 min, network)
.venv/bin/python src/22_build_payload.py  # drive + multimodal fields, mesh, lines
.venv/bin/python src/23_build_dist.py     # dist/index.html + dist/timemap.bin.gz
.venv/bin/python src/24_og_image.py       # dist/og.png for link previews
```

It must be **served over HTTP** — the page fetches `timemap.bin.gz`, which browsers block on
`file://`. `python -m http.server -d dist` is enough to check it locally.

The 253 origins are real county seats: every metro anyone would look for by name, then a
farthest-point fill so no county is stranded. The median county sits 72 km from the nearest origin
(p95 141 km). 253 is not arbitrary — each origin is one array-texture layer and WebGL2 only
promises 256, which is also why an origin's two modes share a layer instead of taking one each.

### Two modes

**Drive** is the map from the report, generalised to 253 origins. **Fly** takes the better of
driving and the best flight itinerary, using the top 70 CONUS airports by route count
(OpenFlights) with drive times to and from each:

| leg | cost |
|---|---|
| get to the gate | 1.6 h |
| taxi, climb, descend | 0.45 h + distance / 800 km/h |
| no direct route between the pair | +1.4 h |
| land and pick up a car | 0.5 h |
| shorter than 300 km | not flown |

Flying is a model, not a timetable: it assumes a seat is available whenever you want one. Drive
times are measured.

**Fly is a crumpled map, and that is the finding, not a bug.** Two things do it. The radial range
collapses — the p95/p5 spread of time-distance is 7.2× driving and **2.9× flying**, because a
flight is ~3.7 h of fixed overhead plus 0.00125 h/km, so an eightfold spread in real distance
becomes a threefold spread in time. And the sheet folds: **36% of triangles invert** against 1.4%
for driving. The folding is not at the drive/fly boundary — triangles whose three counties all
drive invert 0.6% of the time — it is *inside* the flown region, at 41.6%. Once you are flying,
your time is set by which airport you land at and how far you then drive, not by how far the
destination is. At 1 h = 100 km, an extra hour of airport transfer moves a county 100 km, while an
extra 100 km of flight moves it 12.5 km. The signal is eight times weaker than the noise.

No constant in the flight model repairs this, which is worth saying because it looks like it
should. Measured over 14 origins: `CONNECT` 1.4 → 0 gives 37.0% folded, `N_NEAR_AIRPORTS` 5 → 1
gives 35.6%, `MIN_FLY_KM` 300 → 600 gives 36.2%, `CRUISE` 800 → 500 gives 32.4%, and making each
destination use its own nearest airport gives 37.6%. None of them touch the ratio above.

What does fix it is pulling the exponent the *other* way, so **the two modes have different
exaggeration ranges**: Drive runs `sᵏ` from k=1.0 up to 2.6, Fly from k=0.35 up to 1.0, and each
one's notch is its own default.

That is not a cosmetic fudge, and the numbers are the argument. Folding and radial spread improve
*together* as k comes down, because the folds are what was eating the spread:

| k | folded | radial range |
|---|---|---|
| 1.00 | 36.9% | 2.8× |
| 0.70 | 29.3% | 3.5× |
| 0.50 | 20.5% | 4.3× |
| **0.35** | **10.3%** | **5.0×** |
| 0.25 | 3.6% | 5.6× |

Fly at k=0.35 lands within reach of Drive's 1.4% and 7.2×. The slider still reaches the raw metric
at its right-hand end, and the legend reports what is actually drawn — the arms read 0.61×–1.15×
at the default rather than the raw field's 0.24×–1.50×.

Colour is sequential in Drive (nothing is ever faster than the paper scale) and diverging in Fly,
where a place can be closer in time than in space. Each arm's span is set from the 1st and 99th
percentile of that field — the fly field runs 0.24× to 1.50×, so a symmetric ramp would waste an
arm — while the neutral stays pinned at exactly 1.0×.

### A surface, not a picture

The sheet is drawn in 3-D: each point keeps its warped position on the plane and is lifted by its
stretch, so **1.0× is sea level**. Places further in time than in space stand up as ridges; places
the aeroplane brings closer sink below the plane. The relief is the same field the hillshade was
already drawing — here it is the actual geometry, lit by real surface normals taken from two
finite differences of the same warp each vertex just went through.

Height runs through exactly the two-armed normalisation the colour uses — each arm divided by its
own measured span, then `tanh` — so **hue and elevation say the same thing**, and neither arm can
run away from the other. That last part matters: the stretch ratio is unstable within an hour of
the origin (DC scores 3.4 against itself), and a raw log of it spikes taller than the country is
wide. The amplitude is a fixed share of the sheet's own width, halved in Fly because that mode
uses both arms, so a 2,000 km fly map and a 9,000 km drive map stand up by the same amount.

Drag to orbit, drag with **shift** (or two fingers) to pan, scroll or pinch to zoom, double-click
or **R** to reset. The hour rings stay on the flat reference plane, which is what makes the tilt
legible. Framing is solved exactly at every angle: the camera distance comes from projecting the
sampled national outline through the current view, so nothing clips as you turn it.

### Showing the mechanism

**`#` draws the Interstate system, warped along with everything else.** Put the origin in Kansas
and slide from real to time: the network is a regular grid on the real map, and on the time map
four pale lobes open along N/S/E/W while the diagonal quadrants sink. That is the section grid
charging you √2 to travel diagonally, and it is the largest single cause of the deformation from
anywhere in the interior.

The overlay is a skeleton, **not a predictor** — worth saying plainly, because it is the obvious
thing to assume. Distance from a county to the nearest interstate tells you essentially nothing
about how stretched that county is (Spearman −0.01; median stretch is 1.16 within 10 km of an
interstate and 1.13 beyond 100 km). A long trip runs on interstates almost the whole way whatever
sits at the far end, so what matters is the shape of the path, not the last thirty kilometres.

Decomposing `stretch = circuity × (100 / speed)` over the county set says the same thing. Implied
speed barely moves — the median is 100–103 km/h in every bearing band — while circuity climbs from
1.14 for trips due N/S/E/W to 1.31 for trips at 45°. From the Kansas centre, circuity accounts for
88% of the variance in log stretch. **The country is not slow; it is bent.**

Hovering an origin gives you the same fact as a number you have a feel for: `43.1 h · 89 km/h
direct` — what you actually average in a straight line, once the road has finished bending.

### Controls

Search 253 places, or click any dot (the dots move with the map, and sit on the surface). Then:

| control | what it does |
|---|---|
| **real ⟷ stretched** | one axis of distortion. Up to the notch the real map dissolves into the time map; past it the same warp is pushed further, `s → sᵏ`. The range is per mode — Drive k=1.0→2.6, Fly k=0.35→1.0 — because 1.0 means a legible map in one and a folded one in the other. The slider snaps to the notch. |
| **flat ⟷ tall** | height of the relief; at flat the sheet is a plane and the map is the 2-D one |
| **#** | the Interstate system, warped with the map — the network every trip runs along |
| **◔** | shading — real surface normals when raised, a hillshade of `log(stretch)` when flat |
| **◑** | cycles three diverging palettes, all validated for colour-vision deficiency |
| **▷** | walks a fixed route — DC, Fort Myers, Chicago, Milwaukee, Denver, San Francisco, Los Angeles, Bend, Eugene, Seattle — picking up from wherever you already are |

Either slider's end labels are buttons: click one and it goes there.

Any view is a URL: `?from=Denver%2C%20CO&mode=fly&x=2&h=0.8&tilt=30&spin=-25&relief=1&pal=1`
(`tilt` and `spin` in degrees, `tilt=90` is straight down). `t=0…1` sets the dissolve directly and
skips the opening animation; `nochrome=1` hides the interface for stills.

Under `min(drive, fly)` the sheet genuinely folds over itself — places past the crossover jump
inward and land behind their neighbours. In 3-D the fold is real geometry you can look under,
because the flown side sinks and the drive-only core stands up. Flat, there is no depth to sort
by, so the renderer falls back to ordering the fold by true distance; county lines fade out where
the sheet is too compressed to show them.

## Known limits

- Free-flow times only. A rush-hour map would tear the Northeast open much wider.
- The stretch *ratio* is unstable within ~150 km of the origin, where a fixed few minutes of town
  driving is a large fraction of the trip. DC scores its own stretch of 3.4 for this reason; all
  rankings exclude counties inside 150 km.
- The warp is not injective: 2.1% of Delaunay triangles invert from the Capitol, 1.8% from the
  centre. Those are genuine order swaps, not numerical artefacts.
- OSRM routes ferries as roads, which is why Nantucket, Dukes and San Juan counties stretch hard.

## Output

- `report.html` — the write-up, figures inlined, light and dark themes
- `simulation.html` — the live WebGL deformation, 64 origins, self-contained
- `out/*.png` — six figures, rendered per theme
- `data/county_warped.csv` — per county per origin: bearing, hours, road km, implied speed, detour
  factor, stretch, and both the true and time-distance positions
- `data/summary.json` — rankings, per-state medians, variance decomposition
