# weighted_map

Maps of the contiguous US where distance from an origin is replaced by travel time.
`README.md` has the pipeline and the method; this file is only what bites.

## Commands

- Always `.venv/bin/python …`. The venv is CPython 3.12 via `uv` because system python is 3.14
  and geopandas/shapely/pyproj have no 3.14 wheels.
- `python3 -m http.server 8000 -d dist` to view the standalone site. It **must** be served over
  HTTP — the page `fetch`es `timemap.bin.gz` and inflates it with `DecompressionStream`, and
  browsers block the fetch on `file://`.
- The `*_fetch_*.py` scripts save after every row and skip rows already complete, so a killed run
  resumes. To force a refetch, delete the `.npy` first.
- `22_build_payload.py` takes about four minutes and prints nothing until the end. It has not hung.
- Re-picking origins is cheap: `20_pick_nodes.py` snapshots `data/nodes_origins_prev.csv`, and
  `21_fetch_nodes.py` re-indexes the existing matrices onto the new list by fips before fetching.
  Delete that snapshot and every row gets fetched again.

## Invariants kept by hand

- `OSRM_BIAS = 1.1873` is copied into `04_warp.py`, `12_build_payload.py`, `22_build_payload.py`
  and `25_check_air.py`. Re-running `03_calibrate.py` means editing all four.
- The flight model constants (`DEP, AIR_FIX, CONNECT, ARR, MIN_FLY_KM, CRUISE`, `N_NEAR_AIRPORTS`)
  exist in both `22_build_payload.py` and `25_check_air.py`. Tune one and the check silently
  validates a model that is no longer being shipped.
- The warp mesh must stay under 65,536 vertices — indices ship as `uint16`. Lowering `MESH_M`
  trips the assert in `22_build_payload.py`.
- **At most 256 origins.** Each is one layer of a `TEXTURE_2D_ARRAY` and WebGL2 only guarantees
  256 of them, so the two modes are stacked *within* a layer (texture height `nlat * 2`, and
  `raw()` in the shader offsets the row). `MAX_ORIGINS` in `20_pick_nodes.py` asserts it.
- `place()` in `map_template.html` and the vertex shader's `world()` must agree exactly, `lift()`
  included, or the origin dots and the framing float off the surface.
- The per-mode colour spans `23_build_dist.py` measures also set the **terrain height** — `lift()`
  divides by the same `uSpanPos`/`uSpanNeg`. Re-picking those percentiles reshapes the relief.

## Gotchas

- **The stretch grid texture.** A grid row is `nlon * 2` bytes and `nlon` is odd (121), so the
  default 4-byte `UNPACK_ALIGNMENT` makes GL reject `texSubImage3D` and leave the texture zeroed —
  every stretch reads 0 and the map renders at ~1/10 scale, silently.
  `gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)` before the upload.
- **Coincident county points.** Todd/Tripp SD and Fall River/Oglala Lakota SD share a seat, and
  duplicate sites make the thin-plate-spline system singular. `warpers()` in `04_warp.py`
  de-duplicates before fitting; any new interpolator must too.
- **Never `shapely.node()` a road network.** `lines_of()` nodes and line-merges, right for
  boundaries but it splits roads at every interchange — 202k vertices instead of 22k, and minutes
  instead of a second. Use `road_lines()`; drawing needs no topology.
- **geopandas `.plot()` linestyle.** Pass a dash tuple as `linestyle=(0, (5, 3))`, never `ls=`.
  `ls` is not normalised by `_expand_kwargs`, which then tries `np.take` on the tuple and raises
  on any MultiPolygon layer.
- **`backdrop-filter` on the panels** makes each one a containing block for `position:fixed`
  descendants. Never write a bare `canvas { position:fixed }` — `#keybar` lives inside `#key` and
  gets yanked to that panel's corner. Style `#gl, #ov` by id.
- **`#bar` is `left:50%`**, so shrink-to-fit only has half the viewport to work with and the row
  wraps at 700px however wide the screen is. It needs `width:max-content` to size to its content,
  and each label+slider pair needs its own `.grp` wrapper so a wrap never orphans a label.
- **Valhalla's public server** refuses routes over 1,500 km and matrices over 400 km. That is why
  it is only used to calibrate OSRM on short hops, never for the main fetch.
- **County seats come from Wikidata** and a few resolve badly — Connecticut has had planning
  regions and no seats since 2022, and some labels come back as raw `Q…` ids. Read the
  `NOT MATCHED` list `20_pick_nodes.py` prints, and patch via `BY_FIPS`, not the name lookup.

## Verifying the WebGL page

Headless screenshots race the page, and every earlier "bug" found this way was the harness:

- Load `?t=1` (and `nochrome=1` for stills). Without it the opening dissolve animates, and
  SwiftShader is too frame-starved to ever finish an eased animation — captures land mid-flight.
- Anchor any injected script to readiness by polling for `#boot` to be removed. `setTimeout` under
  `--virtual-time-budget` fires long before the payload has downloaded.
- Vary the query string between runs. Chrome caches `index.html` per URL, so a rebuilt page keeps
  serving the old one.
- Give `--virtual-time-budget` at least 90000. The payload is 4.7 MB and inflates before the first
  frame; at 40000 the capture is of the loading bar.
- `--dump-dom` alongside `--screenshot` is the cheap way to tell a page bug from a paint artefact.
  A control whose class the DOM says is right but whose pixels say otherwise is a stale
  `backdrop-filter` layer, not a bug — this cost a chase twice.
- To drive the pointer, inject a script that fires its events the moment `#boot` disappears. Do
  **not** wait N `requestAnimationFrame`s first: a frame of this scene under SwiftShader can eat
  the whole virtual-time budget, and the callback simply never arrives.
- `?tilt=`/`?spin=` (degrees) pin the camera; `tilt=90` is the old flat top-down view.
