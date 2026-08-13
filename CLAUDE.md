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
- **Deploying is `git push`.** The Vercel project is linked to the GitHub repo with its root
  directory set to `dist`, so every push to `main` publishes. Do not deploy with the CLI: a git
  push afterwards overwrites it, and for a while every push was silently republishing the repo
  root — which has no `index.html` — as a 404 site.
- `https://cool-maps.vercel.app` is the honest check. The custom domain is behind a firewall here
  and fails locally even when the deploy is fine.
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
  included **and the span it divides by** — the shader interpolates that across a mode change, so
  reading `SPAN[B.m]` on the CPU floats the dots off the surface for the length of a morph.
- **Anything that frames the map is interpolated between the two endpoints, never measured off
  the blend.** Halfway between two different warps the sheet collapses — points pass through the
  middle — so a fit taken there pulls the camera in and pushes it back out. Simulated over the
  tour route, Seattle→Washington dipped 17.6% and returned. `camera()` fits `hullA` and `hullB`
  separately and travels between them; `fitSpan`, `heightKm`, the pivot and `ghostFit` all lerp
  the same way. Re-measuring the blend "to simplify" puts the lurch straight back.
- The per-mode colour spans `23_build_dist.py` measures also set the **terrain height** — `lift()`
  divides by the same `uSpanPos`/`uSpanNeg`. Re-picking those percentiles reshapes the relief.
- **`EXAG` is per mode** — Drive `s^k` runs 1.0→2.6, Fly 0.35→1.0, and each notch is its default.
  Fly's stretch is mostly below 1, so raising k pulls far places inward past near ones; lowering
  it cuts folding 37%→10% *and* widens the radial spread 2.8×→5.0×. Don't "fix" Fly's crumpling
  with the flight constants instead — `CONNECT`, `N_NEAR_AIRPORTS`, `MIN_FLY_KM` and `CRUISE` were
  all measured and none move folding off ~36%.

## Gotchas

- **The stretch grid texture.** It is `R16F`, not the `R16UI` the payload ships, purely so the
  sampler can filter it — one `texture()` call instead of four `texelFetch` and six mixes by
  hand, on ~476k vertices a frame. Integer textures cannot be filtered at all. Two things are
  load-bearing: `raw()`'s clamp to `uGridSize - 1.001` keeps the bilinear kernel one row short of
  the seam where the two modes meet in a shared layer (without it Drive bleeds into Fly), and
  `gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)` before the upload — `nlon` is odd (121), so any
  narrower texel type gives rows GL rejects outright, leaving the texture zeroed and the map
  rendering at ~1/10 scale with no error at all.
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

## Keeping it fast

Both ends cost. ~600k vertex invocations a frame, each doing trig and texture work — *and*, until
measured on an integrated GPU, a 3840×1910 backbuffer at 4× MSAA, which is ~29M samples a frame.
An earlier note here said resolution scaling buys nothing; that was measured on a discrete GPU
with MSAA already dominating, and it is wrong on integrated hardware. The context now asks for
`antialias: false` and `powerPreference: "high-performance"`, and renders at an adaptive `scale`
(0.75–1.5 device px per css px) that `adapt()` walks from a median of recent frame intervals.

- Distance and bearing from an origin are precomputed by a transform-feedback pass on origin
  change (`computeGeo`), never per frame, and arrive as the `aGA*`/`aGB*` attributes. A morph
  reuses the outgoing B buffer as the incoming A. **A buffer still bound to `ARRAY_BUFFER`
  cannot also be bound as a feedback target** — leave it bound and that one layer silently
  never fills and vanishes, with only a `getError` to show for it. `bindGeo()` unbinds.
- `world()` skips the A origin whenever `uMix >= 1`, halving texture fetches outside a morph.
  Any new per-vertex sampling must stay behind that branch.
- Morph frames are the worst case — the A-origin skip is off *and* the sheet is being re-measured.
  `measure()` caches each endpoint's hull and interpolates; county lines sit morphs and drags out.
- Normals must stay per-vertex. `dFdx` of the position is far cheaper but facets the relief
  visibly: the mesh is 20 km, which is 15–20 px on screen, not the 2–3 px that would hide it.
- **Make frames cheaper, not rarer.** Skipping the paint when nothing moved looks free and is not:
  a page that paints nothing can stop being served frames at all — measured 0 `requestAnimationFrame`
  callbacks across 5 s of idle — and then every timed thing freezes, including the opening
  dissolve and the tour. This was tried and reverted; don't re-add it.
- County lines are ~190k vertices and simplification will *not* shrink them — the count is set by
  how many county arcs there are, not by detail within an arc. Raising the tolerance is wasted.

## Verifying the WebGL page

Headless screenshots race the page, and every earlier "bug" found this way was the harness:

- Load `?t=1` (and `nochrome=1` for stills). Without it the opening dissolve animates, and
  SwiftShader is too frame-starved to ever finish an eased animation — captures land mid-flight.
- Anchor any injected script to readiness by polling for `#boot` to be removed. `setTimeout` under
  `--virtual-time-budget` fires long before the payload has downloaded.
- Vary the query string between runs. Chrome caches `index.html` per URL, so a rebuilt page keeps
  serving the old one.
- Captures that come back as the loading bar have three causes, in this order: a
  `--virtual-time-budget` under 90000 (the payload is 4.7 MB and inflates before the first frame);
  old `--headless` leaving processes behind, so check `pgrep -c chrome` and kill with
  `pgrep -x chrome | xargs -r kill` — a bare `pkill -f chrome` matches its own shell and kills it;
  and the `python3 -m http.server` having quietly died.
- `--dump-dom` alongside `--screenshot` is the cheap way to tell a page bug from a paint artefact.
  A control whose class the DOM says is right but whose pixels say otherwise is a stale
  `backdrop-filter` layer, not a bug — this cost a chase twice.
- To drive the pointer, inject a script that fires its events the moment `#boot` disappears. Do
  **not** wait N `requestAnimationFrame`s first: a frame of this scene under SwiftShader can eat
  the whole virtual-time budget, and the callback simply never arrives.
- `?tilt=`/`?spin=` (degrees) pin the camera; `tilt=90` is the old flat top-down view.
- Driving a real browser is ground truth for *looks*, but **the extension's tab reports
  `document.visibilityState === "hidden"`**, so `requestAnimationFrame` is throttled to roughly
  1 Hz. Clicking the page makes it animate but does not clear the flag. A map that creeps instead
  of animating is that, not a bug — it already cost one wrong diagnosis and a revert. Screenshots
  still force a paint, so use it to judge pixels and never to judge time.
- Test the opening animation at least once *without* `?t=1`. Everything else here uses it, so a
  regression in the intro path can sit unnoticed indefinitely.

**Frame timing cannot be measured in this setup at all.** The tab above is throttled, and headless
virtual time races ahead of real GPU frames, so a frame battery never finishes — raising the budget
makes it *worse*, not better. Judge performance by work removed (samples per frame, `place()` calls
per frame), which is countable, and leave frame rate to the human. The extension's Chrome also
cannot reach `127.0.0.1:8000`; point it at a `vercel deploy` preview instead.
`.iterate/*/make_perf.py` builds `dist/_perf.html` with a frame-phase profiler for a real
foreground tab.
