"""Render the map figures, once per theme."""
import pickle

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

mpl.use("Agg")
mpl.rcParams["font.family"] = "DejaVu Sans"

BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a"], "dark": ["#3987e5", "#d95926", "#199e70"]}
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  hair="#e1e0d9", ramp=BLUE),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 hair="#2c2c2a", ramp=BLUE[::-1]),
}
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}
STRETCH_LO, STRETCH_HI = 1.0, 1.8
PANEL = {"capitol": "From the Capitol", "center": "From the geographic centre"}
ORIGIN_LABEL = {"capitol": "the U.S. Capitol in Washington D.C.",
                "center": "the geographic centre of the 48 states, in Lebanon, Kansas"}
RING_ANGLE = {"capitol": -90, "center": -115}   # bearing to write the hour labels along

blob = pickle.load(open("data/warped_geoms.pkl", "rb"))
G = blob["geoms"]
cty = pd.read_csv("data/county_warped.csv", dtype={"fips": str})

county_shapes = gpd.read_file("data/cb_2023_us_county_5m/cb_2023_us_county_5m.shp")
county_shapes = county_shapes[~county_shapes.STATEFP.isin(NON_CONUS)].to_crs(5070)
county_shapes["fips"] = county_shapes.STATEFP + county_shapes.COUNTYFP
state_shapes = gpd.read_file("data/cb_2023_us_state_5m/cb_2023_us_state_5m.shp")
state_shapes = state_shapes[~state_shapes.STATEFP.isin(NON_CONUS)].to_crs(5070)


def mapfig(extent, ncols=1, width=13.0, header=1.05, footer=1.05, gap=0.35):
    """A figure whose axes exactly fit the map, so there is no letterboxing."""
    (x0, x1, y0, y1) = extent
    panel_w = (width - gap * (ncols - 1)) / ncols
    panel_h = panel_w * (y1 - y0) / (x1 - x0)
    fig, axes = plt.subplots(1, ncols, figsize=(width, panel_h + header + footer))
    fig.subplots_adjust(left=0, right=1, bottom=footer / (panel_h + header + footer),
                        top=1 - header / (panel_h + header + footer), wspace=gap / panel_w)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_axis_off()
        ax.set_aspect("equal")
    return fig, axes


def extent_of(*arrays, pad=0.03):
    xs = np.concatenate([a[0] for a in arrays])
    ys = np.concatenate([a[1] for a in arrays])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - px, x1 + px, y0 - py, y1 + py


def head(fig, t, main, sub):
    h = fig.get_size_inches()[1]
    fig.suptitle(main, color=t["ink"], fontsize=17, fontweight="bold",
                 x=0.005, y=1 - 0.30 / h, ha="left", va="top")
    fig.text(0.005, 1 - 0.62 / h, sub, color=t["ink2"], fontsize=10, ha="left", va="top")


def panel_title(ax, t, text):
    ax.set_title(text, color=t["ink"], fontsize=12, fontweight="bold", loc="left", pad=8)


def hour_rings(ax, t, origin, step_km=500, label_every=1):
    x0, x1, y0, y1 = *ax.get_xlim(), *ax.get_ylim()
    reach = max(np.hypot(x, y) for x in (x0, x1) for y in (y0, y1)) / 1000
    th = np.linspace(0, 2 * np.pi, 720)
    ang = np.deg2rad(RING_ANGLE[origin])
    for i, r_km in enumerate(np.arange(step_km, reach + step_km, step_km), start=1):
        r = r_km * 1000
        ax.plot(r * np.cos(th), r * np.sin(th), color=t["hair"], lw=0.7, zorder=0)
        lx, ly = r * np.cos(ang), r * np.sin(ang)
        if i % label_every == 0 and x0 < lx < x1 and y0 < ly < y1:
            ax.text(lx, ly, f"{r_km / 100:.0f} h", color=t["muted"], fontsize=8,
                    ha="center", va="center", zorder=5,
                    bbox=dict(fc=t["surface"], ec="none", pad=1.5))


def choropleth(ax, gdf, values, t, cmap, lo, hi, edge_lw=0.12):
    gdf = gdf.copy()
    gdf["_v"] = np.clip(values, lo, hi)
    gdf.plot(ax=ax, column="_v", cmap=cmap, vmin=lo, vmax=hi,
             edgecolor=t["surface"], linewidth=edge_lw, zorder=2)


def colorbar(fig, ax, t, cmap, lo, hi, label, ticks, fmt="{:.1f}x"):
    cb = fig.colorbar(mpl.cm.ScalarMappable(Normalize(lo, hi), cmap), ax=ax,
                      orientation="horizontal", fraction=0.045, pad=0.015, aspect=45, shrink=0.62)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"], labelsize=8.5, length=0)
    cb.set_ticks(ticks)
    cb.set_ticklabels([fmt.format(abs(v)) for v in ticks])
    cb.set_label(label, color=t["ink2"], fontsize=9.5, labelpad=6)
    return cb


def spread_labels(d, n=6, min_gc=350):
    """Highest-drift county in each angular sector, so callouts never pile up."""
    f = d[d.gc_km > min_gc].copy()
    f["sector"] = pd.cut(np.degrees(np.arctan2(f.y, f.x)) % 360, np.linspace(0, 360, n + 1))
    return f.loc[f.groupby("sector", observed=True).drift_km.idxmax()]


def render(theme):
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    ser = SERIES[theme]
    plt.rcParams.update({"figure.facecolor": t["surface"], "savefig.facecolor": t["surface"]})

    def save(fig, name):
        fig.savefig(f"out/{name}_{theme}.png", dpi=150, facecolor=t["surface"],
                    bbox_inches="tight", pad_inches=0.30)
        plt.close(fig)

    # ---- 1/2. Hero time maps -------------------------------------------------
    for origin in ("capitol", "center"):
        d = cty[cty.origin == origin]
        wc = G[(origin, "radial", "counties")]
        bx = wc.total_bounds
        fig, (ax,) = mapfig(extent_of((bx[[0, 2]], bx[[1, 3]]), pad=0.045), width=13.0, footer=1.15)
        hour_rings(ax, t, origin)
        choropleth(ax, wc, d.stretch.values, t, cmap, STRETCH_LO, STRETCH_HI)
        G[(origin, "radial", "states")].plot(ax=ax, facecolor="none", edgecolor=t["surface"],
                                             linewidth=0.7, zorder=3)
        G[(origin, "radial", "nation")].plot(ax=ax, facecolor="none", edgecolor=t["ink"],
                                             linewidth=1.1, zorder=4)
        G[(origin, "true", "nation")].plot(ax=ax, facecolor="none", edgecolor=t["muted"],
                                           linewidth=1.1, linestyle=(0, (5, 3)), zorder=5)
        ax.plot(0, 0, marker="*", ms=15, color=t["ink"], mec=t["surface"], mew=1.2, zorder=6)

        for _, r in spread_labels(d).iterrows():
            out = np.array([r.wx, r.wy]) / max(np.hypot(r.wx, r.wy), 1)
            ax.annotate(f"{r.seat}, {r.state}\n{r.hours:.1f} h · {r.drift_km:.0f} km out",
                        (r.wx, r.wy), xytext=(20 * out[0], 20 * out[1]), textcoords="offset points",
                        fontsize=8.2, color=t["ink"], zorder=7,
                        ha="left" if out[0] > -0.3 else "right",
                        va="bottom" if out[1] > -0.3 else "top",
                        bbox=dict(fc=t["surface"], ec=t["hair"], lw=0.6, pad=2.4),
                        arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.7))

        head(fig, t, "The United States redrawn by driving time",
             f"Every county keeps its true bearing from {ORIGIN_LABEL[origin]}, but its distance is "
             f"replaced by driving hours at 1 hour = 100 km.\nThe rings are hours; the dashed grey line "
             f"is where the country really is.")
        colorbar(fig, ax, t, cmap, STRETCH_LO, STRETCH_HI,
                 "time-distance ÷ straight-line distance", [1.0, 1.2, 1.4, 1.6, 1.8])
        ax.legend(handles=[Line2D([], [], color=t["muted"], ls=(0, (5, 3)), lw=1.0, label="true outline"),
                           Line2D([], [], color=t["ink"], lw=1.1, label="time-distance outline")],
                  loc="lower left", frameon=False, fontsize=9, labelcolor=t["ink2"])
        save(fig, f"hero_{origin}")

    # ---- 3. Drift field ------------------------------------------------------
    both = [(cty[cty.origin == o][["wx", "x"]].values.ravel(),
             cty[cty.origin == o][["wy", "y"]].values.ravel()) for o in ("capitol", "center")]
    fig, axes = mapfig(extent_of(*both, pad=0.05), ncols=2, width=15.0, header=1.4)
    for ax, origin in zip(axes, ("capitol", "center")):
        d = cty[cty.origin == origin]
        G[(origin, "true", "states")].plot(ax=ax, facecolor="none", edgecolor=t["hair"], lw=0.5, zorder=1)
        G[(origin, "true", "nation")].plot(ax=ax, facecolor="none", edgecolor=t["muted"], lw=0.9, zorder=2)
        segs = np.stack([np.column_stack([d.x, d.y]), np.column_stack([d.wx, d.wy])], axis=1)
        ax.add_collection(mpl.collections.LineCollection(
            segs, colors=cmap(Normalize(STRETCH_LO, STRETCH_HI)(np.clip(d.stretch, STRETCH_LO, STRETCH_HI))),
            linewidths=0.6, zorder=3))
        ax.scatter(d.wx, d.wy, s=1.5, c=t["ink"], zorder=4, linewidths=0)
        ax.plot(0, 0, marker="*", ms=13, color=t["ink"], mec=t["surface"], mew=1.2, zorder=5)
        panel_title(ax, t, PANEL[origin])
    head(fig, t, "Where every county moves",
         "Each thread runs from a county seat's true position to its time-distance position. "
         "Everything moves outward, away from the origin — the question is how far.")
    colorbar(fig, axes.tolist(), t, cmap, STRETCH_LO, STRETCH_HI, "stretch factor",
             [1.0, 1.2, 1.4, 1.6, 1.8])
    save(fig, "drift")

    # ---- 4. Stretch in true geography ---------------------------------------
    b = county_shapes.total_bounds
    for name, cols, sub in [
        ("stretch_true", [("capitol", "stretch"), ("center", "stretch")],
         "Driving hours x 100 km divided by straight-line distance, county by county, drawn on the "
         "undistorted map. The scale runs from no time penalty at all to a penalty of 1.8x and above.")]:
        fig, axes = mapfig(extent_of((b[[0, 2]], b[[1, 3]]), pad=0.02), ncols=2, width=15.0, header=1.6)
        for ax, (origin, _) in zip(axes, cols):
            d = cty[cty.origin == origin].set_index("fips").loc[county_shapes.fips]
            choropleth(ax, county_shapes, d.stretch.values, t, cmap, STRETCH_LO, STRETCH_HI)
            state_shapes.plot(ax=ax, facecolor="none", edgecolor=t["surface"], lw=0.6, zorder=3)
            panel_title(ax, t, PANEL[origin])
        head(fig, t, "How much further it is in time than in space", sub)
        colorbar(fig, axes.tolist(), t, cmap, STRETCH_LO, STRETCH_HI, "stretch factor",
                 [1.0, 1.2, 1.4, 1.6, 1.8])
        save(fig, name)

    # ---- 5. The two causes ---------------------------------------------------
    d = cty[cty.origin == "capitol"].set_index("fips").loc[county_shapes.fips]
    fig, axes = mapfig(extent_of((b[[0, 2]], b[[1, 3]]), pad=0.02), ncols=2, width=15.0, header=1.6, footer=1.2)
    specs = [(d.detour.values, 1.0, 1.6, "road distance ÷ straight-line distance",
              [1.0, 1.2, 1.4, 1.6], "{:.1f}x", "Circuity — how far the road bends"),
             (-d.speed_kmh.values, -115, -75, "implied door-to-door speed (km/h), slower to the right",
              [-115, -105, -95, -85, -75], "{:.0f}", "Slowness — how fast that road runs")]
    for ax, (vals, lo, hi, lab, ticks, fmt, ttl) in zip(axes, specs):
        choropleth(ax, county_shapes, vals, t, cmap, lo, hi)
        state_shapes.plot(ax=ax, facecolor="none", edgecolor=t["surface"], lw=0.6, zorder=3)
        colorbar(fig, ax, t, cmap, lo, hi, lab, ticks, fmt)
        panel_title(ax, t, ttl)
    head(fig, t, "Stretch = circuity x slowness",
         "A county is pushed outward for one of two reasons: the road detours around an obstacle, or the "
         "road is slow. Splitting the stretch factor into its two parts says which. Both maps are from the Capitol.")
    save(fig, "why")

    # ---- 6. Warping methods --------------------------------------------------
    bx = G[("capitol", "tps", "nation")].total_bounds
    fig, axes = mapfig(extent_of((bx[[0, 2]], bx[[1, 3]]), pad=0.04), ncols=2, width=15.0, header=1.6)
    methods = [("radial", "radial stretch field", 2.8), ("affine", "piecewise affine (Delaunay)", 1.7),
               ("tps", "thin-plate spline", 0.9)]
    axes[1].set_xlim(-1.55e6, -0.25e6)
    axes[1].set_ylim(0.30e6, 1.45e6)
    for ax in axes:
        G[("capitol", "true", "nation")].plot(ax=ax, facecolor="none", edgecolor=t["muted"],
                                              lw=0.9, linestyle=(0, (5, 3)), zorder=1)
        for (m, _, lw), c in zip(methods, ser):
            G[("capitol", m, "nation")].plot(ax=ax, facecolor="none", edgecolor=c, lw=lw, zorder=2)
        ax.plot(0, 0, marker="*", ms=12, color=t["ink"], mec=t["surface"], mew=1.2, zorder=5)
    for (_, lab, lw), c in zip(methods, ser):
        axes[0].plot([], [], color=c, lw=max(lw, 1.6), label=lab)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=t["ink2"])
    panel_title(axes[0], t, "Three ways to carry the borders along")
    panel_title(axes[1], t, "The Great Lakes, magnified")
    head(fig, t, "The county points agree; the space between them does not",
         "All three interpolators reproduce the 3,109 measured county positions exactly, so they are drawn "
         "thick-to-thin to stay visible where they coincide.\nThey part company only where there is no "
         "measurement to constrain them — out over water, and past the edge of the country.")
    save(fig, "methods")


for theme in THEMES:
    render(theme)
    print(f"rendered {theme}")
