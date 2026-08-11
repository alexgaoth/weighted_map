"""Summarise the result: rankings, state aggregates, and a variance decomposition."""
import json
import pickle

import numpy as np
import pandas as pd

cty = pd.read_csv("data/county_warped.csv", dtype={"fips": str})
calib = pd.read_csv("data/calibration_sample.csv")
G = pickle.load(open("data/warped_geoms.pkl", "rb"))["geoms"]

out = {"n_counties": int(cty.fips.nunique()), "n_routes": int(len(cty)),
       "calibration": {"n_pairs": int(len(calib)),
                       "factor": float(calib.osrm_h.sum() / calib.valhalla_h.sum()),
                       "ratio_sd": float(calib.ratio.std())}}

for origin in ("capitol", "center"):
    d = cty[cty.origin == origin]
    far = d[d.gc_km > 150]
    true_a = G[(origin, "true", "nation")].geometry.iloc[0].area
    warp_a = G[(origin, "radial", "nation")].geometry.iloc[0].area

    # stretch = detour x (100 / speed); split the spread of log(stretch) between them.
    ld, ls_ = np.log(far.detour), np.log(100 / far.speed_kmh)
    v = np.var(ld + ls_)
    cols = ["state", "county", "seat", "gc_km", "road_km", "hours", "speed_kmh", "detour",
            "stretch", "drift_km"]
    out[origin] = {
        "hours_median": float(d.hours.median()), "hours_max": float(d.hours.max()),
        "speed_median": float(d.speed_kmh.median()),
        "stretch_median": float(d.stretch.median()),
        "drift_median_km": float(d.drift_km.median()), "drift_max_km": float(d.drift_km.max()),
        "area_ratio": float(warp_a / true_a),
        "var_share_detour": float((np.var(ld) + np.cov(ld, ls_)[0, 1]) / v),
        "var_share_slowness": float((np.var(ls_) + np.cov(ld, ls_)[0, 1]) / v),
        "top": far.nlargest(15, "stretch")[cols].round(2).to_dict("records"),
        "bottom": far.nsmallest(10, "stretch")[cols].round(2).to_dict("records"),
        "by_state": (d.groupby("state").stretch.median().round(3).sort_values(ascending=False)
                     .to_dict()),
    }

json.dump(out, open("data/summary.json", "w"), indent=1)

c = out["capitol"]
print(f"counties {out['n_counties']}  routes {out['n_routes']}")
print(f"calibration k={out['calibration']['factor']:.4f} from {out['calibration']['n_pairs']} pairs")
for o in ("capitol", "center"):
    s = out[o]
    print(f"\n{o}: median {s['hours_median']:.1f} h, max {s['hours_max']:.1f} h, "
          f"median stretch {s['stretch_median']:.3f}, warped area {s['area_ratio']:.2f}x true")
    print(f"   variance of log-stretch: {s['var_share_detour']:.0%} circuity / "
          f"{s['var_share_slowness']:.0%} slowness")
    print("   most stretched states:", ", ".join(f"{k} {v:.2f}" for k, v in
                                                 list(s["by_state"].items())[:6]))
    print("   least stretched states:", ", ".join(f"{k} {v:.2f}" for k, v in
                                                  list(s["by_state"].items())[-6:]))
