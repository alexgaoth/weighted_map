"""Assemble report.html: inline the figures as data URIs and fill the tables."""
import base64
import json
import re

import pandas as pd

summary = json.load(open("data/summary.json"))
cty = pd.read_csv("data/county_warped.csv", dtype={"fips": str})
tpl = open("src/report_template.html").read()

# Valhalla durations measured directly in the calibration work, for the four long
# routes its public server will accept.
VALHALLA = {"36061": ("New York", 4.02), "17031": ("Chicago", 11.25),
            "13121": ("Atlanta", 9.97), "12086": ("Miami", 16.07)}
FLIP = [("PA", "ridges crossed at short range, or after 1,800 km of interstate"),
        ("NM", "open run on I-40, or straight into the Colorado Plateau"),
        ("AZ", "open run on I-40, or straight into the Rockies"),
        ("MD", "Chesapeake in the way, or approached from behind"),
        ("MN", "far either way; the Great Lakes detour never goes away"),
        ("NV", "least distorted in the country from either origin")]


def plate(m):
    name, no, title = m.group(1), m.group(2), m.group(3)
    imgs = ""
    for theme in ("light", "dark"):
        b64 = base64.b64encode(open(f"out/{name}_{theme}.webp", "rb").read()).decode()
        imgs += (f'<img class="fig-{theme}" alt="{title}" '
                 f'src="data:image/webp;base64,{b64}">')
    return (f'<figure class="full"><div class="plate-head">'
            f'<span class="plate-no">Plate {no}</span><h3>{title}</h3></div>'
            f'<div class="plate-img">{imgs}</div>')


def rows_top():
    out = ""
    for r in summary["capitol"]["top"]:
        out += (f'<tr><td class="place">{r["seat"]}, {r["state"]}</td>'
                f'<td class="dim">{r["county"]}</td>'
                f'<td class="num">{r["gc_km"]:,.0f} km</td><td class="num">{r["road_km"]:,.0f} km</td>'
                f'<td class="num">{r["hours"]:.2f} h</td><td class="num">{r["speed_kmh"]:.0f}</td>'
                f'<td class="num">{r["stretch"]:.2f}×</td></tr>')
    return out


def rows_flip():
    a, b = summary["capitol"]["by_state"], summary["center"]["by_state"]
    return "".join(
        f'<tr><td class="place">{s}</td><td class="num">{a[s]:.2f}×</td>'
        f'<td class="num">{b[s]:.2f}×</td><td class="dim note">{why}</td></tr>' for s, why in FLIP)


def rows_validation():
    c = cty[cty.origin == "capitol"].set_index("fips")
    out = ""
    for fips, (label, v) in VALHALLA.items():
        h = c.loc[fips, "hours"]
        out += (f'<tr><td class="place">{label}</td><td class="num">{v:.2f} h</td>'
                f'<td class="num">{h:.2f} h</td>'
                f'<td class="num dim">{(h / v - 1) * 100:+.1f}%</td></tr>')
    return out


html = re.sub(r"\{\{PLATE:([a-z_]+):(\d):([^}]+)\}\}", plate, tpl)
for key, fn in (("capitol_top", rows_top), ("flip", rows_flip), ("validation", rows_validation)):
    html = html.replace("{{TABLE:%s}}" % key, fn())

assert "{{" not in html, "unfilled placeholder"
open("report.html", "w").write(html)
print(f"report.html  {len(html) / 1048576:.2f} MB")
