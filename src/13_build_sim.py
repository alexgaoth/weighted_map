"""Inline the simulation payload into a single self-contained page."""
import base64
import json

meta = json.load(open("data/sim_payload.json"))
blob = open("data/sim_payload.bin", "rb").read()
tpl = open("src/sim_template.html").read()

n_ready = sum(o["ready"] for o in meta["origins"])
html = (tpl.replace("{{META}}", json.dumps(meta, separators=(",", ":")))
           .replace("{{BLOB}}", base64.b64encode(blob).decode()))
assert "{{" not in html

open("simulation.html", "w").write(html)
print(f"simulation.html  {len(html) / 1048576:.2f} MB   origins ready: {n_ready}/{len(meta['origins'])}")
