"""Render dist/og.png — the link preview, straight from the live page."""
import pathlib
import subprocess

URL = ("http://127.0.0.1:8731/index.html"
       "?t=1&from=Washington%20D.C.&mode=fly&nochrome=1&tilt=32&spin=-22&roads=0")
dist = pathlib.Path("dist")
srv = subprocess.Popen(["python3", "-m", "http.server", "8731", "--bind", "127.0.0.1",
                        "-d", str(dist)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    subprocess.run(["sh", "-c", "until curl -sf -o /dev/null http://127.0.0.1:8731/index.html; "
                                "do sleep 0.3; done"], check=True, timeout=40)
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--enable-unsafe-swiftshader", "--hide-scrollbars",
                    "--force-color-profile=srgb", "--window-size=1200,630",
                    "--virtual-time-budget=45000", f"--screenshot={dist / 'og.png'}", URL],
                   check=True, capture_output=True, timeout=180)
finally:
    srv.terminate()
print(f"dist/og.png  {(dist / 'og.png').stat().st_size / 1024:.0f} KB")
