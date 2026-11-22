"""Render drone/README.md as-is into site/reports/drone.html (the drone project has no report page).

Run from the repo root:  uv run --with markdown python site/tools/drone_report.py
"""
import pathlib
import re

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "drone" / "README.md"
OUT = ROOT / "site" / "reports" / "drone.html"
PREFIX = "../../drone/"  # site/reports/ -> drone/

body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])


def fix(m):
    attr, url = m.group(1), m.group(2)
    if re.match(r"^(https?:|mailto:|#|data:)", url):
        return m.group(0)
    return f'{attr}="{PREFIX}{url}"'


body = re.sub(r'(src|href)="([^"]+)"', fix, body)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Drone · Intruder Forecast</title>
<link rel="stylesheet" href="../assets/fonts.css">
<style>
  :root {{ color-scheme: light; --ink: #0f172a; --ink-2: #475569; --line: #e2e8f0; --panel: #f7f8fa; --tab: #2563eb; }}
  body {{ margin: 0; background: #fff; color: var(--ink); font: 16px/1.65 Inter, system-ui, sans-serif; }}
  main {{ max-width: 1040px; margin: 0 auto; padding: 40px 28px 80px; }}
  h1 {{ font-size: 34px; line-height: 1.2; letter-spacing: -0.01em; margin: 0 0 18px; }}
  h2 {{ font-size: 24px; margin: 44px 0 12px; padding-top: 20px; border-top: 1px solid var(--line); }}
  h3 {{ font-size: 18px; margin: 28px 0 8px; }}
  p, li {{ color: #1e293b; }}
  a {{ color: var(--tab); }}
  img {{ max-width: 100%; height: auto; border-radius: 10px; border: 1px solid var(--line); }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14.5px; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: var(--panel); font-weight: 600; }}
  code {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 0.9em; background: var(--panel); padding: 1px 5px; border-radius: 4px; }}
  pre {{ background: var(--panel); padding: 14px 16px; border-radius: 10px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  .src {{ font: 13px 'IBM Plex Mono', monospace; color: var(--ink-2); margin: 0 0 24px; }}
</style>
</head>
<body>
<main>
<p class="src">Rendered from drone/README.md, unchanged.</p>
{body}
</main>
</body>
</html>
""", encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}")
