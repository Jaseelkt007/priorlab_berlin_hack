# Demo site

The project website: an overview page (`index.html`) and one page per project. Each project page is a
shared top header with the project's **own report shown unchanged** underneath (an iframe), in the
reports' white theme. Plain HTML and CSS, no build step, fonts self-hosted.

| Page | Shows |
|---|---|
| `index.html` | overview: one heading, four slanted cards (a real frame, the project name, its question) |
| `picker.html` | `docs/report.html` (suction pick line, delta robot of the ABB IRB 360 FlexPicker class) |
| `crane.html` | `crane/docs/report.html` (Crane Sway) |
| `drone.html` | `site/reports/drone.html`, rendered from `drone/README.md` (the drone project has no report page) |
| `fly.html` | `fly/docs/report.html` (fruit fly, MuJoCo flybody) |

## Open it

Serve the repo root, so the pages can reach the reports in the project folders:

```bash
python3 -m http.server 8765 --bind 127.0.0.1    # then open http://localhost:8765/site/
```

Opening `site/index.html` as a file also works. On GitHub Pages, serve from the repo root.

Keys: `1`–`4` open a project, `Esc` returns to the overview (click the header first if the report has focus).

## Drone page

After `drone/README.md` changes, re-render it from the repo root:

```bash
uv run --no-project --with markdown python site/tools/drone_report.py
```
