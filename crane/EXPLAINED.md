# Crane sway with TabPFN, explained in pictures

> **One line:** an overhead crane's load keeps swinging after every move. TabPFN predicts **where the load will be in
> 1 second** and, before each move, **which anti-sway setting gives the shortest safe cycle**. No training, only
> examples. On a real lab crane it got **3× more accurate after 5 s of a new load**; in a full-size MuJoCo crane the live
> pick and place learned the right anti-sway setting and ran **27% faster** than the typed-in setting.

Interactive explainer with all charts: https://claude.ai/artifact/7rjNJLFjPqkE2g74ekxL31

---

## 1. The problem

![the crane](results/still_b.png)

![anti-sway: two half pushes](figures/antisway.png)

```mermaid
flowchart LR
    P["📦 Beam picked at bay 1<br/>x = 1 m"] -->|"10 m trolley move"| Q["🟨 Bay 2, between racks<br/>x = 11 m, ±5 cm to place"]
    Q --> W["⏳ Load still swinging:<br/>operator waits to set down"]
    A["🔧 Anti-sway<br/>needs the sling length,<br/>typed in by hand"] -.->|"wrong sling →<br/>swing comes back"| W
```

| Thing | Value | Why it matters |
|---|---|---|
| Rope (hoist) | 3–7 m, measured | sets how slowly the load swings |
| Sling (rigging) | 0.5–3 m, **never measured** | the anti-sway guesses it; wrong guess = 20+ s of waiting |
| Rack clearance | 0.45 m | a bigger swing hits the racks |
| Set-down rule | within ±5 cm for 1 s | the wait we want to shorten |

The anti-sway only knows the pendulum someone **typed in**. We want the crane to know how **this** load really swings.

---

## 2. The questions we answer

```mermaid
flowchart LR
    IN["Crane sensors so far"] --> Q1["❓ Where is the load in 1 s?<br/><b>offset from trolley</b> + 90% range<br/>(every 0.1 s)"]
    IN --> Q2["❓ Which anti-sway setting<br/>gives the shortest safe wait?<br/><b>wait + swing left</b> per setting<br/>(once per move)"]
```

---

## 3. The pipeline

```mermaid
flowchart LR
    A["MuJoCo crane<br/>rail, trolley, rope, hook,<br/>sling, beam, racks"] --> B["Pick and place cycles<br/>random rope, sling,<br/>payload, speed"]
    B --> C["Sensors<br/>rope angle, encoders,<br/>load cell, plan"]
    C --> D["Table<br/>1 row per 0.1 s<br/>12 inputs"]
    D --> E1["Physics model<br/>(pendulum, typed-in sling)"]
    D --> E2["TabPFN<br/>(examples in context)"]
    E1 --> F["Load position in 1 s<br/>+ anti-sway choice"]
    E2 --> F
```

---

## 4. Learn on some loads, test on new ones

```mermaid
flowchart TB
    subgraph REAL["Real lab tower crane (Nazarbayev Univ., CC BY 4.0)"]
        R1["33 runs, 100 Hz<br/>cables 0.1–0.8 m, 3 payloads"]
    end
    subgraph SIM["MuJoCo overhead crane"]
        F["40 fleet jobs → 3,000 rows<br/>120 commissioning moves"]
        N["new loads never in the table<br/>(long slings, light beams)"]
    end
    R1 -->|"hold out a whole run:<br/>cable 0.5 m / beam / hoisting"| T1["🎯 Test on the real crane"]
    F -->|"context"| T2["🎯 Test in MuJoCo"]
    N --> T2
```

---

## 5. What one table row looks like

```mermaid
flowchart LR
    subgraph IN["12 INPUTS: crane sensors + plan"]
        direction TB
        a["<b>Load moving now</b><br/>off · doff"]
        b["<b>Memory</b><br/>off_m02 · off_m05 · amp2s"]
        c["<b>Pendulum</b><br/>L1 (rope) · payload_t"]
        d["<b>Push</b><br/>x · dx · ddx · plan_dx05 · plan_dx10"]
    end
    subgraph OUT["OUTPUT"]
        direction TB
        o1["offset from trolley in 1 s<br/>5% · 50% · 95%"]
    end
    IN --> OUT
```

The sling length is **never** a column: TabPFN has to read it from how the load moves. The anti-sway choice uses a
second, smaller row: **sway period · speed · tuning error** → **wait** and **swing left**.

---

## 6. How TabPFN predicts (no training)

```mermaid
flowchart LR
    CTX["📚 Context: 3,000 example rows<br/>inputs + what really happened 1 s later"] --> TP["TabPFN<br/>pre-trained foundation model<br/>(v3.5, Prior Labs)"]
    Q["❓ New moment<br/>12 inputs, answer unknown"] --> TP
    TP --> O["offset in 1 s<br/>5% · 50% · 95%"]
    O -.->|"1 s later the sensor shows the truth:<br/>the row joins the table"| CTX
```

- **No training, no tuning:** "fit" stores the table; a new load is handled by adding its rows.
- **Every moment is predicted on its own**; the memory columns carry the recent history.
- **2 models:** the forecast (regression with a range) and the anti-sway choice (wait + swing, per setting).

---

## 7. The opponent: physics + anti-sway

```mermaid
flowchart LR
    S["Typed-in sling<br/>0.5 m"] --> L["Pendulum length<br/>rope + hook + sling"]
    L --> P["Physics forecast<br/>(single pendulum)"]
    L --> Z["ZV anti-sway<br/>two half pushes,<br/>half a period apart"]
```

What cranes use today: exact if the length is right, wrong when the sling or the load shape differs.

---

## 8. Results

| Test | Physics / industry | TabPFN | Where |
|---|---|---|---|
| Forecast, new payload (beam), after 5 s of its rows | 0.54 cm (fitted pendulum) | **0.10 cm** (3× better), 90% range holds 91% | real lab crane |
| Forecast, hidden 2.5 m sling | 4.7 cm | **0.7 cm** | MuJoCo, 27 runs |
| Forecast, all 81 settings | 2.2 cm | **0.8 cm** | MuJoCo playground |
| 12 new loads, total time | 403 s | **261 s (−35%)** | MuJoCo closed loop |
| Live pick and place, 9 cycles | about 68 s / cycle | **49.6 s / cycle**, 7 of 9 OK | MuJoCo live |

```mermaid
flowchart LR
    A["Cycle 1<br/>only commissioning rows<br/>picks 0.5 m: 23.7 s wait"] -->|"its own move<br/>joins the table"| B["Cycles 2–4<br/>tries 1.5 m and 1.0 m<br/>1.0 s and 3.0 s wait"]
    B -->|"measured beats predicted"| C["Cycles 5–9<br/>1.5 m every time<br/>1.0 s wait, all OK"]
```

### Checks a judge will ask about

| Check | Answer |
|---|---|
| Real data? | The forecast is checked on a real lab crane; the full-size crane and closed loop are simulation. Every number says which. |
| Fair opponent? | Physics gets a fitted pendulum length on the lab crane; with the right sling it wins, and we show that. |
| Fewer inputs? | The forecast needs all 12 (5 nearly double the error); the anti-sway choice is as good with 3 (`ablation.py`). |
| Safer? | Not shown: nothing collided, so the gain is time. The 99% range is used as a "clear of the racks" rule. |
| What failed? | Timing the set-down alone: the position was right, but the load still moved at 15 cm/s. |

![real crane: textbook period is wrong](figures/real_crane_period.png)
![real crane: adaptation](figures/real_crane_adaptation.png)
![MuJoCo: forecast by sling](figures/forecast_by_sling.png)
![MuJoCo: closed loop](figures/closed_loop.png)
![live pick and place](figures/live_cycles.png)

---

## 9. Try it

```sh
crane/run_live.sh                      # Linux / macOS: live pick and place, TabPFN chooses the anti-sway
powershell -File crane\run_live.ps1    # Windows
crane/.venv/bin/python crane/check_setup.py
```

---

The pitch slides and video script: [`PITCH.md`](PITCH.md). Figures: `figures/` (made by `make_figures.py`).
