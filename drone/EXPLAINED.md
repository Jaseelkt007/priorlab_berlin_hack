# Drone defence with TabPFN, explained in pictures

> **One line:** a radar sees an intruder drone. TabPFN predicts **where it will be in 1 second** and **whether it
> breaks into the protected zone within 2 seconds**. No training, only examples, and it beats the radar industry's
> standard tracker (IMM) by **8%**, even after the IMM is re-tuned on the same data and against a pilot it never saw.

---

## 1. The problem

A radar knows where a drone **is**. To stop it, you need where it **will be**, and a small drone can change course within a second. This matters wherever small drones are a threat: airports, stadiums, power plants, prisons.

![the world: building, red zone, an intruder flying in, the X2 defender hovering](results/world.gif)

```mermaid
flowchart LR
    S["🛸 Intruder drone<br/>starts 60 m away"] -->|"20 to 30 s of flight"| Z["🔴 Red zone<br/>10 m radius, 30 m high"]
    R["📡 Radar on the building<br/>sees the drone the whole time"] -.->|"10 readings per second<br/>noisy, about 11% missing"| S
    Z --> B["🏢 Protected building"]
```

| Circle | Size | Job |
|---|---|---|
| Radar coverage | 60 to 70 m | **see** the drone |
| Red zone | 10 m | **no-go area**: entering it is a *breach* |
| Gap between them | 20 to 30 s | **time to predict and warn** |

The radar only says where the drone **is**. We want to know where it **will be**.

---

## 2. The questions we answer at every radar reading

```mermaid
flowchart LR
    IN["Radar readings so far"] --> Q1["❓ Where is the drone in 1 s?<br/><b>x, y, z</b> + 90% range"]
    IN --> Q2["❓ Will it enter the red zone<br/>within 2 s?<br/><b>alarm probability</b>"]
```

---

## 3. The pipeline

```mermaid
flowchart LR
    A["MuJoCo simulator<br/>real drone models<br/>wind, physics"] --> B["Intruder pilot<br/>attack / fly-by / loiter<br/>7 manoeuvres"]
    B --> C["Radar<br/>noise + dropouts"]
    C --> D["Table<br/>1 row per reading<br/>16 inputs"]
    D --> E1["IMM<br/>(radar standard)"]
    D --> E2["TabPFN<br/>(examples in context)"]
    E1 --> F["Prediction in 1 s<br/>+ alarm"]
    E2 --> F
```

---

## 4. Two drones: learn on one, test on the other

```mermaid
flowchart TB
    subgraph BIG["Skydio X2: big drone, up to 10 m/s"]
        X["2,000 flights<br/>≈400,000 rows"]
    end
    subgraph SMALL["Crazyflie: small drone, about 3 m/s (the attacker)"]
        C1["flights 0 to 49<br/>a few examples"]
        C2["flights 50 to 299<br/><b>250 TEST flights</b><br/>never seen by any model"]
    end
    X -->|"4,000 rows"| A["TabPFN A"]
    X -->|"4,000 rows"| B["TabPFN B"]
    C1 -->|"+1,000 rows"| B
    X -->|"tuned on"| I["IMM"]
    C2 --> T["🎯 Test: same rows for all 3 models"]
```

---

## 5. What one table row looks like

```mermaid
flowchart LR
    subgraph IN["16 INPUTS: radar only"]
        direction TB
        a["<b>Where</b><br/>range · height · zone_fwd · zone_left"]
        b["<b>Moving now</b><br/>speed · climb · acc_fwd · acc_left · acc_up · turn_rate"]
        c["<b>Memory</b><br/>speed_1s · turn_rate_1s · climb_1s · turn_rate_2s"]
        d["<b>Radar quality</b><br/>n_missing · fit_error"]
    end
    subgraph OUT["OUTPUTS"]
        direction TB
        o1["moves in 1 s: forward · left · up<br/>→ converted to <b>x, y, z</b>"]
        o2["breach within 2 s: <b>yes / no</b>"]
    end
    IN --> OUT
```

The movement is learned in the **drone's own frame** (forward, left, up), so a turn to the left looks the same in
every direction. We convert back to x, y, z for scoring and the video.

---

## 6. How TabPFN predicts (no training)

```mermaid
flowchart LR
    CTX["📚 Context: 4,000 to 5,000 example rows<br/>inputs + what really happened"] --> TP["TabPFN<br/>pre-trained foundation model<br/>(v3.5, Prior Labs)"]
    Q["❓ New radar reading<br/>16 inputs, answer unknown"] --> TP
    TP --> O["x, y, z in 1 s<br/>5% · 50% · 95%<br/>+ breach probability"]
```

- **No training, no tuning:** the examples are given to it, like showing a person a few cases.
- **Every reading is predicted on its own**; the "memory" columns carry the recent history.
- **4 models in total:** forward, left and up (regression with a range), plus the alarm (classification).

---

## 7. The opponent: IMM

```mermaid
flowchart LR
    R["Radar reading"] --> K1["Constant speed<br/>model"]
    R --> K2["Accelerating<br/>model"]
    R --> K3["Turning<br/>model"]
    K1 --> M["Mix by which model<br/>fits best right now"]
    K2 --> M
    K3 --> M
    M --> P["Position in 1 s<br/>+ uncertainty"]
```

This is what real radar trackers use. It is physics with 5 hand-set knobs, which we tuned on the big drone.

---

## 8. Results (250 test flights, 112,501 moments)

| Model | Error 1 s ahead | New pilot (never seen) | Breaches caught |
|---|---|---|---|
| IMM (radar standard), tuned on big drone | 1.54 m | 1.80 m | 99% |
| IMM re-tuned on the same small-drone flights | 1.50 m | 1.82 m | 98% |
| TabPFN A (big drone examples only) | 1.70 m | 2.25 m | 95% |
| **TabPFN B (+ 1,000 small drone examples)** | **1.37 m** ✅ | **1.67 m** ✅ | 98% |

![error by model](results/charts/error_by_model.png)

```mermaid
flowchart LR
    A["TabPFN A<br/>never saw the small drone<br/><b>10% worse</b> than IMM"] -->|"+ small-drone example rows<br/>no training"| B["TabPFN B<br/><b>8% better</b> than the<br/>fairly re-tuned IMM"]
    B -->|"new pilot style<br/>nothing changed"| C["still <b>8% better</b><br/>better on 217 of 250 flights"]
```

### Three checks a judge will ask about

```mermaid
flowchart TB
    Q1["❓ Did TabPFN only win because it saw small-drone data?"] --> A1["IMM re-tuned on the SAME data: 1.50 m<br/>TabPFN B: 1.37 m → <b>still 8% better</b>"]
    Q2["❓ How much data on a new drone is needed?"] --> A2["<b>250 rows ≈ 25 s of radar</b><br/>beats both IMMs"]
    Q3["❓ Did it just learn the pilot's habits?"] --> A3["new pilot, nothing re-tuned<br/><b>still 8% better</b>"]
```

![example sweep](results/charts/example_sweep.png)

- **Better on:** straight, zigzag, orbit, hover, dash. **Slightly worse on:** dive, climb.
- **Alarm:** all models are near perfect at 2 s (AUC 0.99).
- **In the 250 test flights 108 drones breached**; TabPFN B warned for 98% of them, about 4 s before.
- **The gain is solid:** resampling the 250 flights 1,000 times keeps it between 7% and 10%.

![error by manoeuvre](results/charts/error_by_manoeuvre.png)

---

### IMM + TabPFN: TabPFN corrects the tracker the radar already runs

```mermaid
flowchart LR
    R["📡 radar readings"] --> I["IMM (physics)<br/>'in 1 s it moves 2.6 m forward'<br/>'70% sure: straight'"]
    R --> F["16 radar inputs"]
    I -->|"6 extra columns"| T["TabPFN<br/>examples show when the IMM<br/>is right and how it is usually wrong"]
    F --> T
    T --> O["corrected position in 1 s"]
```

| On pilot b (never seen) | Error 1 s ahead |
|---|---|
| IMM | 1.80 m |
| TabPFN B | 1.67 m |
| **IMM + TabPFN** | **1.62 m: 10% better than the IMM**, better on 232 of 250 flights |

---

### Live: does it still work in real time?

TabPFN needs 252 ms per answer, the radar reads every 100 ms. So the IMM tracks every 0.1 s and TabPFN sends a new correction every 0.3 s.

```mermaid
flowchart LR
    R["📡 radar, every 0.1 s"] --> I["IMM<br/>fresh guess every 0.1 s"]
    R -->|"every 0.3 s"| T["TabPFN (252 ms)<br/>'the IMM is off by this much'"]
    T -->|"correction, 0.3 s later"| P["IMM guess + latest correction"]
    I --> P
```

| On pilot b | Error 1 s ahead |
|---|---|
| IMM alone | 1.80 m |
| IMM + TabPFN, an answer at every reading (not live) | 1.56 m: 13% better |
| **Live: correction every 0.3 s** | **1.72 m: 4% better**, on 201 of 250 flights |

**The gap is speed, not accuracy:** a faster TabPFN gets back toward 13%.

---

## 9. See it

![IMM, TabPFN B and IMM + TabPFN on the same flight](results/compare.gif)

| Colour | Meaning |
|---|---|
| 🟢 green line | the real path |
| 🔵 blue | IMM's predicted path; its ball = its guess for *now*, made 1 s ago |
| 🟠 orange | TabPFN B |
| 🟣 purple | IMM + TabPFN |

All videos (not in git; `drone/make_videos.sh` makes them): list in `drone/results/README.md`, "Videos".

---

## 10. In a real system

```mermaid
flowchart LR
    R["📡 Radar<br/>on the building<br/>measures"] -->|"readings"| G["🖥️ Ground station<br/>GPU server<br/>TabPFN predicts + alarm"]
    G -->|"radio: target point"| D["🚁 Defender X2<br/>flies to where the<br/>intruder <b>will be</b>"]
    G --> O["👤 Operator screen<br/>+ alarm"]
```

In our video the X2 defender only hovers. Flying it to where the intruder **will be** is what the predictions enable; building that is the next step.

---

## 11. Honest limits

| Limit | What we say |
|---|---|
| Simulation only | real drone models and physics, realistic radar; real radar data is next |
| Close range (60 m) | close-in defence: the last 20 to 30 seconds |
| Scripted pilots | two styles tested (TabPFN wins on both); human pilots are next |
| Example rows are scattered | "25 s of radar" means rows from 50 flights, not one continuous track |
| Perfect example answers | in real use "where it really went" comes from smoothed radar tracks of past flights, slightly noisier; the live drone's future is never shown |
| Speed | measured: 5.2 s → **252 ms per reading** with TabPFN's example cache and 1 pass, no accuracy lost; not yet under the radar's 100 ms, so live the IMM tracks and TabPFN corrects every 0.3 s: 4% better than the IMM instead of 13% |

---

**Run it:** `drone/README.md` · **all numbers:** `drone/results/README.md` · **every experiment:** `drone/EXPERIMENTS.md`
