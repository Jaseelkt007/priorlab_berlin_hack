# Drone defence with TabPFN, explained in pictures

> **One line:** a radar sees an intruder drone. TabPFN predicts **where it will be in 1 second** and **whether it
> breaks into the protected zone within 2 seconds**. No training, only examples, and it beats the radar industry's
> standard tracker (IMM) by **11%**.

---

## 1. The problem

![the scene](../results/charts/scene.jpg)

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

| Model | Error 1 s ahead | Breaches caught | Warning before breach |
|---|---|---|---|
| IMM (radar standard) | 1.54 m | 99% | 3.9 s |
| TabPFN A (big drone examples only) | 1.70 m | 95% | 4.5 s |
| **TabPFN B (+ 1,000 small drone examples)** | **1.37 m** ✅ | 98% | 4.4 s |

![error by model](../results/charts/error_by_model.png)

```mermaid
flowchart LR
    A["TabPFN A<br/>never saw the small drone<br/><b>10% worse</b> than IMM"] -->|"+ 1,000 example rows<br/>no training"| B["TabPFN B<br/><b>11% better</b> than IMM<br/>closer at 62% of moments"]
```

- **Better on:** straight, zigzag, orbit, hover, dash. **Slightly worse on:** dive, climb.
- **Alarm:** all models are near perfect at 2 s (AUC 0.99).
- **In the 250 test flights 108 drones breached**; TabPFN B warned for 98% of them, about 4 s before.

![error by manoeuvre](../results/charts/error_by_manoeuvre.png)

---

## 9. See it

![IMM vs TabPFN](../results/charts/imm_vs_tabpfn.jpg)

| Colour | Meaning |
|---|---|
| ⚪ white dots | radar readings |
| 🟢 green dots | the real path |
| 🔵 blue ball / 🟡 yellow ball | IMM: in 1 s / its guess for *now*, made 1 s ago |
| 🩷 pink ball / pink cube | TabPFN: in 1 s / its guess for *now*, made 1 s ago |

Videos (shared separately, not in git): `scene_54.mp4`, `imm_vs_tabpfn_chase_54.mp4`, `imm_vs_tabpfn_54.mp4`.

---

## 10. In a real system

```mermaid
flowchart LR
    R["📡 Radar<br/>on the building<br/>measures"] -->|"readings"| G["🖥️ Ground station<br/>GPU server<br/>TabPFN predicts + alarm"]
    G -->|"radio: target point"| D["🚁 Defender X2<br/>flies to where the<br/>intruder <b>will be</b>"]
    G --> O["👤 Operator screen<br/>+ alarm"]
```

In our video the X2 defender only hovers; using the predictions to intercept is the next step.

---

## 11. Honest limits

| Limit | What we say |
|---|---|
| Simulation only | real drone models and physics, realistic radar; real radar data is next |
| Close range (60 m) | close-in defence: the last 20 to 30 seconds |
| One scripted pilot | other pilot styles and humans are next |
| IMM tuned on the big drone only | fairness check: re-tune it on the same 50 small-drone flights |
| Speed measured in batch | real-time delay per reading is not measured yet |

---

**Run it:** `drone/README.md` · **all numbers:** `results/README.md` · **every experiment:** `drone/EXPERIMENTS.md`
