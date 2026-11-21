# Crane Sway: pitch and video script

Every number below is from our own result files, or from a source linked on the page:
- page: https://claude.ai/artifact/7rjNJLFjPqkE2g74ekxL31 (private until shared)
- `results/exp1_real.json`, `exp2_real.json`, `mujoco_demo.json`

Say **"in simulation"** for MuJoCo numbers, and **"on a lab crane"** for the real-data numbers.

## One-line pitch
Anti-sway control breaks when the load is new. TabPFN reads the crane's own recent moves and predicts the swing 1 s ahead with a calibrated range. It adapts within seconds by adding rows, with no retraining.

---

## Slides (5-minute live pitch, 7 slides, about 40 s each)

**1 · The problem**
- Title: *"When a crane stops, the load keeps going."*
- Visual: MuJoCo still (`results/still_a.png`).
- Text:
  - Operators wait for the swing to settle before every set-down.
  - US: 297 crane-related work deaths in 2011–17; 154 of them were workers struck by an object or equipment (BLS).
- Say: "Every move ends with waiting, and sometimes with something being hit."

**2 · Why today's fix breaks**
- Title: *"Anti-sway needs the right pendulum."*
- Visual: the page's section 2 chart (residual swing vs. push duration).
- Text:
  - Input shaping cancels swing, but only for the pendulum it was tuned for.
  - A wrong sling entry, or a long-sling double pendulum, brings the swing back.
- Say: "Vendors claim 10–40% faster cycles, when the model is right."

**3 · Our idea**
- Title: *"Physics plus a table that learns in context."*
- Visual: the four-step diagram from page section 4 (sense → ask → choose → log).
- Text:
  - One row per 0.1 s: swing now and recent, trolley state, rope length, payload, planned move.
  - TabPFN predicts the swing 1 s ahead with a 90% range.
  - New load → add its rows. No training loop.

**4 · Real data first**
- Title: *"On a real crane, the textbook pendulum is wrong in every run."*
- Visual: the page's section 6 period chart.
- Text:
  - Nazarbayev University lab tower crane, 33 runs, CC BY 4.0.
  - Measured swing is always slower than 2π√(L/g), as if the rope were 12–17 cm longer (beam: +28 cm).

**5 · Result: it adapts in seconds**
- Title: *"5 seconds of a new load → 3× lower error."*
- Visual: the adaptation chart (page section 6).
- Text, on an unseen beam payload:
  - TabPFN's 1-s error goes from 0.30 to 0.10 cm after 5 s of the beam's rows.
  - Its 90% range holds 32% of outcomes at first, then 91%.
  - Trees retrained on the same rows reach 0.37 cm; the refitted pendulum 0.50 cm.
- Footnote: on a new cable length alone, physics wins. Physics + TabPFN is best or tied almost everywhere.

**6 · Demo**
- Title: *"Full-size crane, new load, live predictions."*
- Visual: 20 s of `results/mujoco_demo.mp4`, from about t = 50 s.
- Text, in simulation, 400 kg beam on 2.6 m slings, anti-sway set for 0.5 m:
  - First move: TabPFN 3.5 cm vs. physics 7.0 cm.
  - Later moves: 2.6 cm vs. 10.9 cm, and the 90% range holds 90%.

**7 · Honest limits, next steps**
- Limits:
  - Lab crane is about 10× smaller than an industrial one; one run per condition.
  - The MuJoCo demo was rendered offline (7–8 s fits on our GTX 1650).
- Next:
  - Closed loop: choose speed and shaper tuning per move from the predicted risk.
  - Then an industrial crane's logs, in shadow mode first.
- Close: "Physics where it's known. TabPFN where the load surprises you."

---

## Video (3 minutes, a person on camera)

| Time | Picture | Words (about 140 words/min) |
|---|---|---|
| 0:00–0:20 | **You on camera** | "When an overhead crane stops, the load keeps swinging. Operators wait for it to settle before every set-down. And when it doesn't settle, things and people get hit: in the US, more than half of crane deaths are workers struck by an object or equipment." |
| 0:20–0:45 | Page section 2, drag the sling slider | "Anti-sway systems cancel the swing, but only if they know the pendulum. Put a light beam on long slings and the model is wrong. The swing comes back." |
| 0:45–1:10 | Page section 3 table + section 4 steps | "We turn the crane's sensors into a table: swing now and a moment ago, trolley motion, rope length, payload. TabPFN reads that table and predicts where the load will be one second ahead, with a range. When a new load arrives, we add its rows. Nothing is retrained." |
| 1:10–1:50 | **MuJoCo video**, t = 0–15 s then t = 50–65 s (speed up the middle) | "Here, in MuJoCo, is a full-size crane with a load it has never seen. Teal is TabPFN's prediction, orange is the physics model set for the wrong sling. On the first move, TabPFN is already twice as accurate. By the fourth move, it is within about 2.6 cm on average, four times better, and its 90% range really holds 90% of the time." |
| 1:50–2:30 | Page section 6: period chart, then adaptation chart | "Then we tested it on real data, a lab crane from Nazarbayev University. The textbook pendulum is wrong in every run. On a payload TabPFN had never seen, five seconds of watching cut its error three times, with no retraining. Physics still wins when only the rope length changes, so we combine the two." |
| 2:30–3:00 | **You on camera** | "It's a lab crane and a simulation, not a factory yet. Next, we let the crane choose its speed from the predicted risk, and test on real industrial logs in shadow mode. Physics where it's known, TabPFN where the load surprises you. Thanks." |

**Recording tips:**
- Screen-record the page in the browser, and play the MP4 full screen for the demo part.
- Do the camera parts last, in one take each.
- Keep the total under 3:00; the organisers require a person on camera.
