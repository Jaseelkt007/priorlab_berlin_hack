# Real crane datasets: search results (2026-11-21)

I searched for public overhead, gantry, tower and container crane datasets that record sway. **Only one fits our use case**: a lab tower-crane dataset from Nazarbayev University. It is downloaded into `data/tower_crane_mendeley_v2/`. I found no public overhead-crane (bridge crane) sway dataset.

## Candidates, checked by opening each page

| Dataset | What it records | Size | Licence / access | Fit |
|---|---|---|---|---|
| **[Tower crane trajectories and payload oscillations](https://data.mendeley.com/datasets/x69hyyv3zw/2)** (Alhassan, Arif, Talapiden, Shehu, Haruna, Do; Nazarbayev Univ.; v2, 16 May 2025) | cable length, reference + actual trolley displacement, payload swing angle θ, at 100 Hz for 99 s | 33 runs + time file + ReadMe, 11 MB, `.xlsx` | **CC BY 4.0**, free download (no account; we downloaded it via Mendeley's public API) | **Good**: trolley moves at 8 cable lengths (0.1–0.8 m), 3 payload shapes (ball, cylinder, beam), and hoisting |
| [Quay crane spreader sway speed X-axis](https://ieee-dataport.org/node/2439) (S. Jakovlev, 2020, Klaipėda container terminal) | sway **speed** only, 10 use cases, plus timestamps; no trolley position, rope length or angle | one `.mat`, 285 KB | CC BY; IEEE login needed (the page is contradictory about whether a subscription is needed) | Weak: real port data, but too few signals |
| [UCI Container Crane Controller](https://archive.ics.uci.edu/dataset/436/container+crane+controller+data+set) (Ferreira et al., 2016) | 15 rows of speed / angle / power levels | 161 bytes | CC BY 4.0, free | No: a toy fuzzy-control table, not a time series |
| [MuJoCo MPC for double-pendulum sway on a shipboard crane](https://arxiv.org/abs/2603.16407) (Pang et al., 2026) | a paper, not a dataset; MuJoCo + real hardware | – | the abstract page doesn't state a code or data release | Useful reference for our MuJoCo plan |

Not usable: the gantry-crane and overhead-crane papers I found show sway only as plots, with no data files.

## The dataset we downloaded

**Files:** `data/tower_crane_mendeley_v2/`
- `(1)`–`(8)`: trolley motion at fixed L = 0.1 … 0.8 m
- `(9)`: trolley motion with hoisting from 0.1 to 0.5 m
- `(10)`–`(12)`: trolley motion at L = 0.5 m with a 75 g ball, a 170 g cylinder, a 500 g beam
- `(13)`: beam payload with hoisting
- `(14)`–`(23)`: jib rotation
- `(24)`–`(33)`: simultaneous trolley motion and jib rotation
- `34)`: the time column, 0–99 s
- `ReadMe.pdf`: the authors' description

**Columns** (one sheet per file, 9,901 rows = 99 s at 0.01 s):

| Column (as in the file) | Unit |
|---|---|
| Cable length, L | m |
| Reference trolley displacement, x_r | m |
| Actual trolley displacement, x | m |
| Payload oscillation in x-axis, θ | rad |

**How to cite it.** Dataset: Alhassan, A., Arif, S. M. U., Talapiden, K., Shehu, M. A., Haruna, A., Do, T. (2025). *Trajectories and Payload Oscillations Data of Tower Crane Systems with varied Payload Masses, Shapes, and Cable Lengths for Machine Learning-Based Control, System Identification, and Industrial Automation*. Mendeley Data, V2. https://doi.org/10.17632/x69hyyv3zw.2. CC BY 4.0.

**The paper that describes it:** "Comprehensive tower crane dynamics: An experimental dataset of trajectories and payload oscillations for system identification and machine learning-based control", *Data in Brief*, 2025 ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925005499)). I couldn't open the full text because the PMC page served a captcha, so check the author list on the article page before citing it.

**Licence duty (CC BY 4.0).** Credit the authors and link the licence. The repo goes public at submission, and sharing is allowed with credit, but committing 11 MB of xlsx is heavy. **Better: commit a download script, not the files.**

## First finding from the real data

Script: `python3 -I crane/data_tools/period_check.py crane/data/tower_crane_mendeley_v2`

| Run | Cable L | Measured period | Textbook 2π√(L/g) | Length that explains it |
|---|---|---|---|---|
| fixed-L runs (1)–(8), default payload, L = 0.1 … 0.8 m | 0.10–0.81 m | 1.03–1.93 s | 0.63–1.80 s | **L + 0.12 … 0.17 m** |
| 75 g ball, L = 0.5 | 0.50 m | 1.60 s | 1.41 s | L + 0.14 m |
| 170 g cylinder, L = 0.5 | 0.50 m | 1.71 s | 1.41 s | L + 0.23 m |
| 500 g beam, L = 0.5 | 0.50 m | 1.77 s | 1.41 s | L + 0.28 m |

**What it means.** On real hardware, the textbook model using the logged cable length is wrong in every run. The swing is always slower, as if the rope were 12–17 cm longer (the hook and payload hang below the rope end), and up to 28 cm longer for the beam. The payload's shape changes the period, and the cable-length column can't see that. This is the gap we claim TabPFN closes by learning from the crane's own data. It's real evidence for the pitch, not just simulated.

**Caveat.** This is a quick check: upward zero crossings over the last 40 s of each run, one run per condition. Report it as a preliminary observation.

## How to use it (next steps)

1. **Loader** (`crane/real_data.py`): read the xlsx files (std-lib zip parsing, as in `data_tools/`, or `openpyxl`). Join the time file and label each run with its L, payload and motion type.
2. **Features:** downsample to 10 Hz (lesson 3, summaries). Then compute θ, θ̇, θ 0.2 s and 0.5 s ago, trolley x, ẋ, ẍ (from x), reference x_r − x, and L.
3. **Targets:** θ at +0.5 s and +1.0 s (experiment A, forward), and the residual swing amplitude after each trolley stop.
4. **Held-out splits** (lesson 4):
   - context L ∈ {0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8}, test L = 0.5;
   - context ball and plain-hook runs, test **beam** (the shape that breaks the textbook period);
   - context fixed-L runs, test **hoisting** runs.
5. **Baselines on the same rows:**
   - linear pendulum with the logged L;
   - pendulum with L_eff fitted on the context runs (the strong physics baseline);
   - gradient-boosted trees.
6. **TabPFN:** `TabPFNRegressor` with quantiles; report RMSE (rad and cm at the payload), coverage₉₀, and ms per query. Also run experiment B: add 5, 20, 50 rows of the beam run and plot error vs. N.
7. **Pitch line:** "Validated on a real lab crane (Nazarbayev University dataset, CC BY 4.0)", then scale the story up with our MuJoCo overhead-crane sim.

Limit: it's a small lab **tower** crane (cables under 1 m, payloads under 1 kg), not an industrial overhead crane. Say so. The physics of the trolley axis is the same pendulum, but the sizes differ by about 10×.
