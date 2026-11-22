# The four lessons: what they are and where to read about them

These come from the **Prior Labs slide deck** given to hackathon teams (Diana Kriuchkova, Product @ Prior Labs).
- Local copy: [`docs/HackLab_Berlin_x_TabPFN_Prior_Labs.pdf`](../docs/HackLab_Berlin_x_TabPFN_Prior_Labs.pdf)
- Our team's summary: [`docs/HACKATHON.md`](../docs/HACKATHON.md), section "What Prior Labs' deck says"
- The judging criteria aren't published, so treat these four lessons as the rubric.

## Start here if TabPFN is new to you
| What | Link | Why read it |
|---|---|---|
| Quickstart | https://docs.priorlabs.ai/quickstart | install it, get a key, make your first prediction (15 min) |
| TabPFN code | https://github.com/PriorLabs/TabPFN | run the model locally on your GPU (no API usage) |
| API client | https://github.com/PriorLabs/tabpfn-client | cloud version that uses `TABPFN_TOKEN` |
| Extensions | https://github.com/PriorLabs/tabpfn-extensions | extra tools (feature importance, etc.) |
| Nature paper (2025) | https://www.nature.com/articles/s41586-024-08328-6 | how it works: a model pretrained on millions of synthetic tables, which then predicts by "reading" your table |

**The core idea in one line:** `fit(X, y)` doesn't train anything. It stores your example rows (the "context"). `predict` reads that context and answers immediately. Changing what it knows means changing the rows.

## The lessons, with examples you can read

### 1. Mind the loop rate
A robot controller runs at a fixed speed (e.g. 10–50 times a second), so every prediction has to finish in time. Keep the number of rows and columns small, run the model locally, and cache the context so it isn't reprocessed on every call.
- Example: **duckpfn** runs at 50 Hz with a frozen, cached context (`fit_with_cache`): https://github.com/nikikilbertus/duckpfn
- Our own speed numbers: commit `caae036` on `main` (TabPFN-3.5 benchmark on an RTX 4050).

### 2. Choose the rows you show
Don't send the whole dataset. For each query, send only the most relevant rows, such as the 256 past situations most similar to the current one. You get better predictions, faster, with less usage.
- Example from the deck: the Breakout project used the 256 most similar "teacher" game states for each frame.

### 3. Features before raw data
Summarize signals into a few meaningful numbers (speed, angle, averages) instead of feeding in raw samples or pixels. From the deck: PCA features beat raw pixels, 0.87 m vs 0.28 m.
- Example: the **sEMG winner** turns each 0.5 s of EMG signal into 144 standard features: https://github.com/simon-bachhuber/tabpfn-sEMG-human-gesture-detection

### 4. Test on what you haven't seen
Prove it generalizes by holding out whole people, robots, starting states, or (for us) **cable lengths and loads**, not just random rows.
- Example: the sEMG winner tested on 4 people the model had never seen and beat the dataset authors' own published baselines. Credibility won.

## What judges rewarded last time
Read [`docs/research/prior-winners.md`](../docs/research/prior-winners.md) (about 5 minutes):
- **1st, sEMG:** real problem, unseen users, fair baselines, reproducible.
- **2nd, duckpfn:** TabPFN *imitates* a pretrained policy in MuJoCo. It had great presentation (GIF, video, notebook) but **no baselines** and a frozen context. Our crane plan fixes both: it compares against baselines and adapts by adding rows.
