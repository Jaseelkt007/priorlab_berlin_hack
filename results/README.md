# Result data (every number in README.md and docs/research/evaluation.md comes from these files)

| Path | What | Made by |
|---|---|---|
| `logs/commission_lap.csv`, `logs/commission_srv.csv` | commissioning log, 2,700 picks: 6 products × 9 speed settings × 50 random line states (seed s 0–24 laptop, 25–49 server) | `scripts/sweep_picks.py` |
| `logs/commission_small.csv` | the 540-pick small log (s 0–9 of the laptop half) | subset of the above |
| `logs/bank_lap.csv`, `logs/bank_srv.csv` | evaluation bank: 940 pick instances × all 9 settings (exact counterfactuals) | `scripts/make_bank.py` |
| `logs/sweep_v2.csv` | gate-5 validation sweep, 1,080 picks | `scripts/sweep_picks.py` |
| `eval/learning_curve.csv`, `eval/placement.csv` | prediction and decision quality vs context size; placement prediction | `scripts/learning_curve.py`, `scripts/placement.py` |
| `eval/runs_*.csv.gz`, `eval/summary_*.csv` | closed-loop replays (large log): cost_safe (headline), cost, main (risk targets), frozen, explore, tabpfn_3.5-fast | `scripts/run_eval.py` |
| `eval_small/runs_cost_safe.csv.gz`, `eval_small/runs_margin.csv.gz` | closed-loop replays with the small log; recipes with 1–2 step margins | `scripts/run_eval.py --suite cost --safe-step`, `--suite margin` |

Tables: `.venv/bin/python scripts/report.py results/eval` (and `results/eval_small`); it reads the gzipped runs directly.
