# Results

One folder per experiment. **Data version matters:** the first data collection stored wrong wing labels (flybody adds the wingbeat pattern to the wing actions in place inside `env.step()`). It was fixed and all six passes were re-collected on 21 Nov, ~22:00. Runs of the trained **policy** are unaffected (they never read the stored labels). TabPFN runs before the fix are kept as history only.

| Folder | What | Data | Produced by |
|---|---|---|---|
| `exp_a/` | Future position in 1/3/10 wingbeats: TabPFN vs trees vs constant velocity (+ API cache) | corrected | `experiments/exp_a_forward.py` |
| `exp_b/` | Adapting to a weak wing / heavier body by adding rows (+ API cache) | corrected | `experiments/exp_b_adapt.py` |
| `exp_c/` | Copying the policy's actions offline, R² by context size (+ API cache) | corrected | `experiments/exp_c_clone.py` |
| `exp_d/` | TabPFN flies 3 held-out recorded flights, 256 and 512 context steps | **first data (wrong labels)** | `collect/closed_loop.py` |
| `exp_e/` | Synthetic paths, diagnostics and split tests, 256–512 context steps | mixed, see below | `collect/synthetic_paths.py` |
| `exp_f/` | DAgger rounds on the smooth path, 512 context steps | **first data (wrong labels)** | `collect/dagger.py` |
| `showreel/` | 2-minute labelled showreels: policy and TabPFN at 16,384 steps (`.mp4` local only) | corrected | `collect/showreel.py` |
| `sweep/` | Context-size sweep, 4k to 83k steps (49k to 1M rows), five paths each, per-step drift (`.mp4` local only) | corrected | `collect/showreel.py --once`, `experiments/sweep_summary.py` |

**`exp_e/` by file:**

| Files | Data |
|---|---|
| `*_policy.*`, `summary_policy.json` | policy, unaffected |
| `*_tabpfn_ctx512.*`, `drift_vs_time.png`, `curve_tabpfn_ctx256.*`, `curve_tabpfn_ctx256_delta.*` | first data (wrong labels) |
| `curve_tabpfn_ctx256_pass1*` (incl. the `_policy-*` split tests) | corrected (clean pass only) |

API caches (`exp_a`–`exp_c/cache/`) are keyed by name, not by data: delete them before rerunning on new data, or the old predictions are reused. Videos (`*.mp4`) are git-ignored and live only on the machine that made them.
