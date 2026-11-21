# TabPFN latency on our hardware (measured 2026-11-21)

Script: `scripts/bench_tabpfn.py --quick` · GPU: RTX 4050 Laptop 6 GB · torch 2.15 + CUDA 13.2 · tabpfn 10.2.0 · local weights · `fit_mode="fit_with_cache"`.
Data: toy delta pick log (`sim/delta_toy.py`), 11 features, coffee-pouch product, 9.6% drops. Times are medians over repeated calls (p95 in brackets).

| Model | Context | Estimators | Fit + cache | 1 row | 15 rows (one pick, all speeds) | 3,000-row speed map | AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| 3.5-Fast | 300 | 1 | 0.3 s* | 36 ms | **26 ms** (43) | **92 ms** | 0.986 | 0.029 |
| 3.5-Fast | 300 | 4 | 0.28 s | 41 ms | 41 ms (48) | 357 ms | 0.986 | 0.029 |
| 3.5-Fast | 1000 | 1 | 0.32 s | 35 ms | **22 ms** (23) | **84 ms** | 0.986 | 0.029 |
| 3.5-Fast | 1000 | 4 | 0.36 s | 36 ms | 38 ms (47) | 354 ms | 0.986 | 0.028 |
| 3.5 | 300 | 1 | 0.4 s* | 48 ms | 46 ms (56) | 135 ms | 0.985 | 0.030 |
| 3.5 | 300 | 4 | 0.39 s | 48 ms | 55 ms (94) | 574 ms | 0.985 | 0.030 |
| 3.5 | 1000 | 1 | 0.44 s | 43 ms | 64 ms (78) | 139 ms | 0.986 | 0.028 |
| 3.5 | 1000 | 4 | 0.56 s | 57 ms | 52 ms (65) | 602 ms | 0.987 | 0.028 |

\* The first fit of each model also downloaded the weights (20 s and 56 s). Those runs are excluded from the fit time shown.

Peak GPU memory: 1.76 GB.

## What it means for the line

- **A per-pick query is about 22–45 ms, and it is mostly fixed overhead** (preprocessing): 1 row costs as much as 15. That is too slow to sit between the seal reading and the start of acceleration (10–30 ms) with any safety margin.
- **The speed map is cheap.** Refitting on new rows (about 0.3 s) plus predicting a 3,000-row map (about 0.1 s with Fast and 1 estimator) takes about 0.4 s per product. It can refresh about every second in the background, which is fast enough to track humidity, wear and changeovers.
- **Design:** TabPFN-3.5-Fast with 1 estimator rewrites a per-product speed map about once a second. The robot or PLC looks up the speed from the seal reading in microseconds.
- **Quality on toy data:** AUC 0.986 and Brier 0.029 already with 300 context rows. The toy physics is easier than real data, so the MuJoCo cell will tell us more.
- **Not tried yet:** `enable_torch_compile`, int8/fp8 KV cache, and avoiding per-call preprocessing. Any of these could cut the fixed overhead.
