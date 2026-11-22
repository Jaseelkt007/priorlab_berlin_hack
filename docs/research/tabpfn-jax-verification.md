# Independent verification of the JAX port (measured 2026-11-21)

Script: `scripts/verify_jax_port.py` (branch `verify-jax`). It uses only the port's public API (`export.bundle_from_classifier`, `JaxTabPFN`, `update_context`) and none of `jax_tabpfn/parity.py` or `bench.py`. It uses new data (seeds 12345, 777, 2024; the port was built on seed 0).

**Setup:** RTX 4050 Laptop 6 GB (WSL2). No other GPU compute process ran during the latency section. TabPFN-3.5-Fast, 1 estimator. JAX 0.12.1, torch 2.15, tabpfn 10.2.0, in one process.

## Verdict

| Claim from `tabpfn-jax-port.md` | Verified? | Evidence |
|---|---|---|
| JAX fp32 matches torch fp32 within 1e-4 | **Yes.** Worst 1.8e-5, mean ≤ 1.2e-6 | 9 settings × (20 picks + 5 edge picks + 2 maps); 11 refits |
| Same decisions as torch fp32 (fp32 mode) | **Yes.** 0 of 3,825 chosen speeds differ | Same queries; 0 of 55 in the refit run |
| Fast mode drifts no more than TabPFN's own fp16 default | **Yes.** vs torch fp32: max 1.1e-2, 17 of 3,825 decisions differ. Torch default vs torch fp32: max 1.3e-1, 28 of 3,825 | Same queries |
| 15-row pick 3.7 ms, beats CUDA graph | **Yes.** 3.68 ms (p95 4.88, p99 5.38) vs 6.81 ms (p95 9.01, p99 9.73) | 200 picks, same process |
| Map 52–57 ms vs ~82 ms | **Yes.** 52.6 ms (p99 55.7) vs CUDA graph 80.8 ms, torch normal 90.2 ms | 20 maps, same process |
| Refit: ~20 ms context swap, no recompile | **Yes.** 16–31 ms; first query after a swap 4–9 ms | 10 refits of a rolling window |
| Warm-cache first call ~0.5 s | **Yes.** 0.5–1.1 s | Fresh process |

**What fails, or needs saying plainly:**

1. **A cold XLA cache is slow.** In a fresh process with an empty compile cache:
   - fast mode: 12 s for the first pick, 108 s for the first map;
   - fp32 mode: 86 s for the first pick, 145 s for the first map.

   A new demo machine, a cleared `~/.cache/jax_tabpfn`, or a new context size pays this once. Warm the cache on the demo machine before going on stage.
2. **Fast mode does not always make the same decision as torch.** 17 of 3,825 chosen speeds (0.44%) differ from torch fp32, and 31 (0.81%) from torch default. In every case a P(drop) lies within 0.0001 (fp32) or 0.001 (default) of the 0.5% budget, so these are ties at the threshold, not errors. Torch's own fp16 default flips 28 of the same decisions against torch fp32. **If the pitch says "identical decisions to PyTorch", that holds for the fp32 mode (4.9 ms per pick), not the fast mode.**
3. **TabPFN's own default (fp16 autocast + int8 KV) drifts up to 0.13** from fp32 on one biscuit map (n=1000, mean 9.7e-3). The JAX fast mode stays within 1.1e-2 of fp32 there. This is torch behaviour, not the port.
4. **A bug in this harness, found and fixed.** The first refit run reported 1 fp32 "flip". Both sides actually said "no safe speed" (NaN), and the refit section compared with `!=`, where NaN ≠ NaN. Fixed with `same_choice`; the table below is from the rerun. The parity section was not affected.

Two observations about the model, not the port, that matter for the decision rule:
- **P(drop) is not monotone at the slowest speed.** At 0.30 it is often higher than at 0.35–0.45: for example 0.0138 at 0.30 against 0.0005 at 0.35 on one cheese map item. "The fastest speed under budget" is still well defined, but a rule like "slow down until safe" would misbehave.
- **Columns that are constant in the context are invisible to the model.** Here that is `humidity_pct` and `cup_d_mm`. TabPFN drops them, and the port reproduces that, so changing humidity in a query has no effect unless humidity varied in the context rows.

## 1. Parity on unseen data

- **Settings:**
  - products biscuit (0.3% drops), pouch (8–10%) and cheese (34–37%);
  - context sizes 300, 1000 and 2000;
  - one `JaxTabPFN` per mode, moved between settings with `update_context`, so product and size changes under one instance are also tested.
- **Queries per setting:**
  - 20 fresh 15-row picks;
  - 5 edge picks: seal 3 kPa, seal 65 kPa, every feature far above the context (×3 to ×5, seal 150 kPa, cup cycles 1e6), far below it (negative offset and cycles, seal 0.5 kPa), and seal = NaN;
  - 2 speed maps (200 items × 15 speeds).

  Every pick includes speeds 0.30 and 1.00.
- **Torch references:** torch fp32 (unquantised cache) and torch default (fp16 autocast, int8 KV).
- **JAX runs:**
  - fp32 and fast use the fp32-cache bundle;
  - `fast[int8 bundle]` runs on the bundle exported from the default classifier, which tests the int8 dequantisation path.
- **Chosen speed:** the fastest of the 15 speeds with P(drop) ≤ 0.5%, or "none". It is compared per item: 1 per pick, 200 per map.

### Summary over all 9 settings

| Comparison | Queries | max \|Δp\| | mean \|Δp\| | Chosen-speed differences |
|---|---|---|---|---|
| JAX fp32 vs torch fp32 | picks | 1.1e-05 | 1.9e-07 | 0 / 180 |
| JAX fp32 vs torch fp32 | edge | 9.7e-06 | 3.7e-07 | 0 / 45 |
| JAX fp32 vs torch fp32 | maps | 1.8e-05 | 5.3e-07 | 0 / 3600 |
| JAX fp32 vs torch default | picks | 9.1e-03 | 1.5e-04 | 3 / 180 |
| JAX fp32 vs torch default | edge | 3.2e-02 | 1.3e-03 | 0 / 45 |
| JAX fp32 vs torch default | maps | 1.3e-01 | 1.7e-03 | 25 / 3600 |
| JAX fast vs torch fp32 | picks | 5.0e-03 | 1.5e-04 | 2 / 180 |
| JAX fast vs torch fp32 | edge | 7.4e-03 | 3.2e-04 | 0 / 45 |
| JAX fast vs torch fp32 | maps | 1.1e-02 | 4.7e-04 | 15 / 3600 |
| JAX fast vs torch default | picks | 1.1e-02 | 2.3e-04 | 3 / 180 |
| JAX fast vs torch default | edge | 3.2e-02 | 1.5e-03 | 0 / 45 |
| JAX fast vs torch default | maps | 1.3e-01 | 1.8e-03 | 28 / 3600 |
| JAX fast[int8 bundle] vs torch default | picks | 4.9e-03 | 1.2e-04 | 1 / 180 |
| JAX fast[int8 bundle] vs torch default | edge | 3.9e-03 | 2.7e-04 | 0 / 45 |
| JAX fast[int8 bundle] vs torch default | maps | 6.8e-03 | 3.2e-04 | 10 / 3600 |
| *reference:* torch default vs torch fp32 | picks | 9.1e-03 | 1.5e-04 | 3 / 180 |
| *reference:* torch default vs torch fp32 | edge | 3.2e-02 | 1.3e-03 | 0 / 45 |
| *reference:* torch default vs torch fp32 | maps | 1.3e-01 | 1.8e-03 | 25 / 3600 |

How close the disagreeing decisions are to the threshold. "Decisive P(drop)" is a value at a speed where only one side is within budget:

| Comparison | Differences | Decisive P(drop), max distance from 0.005 | Median distance | Tied at 4 decimals |
|---|---|---|---|---|
| JAX fast vs torch fp32 | 17 | 0.0001 | 0.0000 | 10 |
| JAX fast[int8 bundle] vs torch default | 11 | 0.0001 | 0.0000 | 8 |
| JAX fast vs torch default | 31 | 0.0010 | 0.0001 | 10 |
| JAX fp32 vs torch default | 28 | 0.0010 | 0.0001 | 13 |

### Edge queries

- **fp32 mode:** |Δp| ≤ 9.7e-6 on every edge pick in every setting, including seal = NaN and values far outside the context.
- **Fast mode against torch default:** ≤ 3.2e-2. That figure is torch default's own drift; JAX fast against torch fp32 is ≤ 7.4e-3.
- **Torch** accepted every edge query, so none had to be skipped.

The per-query list is in the appendix.

## 2. Latency in fresh processes

Context: 1,000 rows of pouch. Queries: 200 distinct 15-row picks and 20 distinct 3,000-row maps, the same for every path, from raw rows to probabilities (`predict_proba`). Each path ran in one block after warm-up, all in the same process.

### First call in a fresh process

"Cold" means an empty XLA compile cache, pointed at a temp dir. "Warm" means the persistent cache in `~/.cache/jax_tabpfn/xla`.

| XLA cache | fast: load | fast: 1st pick | fast: 1st map | fp32: load | fp32: 1st pick | fp32: 1st map |
|---|---|---|---|---|---|---|
| cold | 3.3 s | 12.4 s | 107.7 s | 5.6 s | 85.8 s | 144.7 s |
| warm | 1.1 s | 0.5 s | 0.7 s | 0.6 s | 0.5 s | 0.6 s |

### Steady state, same process, same queries

| Path | Pick median | Pick p95 | Pick p99 | Map median | Map p95 | Map p99 |
|---|---|---|---|---|---|---|
| JAX fast | 3.68 ms | 4.88 ms | 5.38 ms | 52.6 ms | 54.2 ms | 55.7 ms |
| JAX fp32 | 4.92 ms | 6.34 ms | 6.78 ms | 153.0 ms | 154.6 ms | 155.6 ms |
| torch normal (default) | 34.10 ms | 48.92 ms | 57.54 ms | 90.2 ms | 106.3 ms | 206.4 ms |
| torch CUDA graph (default) | 6.81 ms | 9.01 ms | 9.73 ms | 80.8 ms | 82.1 ms | 82.7 ms |

The CUDA-graph path is a copy of the wrapper in `scripts/spike_cudagraph.py`, which runs its experiment at import and so can't be imported. One change: it takes the classifier's own model object instead of searching `gc`, because two classifiers live in this process.

## 3. Refit at a fixed 1,000-row window

Pouch stream. Each refit adds 50 new rows and drops the 50 oldest. After each refit, 5 fresh picks are drawn from rows never in any window. "fp32 max |Δp|" is JAX fp32 against torch fp32; the fast column shows JAX fast against torch fp32, then against torch default.

| Refit | Window rows | Drops | torch fp32 fit | Export | update_context fp32 / fast | 1st query fp32 / fast | fp32 max \|Δp\| | fast max \|Δp\| | Chosen-speed differences fp32 / fast |
|---|---|---|---|---|---|---|---|---|---|
| 0 (initial) | 0-1000 | 10.7% | 3421 ms | 3 ms | 1201 / 407 ms | 472.4 / 770.2 ms | 2.6e-06 | 3.3e-03 / 1.3e-02 | 0 / 0 |
| 1 | 50-1050 | 10.8% | 242 ms | 3 ms | 20 / 20 ms | 6.2 / 5.3 ms | 2.7e-06 | 1.1e-03 / 1.6e-03 | 0 / 0 |
| 2 | 100-1100 | 11.1% | 218 ms | 3 ms | 21 / 31 ms | 7.3 / 6.5 ms | 4.1e-06 | 8.9e-03 / 1.0e-02 | 0 / 0 |
| 3 | 150-1150 | 10.5% | 280 ms | 3 ms | 23 / 24 ms | 7.0 / 4.7 ms | 2.6e-06 | 2.7e-03 / 2.0e-03 | 0 / 0 |
| 4 | 200-1200 | 10.9% | 204 ms | 3 ms | 20 / 18 ms | 5.7 / 4.6 ms | 2.3e-06 | 2.3e-03 / 1.9e-03 | 0 / 0 |
| 5 | 250-1250 | 11.3% | 219 ms | 2 ms | 18 / 18 ms | 6.8 / 4.7 ms | 6.5e-06 | 5.0e-03 / 2.3e-03 | 0 / 0 |
| 6 | 300-1300 | 11.2% | 239 ms | 2 ms | 18 / 18 ms | 5.6 / 4.3 ms | 2.8e-06 | 4.7e-03 / 5.2e-03 | 0 / 0 |
| 7 | 350-1350 | 10.9% | 222 ms | 3 ms | 20 / 20 ms | 6.8 / 4.4 ms | 3.1e-06 | 3.1e-03 / 2.8e-03 | 0 / 0 |
| 8 | 400-1400 | 10.6% | 373 ms | 7 ms | 29 / 29 ms | 9.2 / 8.6 ms | 1.2e-06 | 4.8e-03 / 5.8e-03 | 0 / 0 |
| 9 | 450-1450 | 10.4% | 239 ms | 2 ms | 16 / 17 ms | 5.4 / 4.1 ms | 1.1e-05 | 1.2e-02 / 9.2e-03 | 0 / 0 |
| 10 | 500-1500 | 10.1% | 261 ms | 4 ms | 19 / 17 ms | 6.2 / 5.4 ms | 9.6e-06 | 7.3e-03 / 7.1e-03 | 0 / 0 |

- **Refit 0** includes loading the weights and the (warm-cache) compile.
- **Refits 1–10** swap the context in 16–31 ms. The first query after a swap takes 4–9 ms against ~4–5 ms in steady state, so nothing recompiles.
- **The first-query times** in this table include torch's interleaved predicts on the same GPU.

## Reproduce

```bash
# from the repo root, torch + jax in one interpreter (DEBT.md)
export PYTHONPATH=.:../priorlab_berlin_hack/.venv/lib/python3.12/site-packages
.venv-jax/bin/python scripts/verify_jax_port.py parity    # ~15 min with new context sizes to compile
.venv-jax/bin/python scripts/verify_jax_port.py refit     # ~1 min
.venv-jax/bin/python scripts/verify_jax_port.py latency   # ~8 min, mostly the cold-cache compile
```

## Appendix: raw parity output

### Per setting

| setting | queries | comparison | max abs dp | mean abs dp | chosen-speed flips |
|---|---|---|---|---|---|
| biscuit n=300 (drops 0.3%) | pick | JAX fp32 vs torch fp32 | 2.6e-07 | 2.0e-08 | 0 / 20 |
| biscuit n=300 (drops 0.3%) | pick | JAX fp32 vs torch default | 5.1e-04 | 2.6e-05 | 2 / 20 |
| biscuit n=300 (drops 0.3%) | pick | JAX fast vs torch fp32 | 4.0e-04 | 1.9e-05 | 1 / 20 |
| biscuit n=300 (drops 0.3%) | pick | JAX fast vs torch default | 3.9e-04 | 2.7e-05 | 1 / 20 |
| biscuit n=300 (drops 0.3%) | pick | JAX fast[int8 bundle] vs torch default | 1.9e-04 | 1.0e-05 | 1 / 20 |
| biscuit n=300 (drops 0.3%) | pick | torch default vs torch fp32 | 5.1e-04 | 2.6e-05 | 2 / 20 |
| biscuit n=300 (drops 0.3%) | edge | JAX fp32 vs torch fp32 | 1.5e-06 | 5.5e-07 | 0 / 5 |
| biscuit n=300 (drops 0.3%) | edge | JAX fp32 vs torch default | 4.0e-03 | 8.8e-04 | 0 / 5 |
| biscuit n=300 (drops 0.3%) | edge | JAX fast vs torch fp32 | 1.2e-03 | 2.2e-04 | 0 / 5 |
| biscuit n=300 (drops 0.3%) | edge | JAX fast vs torch default | 2.9e-03 | 9.7e-04 | 0 / 5 |
| biscuit n=300 (drops 0.3%) | edge | JAX fast[int8 bundle] vs torch default | 1.4e-03 | 3.5e-04 | 0 / 5 |
| biscuit n=300 (drops 0.3%) | edge | torch default vs torch fp32 | 4.0e-03 | 8.8e-04 | 0 / 5 |
| biscuit n=300 (drops 0.3%) | map | JAX fp32 vs torch fp32 | 3.5e-06 | 4.1e-07 | 0 / 400 |
| biscuit n=300 (drops 0.3%) | map | JAX fp32 vs torch default | 3.2e-03 | 4.3e-04 | 1 / 400 |
| biscuit n=300 (drops 0.3%) | map | JAX fast vs torch fp32 | 2.3e-03 | 5.1e-04 | 3 / 400 |
| biscuit n=300 (drops 0.3%) | map | JAX fast vs torch default | 4.8e-03 | 5.4e-04 | 4 / 400 |
| biscuit n=300 (drops 0.3%) | map | JAX fast[int8 bundle] vs torch default | 2.6e-03 | 2.9e-04 | 1 / 400 |
| biscuit n=300 (drops 0.3%) | map | torch default vs torch fp32 | 3.2e-03 | 4.3e-04 | 1 / 400 |
| biscuit n=1000 (drops 0.3%) | pick | JAX fp32 vs torch fp32 | 4.0e-07 | 1.3e-08 | 0 / 20 |
| biscuit n=1000 (drops 0.3%) | pick | JAX fp32 vs torch default | 9.1e-03 | 1.5e-04 | 1 / 20 |
| biscuit n=1000 (drops 0.3%) | pick | JAX fast vs torch fp32 | 3.6e-04 | 1.5e-05 | 0 / 20 |
| biscuit n=1000 (drops 0.3%) | pick | JAX fast vs torch default | 8.8e-03 | 1.5e-04 | 1 / 20 |
| biscuit n=1000 (drops 0.3%) | pick | JAX fast[int8 bundle] vs torch default | 2.7e-04 | 6.5e-06 | 0 / 20 |
| biscuit n=1000 (drops 0.3%) | pick | torch default vs torch fp32 | 9.1e-03 | 1.5e-04 | 1 / 20 |
| biscuit n=1000 (drops 0.3%) | edge | JAX fp32 vs torch fp32 | 2.0e-06 | 3.8e-07 | 0 / 5 |
| biscuit n=1000 (drops 0.3%) | edge | JAX fp32 vs torch default | 3.2e-02 | 8.2e-03 | 0 / 5 |
| biscuit n=1000 (drops 0.3%) | edge | JAX fast vs torch fp32 | 1.2e-03 | 2.0e-04 | 0 / 5 |
| biscuit n=1000 (drops 0.3%) | edge | JAX fast vs torch default | 3.2e-02 | 8.3e-03 | 0 / 5 |
| biscuit n=1000 (drops 0.3%) | edge | JAX fast[int8 bundle] vs torch default | 8.4e-04 | 2.2e-04 | 0 / 5 |
| biscuit n=1000 (drops 0.3%) | edge | torch default vs torch fp32 | 3.2e-02 | 8.2e-03 | 0 / 5 |
| biscuit n=1000 (drops 0.3%) | map | JAX fp32 vs torch fp32 | 7.0e-06 | 5.0e-07 | 0 / 400 |
| biscuit n=1000 (drops 0.3%) | map | JAX fp32 vs torch default | 1.3e-01 | 9.7e-03 | 7 / 400 |
| biscuit n=1000 (drops 0.3%) | map | JAX fast vs torch fp32 | 9.8e-03 | 4.6e-04 | 0 / 400 |
| biscuit n=1000 (drops 0.3%) | map | JAX fast vs torch default | 1.3e-01 | 9.5e-03 | 7 / 400 |
| biscuit n=1000 (drops 0.3%) | map | JAX fast[int8 bundle] vs torch default | 4.1e-03 | 2.6e-04 | 2 / 400 |
| biscuit n=1000 (drops 0.3%) | map | torch default vs torch fp32 | 1.3e-01 | 9.7e-03 | 7 / 400 |
| biscuit n=2000 (drops 0.3%) | pick | JAX fp32 vs torch fp32 | 2.9e-06 | 3.6e-08 | 0 / 20 |
| biscuit n=2000 (drops 0.3%) | pick | JAX fp32 vs torch default | 7.2e-03 | 4.9e-05 | 0 / 20 |
| biscuit n=2000 (drops 0.3%) | pick | JAX fast vs torch fp32 | 9.6e-04 | 2.0e-05 | 1 / 20 |
| biscuit n=2000 (drops 0.3%) | pick | JAX fast vs torch default | 6.8e-03 | 5.1e-05 | 1 / 20 |
| biscuit n=2000 (drops 0.3%) | pick | JAX fast[int8 bundle] vs torch default | 3.0e-03 | 2.9e-05 | 0 / 20 |
| biscuit n=2000 (drops 0.3%) | pick | torch default vs torch fp32 | 7.2e-03 | 4.9e-05 | 0 / 20 |
| biscuit n=2000 (drops 0.3%) | edge | JAX fp32 vs torch fp32 | 1.5e-06 | 2.2e-07 | 0 / 5 |
| biscuit n=2000 (drops 0.3%) | edge | JAX fp32 vs torch default | 1.8e-03 | 3.7e-04 | 0 / 5 |
| biscuit n=2000 (drops 0.3%) | edge | JAX fast vs torch fp32 | 8.5e-04 | 1.3e-04 | 0 / 5 |
| biscuit n=2000 (drops 0.3%) | edge | JAX fast vs torch default | 1.6e-03 | 3.7e-04 | 0 / 5 |
| biscuit n=2000 (drops 0.3%) | edge | JAX fast[int8 bundle] vs torch default | 5.5e-04 | 1.1e-04 | 0 / 5 |
| biscuit n=2000 (drops 0.3%) | edge | torch default vs torch fp32 | 1.8e-03 | 3.7e-04 | 0 / 5 |
| biscuit n=2000 (drops 0.3%) | map | JAX fp32 vs torch fp32 | 4.9e-06 | 3.9e-07 | 0 / 400 |
| biscuit n=2000 (drops 0.3%) | map | JAX fp32 vs torch default | 1.2e-02 | 1.8e-03 | 2 / 400 |
| biscuit n=2000 (drops 0.3%) | map | JAX fast vs torch fp32 | 5.5e-03 | 3.2e-04 | 1 / 400 |
| biscuit n=2000 (drops 0.3%) | map | JAX fast vs torch default | 1.4e-02 | 1.9e-03 | 3 / 400 |
| biscuit n=2000 (drops 0.3%) | map | JAX fast[int8 bundle] vs torch default | 4.4e-03 | 2.4e-04 | 1 / 400 |
| biscuit n=2000 (drops 0.3%) | map | torch default vs torch fp32 | 1.2e-02 | 1.8e-03 | 2 / 400 |
| pouch n=300 (drops 7.7%) | pick | JAX fp32 vs torch fp32 | 2.3e-06 | 1.7e-07 | 0 / 20 |
| pouch n=300 (drops 7.7%) | pick | JAX fp32 vs torch default | 8.3e-03 | 3.5e-04 | 0 / 20 |
| pouch n=300 (drops 7.7%) | pick | JAX fast vs torch fp32 | 3.1e-03 | 1.6e-04 | 0 / 20 |
| pouch n=300 (drops 7.7%) | pick | JAX fast vs torch default | 1.1e-02 | 3.9e-04 | 0 / 20 |
| pouch n=300 (drops 7.7%) | pick | JAX fast[int8 bundle] vs torch default | 3.8e-03 | 2.1e-04 | 0 / 20 |
| pouch n=300 (drops 7.7%) | pick | torch default vs torch fp32 | 8.3e-03 | 3.5e-04 | 0 / 20 |
| pouch n=300 (drops 7.7%) | edge | JAX fp32 vs torch fp32 | 1.5e-06 | 2.8e-07 | 0 / 5 |
| pouch n=300 (drops 7.7%) | edge | JAX fp32 vs torch default | 2.0e-03 | 2.9e-04 | 0 / 5 |
| pouch n=300 (drops 7.7%) | edge | JAX fast vs torch fp32 | 1.4e-03 | 2.6e-04 | 0 / 5 |
| pouch n=300 (drops 7.7%) | edge | JAX fast vs torch default | 2.9e-03 | 3.8e-04 | 0 / 5 |
| pouch n=300 (drops 7.7%) | edge | JAX fast[int8 bundle] vs torch default | 9.3e-04 | 2.4e-04 | 0 / 5 |
| pouch n=300 (drops 7.7%) | edge | torch default vs torch fp32 | 2.0e-03 | 2.9e-04 | 0 / 5 |
| pouch n=300 (drops 7.7%) | map | JAX fp32 vs torch fp32 | 4.2e-06 | 3.2e-07 | 0 / 400 |
| pouch n=300 (drops 7.7%) | map | JAX fp32 vs torch default | 5.5e-03 | 4.4e-04 | 1 / 400 |
| pouch n=300 (drops 7.7%) | map | JAX fast vs torch fp32 | 4.4e-03 | 3.6e-04 | 0 / 400 |
| pouch n=300 (drops 7.7%) | map | JAX fast vs torch default | 7.0e-03 | 4.7e-04 | 1 / 400 |
| pouch n=300 (drops 7.7%) | map | JAX fast[int8 bundle] vs torch default | 3.5e-03 | 2.9e-04 | 0 / 400 |
| pouch n=300 (drops 7.7%) | map | torch default vs torch fp32 | 5.5e-03 | 4.4e-04 | 1 / 400 |
| pouch n=1000 (drops 9.3%) | pick | JAX fp32 vs torch fp32 | 4.8e-06 | 1.9e-07 | 0 / 20 |
| pouch n=1000 (drops 9.3%) | pick | JAX fp32 vs torch default | 3.6e-03 | 1.2e-04 | 0 / 20 |
| pouch n=1000 (drops 9.3%) | pick | JAX fast vs torch fp32 | 2.9e-03 | 1.2e-04 | 0 / 20 |
| pouch n=1000 (drops 9.3%) | pick | JAX fast vs torch default | 3.5e-03 | 1.3e-04 | 0 / 20 |
| pouch n=1000 (drops 9.3%) | pick | JAX fast[int8 bundle] vs torch default | 3.9e-03 | 1.3e-04 | 0 / 20 |
| pouch n=1000 (drops 9.3%) | pick | torch default vs torch fp32 | 3.6e-03 | 1.2e-04 | 0 / 20 |
| pouch n=1000 (drops 9.3%) | edge | JAX fp32 vs torch fp32 | 1.7e-06 | 2.5e-07 | 0 / 5 |
| pouch n=1000 (drops 9.3%) | edge | JAX fp32 vs torch default | 3.5e-03 | 2.8e-04 | 0 / 5 |
| pouch n=1000 (drops 9.3%) | edge | JAX fast vs torch fp32 | 3.1e-03 | 3.7e-04 | 0 / 5 |
| pouch n=1000 (drops 9.3%) | edge | JAX fast vs torch default | 4.4e-03 | 5.0e-04 | 0 / 5 |
| pouch n=1000 (drops 9.3%) | edge | JAX fast[int8 bundle] vs torch default | 2.8e-03 | 3.1e-04 | 0 / 5 |
| pouch n=1000 (drops 9.3%) | edge | torch default vs torch fp32 | 3.5e-03 | 2.8e-04 | 0 / 5 |
| pouch n=1000 (drops 9.3%) | map | JAX fp32 vs torch fp32 | 8.9e-06 | 4.2e-07 | 0 / 400 |
| pouch n=1000 (drops 9.3%) | map | JAX fp32 vs torch default | 6.2e-03 | 5.3e-04 | 4 / 400 |
| pouch n=1000 (drops 9.3%) | map | JAX fast vs torch fp32 | 6.1e-03 | 3.6e-04 | 3 / 400 |
| pouch n=1000 (drops 9.3%) | map | JAX fast vs torch default | 7.5e-03 | 4.2e-04 | 3 / 400 |
| pouch n=1000 (drops 9.3%) | map | JAX fast[int8 bundle] vs torch default | 3.9e-03 | 2.9e-04 | 0 / 400 |
| pouch n=1000 (drops 9.3%) | map | torch default vs torch fp32 | 6.2e-03 | 5.3e-04 | 4 / 400 |
| pouch n=2000 (drops 9.7%) | pick | JAX fp32 vs torch fp32 | 4.9e-06 | 2.6e-07 | 0 / 20 |
| pouch n=2000 (drops 9.7%) | pick | JAX fp32 vs torch default | 3.2e-03 | 1.3e-04 | 0 / 20 |
| pouch n=2000 (drops 9.7%) | pick | JAX fast vs torch fp32 | 5.0e-03 | 2.1e-04 | 0 / 20 |
| pouch n=2000 (drops 9.7%) | pick | JAX fast vs torch default | 8.3e-03 | 2.7e-04 | 0 / 20 |
| pouch n=2000 (drops 9.7%) | pick | JAX fast[int8 bundle] vs torch default | 4.9e-03 | 1.8e-04 | 0 / 20 |
| pouch n=2000 (drops 9.7%) | pick | torch default vs torch fp32 | 3.2e-03 | 1.3e-04 | 0 / 20 |
| pouch n=2000 (drops 9.7%) | edge | JAX fp32 vs torch fp32 | 2.6e-06 | 1.9e-07 | 0 / 5 |
| pouch n=2000 (drops 9.7%) | edge | JAX fp32 vs torch default | 2.8e-03 | 2.9e-04 | 0 / 5 |
| pouch n=2000 (drops 9.7%) | edge | JAX fast vs torch fp32 | 4.6e-03 | 3.0e-04 | 0 / 5 |
| pouch n=2000 (drops 9.7%) | edge | JAX fast vs torch default | 2.8e-03 | 2.5e-04 | 0 / 5 |
| pouch n=2000 (drops 9.7%) | edge | JAX fast[int8 bundle] vs torch default | 1.2e-03 | 2.1e-04 | 0 / 5 |
| pouch n=2000 (drops 9.7%) | edge | torch default vs torch fp32 | 2.8e-03 | 2.9e-04 | 0 / 5 |
| pouch n=2000 (drops 9.7%) | map | JAX fp32 vs torch fp32 | 8.6e-06 | 5.2e-07 | 0 / 400 |
| pouch n=2000 (drops 9.7%) | map | JAX fp32 vs torch default | 4.5e-03 | 4.4e-04 | 4 / 400 |
| pouch n=2000 (drops 9.7%) | map | JAX fast vs torch fp32 | 6.6e-03 | 3.5e-04 | 7 / 400 |
| pouch n=2000 (drops 9.7%) | map | JAX fast vs torch default | 8.2e-03 | 5.9e-04 | 3 / 400 |
| pouch n=2000 (drops 9.7%) | map | JAX fast[int8 bundle] vs torch default | 4.5e-03 | 3.1e-04 | 3 / 400 |
| pouch n=2000 (drops 9.7%) | map | torch default vs torch fp32 | 4.6e-03 | 4.4e-04 | 4 / 400 |
| cheese n=300 (drops 37.3%) | pick | JAX fp32 vs torch fp32 | 2.2e-06 | 1.8e-07 | 0 / 20 |
| cheese n=300 (drops 37.3%) | pick | JAX fp32 vs torch default | 2.2e-03 | 2.2e-04 | 0 / 20 |
| cheese n=300 (drops 37.3%) | pick | JAX fast vs torch fp32 | 3.0e-03 | 2.9e-04 | 0 / 20 |
| cheese n=300 (drops 37.3%) | pick | JAX fast vs torch default | 3.1e-03 | 3.2e-04 | 0 / 20 |
| cheese n=300 (drops 37.3%) | pick | JAX fast[int8 bundle] vs torch default | 2.4e-03 | 1.4e-04 | 0 / 20 |
| cheese n=300 (drops 37.3%) | pick | torch default vs torch fp32 | 2.2e-03 | 2.2e-04 | 0 / 20 |
| cheese n=300 (drops 37.3%) | edge | JAX fp32 vs torch fp32 | 1.3e-06 | 2.4e-07 | 0 / 5 |
| cheese n=300 (drops 37.3%) | edge | JAX fp32 vs torch default | 2.1e-03 | 4.3e-04 | 0 / 5 |
| cheese n=300 (drops 37.3%) | edge | JAX fast vs torch fp32 | 2.3e-03 | 4.8e-04 | 0 / 5 |
| cheese n=300 (drops 37.3%) | edge | JAX fast vs torch default | 3.6e-03 | 6.9e-04 | 0 / 5 |
| cheese n=300 (drops 37.3%) | edge | JAX fast[int8 bundle] vs torch default | 2.2e-03 | 3.2e-04 | 0 / 5 |
| cheese n=300 (drops 37.3%) | edge | torch default vs torch fp32 | 2.1e-03 | 4.3e-04 | 0 / 5 |
| cheese n=300 (drops 37.3%) | map | JAX fp32 vs torch fp32 | 3.7e-06 | 4.6e-07 | 0 / 400 |
| cheese n=300 (drops 37.3%) | map | JAX fp32 vs torch default | 4.8e-03 | 5.3e-04 | 1 / 400 |
| cheese n=300 (drops 37.3%) | map | JAX fast vs torch fp32 | 5.1e-03 | 7.3e-04 | 0 / 400 |
| cheese n=300 (drops 37.3%) | map | JAX fast vs torch default | 4.1e-03 | 6.6e-04 | 1 / 400 |
| cheese n=300 (drops 37.3%) | map | JAX fast[int8 bundle] vs torch default | 2.5e-03 | 3.1e-04 | 0 / 400 |
| cheese n=300 (drops 37.3%) | map | torch default vs torch fp32 | 4.8e-03 | 5.3e-04 | 1 / 400 |
| cheese n=1000 (drops 34.9%) | pick | JAX fp32 vs torch fp32 | 1.1e-05 | 6.3e-07 | 0 / 20 |
| cheese n=1000 (drops 34.9%) | pick | JAX fp32 vs torch default | 2.0e-03 | 1.8e-04 | 0 / 20 |
| cheese n=1000 (drops 34.9%) | pick | JAX fast vs torch fp32 | 4.6e-03 | 2.9e-04 | 0 / 20 |
| cheese n=1000 (drops 34.9%) | pick | JAX fast vs torch default | 5.3e-03 | 3.9e-04 | 0 / 20 |
| cheese n=1000 (drops 34.9%) | pick | JAX fast[int8 bundle] vs torch default | 2.5e-03 | 2.3e-04 | 0 / 20 |
| cheese n=1000 (drops 34.9%) | pick | torch default vs torch fp32 | 2.0e-03 | 1.8e-04 | 0 / 20 |
| cheese n=1000 (drops 34.9%) | edge | JAX fp32 vs torch fp32 | 9.7e-06 | 8.9e-07 | 0 / 5 |
| cheese n=1000 (drops 34.9%) | edge | JAX fp32 vs torch default | 2.7e-03 | 4.7e-04 | 0 / 5 |
| cheese n=1000 (drops 34.9%) | edge | JAX fast vs torch fp32 | 5.4e-03 | 3.6e-04 | 0 / 5 |
| cheese n=1000 (drops 34.9%) | edge | JAX fast vs torch default | 5.6e-03 | 7.6e-04 | 0 / 5 |
| cheese n=1000 (drops 34.9%) | edge | JAX fast[int8 bundle] vs torch default | 3.9e-03 | 4.3e-04 | 0 / 5 |
| cheese n=1000 (drops 34.9%) | edge | torch default vs torch fp32 | 2.7e-03 | 4.7e-04 | 0 / 5 |
| cheese n=1000 (drops 34.9%) | map | JAX fp32 vs torch fp32 | 1.8e-05 | 1.2e-06 | 0 / 400 |
| cheese n=1000 (drops 34.9%) | map | JAX fp32 vs torch default | 7.4e-03 | 1.1e-03 | 2 / 400 |
| cheese n=1000 (drops 34.9%) | map | JAX fast vs torch fp32 | 6.3e-03 | 3.7e-04 | 1 / 400 |
| cheese n=1000 (drops 34.9%) | map | JAX fast vs torch default | 1.0e-02 | 1.3e-03 | 3 / 400 |
| cheese n=1000 (drops 34.9%) | map | JAX fast[int8 bundle] vs torch default | 6.8e-03 | 5.2e-04 | 1 / 400 |
| cheese n=1000 (drops 34.9%) | map | torch default vs torch fp32 | 7.4e-03 | 1.1e-03 | 2 / 400 |
| cheese n=2000 (drops 34.4%) | pick | JAX fp32 vs torch fp32 | 2.3e-06 | 1.7e-07 | 0 / 20 |
| cheese n=2000 (drops 34.4%) | pick | JAX fp32 vs torch default | 3.2e-03 | 1.5e-04 | 0 / 20 |
| cheese n=2000 (drops 34.4%) | pick | JAX fast vs torch fp32 | 4.4e-03 | 2.6e-04 | 0 / 20 |
| cheese n=2000 (drops 34.4%) | pick | JAX fast vs torch default | 4.4e-03 | 3.0e-04 | 0 / 20 |
| cheese n=2000 (drops 34.4%) | pick | JAX fast[int8 bundle] vs torch default | 1.6e-03 | 1.5e-04 | 0 / 20 |
| cheese n=2000 (drops 34.4%) | pick | torch default vs torch fp32 | 3.2e-03 | 1.5e-04 | 0 / 20 |
| cheese n=2000 (drops 34.4%) | edge | JAX fp32 vs torch fp32 | 3.2e-06 | 3.0e-07 | 0 / 5 |
| cheese n=2000 (drops 34.4%) | edge | JAX fp32 vs torch default | 3.6e-03 | 8.2e-04 | 0 / 5 |
| cheese n=2000 (drops 34.4%) | edge | JAX fast vs torch fp32 | 7.4e-03 | 5.9e-04 | 0 / 5 |
| cheese n=2000 (drops 34.4%) | edge | JAX fast vs torch default | 9.9e-03 | 1.1e-03 | 0 / 5 |
| cheese n=2000 (drops 34.4%) | edge | JAX fast[int8 bundle] vs torch default | 1.4e-03 | 2.6e-04 | 0 / 5 |
| cheese n=2000 (drops 34.4%) | edge | torch default vs torch fp32 | 3.6e-03 | 8.2e-04 | 0 / 5 |
| cheese n=2000 (drops 34.4%) | map | JAX fp32 vs torch fp32 | 8.2e-06 | 5.6e-07 | 0 / 400 |
| cheese n=2000 (drops 34.4%) | map | JAX fp32 vs torch default | 9.8e-03 | 7.8e-04 | 3 / 400 |
| cheese n=2000 (drops 34.4%) | map | JAX fast vs torch fp32 | 1.1e-02 | 7.7e-04 | 0 / 400 |
| cheese n=2000 (drops 34.4%) | map | JAX fast vs torch default | 1.1e-02 | 1.1e-03 | 3 / 400 |
| cheese n=2000 (drops 34.4%) | map | JAX fast[int8 bundle] vs torch default | 5.3e-03 | 3.7e-04 | 2 / 400 |
| cheese n=2000 (drops 34.4%) | map | torch default vs torch fp32 | 9.8e-03 | 7.9e-04 | 3 / 400 |

### Edge queries

- biscuit n=300 seal=3kPa: fp32 |dp| 1.0e-06, fast-vs-default |dp| 2.1e-03, P(drop) torch fp32 0.623-0.667
- biscuit n=300 seal=65kPa: fp32 |dp| 8.0e-07, fast-vs-default |dp| 1.1e-03, P(drop) torch fp32 0.388-0.482
- biscuit n=300 far-above: fp32 |dp| 6.9e-07, fast-vs-default |dp| 1.5e-03, P(drop) torch fp32 0.409-0.420
- biscuit n=300 far-below: fp32 |dp| 8.0e-07, fast-vs-default |dp| 1.3e-03, P(drop) torch fp32 0.423-0.447
- biscuit n=300 seal=NaN: fp32 |dp| 1.5e-06, fast-vs-default |dp| 2.9e-03, P(drop) torch fp32 0.114-0.258
- biscuit n=1000 seal=3kPa: fp32 |dp| 7.2e-07, fast-vs-default |dp| 2.6e-02, P(drop) torch fp32 0.584-0.700
- biscuit n=1000 seal=65kPa: fp32 |dp| 2.0e-06, fast-vs-default |dp| 1.9e-02, P(drop) torch fp32 0.146-0.273
- biscuit n=1000 far-above: fp32 |dp| 4.8e-07, fast-vs-default |dp| 1.6e-02, P(drop) torch fp32 0.385-0.409
- biscuit n=1000 far-below: fp32 |dp| 6.9e-07, fast-vs-default |dp| 3.2e-02, P(drop) torch fp32 0.369-0.420
- biscuit n=1000 seal=NaN: fp32 |dp| 3.8e-08, fast-vs-default |dp| 1.7e-03, P(drop) torch fp32 0.001-0.014
- biscuit n=2000 seal=3kPa: fp32 |dp| 1.5e-06, fast-vs-default |dp| 1.6e-03, P(drop) torch fp32 0.597-0.698
- biscuit n=2000 seal=65kPa: fp32 |dp| 5.4e-07, fast-vs-default |dp| 9.2e-04, P(drop) torch fp32 0.175-0.275
- biscuit n=2000 far-above: fp32 |dp| 2.7e-07, fast-vs-default |dp| 6.0e-04, P(drop) torch fp32 0.356-0.381
- biscuit n=2000 far-below: fp32 |dp| 3.6e-07, fast-vs-default |dp| 8.2e-04, P(drop) torch fp32 0.325-0.340
- biscuit n=2000 seal=NaN: fp32 |dp| 3.5e-08, fast-vs-default |dp| 5.5e-04, P(drop) torch fp32 0.001-0.020
- pouch n=300 seal=3kPa: fp32 |dp| 1.5e-06, fast-vs-default |dp| 2.6e-03, P(drop) torch fp32 0.326-0.815
- pouch n=300 seal=65kPa: fp32 |dp| 4.2e-07, fast-vs-default |dp| 5.9e-04, P(drop) torch fp32 0.002-0.035
- pouch n=300 far-above: fp32 |dp| 3.3e-07, fast-vs-default |dp| 2.5e-04, P(drop) torch fp32 0.317-0.413
- pouch n=300 far-below: fp32 |dp| 3.3e-07, fast-vs-default |dp| 4.0e-04, P(drop) torch fp32 0.291-0.425
- pouch n=300 seal=NaN: fp32 |dp| 7.0e-07, fast-vs-default |dp| 2.9e-03, P(drop) torch fp32 0.001-0.579
- pouch n=1000 seal=3kPa: fp32 |dp| 1.7e-06, fast-vs-default |dp| 4.4e-03, P(drop) torch fp32 0.341-0.937
- pouch n=1000 seal=65kPa: fp32 |dp| 1.6e-07, fast-vs-default |dp| 3.2e-04, P(drop) torch fp32 0.001-0.011
- pouch n=1000 far-above: fp32 |dp| 3.6e-07, fast-vs-default |dp| 3.2e-04, P(drop) torch fp32 0.337-0.428
- pouch n=1000 far-below: fp32 |dp| 4.5e-07, fast-vs-default |dp| 4.1e-04, P(drop) torch fp32 0.302-0.420
- pouch n=1000 seal=NaN: fp32 |dp| 7.2e-07, fast-vs-default |dp| 3.7e-03, P(drop) torch fp32 0.008-0.987
- pouch n=2000 seal=3kPa: fp32 |dp| 2.6e-06, fast-vs-default |dp| 2.8e-03, P(drop) torch fp32 0.495-0.958
- pouch n=2000 seal=65kPa: fp32 |dp| 3.3e-07, fast-vs-default |dp| 2.0e-04, P(drop) torch fp32 0.002-0.012
- pouch n=2000 far-above: fp32 |dp| 3.6e-07, fast-vs-default |dp| 6.5e-04, P(drop) torch fp32 0.325-0.393
- pouch n=2000 far-below: fp32 |dp| 4.5e-07, fast-vs-default |dp| 5.2e-04, P(drop) torch fp32 0.284-0.399
- pouch n=2000 seal=NaN: fp32 |dp| 6.0e-07, fast-vs-default |dp| 2.5e-04, P(drop) torch fp32 0.000-0.141
- cheese n=300 seal=3kPa: fp32 |dp| 1.3e-06, fast-vs-default |dp| 3.6e-03, P(drop) torch fp32 0.262-0.805
- cheese n=300 seal=65kPa: fp32 |dp| 6.0e-07, fast-vs-default |dp| 2.4e-03, P(drop) torch fp32 0.001-0.901
- cheese n=300 far-above: fp32 |dp| 6.6e-07, fast-vs-default |dp| 8.0e-04, P(drop) torch fp32 0.325-0.635
- cheese n=300 far-below: fp32 |dp| 7.2e-07, fast-vs-default |dp| 8.8e-04, P(drop) torch fp32 0.300-0.666
- cheese n=300 seal=NaN: fp32 |dp| 1.1e-06, fast-vs-default |dp| 1.2e-03, P(drop) torch fp32 0.002-0.955
- cheese n=1000 seal=3kPa: fp32 |dp| 3.5e-06, fast-vs-default |dp| 2.8e-03, P(drop) torch fp32 0.315-0.759
- cheese n=1000 seal=65kPa: fp32 |dp| 9.7e-06, fast-vs-default |dp| 5.6e-03, P(drop) torch fp32 0.000-0.928
- cheese n=1000 far-above: fp32 |dp| 8.3e-07, fast-vs-default |dp| 7.0e-04, P(drop) torch fp32 0.301-0.599
- cheese n=1000 far-below: fp32 |dp| 1.4e-06, fast-vs-default |dp| 9.6e-04, P(drop) torch fp32 0.342-0.662
- cheese n=1000 seal=NaN: fp32 |dp| 4.4e-06, fast-vs-default |dp| 3.7e-03, P(drop) torch fp32 0.000-0.956
- cheese n=2000 seal=3kPa: fp32 |dp| 1.1e-06, fast-vs-default |dp| 5.0e-03, P(drop) torch fp32 0.309-0.922
- cheese n=2000 seal=65kPa: fp32 |dp| 3.2e-06, fast-vs-default |dp| 9.9e-03, P(drop) torch fp32 0.000-0.966
- cheese n=2000 far-above: fp32 |dp| 3.3e-07, fast-vs-default |dp| 1.6e-03, P(drop) torch fp32 0.264-0.639
- cheese n=2000 far-below: fp32 |dp| 9.5e-07, fast-vs-default |dp| 4.5e-03, P(drop) torch fp32 0.300-0.669
- cheese n=2000 seal=NaN: fp32 |dp| 4.8e-07, fast-vs-default |dp| 1.6e-03, P(drop) torch fp32 0.000-0.998

### Every chosen-speed disagreement, with both P(drop) vectors (speeds 0.30 … 1.00)

```
- biscuit n=300 (drops 0.3%) pick pick | JAX fp32 vs torch default: 0.8 vs 0.85
  P(drop) JAX fp32: [0.0065 0.0042 0.0038 0.0041 0.0033 0.0038 0.0034 0.0035 0.004  0.0038
 0.004  0.005  0.0054 0.0077 0.014 ]
  P(drop) torch default: [0.0065 0.0042 0.0038 0.0041 0.0033 0.0038 0.0034 0.0035 0.0039 0.0037
 0.004  0.005  0.0053 0.0077 0.014 ]
- biscuit n=300 (drops 0.3%) pick pick | JAX fast vs torch fp32: 0.85 vs 0.8
  P(drop) JAX fast: [0.0065 0.0042 0.0038 0.0041 0.0033 0.0038 0.0034 0.0035 0.0039 0.0037
 0.004  0.005  0.0054 0.0077 0.0138]
  P(drop) torch fp32: [0.0065 0.0042 0.0038 0.0041 0.0033 0.0038 0.0034 0.0035 0.004  0.0038
 0.004  0.005  0.0054 0.0077 0.014 ]
- biscuit n=300 (drops 0.3%) pick pick | JAX fp32 vs torch default: 0.85 vs 0.95
  P(drop) JAX fp32: [0.0053 0.0038 0.0034 0.003  0.0037 0.004  0.0034 0.0028 0.0031 0.0039
 0.0033 0.0041 0.0056 0.005  0.0094]
  P(drop) torch default: [0.0053 0.0038 0.0034 0.003  0.0037 0.004  0.0034 0.0028 0.0031 0.0039
 0.0032 0.0041 0.0055 0.005  0.0093]
- biscuit n=300 (drops 0.3%) pick pick | JAX fast vs torch default: 0.85 vs 0.95
  P(drop) JAX fast: [0.0053 0.0038 0.0034 0.003  0.0037 0.0041 0.0034 0.0029 0.0031 0.0039
 0.0033 0.0041 0.0056 0.005  0.0092]
  P(drop) torch default: [0.0053 0.0038 0.0034 0.003  0.0037 0.004  0.0034 0.0028 0.0031 0.0039
 0.0032 0.0041 0.0055 0.005  0.0093]
- biscuit n=300 (drops 0.3%) pick pick | JAX fast[int8 bundle] vs torch default: 0.85 vs 0.95
  P(drop) JAX fast[int8 bundle]: [0.0053 0.0038 0.0034 0.003  0.0037 0.004  0.0034 0.0028 0.0031 0.0039
 0.0032 0.0041 0.0056 0.005  0.0094]
  P(drop) torch default: [0.0053 0.0038 0.0034 0.003  0.0037 0.004  0.0034 0.0028 0.0031 0.0039
 0.0032 0.0041 0.0055 0.005  0.0093]
- biscuit n=300 (drops 0.3%) map map | JAX fast vs torch fp32: 1.0 vs 0.95
  P(drop) JAX fast: [0.003  0.002  0.0018 0.0017 0.0016 0.0016 0.0016 0.0017 0.0021 0.0016
 0.0018 0.0023 0.0024 0.0025 0.005 ]
  P(drop) torch fp32: [0.003  0.002  0.0018 0.0017 0.0016 0.0016 0.0016 0.0017 0.0021 0.0016
 0.0018 0.0023 0.0024 0.0025 0.005 ]
- biscuit n=300 (drops 0.3%) map map | JAX fast vs torch default: 1.0 vs 0.95
  P(drop) JAX fast: [0.003  0.002  0.0018 0.0017 0.0016 0.0016 0.0016 0.0017 0.0021 0.0016
 0.0018 0.0023 0.0024 0.0025 0.005 ]
  P(drop) torch default: [0.003  0.002  0.0018 0.0017 0.0016 0.0016 0.0016 0.0017 0.0021 0.0016
 0.0018 0.0022 0.0024 0.0025 0.005 ]
- biscuit n=300 (drops 0.3%) map map | JAX fp32 vs torch default: 0.85 vs 0.9
  P(drop) JAX fp32: [0.0084 0.0042 0.0042 0.004  0.0038 0.0038 0.0037 0.0041 0.0044 0.0037
 0.0041 0.0047 0.005  0.0079 0.0174]
  P(drop) torch default: [0.0083 0.0042 0.0041 0.004  0.0037 0.0038 0.0037 0.0041 0.0043 0.0037
 0.0041 0.0046 0.005  0.0078 0.0172]
- biscuit n=300 (drops 0.3%) map map | JAX fast vs torch fp32: 0.8 vs 0.85
  P(drop) JAX fast: [0.0064 0.0065 0.0044 0.0038 0.0042 0.0038 0.0048 0.0049 0.0038 0.0041
 0.0043 0.005  0.0056 0.0064 0.0125]
  P(drop) torch fp32: [0.0064 0.0064 0.0044 0.0038 0.0042 0.0037 0.0048 0.0049 0.0038 0.0041
 0.0042 0.005  0.0056 0.0063 0.0126]
- biscuit n=300 (drops 0.3%) map map | JAX fast vs torch fp32: 0.55 vs 0.65
  P(drop) JAX fast: [0.0077 0.0052 0.005  0.005  0.0049 0.0049 0.0055 0.005  0.0051 0.0054
 0.0056 0.0061 0.0086 0.0074 0.0179]
  P(drop) torch fp32: [0.0076 0.0052 0.005  0.0049 0.0048 0.0048 0.0054 0.005  0.005  0.0053
 0.0056 0.006  0.0085 0.0073 0.018 ]
- biscuit n=300 (drops 0.3%) map map | JAX fast vs torch default: 0.8 vs 0.85
  P(drop) JAX fast: [0.0064 0.0065 0.0044 0.0038 0.0042 0.0038 0.0048 0.0049 0.0038 0.0041
 0.0043 0.005  0.0056 0.0064 0.0125]
  P(drop) torch default: [0.0064 0.0064 0.0044 0.0038 0.0042 0.0037 0.0047 0.0049 0.0038 0.0041
 0.0042 0.005  0.0056 0.0063 0.0125]
- biscuit n=300 (drops 0.3%) map map | JAX fast vs torch default: 0.55 vs 0.65
  P(drop) JAX fast: [0.0077 0.0052 0.005  0.005  0.0049 0.0049 0.0055 0.005  0.0051 0.0054
 0.0056 0.0061 0.0086 0.0074 0.0179]
  P(drop) torch default: [0.0076 0.0052 0.005  0.005  0.0048 0.0048 0.0054 0.005  0.005  0.0053
 0.0056 0.0061 0.0085 0.0073 0.0179]
- biscuit n=300 (drops 0.3%) map map | JAX fast vs torch default: 0.85 vs 0.9
  P(drop) JAX fast: [0.0084 0.0042 0.0042 0.004  0.0038 0.0039 0.0038 0.0041 0.0044 0.0037
 0.0041 0.0047 0.005  0.0079 0.0172]
  P(drop) torch default: [0.0083 0.0042 0.0041 0.004  0.0037 0.0038 0.0037 0.0041 0.0043 0.0037
 0.0041 0.0046 0.005  0.0078 0.0172]
- biscuit n=300 (drops 0.3%) map map | JAX fast[int8 bundle] vs torch default: 0.55 vs 0.65
  P(drop) JAX fast[int8 bundle]: [0.0076 0.0052 0.005  0.005  0.0049 0.0048 0.0054 0.005  0.0051 0.0054
 0.0056 0.0061 0.0086 0.0074 0.0181]
  P(drop) torch default: [0.0076 0.0052 0.005  0.005  0.0048 0.0048 0.0054 0.005  0.005  0.0053
 0.0056 0.0061 0.0085 0.0073 0.0179]
- biscuit n=1000 (drops 0.3%) pick pick | JAX fp32 vs torch default: 1.0 vs 0.95
  P(drop) JAX fp32: [0.0007 0.0006 0.0004 0.0003 0.0003 0.0003 0.0003 0.0005 0.0003 0.0005
 0.0004 0.0005 0.0005 0.0007 0.0049]
  P(drop) torch default: [0.0007 0.0005 0.0004 0.0003 0.0003 0.0004 0.0003 0.0005 0.0003 0.0005
 0.0004 0.0005 0.0006 0.0008 0.005 ]
- biscuit n=1000 (drops 0.3%) pick pick | JAX fast vs torch default: 1.0 vs 0.95
  P(drop) JAX fast: [0.0007 0.0005 0.0004 0.0003 0.0003 0.0003 0.0003 0.0005 0.0003 0.0005
 0.0004 0.0005 0.0005 0.0007 0.0048]
  P(drop) torch default: [0.0007 0.0005 0.0004 0.0003 0.0003 0.0004 0.0003 0.0005 0.0003 0.0005
 0.0004 0.0005 0.0006 0.0008 0.005 ]
- biscuit n=1000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.5 vs 0.45
  P(drop) JAX fp32: [0.013  0.0049 0.0068 0.0046 0.0046 0.0054 0.0055 0.006  0.008  0.0083
 0.0135 0.0105 0.0113 0.0179 0.0518]
  P(drop) torch default: [0.0131 0.0051 0.0067 0.0049 0.0054 0.006  0.006  0.0065 0.0085 0.009
 0.0142 0.012  0.0139 0.0215 0.0644]
- biscuit n=1000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.9 vs 0.8
  P(drop) JAX fp32: [0.0073 0.0027 0.0023 0.0022 0.0028 0.0025 0.0028 0.0027 0.0029 0.003
 0.0039 0.0065 0.0047 0.0061 0.0478]
  P(drop) torch default: [0.0088 0.0032 0.0028 0.0025 0.0031 0.0028 0.0032 0.003  0.0032 0.0033
 0.0041 0.0053 0.0053 0.0068 0.0433]
- biscuit n=1000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.95 vs 0.9
  P(drop) JAX fp32: [0.0114 0.0044 0.0042 0.003  0.0031 0.0037 0.0028 0.0027 0.0027 0.0029
 0.0029 0.0029 0.0033 0.0047 0.0182]
  P(drop) torch default: [0.0146 0.006  0.0053 0.0041 0.004  0.0045 0.0035 0.0033 0.0033 0.0035
 0.0035 0.0036 0.004  0.0054 0.0204]
- biscuit n=1000 (drops 0.3%) map map | JAX fast vs torch default: 0.5 vs 0.45
  P(drop) JAX fast: [0.013  0.0049 0.0068 0.0046 0.0046 0.0054 0.0055 0.0059 0.008  0.0083
 0.0135 0.0104 0.0113 0.0178 0.0515]
  P(drop) torch default: [0.0131 0.0051 0.0067 0.0049 0.0054 0.006  0.006  0.0065 0.0085 0.009
 0.0142 0.012  0.0139 0.0215 0.0644]
- biscuit n=1000 (drops 0.3%) map map | JAX fast vs torch default: 0.9 vs 0.8
  P(drop) JAX fast: [0.0074 0.0028 0.0023 0.0022 0.0028 0.0025 0.0029 0.0028 0.003  0.003
 0.0039 0.0065 0.0047 0.0062 0.0476]
  P(drop) torch default: [0.0088 0.0032 0.0028 0.0025 0.0031 0.0028 0.0032 0.003  0.0032 0.0033
 0.0041 0.0053 0.0053 0.0068 0.0433]
- biscuit n=1000 (drops 0.3%) map map | JAX fast vs torch default: 0.95 vs 0.9
  P(drop) JAX fast: [0.0115 0.0045 0.0042 0.0031 0.0031 0.0037 0.0028 0.0027 0.0028 0.0029
 0.0029 0.003  0.0033 0.0047 0.0182]
  P(drop) torch default: [0.0146 0.006  0.0053 0.0041 0.004  0.0045 0.0035 0.0033 0.0033 0.0035
 0.0035 0.0036 0.004  0.0054 0.0204]
- biscuit n=1000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.65 vs nan
  P(drop) JAX fp32: [0.0131 0.0053 0.006  0.0066 0.0047 0.0045 0.0052 0.005  0.007  0.0068
 0.0087 0.0074 0.0093 0.0122 0.0659]
  P(drop) torch default: [0.0143 0.0059 0.0064 0.0069 0.0053 0.0051 0.0056 0.0056 0.0073 0.0072
 0.0092 0.0083 0.0103 0.0135 0.0572]
- biscuit n=1000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.95 vs 0.9
  P(drop) JAX fp32: [0.0075 0.0042 0.0021 0.0024 0.0019 0.0026 0.0018 0.0017 0.0018 0.0018
 0.002  0.0021 0.0034 0.0047 0.0159]
  P(drop) torch default: [0.0095 0.0047 0.0026 0.0027 0.0022 0.0024 0.002  0.0019 0.0021 0.002
 0.0021 0.0023 0.0034 0.0053 0.0154]
- biscuit n=1000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.9 vs 0.85
  P(drop) JAX fp32: [0.0147 0.0033 0.0028 0.0045 0.0055 0.0025 0.003  0.0028 0.0027 0.003
 0.0033 0.004  0.0048 0.0075 0.0715]
  P(drop) torch default: [0.0145 0.0036 0.003  0.0041 0.0047 0.0028 0.0032 0.003  0.003  0.0032
 0.0035 0.0042 0.0051 0.0075 0.0602]
- biscuit n=1000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.85 vs nan
  P(drop) JAX fp32: [0.0205 0.008  0.0058 0.0052 0.005  0.0052 0.0049 0.0044 0.0047 0.0047
 0.0046 0.005  0.0053 0.0074 0.0273]
  P(drop) torch default: [0.0228 0.0098 0.0066 0.006  0.0058 0.0061 0.0057 0.005  0.0053 0.0052
 0.0051 0.0054 0.0058 0.008  0.0265]
- biscuit n=1000 (drops 0.3%) map map | JAX fast vs torch default: 0.65 vs nan
  P(drop) JAX fast: [0.0132 0.0053 0.006  0.0066 0.0047 0.0045 0.0052 0.0049 0.007  0.0068
 0.0088 0.0074 0.0093 0.0122 0.0656]
  P(drop) torch default: [0.0143 0.0059 0.0064 0.0069 0.0053 0.0051 0.0056 0.0056 0.0073 0.0072
 0.0092 0.0083 0.0103 0.0135 0.0572]
- biscuit n=1000 (drops 0.3%) map map | JAX fast vs torch default: 0.95 vs 0.9
  P(drop) JAX fast: [0.0075 0.0042 0.0021 0.0024 0.0019 0.0026 0.0018 0.0017 0.0018 0.0018
 0.002  0.0021 0.0034 0.0047 0.0158]
  P(drop) torch default: [0.0095 0.0047 0.0026 0.0027 0.0022 0.0024 0.002  0.0019 0.0021 0.002
 0.0021 0.0023 0.0034 0.0053 0.0154]
- biscuit n=1000 (drops 0.3%) map map | JAX fast vs torch default: 0.9 vs 0.85
  P(drop) JAX fast: [0.0147 0.0033 0.0028 0.0045 0.0054 0.0025 0.003  0.0028 0.0027 0.0029
 0.0033 0.004  0.0047 0.0074 0.0715]
  P(drop) torch default: [0.0145 0.0036 0.003  0.0041 0.0047 0.0028 0.0032 0.003  0.003  0.0032
 0.0035 0.0042 0.0051 0.0075 0.0602]
- biscuit n=1000 (drops 0.3%) map map | JAX fast vs torch default: 0.85 vs nan
  P(drop) JAX fast: [0.0203 0.0079 0.0057 0.0051 0.005  0.0052 0.0048 0.0044 0.0047 0.0046
 0.0046 0.0049 0.0053 0.0073 0.0269]
  P(drop) torch default: [0.0228 0.0098 0.0066 0.006  0.0058 0.0061 0.0057 0.005  0.0053 0.0052
 0.0051 0.0054 0.0058 0.008  0.0265]
- biscuit n=1000 (drops 0.3%) map map | JAX fast[int8 bundle] vs torch default: 0.75 vs 0.85
  P(drop) JAX fast[int8 bundle]: [0.0059 0.0027 0.0029 0.0036 0.0027 0.0027 0.0031 0.004  0.0035 0.004
 0.0055 0.005  0.0158 0.0073 0.0273]
  P(drop) torch default: [0.0058 0.0027 0.0029 0.0036 0.0027 0.0026 0.003  0.0039 0.0035 0.0039
 0.0055 0.0049 0.0157 0.0073 0.027 ]
- biscuit n=1000 (drops 0.3%) map map | JAX fast[int8 bundle] vs torch default: 0.65 vs nan
  P(drop) JAX fast[int8 bundle]: [0.0227 0.0098 0.0065 0.006  0.0058 0.0061 0.0057 0.005  0.0052 0.0051
 0.0051 0.0053 0.0058 0.0079 0.0265]
  P(drop) torch default: [0.0228 0.0098 0.0066 0.006  0.0058 0.0061 0.0057 0.005  0.0053 0.0052
 0.0051 0.0054 0.0058 0.008  0.0265]
- biscuit n=2000 (drops 0.3%) pick pick | JAX fast vs torch fp32: 1.0 vs 0.95
  P(drop) JAX fast: [0.0004 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0004 0.005 ]
  P(drop) torch fp32: [0.0004 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0004 0.005 ]
- biscuit n=2000 (drops 0.3%) pick pick | JAX fast vs torch default: 1.0 vs 0.95
  P(drop) JAX fast: [0.0004 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0004 0.005 ]
  P(drop) torch default: [0.0004 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0004 0.0052]
- biscuit n=2000 (drops 0.3%) map map | JAX fp32 vs torch default: 0.6 vs 0.5
  P(drop) JAX fp32: [0.0071 0.0034 0.004  0.0033 0.0039 0.0051 0.0048 0.0067 0.009  0.0143
 0.0176 0.0298 0.0431 0.0869 0.1482]
  P(drop) torch default: [0.0073 0.0036 0.0043 0.0036 0.0042 0.0055 0.0053 0.0073 0.0098 0.0154
 0.0191 0.0317 0.0453 0.0898 0.1547]
- biscuit n=2000 (drops 0.3%) map map | JAX fast vs torch fp32: 1.0 vs 0.95
  P(drop) JAX fast: [0.0005 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0005 0.005 ]
  P(drop) torch fp32: [0.0005 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0005 0.0051]
- biscuit n=2000 (drops 0.3%) map map | JAX fast vs torch default: 0.6 vs 0.5
  P(drop) JAX fast: [0.0072 0.0034 0.004  0.0034 0.004  0.0052 0.0049 0.0068 0.0091 0.0145
 0.0178 0.0302 0.0435 0.088  0.1481]
  P(drop) torch default: [0.0073 0.0036 0.0043 0.0036 0.0042 0.0055 0.0053 0.0073 0.0098 0.0154
 0.0191 0.0317 0.0453 0.0898 0.1547]
- biscuit n=2000 (drops 0.3%) map map | JAX fast vs torch default: 1.0 vs 0.95
  P(drop) JAX fast: [0.0005 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0005 0.005 ]
  P(drop) torch default: [0.0005 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0002 0.0002 0.0005 0.0052]
- biscuit n=2000 (drops 0.3%) map map | JAX fp32 vs torch default: 1.0 vs 0.95
  P(drop) JAX fp32: [0.0002 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0001 0.0002 0.0005 0.0049]
  P(drop) torch default: [0.0002 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0001 0.0002 0.0005 0.0051]
- biscuit n=2000 (drops 0.3%) map map | JAX fast vs torch default: 1.0 vs 0.95
  P(drop) JAX fast: [0.0002 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0001 0.0002 0.0005 0.0048]
  P(drop) torch default: [0.0002 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0001 0.0002 0.0005 0.0051]
- biscuit n=2000 (drops 0.3%) map map | JAX fast[int8 bundle] vs torch default: 1.0 vs 0.95
  P(drop) JAX fast[int8 bundle]: [0.0002 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0001 0.0002 0.0005 0.005 ]
  P(drop) torch default: [0.0002 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
 0.0001 0.0001 0.0002 0.0005 0.0051]
- pouch n=300 (drops 7.7%) map map | JAX fp32 vs torch default: 0.55 vs 0.6
  P(drop) JAX fp32: [5.000e-04 2.000e-04 3.000e-04 6.000e-04 1.100e-03 2.300e-03 5.000e-03
 1.130e-02 3.920e-02 1.067e-01 2.954e-01 5.672e-01 7.718e-01 8.801e-01
 9.448e-01]
  P(drop) torch default: [5.000e-04 2.000e-04 3.000e-04 6.000e-04 1.100e-03 2.300e-03 5.000e-03
 1.110e-02 3.860e-02 1.055e-01 2.941e-01 5.665e-01 7.721e-01 8.805e-01
 9.443e-01]
- pouch n=300 (drops 7.7%) map map | JAX fast vs torch default: 0.55 vs 0.6
  P(drop) JAX fast: [5.000e-04 2.000e-04 3.000e-04 6.000e-04 1.100e-03 2.300e-03 5.100e-03
 1.130e-02 3.910e-02 1.072e-01 2.962e-01 5.669e-01 7.713e-01 8.793e-01
 9.443e-01]
  P(drop) torch default: [5.000e-04 2.000e-04 3.000e-04 6.000e-04 1.100e-03 2.300e-03 5.000e-03
 1.110e-02 3.860e-02 1.055e-01 2.941e-01 5.665e-01 7.721e-01 8.805e-01
 9.443e-01]
- pouch n=1000 (drops 9.3%) map map | JAX fp32 vs torch default: 0.9 vs 0.85
  P(drop) JAX fp32: [0.0046 0.0012 0.0012 0.0012 0.0011 0.0014 0.0018 0.002  0.0018 0.0028
 0.0031 0.0034 0.0049 0.0072 0.0531]
  P(drop) torch default: [0.0048 0.0013 0.0013 0.0013 0.0012 0.0015 0.0019 0.0021 0.0019 0.0029
 0.0032 0.0036 0.0051 0.0075 0.0539]
- pouch n=1000 (drops 9.3%) map map | JAX fast vs torch default: 0.9 vs 0.85
  P(drop) JAX fast: [0.0046 0.0012 0.0012 0.0012 0.0011 0.0014 0.0018 0.0021 0.0018 0.0028
 0.0032 0.0035 0.0049 0.0073 0.0527]
  P(drop) torch default: [0.0048 0.0013 0.0013 0.0013 0.0012 0.0015 0.0019 0.0021 0.0019 0.0029
 0.0032 0.0036 0.0051 0.0075 0.0539]
- pouch n=1000 (drops 9.3%) map map | JAX fp32 vs torch default: 0.35 vs nan
  P(drop) JAX fp32: [0.0091 0.005  0.0115 0.0186 0.0321 0.0575 0.1138 0.2116 0.3497 0.5271
 0.7161 0.8237 0.9194 0.9675 0.9732]
  P(drop) torch default: [0.0092 0.005  0.0116 0.0187 0.0324 0.0579 0.114  0.2111 0.3477 0.5239
 0.7129 0.8207 0.9179 0.9668 0.9731]
- pouch n=1000 (drops 9.3%) map map | JAX fp32 vs torch default: 0.6 vs 0.55
  P(drop) JAX fp32: [7.000e-04 1.000e-04 1.000e-04 2.000e-04 4.000e-04 1.700e-03 5.000e-03
 3.610e-02 1.936e-01 3.923e-01 7.821e-01 9.360e-01 9.852e-01 9.967e-01
 9.973e-01]
  P(drop) torch default: [7.000e-04 1.000e-04 1.000e-04 2.000e-04 4.000e-04 1.800e-03 5.100e-03
 3.650e-02 1.947e-01 3.926e-01 7.812e-01 9.354e-01 9.851e-01 9.967e-01
 9.973e-01]
- pouch n=1000 (drops 9.3%) map map | JAX fp32 vs torch default: 0.95 vs 0.9
  P(drop) JAX fp32: [3.1e-03 3.0e-04 2.0e-04 1.0e-04 1.0e-04 1.0e-04 1.0e-04 2.0e-04 1.0e-04
 1.0e-04 2.0e-04 7.0e-04 1.2e-03 5.0e-03 1.1e-01]
  P(drop) torch default: [3.100e-03 3.000e-04 2.000e-04 1.000e-04 1.000e-04 1.000e-04 1.000e-04
 2.000e-04 1.000e-04 1.000e-04 2.000e-04 7.000e-04 1.200e-03 5.200e-03
 1.129e-01]
- pouch n=1000 (drops 9.3%) map map | JAX fast vs torch fp32: nan vs 0.35
  P(drop) JAX fast: [0.0091 0.0051 0.0118 0.0189 0.0326 0.0582 0.1141 0.212  0.3496 0.5276
 0.7149 0.8229 0.9189 0.9671 0.9731]
  P(drop) torch fp32: [0.0091 0.005  0.0115 0.0186 0.0321 0.0575 0.1138 0.2116 0.3497 0.5271
 0.7161 0.8237 0.9194 0.9675 0.9732]
- pouch n=1000 (drops 9.3%) map map | JAX fast vs torch fp32: nan vs 0.35
  P(drop) JAX fast: [0.0113 0.005  0.0088 0.0128 0.0204 0.0369 0.0674 0.1183 0.2101 0.3771
 0.5243 0.6829 0.8474 0.9433 0.9703]
  P(drop) torch fp32: [0.0112 0.0049 0.0085 0.0125 0.02   0.0363 0.0665 0.1169 0.2093 0.3765
 0.5239 0.6832 0.8475 0.9436 0.9705]
- pouch n=1000 (drops 9.3%) map map | JAX fast vs torch fp32: 0.55 vs 0.6
  P(drop) JAX fast: [7.000e-04 1.000e-04 1.000e-04 2.000e-04 4.000e-04 1.700e-03 5.000e-03
 3.630e-02 1.934e-01 3.930e-01 7.827e-01 9.360e-01 9.852e-01 9.967e-01
 9.973e-01]
  P(drop) torch fp32: [7.000e-04 1.000e-04 1.000e-04 2.000e-04 4.000e-04 1.700e-03 5.000e-03
 3.610e-02 1.936e-01 3.923e-01 7.821e-01 9.360e-01 9.852e-01 9.967e-01
 9.973e-01]
- pouch n=1000 (drops 9.3%) map map | JAX fast vs torch default: nan vs 0.35
  P(drop) JAX fast: [0.0113 0.005  0.0088 0.0128 0.0204 0.0369 0.0674 0.1183 0.2101 0.3771
 0.5243 0.6829 0.8474 0.9433 0.9703]
  P(drop) torch default: [0.0113 0.0049 0.0086 0.0127 0.0201 0.0367 0.0666 0.1167 0.208  0.374
 0.5197 0.6795 0.8451 0.9425 0.9702]
- pouch n=1000 (drops 9.3%) map map | JAX fast vs torch default: 0.95 vs 0.9
  P(drop) JAX fast: [3.100e-03 3.000e-04 2.000e-04 1.000e-04 1.000e-04 1.000e-04 1.000e-04
 2.000e-04 1.000e-04 1.000e-04 2.000e-04 7.000e-04 1.200e-03 5.000e-03
 1.099e-01]
  P(drop) torch default: [3.100e-03 3.000e-04 2.000e-04 1.000e-04 1.000e-04 1.000e-04 1.000e-04
 2.000e-04 1.000e-04 1.000e-04 2.000e-04 7.000e-04 1.200e-03 5.200e-03
 1.129e-01]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch fp32: 0.7 vs 0.75
  P(drop) JAX fast: [0.0303 0.0111 0.0093 0.0074 0.0057 0.0049 0.0044 0.004  0.0042 0.005
 0.0061 0.0083 0.0137 0.0265 0.1533]
  P(drop) torch fp32: [0.03   0.011  0.0092 0.0073 0.0056 0.0049 0.0043 0.004  0.0042 0.005
 0.0061 0.0082 0.0136 0.0264 0.1552]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch fp32: 0.9 vs 0.95
  P(drop) JAX fast: [0.0093 0.0014 0.0011 0.0009 0.0007 0.0006 0.0006 0.0005 0.0006 0.0006
 0.0009 0.0014 0.0027 0.0051 0.0418]
  P(drop) torch fp32: [0.009  0.0014 0.0011 0.0009 0.0007 0.0006 0.0006 0.0005 0.0006 0.0006
 0.0009 0.0014 0.0027 0.005  0.0425]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch default: 0.7 vs 0.75
  P(drop) JAX fast: [0.0303 0.0111 0.0093 0.0074 0.0057 0.0049 0.0044 0.004  0.0042 0.005
 0.0061 0.0083 0.0137 0.0265 0.1533]
  P(drop) torch default: [0.0302 0.0111 0.0092 0.0073 0.0056 0.0049 0.0043 0.0039 0.0041 0.005
 0.0061 0.0082 0.0136 0.0265 0.1537]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch default: 0.9 vs 0.95
  P(drop) JAX fast: [0.0093 0.0014 0.0011 0.0009 0.0007 0.0006 0.0006 0.0005 0.0006 0.0006
 0.0009 0.0014 0.0027 0.0051 0.0418]
  P(drop) torch default: [0.0091 0.0013 0.0011 0.0009 0.0007 0.0006 0.0005 0.0005 0.0006 0.0006
 0.0009 0.0014 0.0027 0.005  0.0415]
- pouch n=2000 (drops 9.7%) map map | JAX fast[int8 bundle] vs torch default: 0.9 vs 0.95
  P(drop) JAX fast[int8 bundle]: [0.0092 0.0014 0.0011 0.0009 0.0007 0.0006 0.0006 0.0005 0.0006 0.0006
 0.0009 0.0014 0.0027 0.005  0.0416]
  P(drop) torch default: [0.0091 0.0013 0.0011 0.0009 0.0007 0.0006 0.0005 0.0005 0.0006 0.0006
 0.0009 0.0014 0.0027 0.005  0.0415]
- pouch n=2000 (drops 9.7%) map map | JAX fp32 vs torch default: 0.35 vs nan
  P(drop) JAX fp32: [0.0072 0.005  0.0096 0.0145 0.0289 0.0615 0.1325 0.2651 0.4796 0.7024
 0.8778 0.947  0.9791 0.9928 0.9929]
  P(drop) torch default: [0.0073 0.005  0.0096 0.0144 0.0289 0.0616 0.1319 0.2654 0.48   0.702
 0.8775 0.9465 0.9788 0.9928 0.9931]
- pouch n=2000 (drops 9.7%) map map | JAX fp32 vs torch default: 0.5 vs 0.45
  P(drop) JAX fp32: [0.0194 0.0066 0.0055 0.0047 0.005  0.0051 0.005  0.0052 0.0062 0.0079
 0.0087 0.0143 0.0205 0.0439 0.2652]
  P(drop) torch default: [0.0195 0.0065 0.0055 0.0047 0.005  0.0051 0.005  0.0053 0.0063 0.008
 0.0088 0.0145 0.0208 0.0448 0.2635]
- pouch n=2000 (drops 9.7%) map map | JAX fp32 vs torch default: 0.9 vs 0.85
  P(drop) JAX fp32: [0.0171 0.0037 0.0031 0.0023 0.002  0.0018 0.002  0.0016 0.0017 0.0021
 0.0026 0.003  0.005  0.0107 0.0929]
  P(drop) torch default: [0.0172 0.0037 0.0031 0.0023 0.002  0.0018 0.002  0.0016 0.0017 0.0021
 0.0027 0.003  0.0051 0.0108 0.0928]
- pouch n=2000 (drops 9.7%) map map | JAX fp32 vs torch default: 0.95 vs 1.0
  P(drop) JAX fp32: [0.0001 0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.0001 0.0002 0.005 ]
  P(drop) torch default: [0.0001 0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.0001 0.0002 0.0049]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch fp32: nan vs 0.35
  P(drop) JAX fast: [0.0074 0.0051 0.0098 0.0149 0.0292 0.0619 0.1335 0.2659 0.4808 0.7018
 0.8781 0.9469 0.9788 0.9927 0.993 ]
  P(drop) torch fp32: [0.0072 0.005  0.0096 0.0145 0.0289 0.0615 0.1325 0.2651 0.4796 0.7024
 0.8778 0.947  0.9791 0.9928 0.9929]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch fp32: 0.45 vs 0.5
  P(drop) JAX fast: [0.0194 0.0066 0.0055 0.0047 0.005  0.0051 0.005  0.0053 0.0063 0.008
 0.0088 0.0144 0.0205 0.0439 0.2616]
  P(drop) torch fp32: [0.0194 0.0066 0.0055 0.0047 0.005  0.0051 0.005  0.0052 0.0062 0.0079
 0.0087 0.0143 0.0205 0.0439 0.2652]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch fp32: 0.85 vs 0.9
  P(drop) JAX fast: [0.0173 0.0038 0.0032 0.0023 0.002  0.0019 0.002  0.0016 0.0017 0.0021
 0.0027 0.003  0.0051 0.0108 0.0914]
  P(drop) torch fp32: [0.0171 0.0037 0.0031 0.0023 0.002  0.0018 0.002  0.0016 0.0017 0.0021
 0.0026 0.003  0.005  0.0107 0.0929]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch fp32: 1.0 vs 0.95
  P(drop) JAX fast: [0.0001 0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.0001 0.0002 0.0048]
  P(drop) torch fp32: [0.0001 0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.0001 0.0002 0.005 ]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch fp32: 0.75 vs 0.8
  P(drop) JAX fast: [0.0175 0.0068 0.0069 0.0067 0.0056 0.005  0.005  0.0048 0.0046 0.0046
 0.0051 0.0058 0.0074 0.0098 0.0348]
  P(drop) torch fp32: [0.0173 0.0067 0.0068 0.0066 0.0055 0.005  0.0049 0.0047 0.0045 0.0045
 0.005  0.0056 0.0073 0.0096 0.0352]
- pouch n=2000 (drops 9.7%) map map | JAX fast vs torch default: 0.75 vs 0.8
  P(drop) JAX fast: [0.0175 0.0068 0.0069 0.0067 0.0056 0.005  0.005  0.0048 0.0046 0.0046
 0.0051 0.0058 0.0074 0.0098 0.0348]
  P(drop) torch default: [0.0174 0.0067 0.0068 0.0066 0.0055 0.005  0.005  0.0047 0.0045 0.0045
 0.005  0.0057 0.0074 0.0097 0.0351]
- pouch n=2000 (drops 9.7%) map map | JAX fast[int8 bundle] vs torch default: 0.5 vs 0.45
  P(drop) JAX fast[int8 bundle]: [0.0195 0.0066 0.0056 0.0047 0.005  0.0051 0.005  0.0052 0.0063 0.008
 0.0088 0.0144 0.0207 0.0439 0.2601]
  P(drop) torch default: [0.0195 0.0065 0.0055 0.0047 0.005  0.0051 0.005  0.0053 0.0063 0.008
 0.0088 0.0145 0.0208 0.0448 0.2635]
- pouch n=2000 (drops 9.7%) map map | JAX fast[int8 bundle] vs torch default: 0.75 vs 0.8
  P(drop) JAX fast[int8 bundle]: [0.0175 0.0068 0.0068 0.0066 0.0055 0.005  0.005  0.0047 0.0046 0.0045
 0.005  0.0057 0.0073 0.0096 0.0348]
  P(drop) torch default: [0.0174 0.0067 0.0068 0.0066 0.0055 0.005  0.005  0.0047 0.0045 0.0045
 0.005  0.0057 0.0074 0.0097 0.0351]
- cheese n=300 (drops 37.3%) map map | JAX fp32 vs torch default: 0.6 vs 0.65
  P(drop) JAX fp32: [5.100e-03 1.700e-03 7.000e-04 6.000e-04 6.000e-04 9.000e-04 1.600e-03
 5.000e-03 1.860e-02 7.760e-02 1.924e-01 3.949e-01 6.231e-01 7.911e-01
 9.079e-01]
  P(drop) torch default: [5.100e-03 1.700e-03 7.000e-04 6.000e-04 6.000e-04 8.000e-04 1.600e-03
 4.900e-03 1.840e-02 7.680e-02 1.908e-01 3.926e-01 6.211e-01 7.886e-01
 9.074e-01]
- cheese n=300 (drops 37.3%) map map | JAX fast vs torch default: 0.6 vs 0.65
  P(drop) JAX fast: [5.000e-03 1.700e-03 7.000e-04 6.000e-04 6.000e-04 9.000e-04 1.600e-03
 5.000e-03 1.860e-02 7.750e-02 1.920e-01 3.929e-01 6.202e-01 7.890e-01
 9.074e-01]
  P(drop) torch default: [5.100e-03 1.700e-03 7.000e-04 6.000e-04 6.000e-04 8.000e-04 1.600e-03
 4.900e-03 1.840e-02 7.680e-02 1.908e-01 3.926e-01 6.211e-01 7.886e-01
 9.074e-01]
- cheese n=1000 (drops 34.9%) map map | JAX fp32 vs torch default: 0.65 vs 0.6
  P(drop) JAX fp32: [4.000e-04 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e-04 2.000e-04
 5.000e-03 6.540e-02 2.993e-01 6.187e-01 8.592e-01 9.644e-01 9.944e-01
 9.988e-01]
  P(drop) torch default: [4.000e-04 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e-04 2.000e-04
 5.000e-03 6.630e-02 3.006e-01 6.196e-01 8.600e-01 9.644e-01 9.945e-01
 9.988e-01]
- cheese n=1000 (drops 34.9%) map map | JAX fast vs torch fp32: 0.35 vs 0.4
  P(drop) JAX fast: [0.014  0.0027 0.005  0.0111 0.0413 0.1788 0.4853 0.8516 0.9673 0.9935
 0.9983 0.9991 0.9992 0.9988 0.9838]
  P(drop) torch fp32: [0.0135 0.0026 0.005  0.011  0.0407 0.1772 0.4828 0.8503 0.9669 0.9935
 0.9983 0.9991 0.9992 0.9988 0.9838]
- cheese n=1000 (drops 34.9%) map map | JAX fast vs torch default: 0.35 vs 0.4
  P(drop) JAX fast: [0.014  0.0027 0.005  0.0111 0.0413 0.1788 0.4853 0.8516 0.9673 0.9935
 0.9983 0.9991 0.9992 0.9988 0.9838]
  P(drop) torch default: [0.0136 0.0026 0.0049 0.0108 0.0401 0.1748 0.4775 0.8484 0.9662 0.9933
 0.9983 0.9991 0.9992 0.9988 0.9839]
- cheese n=1000 (drops 34.9%) map map | JAX fast vs torch default: 0.65 vs 0.6
  P(drop) JAX fast: [4.000e-04 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e-04 2.000e-04
 5.000e-03 6.550e-02 2.994e-01 6.191e-01 8.587e-01 9.639e-01 9.943e-01
 9.988e-01]
  P(drop) torch default: [4.000e-04 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e-04 2.000e-04
 5.000e-03 6.630e-02 3.006e-01 6.196e-01 8.600e-01 9.644e-01 9.945e-01
 9.988e-01]
- cheese n=1000 (drops 34.9%) map map | JAX fp32 vs torch default: 0.6 vs 0.55
  P(drop) JAX fp32: [3.700e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 1.000e-03 5.000e-03
 8.440e-02 3.753e-01 7.344e-01 9.224e-01 9.815e-01 9.959e-01 9.987e-01
 9.898e-01]
  P(drop) torch default: [3.600e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 1.000e-03 5.000e-03
 8.570e-02 3.773e-01 7.349e-01 9.227e-01 9.816e-01 9.959e-01 9.987e-01
 9.899e-01]
- cheese n=1000 (drops 34.9%) map map | JAX fast vs torch default: 0.6 vs 0.55
  P(drop) JAX fast: [3.800e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 1.000e-03 5.000e-03
 8.490e-02 3.759e-01 7.348e-01 9.228e-01 9.814e-01 9.958e-01 9.987e-01
 9.899e-01]
  P(drop) torch default: [3.600e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 1.000e-03 5.000e-03
 8.570e-02 3.773e-01 7.349e-01 9.227e-01 9.816e-01 9.959e-01 9.987e-01
 9.899e-01]
- cheese n=1000 (drops 34.9%) map map | JAX fast[int8 bundle] vs torch default: 0.6 vs 0.55
  P(drop) JAX fast[int8 bundle]: [3.700e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 1.000e-03 5.000e-03
 8.510e-02 3.763e-01 7.343e-01 9.227e-01 9.815e-01 9.958e-01 9.987e-01
 9.897e-01]
  P(drop) torch default: [3.600e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 1.000e-03 5.000e-03
 8.570e-02 3.773e-01 7.349e-01 9.227e-01 9.816e-01 9.959e-01 9.987e-01
 9.899e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fp32 vs torch default: 0.45 vs 0.5
  P(drop) JAX fp32: [1.330e-02 5.000e-04 7.000e-04 1.200e-03 5.100e-03 4.260e-02 3.253e-01
 7.529e-01 9.298e-01 9.858e-01 9.968e-01 9.991e-01 9.995e-01 9.995e-01
 9.933e-01]
  P(drop) torch default: [1.280e-02 5.000e-04 7.000e-04 1.100e-03 5.000e-03 4.220e-02 3.224e-01
 7.502e-01 9.291e-01 9.858e-01 9.968e-01 9.991e-01 9.995e-01 9.995e-01
 9.932e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fp32 vs torch default: 0.5 vs 0.55
  P(drop) JAX fp32: [7.900e-03 2.000e-04 4.000e-04 5.000e-04 1.400e-03 5.000e-03 4.010e-02
 3.568e-01 7.325e-01 9.217e-01 9.811e-01 9.958e-01 9.987e-01 9.993e-01
 9.956e-01]
  P(drop) torch default: [7.800e-03 2.000e-04 4.000e-04 5.000e-04 1.400e-03 4.900e-03 3.940e-02
 3.542e-01 7.316e-01 9.214e-01 9.812e-01 9.958e-01 9.988e-01 9.993e-01
 9.956e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fp32 vs torch default: 0.55 vs 0.6
  P(drop) JAX fp32: [5.100e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 9.000e-04 5.000e-03
 9.150e-02 3.999e-01 7.321e-01 9.058e-01 9.742e-01 9.952e-01 9.989e-01
 9.952e-01]
  P(drop) torch default: [5.000e-03 1.000e-04 1.000e-04 1.000e-04 3.000e-04 8.000e-04 5.000e-03
 9.120e-02 3.999e-01 7.324e-01 9.060e-01 9.745e-01 9.952e-01 9.989e-01
 9.952e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fast vs torch default: 0.45 vs 0.5
  P(drop) JAX fast: [1.380e-02 5.000e-04 7.000e-04 1.200e-03 5.200e-03 4.340e-02 3.281e-01
 7.542e-01 9.299e-01 9.858e-01 9.968e-01 9.991e-01 9.995e-01 9.995e-01
 9.939e-01]
  P(drop) torch default: [1.280e-02 5.000e-04 7.000e-04 1.100e-03 5.000e-03 4.220e-02 3.224e-01
 7.502e-01 9.291e-01 9.858e-01 9.968e-01 9.991e-01 9.995e-01 9.995e-01
 9.932e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fast vs torch default: 0.5 vs 0.55
  P(drop) JAX fast: [8.400e-03 2.000e-04 4.000e-04 5.000e-04 1.400e-03 5.100e-03 4.060e-02
 3.583e-01 7.325e-01 9.221e-01 9.812e-01 9.957e-01 9.987e-01 9.993e-01
 9.960e-01]
  P(drop) torch default: [7.800e-03 2.000e-04 4.000e-04 5.000e-04 1.400e-03 4.900e-03 3.940e-02
 3.542e-01 7.316e-01 9.214e-01 9.812e-01 9.958e-01 9.988e-01 9.993e-01
 9.956e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fast vs torch default: 0.55 vs 0.6
  P(drop) JAX fast: [5.400e-03 1.000e-04 1.000e-04 2.000e-04 3.000e-04 9.000e-04 5.100e-03
 9.170e-02 3.993e-01 7.321e-01 9.057e-01 9.740e-01 9.951e-01 9.989e-01
 9.957e-01]
  P(drop) torch default: [5.000e-03 1.000e-04 1.000e-04 1.000e-04 3.000e-04 8.000e-04 5.000e-03
 9.120e-02 3.999e-01 7.324e-01 9.060e-01 9.745e-01 9.952e-01 9.989e-01
 9.952e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fast[int8 bundle] vs torch default: 0.45 vs 0.5
  P(drop) JAX fast[int8 bundle]: [1.280e-02 5.000e-04 7.000e-04 1.200e-03 5.100e-03 4.280e-02 3.249e-01
 7.502e-01 9.294e-01 9.858e-01 9.968e-01 9.991e-01 9.995e-01 9.995e-01
 9.931e-01]
  P(drop) torch default: [1.280e-02 5.000e-04 7.000e-04 1.100e-03 5.000e-03 4.220e-02 3.224e-01
 7.502e-01 9.291e-01 9.858e-01 9.968e-01 9.991e-01 9.995e-01 9.995e-01
 9.932e-01]
- cheese n=2000 (drops 34.4%) map map | JAX fast[int8 bundle] vs torch default: 0.5 vs 0.55
  P(drop) JAX fast[int8 bundle]: [7.700e-03 2.000e-04 4.000e-04 5.000e-04 1.400e-03 5.000e-03 4.010e-02
 3.560e-01 7.316e-01 9.214e-01 9.813e-01 9.958e-01 9.988e-01 9.993e-01
 9.956e-01]
  P(drop) torch default: [7.800e-03 2.000e-04 4.000e-04 5.000e-04 1.400e-03 4.900e-03 3.940e-02
 3.542e-01 7.316e-01 9.214e-01 9.812e-01 9.958e-01 9.988e-01 9.993e-01
 9.956e-01]
```
