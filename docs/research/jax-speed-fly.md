# JAX port vs torch at the fly controller's context size (measured 2026-11-22)

**Question:** does the JAX port of TabPFN-3.5-Fast (`jax_tabpfn/`, from `origin/main` PR #1 and #2) speed up our closed-loop decisions, with the same answers?

**Setup:** Nebius H100 80 GB (driver 580) · torch 2.14.1+cu130 + tabpfn 10.2.0 · JAX 0.12.1 (CUDA 13) · TabPFN-3.5-Fast, 1 estimator, cached context · script `scripts/jax_speed_check.py`, raw results `results/jax_speed/`.

## Caveat first: a proxy, not our exact model

The JAX port supports **classification only**, all-numeric columns. Our controller is a **regressor** with `action_id` as a **categorical** column. So the test uses a TabPFNClassifier on the controller's exact table (N steps × 12 action rows, 105 columns, `action_id` numeric) with each standardised action value binned into 10 classes. The expensive part, attention of the 12 query rows over the cached context, is identical; the output head differs. No flights were flown with JAX.

## Result: 12 query rows (one controller decision), median of 100 calls

| Context | Torch default (fp16) | Torch fp32 | JAX fast (fp16) | **JAX fp32** | Our torch regressor (reference) |
|---|---|---|---|---|---|
| 4,096 steps · 49k rows | 15.1 ms | 59.9 ms | 5.3 ms | **3.45 ms** | 30.8 ms |
| 32,768 steps · 393k rows | 42.4 ms | 400.6 ms | 32.0 ms | **13.7 ms** | 55.4 ms |

| Accuracy vs torch fp32 | 4k | 32k |
|---|---|---|
| JAX fp32, max \|Δp\| | **2.4e-5** | **2.7e-5** |
| JAX fp32, same predicted class | 12 / 12 | 12 / 12 |
| JAX fast, max \|Δp\| | 2.1e-2 | 2.4e-2 |
| JAX fast, same predicted class | 12 / 12 | **10 / 12** |

**Speed-up of JAX fp32:** 4.4× (4k) and **3.1× (32k)** over torch's default, 17× and 29× over torch fp32, with fp32-exact answers.

## One-time costs

| | 4k | 32k |
|---|---|---|
| Torch fit, default / fp32 (the export needs the fp32 cache) | 2.6 s / 4.6 s | 24.6 s / **187 s** |
| Export to bundle | 0.3 s | 2.8 s (2.4 GB) |
| JAX load + context prep | 2–5 s | 2–3 s |
| JAX first call (XLA compile; cached on disk after) | 8–39 s | 3–20 s |

## Verdict: good, use JAX fp32, not fast mode

- **JAX fp32 is both exact and the fastest option** at our query size. With only 12 query rows the work is dominated by per-call overhead and memory traffic, which XLA's fused program cuts. The fp16 "fast" mode is slower here and drifts enough to flip 2 of 12 decisions at 32k.
- **Projected for our regressor:** torch spends ~13–16 ms more on the regressor than on the classifier (its 5,000-bin head and post-processing). Ported to JAX, that head is one small MLP and a softmax over 5,000 bins for 12 rows, likely about 1 ms. So a JAX regressor at 32k should take **~15 ms instead of 55 ms, about 3.5–4× faster**. A 1 s flight (5,000 decisions) would take ~1.3 min instead of 4.6 min.
- **Still not real time for the fly:** 13.7 ms is ~70× the 0.2 ms control step. But it is inside a **50 Hz robot loop (20 ms)** with a 393k-row context, which torch's default (42 ms) is not. That matters for the real-robot story.

## To use it for flying (not done)

Port the regression head (`heads.mlp_regression`, `output_projection`, bar-distribution mean over `regression_borders`, target de-standardisation) and TabPFN's `ordinal_shuffled` encoding of the categorical `action_id` into `jax_tabpfn/`, add `bundle_from_regressor`, prove parity with torch, then fly the sweep paths. Estimated 1.5–2.5 h.
