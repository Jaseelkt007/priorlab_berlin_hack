# TabPFN-3.5-Fast query path in JAX (measured 2026-11-21)

Branch `jax-port`, code in `jax_tabpfn/`. Brief: `docs/handoff/jax-port.md`.

**Setup:** RTX 4050 Laptop 6 GB (WSL2) · JAX 0.12.1 (CUDA 13) · torch 2.15 + tabpfn 10.2.0 for reference · TabPFN-3.5-Fast · 1 estimator · 1,000-row cached context of the toy pick log (`sim/delta_toy.py`, same data as `scripts/bench_tabpfn.py`).

## Result

| Raw rows → probabilities | JAX fast (fp16) | JAX fp32 | torch + CUDA graph | torch normal (default fp16) | torch fp32 |
|---|---|---|---|---|---|
| One pick, 15 rows | **3.7 ms** (p95 4.8) | 4.7 ms (p95 5.9) | 6.9 ms (p95 8.4) | 30.9 ms (p95 46) | 24.3 ms |
| Speed map, 3,000 rows | **52–57 ms** | 152 ms | 81.8 ms | 85.8 ms | 155 ms |
| Max \|Δp\| vs torch fp32 | 7.0e-3 | **8.4e-6** | — | 7.1e-3 | 0 |

- All numbers are end to end, the same work as `clf.predict_proba`: CPU preprocessing, upload, model, softmax and download. Medians over 200 calls (pick) and 30 calls (map).
- The CUDA-graph column comes from `docs/research/tabpfn-latency.md`. The torch normal and fp32 columns were re-measured in this session on the same queries.
- **Parity target met:** the fp32 mode matches PyTorch fp32 within 8.4e-6 on probabilities, against a target of 1e-4. Tested on 7 fresh 15-row picks and the 3,000-row map, and across 4 refitted contexts.
- **The fast mode drifts no more than TabPFN's own GPU default.** It differs from fp32 by up to 7.0e-3 on the map; TabPFN's production path (fp16 autocast, int8 KV) differs by 7.1e-3.
- **Per-pick:**
  - fast mode: 3.7 ms, 1.9× faster than the CUDA graph;
  - fp32 mode: 4.7 ms with exact probabilities.

  Either mode fits the 10–30 ms window between touch and acceleration with room to spare.
- **Speed map: 1.6× faster than torch** (52–57 ms against 82–86 ms), because XLA cuts the real GPU compute:
  - 270 kernels per call instead of 1,671;
  - cuDNN flash attention;
  - the ICL matmuls at about 21 TFLOPS, near the fp16 peak of this GPU.

  The map refresh (torch refit ~0.3 s, export ~3 ms, context swap ~20 ms, map ~55 ms) is under 0.4 s.
- Neither fp32 path gains on the map. Both are compute-bound at about 152–155 ms, so the map gain needs fp16.

## What was ported

The cached-context predict path of `tabpfn/architectures/tabpfn_v3_5.py` (`x_is_test_only=True`, multiclass head), plus the per-call preprocessing TabPFN runs around it:

| Step | Where | File |
|---|---|---|
| Drop constant columns, cast to fp32 | CPU (numpy) | `runtime.py` `model_input` |
| Fingerprint column: SHA-256 of each rounded row + salt | CPU (hashlib, 0.02 ms for 15 rows, 3 ms for 3,000) | `runtime.py` `fingerprint` |
| Column shuffle, soft outlier clip | jitted | `runtime.py` `device_preprocess` |
| NaN/inf indicators, mean imputation, ECDF midrank against the cached buckets, standard scaling, circular feature grouping | jitted | `model.py` `preprocess`, `group_features` |
| Fourier + metadata cell embedding | jitted | `model.py` `x_embed` |
| Distribution embedder: `cross_attn_block2` ×3 against the cached inducing states | jitted | `model.py` `dist_block` |
| Column aggregator: 2 RoPE self-attention blocks + CLS readout block | jitted | `model.py` `agg_self_block`, `agg_readout_block` |
| ICL transformer ×8 against the cached K/V (MQA, softmax-scaling MLP) | jitted | `model.py` `icl_block` |
| Pre-head MLP + many-class decoder; class permutation; softmax | jitted | `model.py` `decoder`, `runtime.py` |

- **Context work is folded once per refit** (`prepare_context`), including the K/V projections of the cached inducing states, which torch recomputes on every query.
- **The class count is a static argument,** so there is no `.item()` sync. A new class count compiles once.
- **Every context array is a jit argument, not a baked-in constant.** A refit with the same context size reuses the compiled program, and nothing reads freed memory (the `y` trap from the CUDA-graph spike does not apply).
- **Weights:** loaded at run time from `~/.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors` into a flat pytree keyed by the torch names. `parity.py` traces the forward with a tracking dict and asserts the following:
  - all 307 tensors are accounted for;
  - the query path reads 205 of them;
  - the other 102 are on an explicit list, and a check confirms the query path never reads them. They are:
    - the regression head;
    - the train-label encoders;
    - `cross_attn_block1` and the inducing vectors;
    - the ICL K/V projections and `k_norm`;
    - the decoder key projection.

    All of these are already folded into the exported cache.
- **Fast mode** (`JaxTabPFN(..., fast=True)`):
  - fp16 matmul operands with fp32 accumulation;
  - `jax.nn.dot_product_attention(implementation="cudnn")` for every attention;
  - fp32 residual stream, norms, softmax-scaling MLPs and ECDF.

  This mirrors what TabPFN's own fp16 autocast does. Notes:
  - bf16 was tried and drifts 4× more (2.8e-2), so fast mode uses fp16;
  - an fp32 einsum attention is memory-bound here: the ICL score tensor is 16 × 3000 × 1000 floats per layer, so flash attention matters most.

## How to run

```bash
uv venv -p 3.12 .venv-jax && VIRTUAL_ENV=.venv-jax uv pip install -r jax_tabpfn/requirements.txt
../priorlab_berlin_hack/.venv/bin/python -m jax_tabpfn.dump_golden --fp32   # torch: bundle + goldens
../priorlab_berlin_hack/.venv/bin/python -m jax_tabpfn.dump_golden          # torch default-precision goldens
.venv-jax/bin/python -m jax_tabpfn.parity     # parity table, exits 1 on failure
.venv-jax/bin/python -m jax_tabpfn.bench      # latency table
PYTHONPATH=.:../priorlab_berlin_hack/.venv/lib/python3.12/site-packages \
  .venv-jax/bin/python -m jax_tabpfn.refit_demo   # torch fits, JAX answers, one process
```

Use from code:

```python
from jax_tabpfn.runtime import JaxTabPFN      # import before torch
from jax_tabpfn.export import bundle_from_classifier
clf.fit(X, y)                                 # torch, fit_with_cache, inference_precision=torch.float32, kv_cache_precision="auto"
jm = JaxTabPFN(bundle_from_classifier(clf))   # first time: loads weights, compiles
jm.update_context(bundle_from_classifier(clf))  # after each refit: ~20 ms, no recompile
proba = jm.predict_proba(X_rows)              # same columns as fit
```

- **torch and JAX run in one process on this GPU**, as `refit_demo.py` shows: fit in torch, export the cache in memory (~3 ms), query in JAX.
- **The export reads an unquantised fp32 cache** (`kv_cache_precision="auto"`, `inference_precision=torch.float32`). An int8 cache is dequantised if given.
- **Compiling** with XLA's autotuning takes 15–110 s per new shape. Programs go to a persistent cache (`~/.cache/jax_tabpfn/xla`), so later processes make their first call in about 0.5 s. Warm up the demo shapes once (`jm.warmup(15, 3000)`) before going on stage.
- **Row counts are padded** to 16, 32, …, 1024, then to multiples of 1024. Test rows never attend to each other, so the padding is exact and odd query sizes do not recompile. 3,000 rows run as 3,072.

## Refit check (`refit_demo.py`)

4 refits on different 1,000-row windows. Each row lists the time for one step, then the max |Δp| against torch fp32 on 5 fresh picks:

| Refit | torch fit | export | JAX context swap | JAX fp32 \|Δp\| | JAX fast \|Δp\| |
|---|---|---|---|---|---|
| 1 (cold) | 3.4 s | 11 ms | 820 ms (weights + compile cache) | 3.0e-6 | 4.1e-3 |
| 2–4 | 0.19–0.70 s | 2–3 ms | 18–20 ms | ≤ 1.3e-5 | ≤ 6.5e-3 |

## Known gaps

- **Preprocessing coverage:**
  - **Supported:** what the 3.5 default config does on an all-numeric table (transform `none`, no categoricals, 1 estimator, temperature 1, no balancing).
  - **Refused:** `export.py` raises on anything else rather than give different numbers.
  - **Not ported:** quantile/SVD transforms, categorical encoding, ensembles.
- **Context size is a static shape.** Each new context row count (or column count) compiles once. Use a fixed window (e.g. the latest 1,000 rows) for the live loop.
- **The context-side forward (building the cache) is not ported.** Refit stays in torch (~0.3 s), which keeps the cache bit-identical to TabPFN's.
- **Only TabPFN-3.5-Fast is tested.** The full 3.5 checkpoint has the same structure (24 ICL layers; the layer count is read from the cache) but has no goldens yet.
- **Tail latency on WSL:**
  - p95 and p99 stay within 1.3× and 1.6× of the median in the final runs;
  - one earlier run showed isolated 25–110 ms spikes in a 300-call loop;
  - torch shows the same pattern (p95 46 ms against a 31 ms median).

  This is likely Windows GPU scheduling. Re-measure on the demo machine.
- **The ~5 ms head-dim-16 attentions in the distribution embedder and aggregator** are the main remaining non-ICL cost. An einsum for the 18-token aggregator attention saves about 1 ms; not worth it yet.
