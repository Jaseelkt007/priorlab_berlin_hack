# Handoff: port TabPFN-3.5-Fast inference to JAX

Written 2026-11-21 for a parallel Claude Code session. Read `CLAUDE.md`, `docs/research/tabpfn-latency.md` and `docs/research/tabpfn-facts.md` first.

## Why

One per-pick query (15 candidate speeds, cached context of 1,000 rows, 1 estimator, TabPFN-3.5-Fast, RTX 4050) takes about **25 ms wall time but only 5.8 ms of GPU compute**. The rest is the CPU launching **1,671 small kernels per call**, plus Python preprocessing.

TabPFN's built-in `enable_torch_compile` did not help (32 ms), and fp16 was worse (37 ms). A `jax.jit`-compiled forward pass with static shapes should let XLA fuse this into a few kernels. **Target: under 8 ms per 15-row query, with probabilities identical to PyTorch within 1e-4.**

The main session is trying the PyTorch alternative (CUDA-graph capture) at the same time. Whichever wins goes into the live demo.

## Rules

- **License:** TabPFN-3.5 weights are non-commercial (accepted for this account). A port for research use is fine. **Never commit weights or converted weights.** Load them at run time from `~/.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors` (the full model is `tabpfn-v3.5-20260909.safetensors`).
- **Isolation, so the two sessions don't collide:**
  - Work on branch `jax-port`, ideally in a git worktree.
  - Put all code under `jax_tabpfn/`.
  - Do not edit `sim/`, `scripts/bench_tabpfn.py`, `docs/explainer/` or `docs/PROBLEM_SPEC.md`.
- **Separate environment:** `.venv-jax` (gitignored), so JAX's CUDA libraries do not clash with torch's cu132 build in `.venv`. The owner has run JAX 0.12 with CUDA on this laptop before.
- Hackathon rules apply: log every shortcut in `DEBT.md` on your branch.

## Where things are

- **Reference implementation:** `.venv/lib/python3.12/site-packages/tabpfn/architectures/tabpfn_v3_5.py` (about 3,300 lines). Key classes:
  - `TabPFNV3p5.forward` (around line 2252), `_stages_0_to_2` (around 2769), `_process_row_chunk` (around 2965)
  - `FeatureDistributionEmbedder` (1761), `InducedSelfAttentionBlock` (1639), `ColumnAggregator` (1848)
  - `ICLTransformerBlock` (1526), `ICLAttention` (1171), `MultiTaskHeads` (1983)
  - ECDF helpers `_build_ecdf_context`, `_in_context_ecdf`, `_ecdf_fourier_features` (3095–3260)
  - KV cache class `TabPFNV3p5Cache`
- **Inference driver:**
  - `tabpfn/inference.py` (`_call_model`, `run`, around lines 1434–1514)
  - `tabpfn/classifier.py` (`_raw_predict`, `forward`, around 1850–2181)
  - Preprocessing happens before the model call; see the ensemble/preprocessing configs.
- **Toy data and benchmark harness:**
  - `sim/delta_toy.py`: `make_log`, `run_picks`, `FEATURES`, `SPEEDS`
  - `scripts/bench_tabpfn.py`: copy its setup to get the same context and query.

## Suggested plan

1. **Pin the exact call.** Fit `TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST, n_estimators=1, device="cuda", fit_mode="fit_with_cache", random_state=0)` on 1,000 toy rows. Hook the model call to dump:
   - the preprocessed query tensor,
   - the cache contents,
   - the output logits.

   These are your golden test vectors. Store them in `jax_tabpfn/golden/` (gitignored if large).
2. **Port the hot path first.** With the context cached, a query only runs the query rows through the column embedding (stages 0–2) and the ICL transformer, attending to the cached K/V, then the head. Port exactly that path, taking the exported cache as input.
3. **Load the safetensors weights** into a pytree. Map parameter names one to one and assert every tensor is used.
4. **Parity tests:** logits and probabilities against the golden vectors (max abs diff ≤ 1e-4 in fp32). Then test on fresh queries.
5. **Benchmark:** `jax.jit` with static shapes (15 rows), warm-up, `block_until_ready`. Report median and p95 over 100 calls against the 25 ms baseline and 5.8 ms GPU floor. Also report the 3,000-row speed-map shape.
6. **Optional:** port the context-side forward too (building the cache), so the whole refit + map refresh runs in JAX.

## Report back

Write `docs/research/tabpfn-jax-port.md`:
- what was ported,
- the parity numbers,
- the latency table,
- known gaps.

Then open a PR from `jax-port` into `main`.
