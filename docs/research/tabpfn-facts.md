# TabPFN facts (checked against source, 2026-11-21)

Checked against `tabpfn` 10.2.0, `tabpfn-client` 0.10.1, `tabpfn-extensions` 0.6.3, docs.priorlabs.ai and the TabPFN-3.5 tech report (arXiv 2609.17895). Nothing here was benchmarked by us. **Measure latency on our own hardware before designing around it.**

## Versions and access
- Available: v2, v2.5, v2.6, v3, **v3.5** (default locally) and **v3.5-Fast**. Switch with `TABPFN_MODEL_VERSION` or `create_default_for_version(ModelVersion.V3_5_FAST)`.
- **Open weights:** 3.5 and 3.5-Fast, one multitask checkpoint for classification and regression (`Prior-Labs/tabpfn_3_5`).
- **API only (`tabpfn-client`):** 3.5-**Plus** (API default, best text handling) and 3.5-**Thinking**.
- **Weights license:** non-commercial. Fine-tuned models count as derivatives and are also non-commercial. Fine for a hackathon.
- **Headless or cloud GPU setup:** accept the license at platform.priorlabs.ai → Licenses, then set `TABPFN_TOKEN=<key>` and `TABPFN_NO_BROWSER=1`. The weights are cached after the first download.

## Architecture (3.5)
The 3.5 model works on **one token per row** in its main stage, not per cell as in v2. It runs four stages:
1. **Cell embedding:** Fourier features of the value, an in-context ECDF rank, and a NaN flag.
2. **Per-column distribution embedder:** induced self-attention over the rows.
3. **Per-row aggregator:** RoPE attention over the features compresses each row into 8 CLS tokens.
4. **In-context-learning transformer over rows:** 24 layers, width 1024, 16 heads. Test rows attend only to training rows.

**Size:** 3.5 has 220M parameters; Fast has 84M.

**Limits:** up to 1M training rows, 20k features (6k recommended) and 160 classes. The model has a native text column (best on Plus).

**Regression output** is a bar distribution: logits over bins.

**No native multi-output.** Use `MultiOutputRegressor` (one model per target) or the cookbook's chained or copula approaches.

**Time series:** the separate `tabpfn-time-series` package (TabPFN-TS, defaults to 3.5).

## Outputs we will use
- `predict(output_type=...)`: `mean`, `median`, `mode`, `quantiles` (with `quantiles=[...]`), or `full`. `full` returns the logits plus a `criterion` with `.sample()`, `.icdf`, `.cdf`, `.pdf`, `.variance`, `.ei`, `.pi` and `.ucb`.
- With `.sample()` we can roll out several possible futures; with `.ucb`, `.ei` and quantiles we can make risk-aware choices.

## Speed and GPU
- **`fit_mode="fit_with_cache"`** builds a KV cache at fit time, so `predict` only computes the test rows. In one docs example (4,750 × 20 table, 250 test rows): 0.91 s uncached vs 0.17 s cached.
- **Other speed settings:** Fast checkpoint, `n_estimators=1`, int8 or fp8 KV cache, `keep_cache_on_device`, `enable_torch_compile=True` (torch ≥ 2.6), batching all queries into one `predict` call.
- **Memory (A100, 3.5, 200 features):** about 3 GB at 1k–10k training rows, about 7 GB at 200k. An RTX 4050 with 6 GB should handle small and medium contexts.
- **CPU:** works, but limited to 5k samples.
- **The API is too slow for control loops:** 60 predicts per minute, a minimum charge of 10k tokens per call, and about 5M tokens per day by default. Use it only offline or for occasional calls.

## Fine-tuning and gradients
- **Official:** `tabpfn.finetuning.FinetunedTabPFN{Classifier,Regressor}`.
  - Full-weight AdamW at lr 1e-5; no LoRA.
  - Trains by meta-learning on chunks of our data split 80/20 into context and query.
  - Loss: cross-entropy, or the bar-distribution NLL for regression.
  - Supports DDP. The examples recommend an **80 GB GPU** and at least 1k rows. No published gains or timings.
- **Differentiable path:** `differentiable_input=True` together with `fit_with_differentiable_input()` lets gradients flow to the inputs and to the context rows (`examples/input_gradients.py`, `examples/prompt_tuning_*.py`). In principle any loss can be backpropagated through the predictive distribution, including RL objectives. Nothing RL-specific exists in the repos.

## Ecosystem
- **`tabpfn-extensions` modules:**
  - `bayesian_optimization` (EI, differentiable in x)
  - `unsupervised` (density and outlier scores, synthetic data)
  - `embedding`
  - `interpretability` (SHAP, and which training rows drive a prediction)
  - `image` (DINOv3 features)
  - `many_class`
- **Removed:** AutoTabPFN, RF-PFN.
- **Distillation to an MLP or trees** is enterprise-only; we could distil it ourselves.
- **JAX:** no official port. The only one is a community port of the educational nanoTabPFN. The full model is PyTorch only.
