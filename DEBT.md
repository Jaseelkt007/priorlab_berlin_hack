# Debt

Quick fixes taken to move fast. One line each: file, what was hacked, what the real fix is.
`/potato-start product` starts from this list.

- `collect/preview_flight.py`: re-implements acme's SinglePrecision/CanonicalSpec wrappers and flybody's TestPolicyWrapper inline, because `dm-acme` needs `launchpad`. Real fix: a pinned env file (environment.yml) that installs flybody[tf] reproducibly, then use their wrappers.
- `collect/closed_loop.py`: context capped at 256 steps (3,072 rows) so TabPFN 3.5 Fast fits the GTX 1050's 3 GB; 512 steps runs out of memory. Real fix: run on the RTX 4050 (6 GB) with 1-2k steps, or kv_cache_precision / memory_saving_mode.
- `collect/closed_loop.py`: with `prev_action`, the first step of an episode uses the context's mean action as the "previous action" (recorded step-0 rows are dropped). Real fix: record the env's initial action, or start the controller from the policy's first action.
- jax_tabpfn/refit_demo.py: runs torch and JAX in one interpreter by appending the torch env's site-packages to `PYTHONPATH` of `.venv-jax`. Real fix: one env with torch cu132 + `jax[cuda13]` (they coexist on the RTX 4050 as tested), once the main session agrees.
- jax_tabpfn/export.py: supports only 1 estimator, all-numeric columns and TabPFN-3.5's default preprocessing (transform `none`, no categoricals, T=1, no balancing); anything else raises. Real fix: port the remaining TabPFN preprocessing steps (quantile, SVD, categorical encoding) and ensembles.
- jax_tabpfn/runtime.py: the context size is a static shape, so a new context row count compiles once (15–110 s, then persistent cache). Real fix: mask padded train rows in the ICL/decoder attention and pad the ECDF buckets, so contexts can be bucketed like queries.
- jax_tabpfn/model.py: the fast-mode decoder asserts at most 64 classes (one flash-attention call of one-hot values). Real fix: loop over 64-class chunks like `_class_attention`.
- scripts/verify_jax_port.py: carries a copy of the CUDA-graph wrapper and `.item()` patch from `scripts/spike_cudagraph.py`, because that script runs its experiment at import. Real fix: move the wrapper into an importable module (or drop it if the JAX path wins), and have both scripts import it.
- `scripts/jax_speed_check.py`: times the JAX port with a TabPFNClassifier (actions binned into 10 classes, `action_id` numeric) as a stand-in for our regressor, because the port has no regression head or categorical encoding. Real fix: port the regression head, bar-distribution mean and `ordinal_shuffled` encoding, then fly the same paths with JAX.
