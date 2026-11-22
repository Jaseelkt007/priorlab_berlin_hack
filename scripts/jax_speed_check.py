"""Speed check: torch TabPFN vs the JAX port (jax_tabpfn/) at the fly controller's context size.

The JAX port only covers classification, so this times a TabPFNClassifier fitted on the
controller's exact table (N recorded steps x 12 action rows, 105 columns: 104 senses +
action_id as a numeric column) with each action value binned into 10 classes. The cost
that matters, attention of 12 query rows over the cached context, is the same as for our
regressor; the output head differs (10-class decoder vs 5,000-bin regression head).

    # torch env: fit, time torch, export the context bundle
    MUJOCO_GL=egl .venv-fly/bin/python scripts/jax_speed_check.py export --steps 32768
    # JAX env: parity vs torch fp32 on the same 12 rows, then latency
    .venv-jax/bin/python scripts/jax_speed_check.py bench --steps 32768

Writes results/jax_speed/ctx<N>_{torch,jax}.json. Bundles go to /tmp (they hold training rows).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/jax_speed"
BUNDLE = Path("/tmp/jax_speed_bundles")
N_QUERY = 12      # one controller decision = 12 rows
N_CLASSES = 10


def _timeit(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return dict(median_ms=round(float(np.median(ts)), 2), p95_ms=round(float(np.percentile(ts, 95)), 2), n=n)


def table(steps):
    """Context rows (X, y-classes) exactly as TabPFNController builds them, plus 12 query rows."""
    sys.path.insert(0, str(ROOT / "collect"))
    from closed_loop import recorded_steps
    obs, act, _ = recorded_steps()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(obs), steps, replace=False)
    o, a = obs[pick], act[pick]
    n_act = a.shape[1]
    z = ((a - a.mean(0)) / np.maximum(a.std(0), 1e-6)).reshape(-1)
    edges = np.quantile(z, np.linspace(0, 1, N_CLASSES + 1)[1:-1])
    X = np.column_stack([np.repeat(o, n_act, 0), np.tile(np.arange(n_act), len(o))]).astype(np.float32)
    y = np.digitize(z, edges).astype(np.int64)
    q_obs = obs[rng.integers(len(obs))]
    Xq = np.column_stack([np.repeat(q_obs[None], n_act, 0), np.arange(n_act)]).astype(np.float32)
    return X, y, Xq


def export(args):
    import torch
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    sys.path.insert(0, str(ROOT))
    from jax_tabpfn.export import export_bundle

    X, y, Xq = table(args.steps)
    res = dict(steps=args.steps, rows=len(X), cols=X.shape[1], query_rows=len(Xq))
    for name, kw in (("torch_default", {}),
                     ("torch_fp32", dict(inference_precision=torch.float32, kv_cache_precision="auto"))):
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST, device="cuda", n_estimators=1,
                                                          random_state=0, fit_mode="fit_with_cache", **kw)
        t0 = time.perf_counter()
        clf.fit(X, y)
        res[f"{name}_fit_s"] = round(time.perf_counter() - t0, 1)
        res[name] = _timeit(lambda: clf.predict_proba(Xq), args.calls)
        print(name, res[name], flush=True)
        if name == "torch_fp32":
            BUNDLE.mkdir(exist_ok=True)
            t0 = time.perf_counter()
            export_bundle(clf, BUNDLE / f"ctx{args.steps}.npz")
            res["export_s"] = round(time.perf_counter() - t0, 1)
            np.savez(BUNDLE / f"ctx{args.steps}_query.npz", Xq=Xq, proba=clf.predict_proba(Xq))
        res[f"{name}_peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        del clf
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ctx{args.steps}_torch.json").write_text(json.dumps(res, indent=1))


def bench(args):
    sys.path.insert(0, str(ROOT))
    from jax_tabpfn.runtime import JaxTabPFN
    q = np.load(BUNDLE / f"ctx{args.steps}_query.npz")
    res = dict(steps=args.steps)
    for name, fast in (("jax_fast", True), ("jax_fp32", False)):
        t0 = time.perf_counter()
        m = JaxTabPFN(str(BUNDLE / f"ctx{args.steps}.npz"), fast=fast)
        res[f"{name}_load_s"] = round(time.perf_counter() - t0, 1)
        t0 = time.perf_counter()
        p = m.predict_proba(q["Xq"])
        res[f"{name}_first_call_s"] = round(time.perf_counter() - t0, 1)
        res[f"{name}_max_abs_dp_vs_torch_fp32"] = float(np.abs(p - q["proba"]).max())
        res[f"{name}_same_argmax"] = int((p.argmax(1) == q["proba"].argmax(1)).sum())
        res[name] = _timeit(lambda: m.predict_proba(q["Xq"]), args.calls)
        print(name, res[name], "max|dp|", res[f"{name}_max_abs_dp_vs_torch_fp32"], flush=True)
        del m
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ctx{args.steps}_jax.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "bench"])
    ap.add_argument("--steps", type=int, default=32768)
    ap.add_argument("--calls", type=int, default=100)
    a = ap.parse_args()
    export(a) if a.mode == "export" else bench(a)
