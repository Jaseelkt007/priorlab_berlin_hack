"""Spike: CUDA-graph capture of TabPFN-3.5-Fast's cached forward pass.

A per-pick call is launch-bound (about 25 ms wall vs 5.8 ms GPU, 1,671 kernel launches). Here
the model forward for a fixed query shape is captured once as a CUDA graph and replayed,
so the launches become one. Checks that probabilities match the normal path exactly.
"""
import gc
import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5
from tabpfn.constants import ModelVersion

from sim.delta_toy import FEATURES, SPEEDS, make_log, run_picks


class GraphedForward:
    """Replaces model.forward: capture per input shape, then replay."""

    def __init__(self, model: torch.nn.Module):
        self.model, self.orig, self.graphs = model, model.forward, {}
        self.enabled = True
        self.__signature__ = inspect.signature(model.forward)  # TabPFN picks kwargs from the forward signature

    def __call__(self, x, y, **kw):
        if not self.enabled:
            return self.orig(x, y, **kw)
        key = (tuple(x.shape), kw.get("task_type"))
        if key not in self.graphs:
            # every tensor input needs a fixed address: TabPFN builds y fresh on each call
            sx, sy = x.clone(), y.clone()
            skw = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.orig(sx, sy, **skw)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                so = self.orig(sx, sy, **skw)
            self.graphs[key] = (g, sx, sy, skw, so)
        g, sx, sy, skw, so = self.graphs[key]
        sx.copy_(x)
        sy.copy_(y)
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                skw[k].copy_(v)
        g.replay()
        return so.clone() if isinstance(so, torch.Tensor) else {k: v.clone() for k, v in so.items()}


rng = np.random.default_rng(0)
pool = make_log(rng, 1000, prod="pouch", hum=55)
X, y = pool[FEATURES].to_numpy(), pool["dropped"].to_numpy()
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST, n_estimators=1, device="cuda",
                                                  fit_mode="fit_with_cache", keep_cache_on_device=True, random_state=0)
clf.fit(X, y)
# Patch: the forward reads the class count with .item() (a GPU->CPU sync, illegal during capture).
# Our labels never change, so use a fixed count. Spike-only hack, logged in DEBT.md.
import textwrap
import tabpfn.architectures.tabpfn_v3_5 as _A
_src = textwrap.dedent(inspect.getsource(_A.TabPFNV3p5.forward))
_old = "    num_present_classes = None\n    if task_type == \"multiclass\":"
_new = "    num_present_classes = getattr(self, \"_fixed_num_classes\", None)\n    if task_type == \"multiclass\" and num_present_classes is None:"
assert _old in _src, "forward source changed; update the patch"
_ns = {}
import __future__
exec(compile(_src.replace(_old, _new), _A.__file__, "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True), _A.__dict__, _ns)
_A.TabPFNV3p5.forward = _ns["forward"]
models = [o for o in gc.get_objects() if isinstance(o, TabPFNV3p5)]
assert len(models) == 1, f"expected one TabPFNV3p5 instance, found {len(models)}"
models[0]._fixed_num_classes = 2
gf = GraphedForward(models[0])
models[0].forward = gf


def queries(n_items, seed):
    r = np.random.default_rng(seed)
    base = make_log(r, n_items, prod="pouch", hum=55)
    rows = base.loc[base.index.repeat(len(SPEEDS))].reset_index(drop=True)
    return run_picks(rows, np.tile(SPEEDS, n_items), r)[FEATURES].to_numpy()


def bench(q, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        p = clf.predict_proba(q)
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return np.median(ts), np.percentile(ts, 95), p


for label, n_items, reps in [("one pick (15 rows)", 1, 50), ("speed map (3,000 rows)", 200, 8)]:
    q0 = queries(n_items, 1)
    gf.enabled = False
    clf.predict_proba(q0)
    base_med, base_p95, _ = bench(q0, reps)
    gf.enabled = True
    t = time.perf_counter()
    clf.predict_proba(q0)  # capture
    torch.cuda.synchronize()
    cap = time.perf_counter() - t
    g_med, g_p95, _ = bench(q0, reps)
    # correctness on fresh queries of the same shape
    diffs = []
    for seed in range(2, 7):
        q = queries(n_items, seed)
        gf.enabled = False
        p_ref = clf.predict_proba(q)
        gf.enabled = True
        p_g = clf.predict_proba(q)
        diffs.append(np.abs(p_ref - p_g).max())
    print(f"{label:24s} normal {base_med:6.1f} ms (p95 {base_p95:5.1f}) | CUDA graph {g_med:6.1f} ms (p95 {g_p95:5.1f}) "
          f"| capture {cap:.2f} s | max |Δp| on 5 fresh queries {max(diffs):.2e}")
