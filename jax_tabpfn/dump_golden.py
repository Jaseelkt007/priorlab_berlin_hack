"""Golden tensors from the PyTorch TabPFN-3.5-Fast cached query path.

Fits the classifier from scripts/bench_tabpfn.py (1,000 toy rows, 1 estimator,
fit_with_cache), exports its context bundle (export.py), and for each query saves the
raw rows, the x the model receives (hooked TabPFNV3p5.forward), its logits and
predict_proba. Queries: two 15-row picks, the 3,000-row speed map, and 5 fresh picks.

Run with the torch env:  ../priorlab_berlin_hack/.venv/bin/python -m jax_tabpfn.dump_golden [--fp32]
--fp32: inference_precision=float32 and an unquantised cache (parity goldens).
Default: the classifier's production setting (fp16 autocast, int8 KV cache).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(Path.home() / "priorlab_berlin_hack/.env")

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.architectures import tabpfn_v3_5 as arch
from tabpfn.constants import ModelVersion

from jax_tabpfn.export import export_bundle
from sim.delta_toy import FEATURES, SPEEDS, make_log, run_picks, sample_items

FP32 = "--fp32" in sys.argv
TAG = "fp32" if FP32 else "default"
OUT = Path(__file__).resolve().parent / "golden"
OUT.mkdir(exist_ok=True)

# Same data as scripts/bench_tabpfn.py
rng = np.random.default_rng(0)
pool = make_log(rng, 4000, prod="pouch", hum=55)
test = make_log(rng, 2000, prod="pouch", hum=55)
grid_items = sample_items(rng, "pouch", 200, hum=55)
grid_items["seal_kpa"] = np.repeat(np.linspace(10, 60, 20), 10)
grid_items["weight_g"] = np.tile(np.linspace(54, 66, 10), 20)
queries = {
    "map": run_picks(grid_items.loc[grid_items.index.repeat(len(SPEEDS))].reset_index(drop=True),
                     np.tile(SPEEDS, len(grid_items)), rng)[FEATURES].to_numpy(),
}
for i in range(7):  # pick, pick2 and 5 fresh picks
    name = {0: "pick", 1: "pick2"}.get(i, f"fresh{i - 2}")
    queries[name] = run_picks(test.iloc[[i] * len(SPEEDS)].reset_index(drop=True), SPEEDS, rng)[FEATURES].to_numpy()
X = pool[FEATURES].to_numpy()[:1000]
y = pool["dropped"].to_numpy()[:1000]

kw = dict(n_estimators=1, device="cuda", fit_mode="fit_with_cache", random_state=0)
if FP32:
    kw.update(inference_precision=torch.float32, kv_cache_precision="auto")
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST, **kw)
clf.fit(X, y)
export_bundle(clf, OUT / f"bundle_{TAG}.npz")

calls = []
orig = arch.TabPFNV3p5.forward


def hooked(self, x, y, task_type, **k):
    out = orig(self, x, y, task_type, **k)
    calls.append((x.detach().float().cpu().numpy(), out.detach().float().cpu().numpy()))
    return out


arch.TabPFNV3p5.forward = hooked
for name, q in queries.items():
    calls.clear()
    proba = clf.predict_proba(q)
    assert len(calls) == 1
    x, logits = calls[0]
    np.savez(OUT / f"{name}_{TAG}.npz", x_raw=q, x=x, logits=logits, proba=proba)
    print(f"{name:7s} x {x.shape}  logits {logits.shape}")
print(f"bundle + {len(queries)} queries -> {OUT} ({TAG})")
