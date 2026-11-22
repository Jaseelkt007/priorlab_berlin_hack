"""End-to-end latency of the JAX runtime: raw rows -> probabilities (like clf.predict_proba).

Run: .venv-jax/bin/python -m jax_tabpfn.bench   (after dump_golden.py --fp32)
Median / p95 / p99 over 200 calls (15-row pick) and 30 calls (3,000-row map),
for fp32 (exact) and fast (fp16 + cuDNN flash) modes, plus the CPU-side share.
"""
import time
from pathlib import Path

import jax
import numpy as np

from jax_tabpfn.runtime import JaxTabPFN

GOLD = Path(__file__).resolve().parent / "golden"


def stats(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t))
    return np.percentile(ts, [50, 95, 99])


def main():
    print(f"device: {jax.devices()[0].device_kind}, jax {jax.__version__}")
    print(f"{'mode':6s} {'rows':>5s} | {'median':>7s} {'p95':>7s} {'p99':>7s} ms | {'CPU part ms':>11s} | {'first call s':>12s}")
    for fast in (False, True):
        m = JaxTabPFN(GOLD / "bundle_fp32.npz", fast=fast)
        for q, reps in (("pick", 200), ("map", 30)):
            X = np.load(GOLD / f"{q}_fp32.npz")["x_raw"]
            t = time.perf_counter()
            m.predict_proba(X)
            first = time.perf_counter() - t
            for _ in range(5):
                m.predict_proba(X)
            med, p95, p99 = stats(lambda: m.predict_proba(X), reps)
            cpu = stats(lambda: m.model_input(X), reps)[0]
            print(f"{'fast' if fast else 'fp32':6s} {len(X):5d} | {med:7.2f} {p95:7.2f} {p99:7.2f}    | {cpu:11.2f} | {first:12.1f}")


if __name__ == "__main__":
    main()
