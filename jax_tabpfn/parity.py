"""Parity of the JAX port against the PyTorch goldens (run dump_golden.py first).

Run: .venv-jax/bin/python -m jax_tabpfn.parity
1. preprocessing: raw rows -> the exact x TabPFN feeds the model;
2. model, fp32: hooked x -> logits and probabilities vs torch fp32 (target <= 1e-4);
3. end to end: raw rows -> probabilities, fp32 and fast (fp16) mode, vs torch fp32 and
   vs torch's own production setting (fp16 autocast + int8 KV).
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_tabpfn.model import check_param_coverage, query_logits
from jax_tabpfn.runtime import JaxTabPFN, device_preprocess

GOLD = Path(__file__).resolve().parent / "golden"
QUERIES = ["pick", "pick2", "fresh0", "fresh1", "fresh2", "fresh3", "fresh4", "map"]
TOL = 1e-4


def main() -> bool:
    ok = True
    exact = JaxTabPFN(GOLD / "bundle_fp32.npz", fast=False)
    fast = JaxTabPFN(GOLD / "bundle_fp32.npz", fast=True)
    n_classes, perm = 2, exact.ctx["class_permutation"]
    g0 = dict(np.load(GOLD / "pick_fp32.npz"))
    with jax.default_matmul_precision("highest"):
        used, total = check_param_coverage(exact.params, exact.ctx, jnp.asarray(g0["x"]), n_classes)
    print(f"weights: {used} read per query, {total - used} context-only or regression-only; none unaccounted for\n")
    model_fn = jax.jit(lambda p, c, x: query_logits(p, c, x, n_classes=n_classes))

    print(f"{'query':7s} {'rows':>5s} | {'prep max|dx|':>12s} | {'model fp32 |dlogit|':>19s} {'|dp|':>8s} | "
          f"{'e2e fp32 |dp|':>13s} | {'fast |dp| vs fp32':>17s} {'vs torch-default':>16s} {'torch-default vs fp32':>21s}")
    for q in QUERIES:
        g = dict(np.load(GOLD / f"{q}_fp32.npz"))
        gd = dict(np.load(GOLD / f"{q}_default.npz"))
        x_dev = np.asarray(device_preprocess(exact.ctx, jnp.asarray(exact.model_input(g["x_raw"]))))
        dx = np.abs(x_dev - g["x"][:, 0]).max()
        with jax.default_matmul_precision("highest"):
            logits = np.asarray(model_fn(exact.params, exact.ctx, jnp.asarray(g["x"])))[:, 0, :n_classes][:, perm]
        p_model = np.asarray(jax.nn.softmax(logits, -1))
        dl = np.abs(logits - g["logits"][:, 0, :n_classes][:, perm]).max()
        dpm = np.abs(p_model - g["proba"]).max()
        dpe = np.abs(exact.predict_proba(g["x_raw"]) - g["proba"]).max()
        pf = fast.predict_proba(g["x_raw"])
        dff, dfd = np.abs(pf - g["proba"]).max(), np.abs(pf - gd["proba"]).max()
        dtd = np.abs(gd["proba"] - g["proba"]).max()
        good = dx == 0 and dpm <= TOL and dpe <= TOL
        ok &= good
        print(f"{q:7s} {len(g['x_raw']):5d} | {dx:12.1e} | {dl:19.2e} {dpm:8.1e} | {dpe:13.1e} | "
              f"{dff:17.1e} {dfd:16.1e} {dtd:21.1e}  {'OK' if good else 'FAIL'}")
    print(f"\nfp32 parity (model and end to end, <= {TOL:g}): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
