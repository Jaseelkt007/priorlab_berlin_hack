"""Independent verification of the JAX port (jax_tabpfn/) against PyTorch TabPFN.

Uses only the port's public API (export.bundle_from_classifier, JaxTabPFN,
update_context); none of jax_tabpfn/parity.py or bench.py. Three sections:

  parity   products x context sizes, fresh + edge-value picks and speed maps; |dp| and the
           chosen speed (fastest speed with P(drop) <= 0.5 %) for JAX fp32 / fast vs torch
           fp32 / default (fp16 autocast + int8 KV)
  latency  fresh processes: first call with an empty vs the warm persistent XLA cache, then
           steady-state JAX fast / fp32 vs torch normal / CUDA graph in one process
  refit    10 successive refits of a rolling 1,000-row window; parity and update_context time

torch and jax must share one interpreter (see DEBT.md); run from the repo root:
  PYTHONPATH=.:../priorlab_berlin_hack/.venv/lib/python3.12/site-packages \
    .venv-jax/bin/python scripts/verify_jax_port.py {parity,latency,refit}
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jax_tabpfn.runtime import JaxTabPFN  # noqa: E402  (jax before torch)
import jax_tabpfn.runtime as jrt  # noqa: E402  (only to point the XLA cache at an empty dir)
from jax_tabpfn.export import bundle_from_classifier  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
load_dotenv(Path.home() / "priorlab_berlin_hack/.env")
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

from sim.delta_toy import FEATURES, PRODUCTS, SPEEDS, make_log, run_picks, sample_items  # noqa: E402

P_MAX = 0.005  # drop-probability budget for the speed decision
HUM = 55


def torch_clf(fp32: bool, keep_on_device: bool = False) -> TabPFNClassifier:
    kw = dict(n_estimators=1, device="cuda", fit_mode="fit_with_cache", random_state=0)
    if fp32:
        kw.update(inference_precision=torch.float32, kv_cache_precision="auto")
    if keep_on_device:
        kw.update(keep_cache_on_device=True)
    return TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST, **kw)


# ------------------------------------------------------------------ queries ----

def pick(items: pd.DataFrame, rng) -> np.ndarray:
    """One pick = one item at all 15 candidate speeds (0.30 ... 1.00)."""
    return run_picks(items.iloc[[0] * len(SPEEDS)].reset_index(drop=True), SPEEDS, rng)[FEATURES].to_numpy()


def speed_map(prod: str, rng) -> np.ndarray:
    """200 (seal, weight) grid items x 15 speeds = 3,000 rows, item-major."""
    p = PRODUCTS[prod]
    g = sample_items(rng, prod, 200, hum=HUM)
    g["seal_kpa"] = np.repeat(np.linspace(5, 64, 20), 10)
    g["weight_g"] = np.tile(np.linspace(p["w"][0] * 0.85, p["w"][0] * 1.15, 10), 20)
    rows = g.loc[g.index.repeat(len(SPEEDS))].reset_index(drop=True)
    return run_picks(rows, np.tile(SPEEDS, len(g)), rng)[FEATURES].to_numpy()


def edge_picks(prod: str, rng) -> dict[str, np.ndarray]:
    """Edge cases: seal at the simulator's clip limits, far outside the context, NaN."""
    out = {}
    for name, edit in {
        "seal=3kPa": dict(seal_kpa=3.0, evac_ms=14 + 520 / 3.0),
        "seal=65kPa": dict(seal_kpa=65.0, evac_ms=14 + 520 / 65.0),
        "far-above": dict(weight_g=5.0, len_mm=3.0, wid_mm=3.0, hgt_mm=3.0, pick_off_mm=40.0, seal_kpa=150.0,
                          evac_ms=2000.0, cup_cycles=1e6),
        "far-below": dict(weight_g=0.05, len_mm=0.1, wid_mm=0.1, hgt_mm=0.1, pick_off_mm=-10.0, seal_kpa=0.5,
                          evac_ms=0.0, cup_cycles=-100.0),
        "seal=NaN": dict(seal_kpa=np.nan),
    }.items():
        it = sample_items(rng, prod, 1, hum=HUM)
        it["speed"] = 0.0  # set per row below
        for col, v in edit.items():
            # multiplicative for dims/weight in the "far" cases, absolute otherwise
            it[col] = it[col] * v if name.startswith("far") and col in ("weight_g", "len_mm", "wid_mm", "hgt_mm") else v
        q = np.repeat(it[FEATURES].to_numpy().astype(float), len(SPEEDS), axis=0)
        q[:, FEATURES.index("speed")] = SPEEDS
        out[name] = q
    return out


def p_drop(proba: np.ndarray) -> np.ndarray:
    return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(proba))


def chosen_speeds(pd_: np.ndarray) -> np.ndarray:
    """Per item (rows item-major, 15 speeds each): fastest speed with P(drop) <= P_MAX, NaN if none."""
    ok = pd_.reshape(-1, len(SPEEDS)) <= P_MAX
    idx = np.where(ok.any(1), len(SPEEDS) - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
    return np.where(idx >= 0, SPEEDS[np.clip(idx, 0, None)], np.nan)


def same_choice(a, b):
    """Equal chosen speeds; NaN (no safe speed) on both sides counts as agreement."""
    return (a == b) | (np.isnan(a) & np.isnan(b))


# ------------------------------------------------------------------- parity ----

PAIRS = [("JAX fp32", "torch fp32"), ("JAX fp32", "torch default"), ("JAX fast", "torch fp32"),
         ("JAX fast", "torch default"), ("JAX fast[int8 bundle]", "torch default"), ("torch default", "torch fp32")]


def compare(res: dict, kind: str, a: str, b: str):
    if not res[kind]:
        return np.nan, np.nan, 0, 0
    da = np.concatenate([np.abs(p_drop(x[a]) - p_drop(x[b])) for x in res[kind]])
    ca = np.concatenate([chosen_speeds(p_drop(x[a])) for x in res[kind]])
    cb = np.concatenate([chosen_speeds(p_drop(x[b])) for x in res[kind]])
    flips = int(np.sum(~same_choice(ca, cb)))
    return da.max(), da.mean(), flips, len(ca)


def section_parity(products, sizes):
    rng = np.random.default_rng(12345)  # data the port has never seen (goldens used seed 0)
    models = {}
    rows, flips_detail, edge_rows = [], [], []
    for prod in products:
        stream = make_log(rng, max(sizes) + 2000, prod=prod, hum=HUM)
        for n in sizes:
            ctx = stream.iloc[:n]
            X, y = ctx[FEATURES].to_numpy(), ctx["dropped"].to_numpy()
            c32, cdef = torch_clf(True), torch_clf(False)
            c32.fit(X, y)
            cdef.fit(X, y)
            b32, bdef = bundle_from_classifier(c32), bundle_from_classifier(cdef)
            for key, b, fast in (("JAX fp32", b32, False), ("JAX fast", b32, True), ("JAX fast[int8 bundle]", bdef, True)):
                if key not in models:
                    models[key] = JaxTabPFN(b, fast=fast)
                else:
                    models[key].update_context(b)  # products and sizes change under the same instance
            fresh = make_log(rng, 20, prod=prod, hum=HUM)
            queries = {"pick": [pick(fresh.iloc[[i]], rng) for i in range(20)],
                       "edge": list(edge_picks(prod, rng).items()),
                       "map": [speed_map(prod, rng) for _ in range(2)]}
            res = {"pick": [], "edge": [], "map": []}
            for kind, qs in queries.items():
                for q in qs:
                    label, q = q if kind == "edge" else (kind, q)
                    r = {"label": label}
                    try:
                        r["torch fp32"], r["torch default"] = c32.predict_proba(q), cdef.predict_proba(q)
                    except Exception as e:  # noqa: BLE001
                        edge_rows.append((prod, n, label, f"torch raised {type(e).__name__}: {str(e)[:80]}"))
                        continue
                    for key, m in models.items():
                        r[key] = m.predict_proba(q)
                    res[kind].append(r)
            setting = f"{prod} n={n} (drops {y.mean():.1%})"
            for kind in ("pick", "edge", "map"):
                for a, b in PAIRS:
                    mx, mean, fl, nd = compare(res, kind, a, b)
                    rows.append((setting, kind, f"{a} vs {b}", mx, mean, fl, nd))
            for r in res["edge"]:
                d = np.abs(p_drop(r["JAX fp32"]) - p_drop(r["torch fp32"])).max()
                df_ = np.abs(p_drop(r["JAX fast"]) - p_drop(r["torch default"])).max()
                edge_rows.append((prod, n, r["label"], f"fp32 |dp| {d:.1e}, fast-vs-default |dp| {df_:.1e}, "
                                  f"P(drop) torch fp32 {p_drop(r['torch fp32']).min():.3f}-{p_drop(r['torch fp32']).max():.3f}"))
            for kind in ("pick", "edge", "map"):
                for r in res[kind]:
                    for a, b in PAIRS[:5]:
                        ca, cb = chosen_speeds(p_drop(r[a])), chosen_speeds(p_drop(r[b]))
                        for i in np.where(~same_choice(ca, cb))[0]:
                            sl = slice(i * len(SPEEDS), (i + 1) * len(SPEEDS))
                            flips_detail.append((setting, kind, r["label"], f"{a} vs {b}", ca[i], cb[i],
                                                 np.round(p_drop(r[a])[sl], 4), np.round(p_drop(r[b])[sl], 4)))
            print(f"done {setting}", flush=True)

    print("\n### Parity (P(drop); chosen speed = fastest speed with P(drop) <= 0.5 %)\n")
    print("| setting | queries | comparison | max abs dp | mean abs dp | chosen-speed flips |")
    print("|---|---|---|---|---|---|")
    for s, k, c, mx, mean, fl, nd in rows:
        print(f"| {s} | {k} | {c} | {mx:.1e} | {mean:.1e} | {fl} / {nd} |")
    print("\n### Edge queries\n")
    for r in edge_rows:
        print(f"- {r[0]} n={r[1]} {r[2]}: {r[3]}")
    print("\n### Every chosen-speed disagreement (JAX vs torch)\n")
    for f in flips_detail:
        print(f"- {f[0]} {f[1]} {f[2]} | {f[3]}: {f[4]} vs {f[5]}\n  P(drop) {f[3].split(' vs ')[0]}: {f[6]}\n"
              f"  P(drop) {f[3].split(' vs ')[1]}: {f[7]}")
    worst = max(mx for s, k, c, mx, *_ in rows if c == "JAX fp32 vs torch fp32")
    fp32_flips = sum(fl for s, k, c, mx, mean, fl, nd in rows if c == "JAX fp32 vs torch fp32")
    print(f"\nJAX fp32 vs torch fp32: worst max|dp| {worst:.1e} (target 1e-4: {'PASS' if worst <= 1e-4 else 'FAIL'}), "
          f"chosen-speed flips {fp32_flips}")


# ------------------------------------------------------------------ latency ----

def install_cuda_graph(clf):
    """CUDA-graph replay of the cached forward, as in scripts/spike_cudagraph.py (copied: that
    script runs its experiment at import). Patches out the `.item()` class-count sync."""
    import __future__
    import inspect
    import textwrap

    import tabpfn.architectures.tabpfn_v3_5 as A

    src = textwrap.dedent(inspect.getsource(A.TabPFNV3p5.forward))
    old = "    num_present_classes = None\n    if task_type == \"multiclass\":"
    new = ("    num_present_classes = getattr(self, \"_fixed_num_classes\", None)\n"
           "    if task_type == \"multiclass\" and num_present_classes is None:")
    assert old in src
    ns = {}
    exec(compile(src.replace(old, new), A.__file__, "exec", flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), A.__dict__, ns)
    A.TabPFNV3p5.forward = ns["forward"]
    model = clf.executor_.model_caches[0].get_any()  # this classifier's own model object
    assert isinstance(model, A.TabPFNV3p5)
    model._fixed_num_classes = clf.n_classes_
    orig, graphs = model.forward, {}

    def graphed(x, y, **kw):
        key = (tuple(x.shape), kw.get("task_type"))
        if key not in graphs:
            sx, sy = x.clone(), y.clone()
            skw = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    orig(sx, sy, **skw)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                so = orig(sx, sy, **skw)
            graphs[key] = (g, sx, sy, skw, so)
        g, sx, sy, skw, so = graphs[key]
        sx.copy_(x)
        sy.copy_(y)
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                skw[k].copy_(v)
        g.replay()
        return so.clone()

    graphed.__signature__ = inspect.signature(orig)
    model.forward = graphed


def pct(ts):
    return np.percentile(ts, [50, 95, 99])


def latency_queries(rng, n_picks=200, n_maps=20):
    items = make_log(rng, n_picks, prod="pouch", hum=HUM)
    return [pick(items.iloc[[i]], rng) for i in range(n_picks)], [speed_map("pouch", rng) for _ in range(n_maps)]


def section_latency_child(mode: str):
    """Runs in its own process. mode: 'cold' (empty XLA cache) or 'warm' (persistent cache)."""
    rng = np.random.default_rng(777)
    log = make_log(rng, 1000, prod="pouch", hum=HUM)
    X, y = log[FEATURES].to_numpy(), log["dropped"].to_numpy()
    c32 = torch_clf(True)
    c32.fit(X, y)
    b = bundle_from_classifier(c32)
    picks, maps = latency_queries(rng)
    if mode == "cold":
        tmp = tempfile.mkdtemp(prefix="xla-empty-")
        orig = jrt.enable_compile_cache
        jrt.enable_compile_cache = lambda: orig(Path(tmp))
    first = {}
    models = {}
    for fast in (True, False):
        t = time.perf_counter()
        m = JaxTabPFN(b, fast=fast)
        first[(fast, "load")] = time.perf_counter() - t
        for name, q in (("pick", picks[0]), ("map", maps[0])):
            t = time.perf_counter()
            m.predict_proba(q)
            first[(fast, name)] = time.perf_counter() - t
        models[fast] = m
    print(f"| {mode} XLA cache | " + " | ".join(
        f"{first[(f, k)]:.1f} s" for f in (True, False) for k in ("load", "pick", "map")) + " |", flush=True)
    if mode == "cold":
        return

    cdef = torch_clf(False, keep_on_device=True)
    cdef.fit(X, y)
    paths = {"JAX fast": models[True].predict_proba, "JAX fp32": models[False].predict_proba,
             "torch normal (default)": cdef.predict_proba}
    out = {}
    for name, fn in paths.items():
        out[name] = run_steady(fn, picks, maps)
    install_cuda_graph(cdef)
    out["torch CUDA graph (default)"] = run_steady(cdef.predict_proba, picks, maps)
    print("\n| path | pick median | pick p95 | pick p99 | map median | map p95 | map p99 |\n|---|---|---|---|---|---|---|")
    for name, (pp, mp) in out.items():
        print(f"| {name} | {pp[0]:.2f} ms | {pp[1]:.2f} ms | {pp[2]:.2f} ms | {mp[0]:.1f} ms | {mp[1]:.1f} ms | {mp[2]:.1f} ms |")


def run_steady(fn, picks, maps):
    for q in picks[:5] + maps[:2]:
        fn(q)  # warm-up (and graph capture)
    tp, tm = [], []
    for q in picks:
        t = time.perf_counter()
        fn(q)
        torch.cuda.synchronize()
        tp.append(1000 * (time.perf_counter() - t))
    for q in maps:
        t = time.perf_counter()
        fn(q)
        torch.cuda.synchronize()
        tm.append(1000 * (time.perf_counter() - t))
    return pct(tp), pct(tm)


def section_latency():
    print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                         capture_output=True, text=True).stdout)
    print("### First call in a fresh process (context 1,000 rows, pouch)\n")
    print("| XLA cache | fast: load | fast: 1st pick | fast: 1st map | fp32: load | fp32: 1st pick | fp32: 1st map |")
    print("|---|---|---|---|---|---|---|", flush=True)
    for mode in ("cold", "warm"):
        r = subprocess.run([sys.executable, __file__, "_latency_child", mode], capture_output=True, text=True,
                           env=os.environ)
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith("|") or l.startswith("\n")) or r.stderr[-3000:],
              flush=True)


# -------------------------------------------------------------------- refit ----

def section_refit(n_refits=10, window=1000, arrivals=50):
    rng = np.random.default_rng(2024)
    stream = make_log(rng, window + arrivals * n_refits + 400, prod="pouch", hum=HUM)
    later = stream.iloc[window + arrivals * n_refits:]  # rows never in any window, for queries
    models = {}
    print("| refit | window rows | drops | torch fp32 fit | export | update_context fp32 / fast | 1st query fp32 / fast "
          "| fp32 max abs dp | fast max abs dp vs fp32 / vs default | chosen-speed flips fp32 / fast |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in range(n_refits + 1):
        ctx = stream.iloc[r * arrivals: r * arrivals + window]
        X, y = ctx[FEATURES].to_numpy(), ctx["dropped"].to_numpy()
        c32, cdef = torch_clf(True), torch_clf(False)
        t0 = time.perf_counter()
        c32.fit(X, y)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cdef.fit(X, y)
        t2 = time.perf_counter()
        b = bundle_from_classifier(c32)
        t3 = time.perf_counter()
        upd, firstq = {}, {}
        qs = [pick(later.iloc[[(37 * r + i) % len(later)]], rng) for i in range(5)]
        for fast in (False, True):
            t = time.perf_counter()
            if fast in models:
                models[fast].update_context(b)
            else:
                models[fast] = JaxTabPFN(b, fast=fast)
            upd[fast] = time.perf_counter() - t
            t = time.perf_counter()
            models[fast].predict_proba(qs[0])
            firstq[fast] = time.perf_counter() - t
        d32, dff, dfd, fl32, flf = 0.0, 0.0, 0.0, 0, 0
        for q in qs:
            p32, pdef = p_drop(c32.predict_proba(q)), p_drop(cdef.predict_proba(q))
            j32, jf = p_drop(models[False].predict_proba(q)), p_drop(models[True].predict_proba(q))
            d32, dff, dfd = max(d32, np.abs(j32 - p32).max()), max(dff, np.abs(jf - p32).max()), max(dfd, np.abs(jf - pdef).max())
            fl32 += int(not same_choice(chosen_speeds(j32)[0], chosen_speeds(p32)[0]))
            flf += int(not same_choice(chosen_speeds(jf)[0], chosen_speeds(pdef)[0]))
        print(f"| {r if r else '0 (initial)'} | {r * arrivals}-{r * arrivals + window} | {y.mean():.1%} | "
              f"{1000 * (t1 - t0):.0f} ms | {1000 * (t3 - t2):.0f} ms | {1000 * upd[False]:.0f} / {1000 * upd[True]:.0f} ms | "
              f"{1000 * firstq[False]:.1f} / {1000 * firstq[True]:.1f} ms | {d32:.1e} | {dff:.1e} / {dfd:.1e} | {fl32} / {flf} |",
              flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "parity":
        section_parity(["biscuit", "pouch", "cheese"], [300, 1000, 2000])
    elif cmd == "latency":
        section_latency()
    elif cmd == "_latency_child":
        section_latency_child(sys.argv[2])
    elif cmd == "refit":
        section_refit()
    else:
        raise SystemExit(__doc__)
