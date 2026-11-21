"""Raw rows in, probabilities out: the JAX replacement for `clf.predict_proba` on a cached context.

    from jax_tabpfn.runtime import JaxTabPFN
    m = JaxTabPFN("bundle.npz")            # bundle from export.export_bundle(clf, ...)
    proba = m.predict_proba(X_raw)         # same columns as clf.fit(X, y)

Per call: the per-row fingerprint hash on the CPU (SHA-256, as TabPFN does), then one
jitted XLA program for the column shuffle, soft clip, model and softmax. Row counts are
padded up to a few fixed sizes (test rows never attend to each other, so padding is
exact) to keep the number of compiled programs small. Compiled programs are kept in a
persistent cache, so only the first process on a machine pays the ~20-100 s autotuning.
"""
from __future__ import annotations

import hashlib
import os
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # share the GPU with torch
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_tabpfn.model import load_params, prepare_context, query_logits

WEIGHTS = Path.home() / ".cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors"
COMPILE_CACHE = Path.home() / ".cache/jax_tabpfn/xla"
_MASK64 = 2**64 - 1


def enable_compile_cache(path: Path = COMPILE_CACHE) -> None:
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def fingerprint(x32: np.ndarray, salt: int) -> np.ndarray:
    """TabPFN's test-row fingerprint: SHA-256 of the rounded fp32 row bytes + salt -> [0, 1)."""
    rounded = np.ascontiguousarray(np.around(x32, decimals=12))
    salt_b = int(salt).to_bytes(8, "little", signed=False)
    return np.array([(int(hashlib.sha256(r.tobytes() + salt_b).hexdigest(), 16) & _MASK64) / _MASK64
                     for r in rounded], dtype=np.float32)


def device_preprocess(ctx, x):
    """TabPFN's GPU steps after the fingerprint: column shuffle, then soft outlier clip."""
    x = x[:, ctx["permutation"]]
    lo, hi = ctx["clip_lower"], ctx["clip_upper"]
    x = jnp.maximum(-jnp.log(1 + jnp.abs(x)) + lo, x)
    return jnp.minimum(jnp.log(1 + jnp.abs(x)) + hi, x)


def bucket(m: int) -> int:
    """Padded row count: 16, 32, ..., 1024, then multiples of 1024."""
    if m <= 1024:
        return max(16, 1 << (m - 1).bit_length())
    return -(-m // 1024) * 1024


class JaxTabPFN:
    def __init__(self, bundle: str | Path | dict, *, fast: bool = True, weights: Path = WEIGHTS):
        """fast=True: fp16 matmuls + cuDNN flash attention (TabPFN's own GPU default is
        fp16 autocast). fast=False: fp32 throughout, matches PyTorch fp32 to ~1e-5."""
        enable_compile_cache()
        self.params = load_params(str(weights))
        self.precision = "default" if fast else "highest"
        dtype = jnp.float16 if fast else None

        @partial(jax.jit, static_argnames="n_classes")
        def _forward(p, ctx, x, n_classes):  # x: (M, C) fingerprinted rows
            x = device_preprocess(ctx, x)
            logits = query_logits(p, ctx, x[:, None, :], n_classes=n_classes, dtype=dtype)[:, 0]
            return jax.nn.softmax(logits[:, ctx["class_permutation"]], axis=-1)

        self._forward = _forward
        self.update_context(bundle)

    def update_context(self, bundle: str | Path | dict) -> None:
        """Swap in a refitted context (from export.bundle_from_classifier).

        Every context array is a jit argument, not a constant, so a refit with the
        same context size and column count reuses the compiled program. A new size
        compiles once (and is then in the persistent cache)."""
        b = dict(np.load(bundle)) if not isinstance(bundle, dict) else bundle
        with jax.default_matmul_precision(self.precision):
            ctx = prepare_context(self.params, b)
        for k in ("permutation", "clip_lower", "clip_upper", "class_permutation"):
            ctx[k] = jnp.asarray(b[k])
        self.sel = b["sel"]
        self.salt = int(b["fingerprint_salt"])
        self.n_features_in = len(self.sel)
        self.n_classes = int(b["n_classes"])
        self.ctx = ctx

    def model_input(self, X) -> np.ndarray:
        """CPU part: drop constant columns, fp32, append the fingerprint column."""
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == self.n_features_in, X.shape
        x = X[:, self.sel].astype(np.float32)
        return np.concatenate([x, fingerprint(x, self.salt)[:, None]], axis=1)

    def predict_proba(self, X) -> np.ndarray:
        x = self.model_input(X)
        m = len(x)
        pad = bucket(m) - m
        if pad:
            x = np.concatenate([x, np.zeros((pad, x.shape[1]), np.float32)])
        with jax.default_matmul_precision(self.precision):
            return np.asarray(self._forward(self.params, self.ctx, x, n_classes=self.n_classes))[:m]

    def warmup(self, *row_counts: int) -> None:
        for m in row_counts:
            self.predict_proba(np.zeros((m, self.n_features_in)))
