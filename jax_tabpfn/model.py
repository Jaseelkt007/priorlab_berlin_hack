"""TabPFN-3.5(-Fast) cached-context query path in JAX.

Mirrors `tabpfn/architectures/tabpfn_v3_5.py` (tabpfn 10.2.0) for the predict path
with a KV cache (`x_is_test_only=True`, multiclass head): preprocessing of the test
cells against the cached scaler/ECDF context, cell embedding, distribution embedder
(cross_attn_block2 against the cached inducing states), column aggregator, ICL
transformer against the cached K/V, many-class decoder.

Everything that depends only on the context (the cache plus the weights) is folded
once by `prepare_context`, e.g. the K/V projections of the cached inducing states.
`query_logits` is then a pure function of (params, ctx, x) and is `jax.jit`-able
with static shapes. The class count is static (no `.item()` sync).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

F32_EPS = float(np.finfo(np.float32).eps)  # torch RMSNorm eps=None uses finfo(dtype).eps


# ---------------------------------------------------------------- weights ----

def load_params(path: str) -> dict[str, jax.Array]:
    """Flat {torch name: array} pytree from the safetensors checkpoint."""
    from safetensors.numpy import load_file

    return {k: jnp.asarray(v) for k, v in load_file(path).items()}


# Weights the cached query path never reads: regression head, train-side label
# encoders, the context-only half of the induced attention (block1 + inducing
# vectors, already folded into the cache's inducing_hidden), the ICL K/V
# projections (folded into the cached K/V) and the decoder key projection
# (folded into decoder_keys). `check_param_coverage` asserts the rest is used.
CONTEXT_ONLY_PREFIXES = (
    "col_y_encoder.", "col_y_layernorm.", "icl_y_encoder.", "icl_y_layernorm.",
    "heads.mlp_regression.", "heads.output_projection.", "heads.regression_borders",
    "heads.many_class_decoder.k_projection.",
)
CONTEXT_ONLY_PATTERNS = (".cross_attn_block1.", ".inducing_vectors",
                         ".icl_attention.k_projection.", ".icl_attention.v_projection.",
                         ".icl_attention.k_norm.")


def is_context_only(name: str) -> bool:
    return name.startswith(CONTEXT_ONLY_PREFIXES) or any(p in name for p in CONTEXT_ONLY_PATTERNS)


class _Tracker(dict):
    """dict that records which keys were read, to prove every weight is mapped."""

    def __init__(self, d):
        super().__init__(d)
        self.used = set()

    def __getitem__(self, k):
        self.used.add(k)
        return super().__getitem__(k)


# ------------------------------------------------------------- primitives ----

def linear(x, w, b=None, dtype=None):
    """torch nn.Linear: x @ w.T + b. `dtype` casts both operands (bf16 path)."""
    if dtype is not None:
        y = jnp.matmul(x.astype(dtype), w.T.astype(dtype), preferred_element_type=jnp.float32)
    else:
        y = x @ w.T
    return y if b is None else y + b


def rms_norm(x, w, eps=F32_EPS):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps) * w).astype(x.dtype)


def layer_norm(x, w, b, eps=1e-5):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * w + b


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def mlp(p, pre, x, dtype=None):
    """MLP = Linear(no bias) -> GELU -> Linear(no bias)."""
    return linear(gelu(linear(x, p[pre + "0.weight"], dtype=dtype)), p[pre + "2.weight"], dtype=dtype)


def sdpa(q, k, v, dtype=None):
    """Scaled dot-product attention, (B,S,H,D) x (B,N,J,D) -> (B,S,H,D), GQA when J<H."""
    B, S, H, D = q.shape
    if dtype is not None:
        # Fast mode: cuDNN flash attention (fused, fp32 accumulation), like torch's
        # flash SDPA under autocast. Never materialises the (S, N) score matrix.
        return jax.nn.dot_product_attention(q.astype(dtype), k.astype(dtype), v.astype(dtype),
                                            implementation="cudnn").astype(jnp.float32)
    # Exact fp32 path.
    J = k.shape[2]
    qg = q.reshape(B, S, J, H // J, D)  # head h uses kv head h // (H // J), like enable_gqa
    s = jnp.einsum("bsjgd,bnjd->bjgsn", qg, k) / math.sqrt(D)
    o = jnp.einsum("bjgsn,bnjd->bsjgd", jax.nn.softmax(s, axis=-1), v)
    return o.reshape(B, S, H, D)


def softmax_scaling(p, pre, q, n):
    """SoftmaxScalingMLP: q * base_mlp(log n) * (1 + tanh(query_mlp(q)))."""
    H, D = q.shape[-2:]
    logn = jnp.log(jnp.maximum(jnp.float32(n), 1.0)).reshape(1, 1)
    h = gelu(linear(logn, p[pre + "base_mlp.0.weight"], p[pre + "base_mlp.0.bias"]))
    base = linear(h, p[pre + "base_mlp.2.weight"], p[pre + "base_mlp.2.bias"]).reshape(1, 1, H, D)
    hq = gelu(linear(q, p[pre + "query_mlp.0.weight"], p[pre + "query_mlp.0.bias"]))
    mod = 1 + jnp.tanh(linear(hq, p[pre + "query_mlp.2.weight"], p[pre + "query_mlp.2.bias"]))
    return q * (base * mod)


def apply_rope(t, inv_freq):
    """Non-interleaved RoPE along axis -3 of (..., S, H, D)."""
    S, D = t.shape[-3], t.shape[-1]
    freqs = jnp.arange(S, dtype=jnp.float32)[:, None] * inv_freq[None, :]  # (S, D/2)
    cos = jnp.concatenate([jnp.cos(freqs)] * 2, -1)[:, None, :]  # (S, 1, D)
    sin = jnp.concatenate([jnp.sin(freqs)] * 2, -1)[:, None, :]
    half = D // 2
    rot = jnp.concatenate([-t[..., half:], t[..., :half]], -1)
    return t * cos + rot * sin


# --------------------------------------------------------- preprocessing ----

def ecdf_ranks(x_BMC, ecdf_ctx):
    """`_in_context_ecdf` against the cached `(3, B, C, K)` bucket context."""
    edges, below, at_most = ecdf_ctx  # (B, C, K)
    K = edges.shape[-1]
    v = jnp.swapaxes(x_BMC, 1, 2)[..., None]  # (B, C, M, 1)
    e = edges[:, :, None, :]  # (B, C, 1, K)
    # searchsorted on a sorted row == count of edges below / at most the value
    left = jnp.sum(e < v, axis=-1).astype(jnp.int32)  # (B, C, M)
    right = jnp.sum(e <= v, axis=-1).astype(jnp.int32)
    lo = jnp.clip(left - 1, 0, None)
    hi = jnp.clip(left, None, K - 1)
    g = lambda a, i: jnp.take_along_axis(a, i, axis=-1)
    edge_lo, edge_hi = g(edges, lo), g(edges, hi)
    at_most_lo, below_hi = g(at_most, lo), g(below, hi)
    vals = v[..., 0]
    width = edge_hi - edge_lo
    inside = width > 0
    weight = jnp.where(inside, (vals - edge_lo) / jnp.where(inside, width, 1.0), 0.0)
    counts = at_most_lo + weight * (below_hi - at_most_lo)
    is_edge = right > left
    exact = 0.5 * (below_hi + g(at_most, hi))
    counts = jnp.where(is_edge, exact, counts)
    counts = jnp.where((left == 0) & ~is_edge, 0.0, counts)
    mid_lo = 0.5 * (below[..., :1] + at_most[..., :1])
    mid_hi = 0.5 * (below[..., -1:] + at_most[..., -1:])
    counts = jnp.clip(counts, mid_lo, mid_hi)
    n = at_most[..., -1:]
    return jnp.swapaxes(counts / n, 1, 2)  # (B, M, C)


def preprocess(x_BMC, ctx):
    """NaN/inf indicators -> mean imputation -> ECDF rank -> standard scaling."""
    nan_ind = (jnp.isnan(x_BMC) * -2.0 + jnp.isposinf(x_BMC) * 2.0
               + jnp.isneginf(x_BMC) * 4.0).astype(x_BMC.dtype)
    mean, std = ctx["scaler_mean"], ctx["scaler_std"]  # (B, C)
    x = jnp.where(jnp.isfinite(x_BMC), x_BMC, mean[:, None, :])
    ecdf = ecdf_ranks(x, ctx["ecdf_context"])
    x = jnp.clip((x - mean[:, None, :]) / (std[:, None, :] + F32_EPS), -100, 100)
    return x, nan_ind, ecdf


def group_features(x, nan_ind, ecdf, group_size=3):
    """Circular-shift grouping: channel g of column c reads column (c + 2**g) % C."""
    return jnp.concatenate(
        [jnp.stack([jnp.roll(t, -(2 ** i), axis=2) for i in range(group_size)], -1)
         for t in (x, nan_ind, ecdf)], -1)


def x_embed(p, xg, group_size=3, ecdf_freqs=4):
    """FourierPlusMetadataFeatureGroupEmbedder: (B, M, C, 3G) -> (B, M, C, E)."""
    proj = xg[..., :group_size, None] * p["x_embed.fourier.frequencies"]  # (..., G, F)
    feats = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], -1).sum(-2)  # (..., 2F)
    fourier = linear(feats, p["x_embed.fourier.in_linear.weight"])
    ranks = xg[..., -group_size:]
    k = jnp.arange(1, ecdf_freqs + 1, dtype=jnp.float32)
    phase = ranks[..., None] * (math.pi * k)
    rank_feats = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], -1)
    meta = jnp.concatenate([xg[..., :-group_size], rank_feats.reshape(*rank_feats.shape[:-2], -1)], -1)
    meta_out = linear(meta, p["x_embed.metadata_linear.weight"])
    return layer_norm(fourier + meta_out, p["x_embed.layernorm.weight"], p["x_embed.layernorm.bias"])


# ---------------------------------------------------------- model blocks ----

def prepare_context(p, cache, *, n_dist_blocks=3, n_heads_dist=8):
    """Fold the context-only work into arrays the jitted query reads.

    `cache` holds numpy arrays exported from `TabPFNV3p5Cache` (see dump_golden.py):
    scaler_mean/std (C,), ecdf_context (3, 1, C, K), inducing_hidden_i (C, I, E),
    kv_i_key/value (1, N, 1, Dh), decoder_keys (1, N, Hdec, Ddec), y (N,).
    """
    ctx = {
        "scaler_mean": jnp.asarray(cache["scaler_mean"]).reshape(1, -1),
        "scaler_std": jnp.asarray(cache["scaler_std"]).reshape(1, -1),
        "ecdf_context": jnp.asarray(cache["ecdf_context"]),
        "decoder_keys": jnp.asarray(cache["decoder_keys"]),
        "y": jnp.asarray(cache["y"]).reshape(1, -1),
    }
    nl = len([k for k in cache if k.startswith("kv_") and k.endswith("_key")])
    ctx["icl_k"] = jnp.stack([jnp.asarray(cache[f"kv_{i}_key"]) for i in range(nl)])
    ctx["icl_v"] = jnp.stack([jnp.asarray(cache[f"kv_{i}_value"]) for i in range(nl)])
    # Distribution embedder: K/V of the cached inducing states, per block.
    dk, dv = [], []
    for i in range(n_dist_blocks):
        pre = f"feature_distribution_embedder.layers.{i}.cross_attn_block2."
        h = rms_norm(jnp.asarray(cache[f"inducing_hidden_{i}"]), p[pre + "layernorm_kv.weight"])
        Bc, I, E = h.shape
        k = linear(h, p[pre + "attn.k_projection.weight"]).reshape(Bc, I, n_heads_dist, -1)
        v = linear(h, p[pre + "attn.v_projection.weight"]).reshape(Bc, I, n_heads_dist, -1)
        dk.append(rms_norm(k, p[pre + "attn.k_norm.weight"]))
        dv.append(v)
    ctx["dist_k"], ctx["dist_v"] = jnp.stack(dk), jnp.stack(dv)
    return ctx


def dist_block(p, i, x, k, v, n_heads=8, dtype=None):
    """cross_attn_block2 of induced self-attention block i, K/V precomputed."""
    pre = f"feature_distribution_embedder.layers.{i}.cross_attn_block2."
    Bc, M, E = x.shape
    q = linear(rms_norm(x, p[pre + "layernorm_q.weight"]), p[pre + "attn.q_projection.weight"], dtype=dtype)
    q = rms_norm(q.reshape(Bc, M, n_heads, -1), p[pre + "attn.q_norm.weight"])
    o = sdpa(q, k, v, dtype=dtype).reshape(Bc, M, E)
    x = x + linear(o, p[pre + "attn.out_projection.weight"], dtype=dtype)
    return x + mlp(p, pre + "mlp.", rms_norm(x, p[pre + "layernorm2.weight"]), dtype=dtype)


def agg_self_block(p, j, x, n_heads=8, dtype=None):
    """ColumnAggregator TransformerBlock j: RoPE self-attention over cls + columns."""
    pre = f"column_aggregator.blocks.{j}."
    rope = p["column_aggregator.rope.freqs"]
    *lead, S, E = x.shape
    h = rms_norm(x, p[pre + "layernorm.weight"])
    q = linear(h, p[pre + "attention.q_projection.weight"], dtype=dtype).reshape(*lead, S, n_heads, -1)
    k = linear(h, p[pre + "attention.k_projection.weight"], dtype=dtype).reshape(*lead, S, n_heads, -1)
    v = linear(h, p[pre + "attention.v_projection.weight"], dtype=dtype).reshape(*lead, S, n_heads, -1)
    q = rms_norm(apply_rope(q, rope), p[pre + "attention.q_norm.weight"])
    k = rms_norm(apply_rope(k, rope), p[pre + "attention.k_norm.weight"])
    R = int(np.prod(lead))
    o = sdpa(q.reshape(R, S, n_heads, -1), k.reshape(R, S, n_heads, -1), v.reshape(R, S, n_heads, -1),
             dtype=dtype).reshape(*lead, S, E)
    x = x + linear(o, p[pre + "attention.out_projection.weight"], dtype=dtype)
    return x + mlp(p, pre + "mlp.", rms_norm(x, p[pre + "layernorm_mlp.weight"]), dtype=dtype)


def agg_readout_block(p, j, x, n_cls, n_heads=8, dtype=None):
    """Last aggregator block (`forward_cross`): CLS tokens query the full sequence."""
    pre = f"column_aggregator.blocks.{j}."
    rope = p["column_aggregator.rope.freqs"]
    *lead, S, E = x.shape
    cls = x[..., :n_cls, :]
    hq = rms_norm(cls, p[pre + "layernorm.weight"])
    hc = rms_norm(x, p[pre + "layernorm.weight"])
    q = linear(hq, p[pre + "attention.q_projection.weight"], dtype=dtype).reshape(*lead, n_cls, n_heads, -1)
    k = linear(hc, p[pre + "attention.k_projection.weight"], dtype=dtype).reshape(*lead, S, n_heads, -1)
    v = linear(hc, p[pre + "attention.v_projection.weight"], dtype=dtype).reshape(*lead, S, n_heads, -1)
    q = rms_norm(apply_rope(q, rope), p[pre + "attention.q_norm.weight"])
    k = rms_norm(apply_rope(k, rope), p[pre + "attention.k_norm.weight"])
    R = int(np.prod(lead))
    o = sdpa(q.reshape(R, n_cls, n_heads, -1), k.reshape(R, S, n_heads, -1), v.reshape(R, S, n_heads, -1),
             dtype=dtype).reshape(*lead, n_cls, E)
    x = cls + linear(o, p[pre + "attention.out_projection.weight"], dtype=dtype)
    return x + mlp(p, pre + "mlp.", rms_norm(x, p[pre + "layernorm_mlp.weight"]), dtype=dtype)


def icl_block(p, i, x, k, v, n_heads=16, dtype=None):
    """ICLTransformerBlock on the cache path: test rows attend to cached train K/V."""
    pre = f"icl_blocks.{i}."
    B, M, D = x.shape
    h = rms_norm(x, p[pre + "layernorm.weight"])
    q = linear(h, p[pre + "icl_attention.q_projection.weight"], dtype=dtype).reshape(B, M, n_heads, -1)
    q = rms_norm(q, p[pre + "icl_attention.q_norm.weight"])
    q = softmax_scaling(p, pre + "icl_attention.softmax_scaling_layer.", q, k.shape[1])
    o = sdpa(q, k, v, dtype=dtype).reshape(B, M, D)
    x = x + linear(o, p[pre + "icl_attention.out_projection.weight"], dtype=dtype)
    return x + mlp(p, pre + "mlp.", rms_norm(x, p[pre + "layernorm_mlp.weight"]), dtype=dtype)


def decoder(p, x, keys, y, n_classes, n_heads=6, dtype=None):
    """MultiTaskHeads multiclass: pre-head MLP + ManyClassDecoder -> (M, B, n_classes) logits.

    Attention values are one-hot train labels; only the first `n_classes` columns are
    computed (the reference computes 64 and slices, which gives the same numbers).
    The reference pads to max_num_classes; those columns are the constant
    log(1e-5 + 3e-5) and never reach the probabilities, so they are dropped here.
    """
    x = x + mlp(p, "heads.mlp_classification.mlp.",
                rms_norm(x, p["heads.mlp_classification.norm.weight"]), dtype=dtype)
    B, M, _ = x.shape
    pre = "heads.many_class_decoder."
    q = linear(x, p[pre + "q_projection.weight"], p[pre + "q_projection.bias"], dtype=dtype)
    q = q.reshape(B, M, n_heads, -1)
    q = softmax_scaling(p, pre + "softmax_scaling_layer.", q, keys.shape[1])
    yi = jnp.where(jnp.isfinite(y), y, -1).astype(jnp.int32)  # (B, N)
    D = q.shape[-1]
    if dtype is not None:
        # Same as the reference: values are the one-hot labels padded to head_dim
        # columns, shared by all heads, through flash attention.
        assert n_classes <= D, "more classes than head_dim needs the chunked class loop"
        onehot = (yi[..., None] == jnp.arange(D)).astype(dtype)  # (B, N, D)
        v = jnp.broadcast_to(onehot[:, :, None, :], keys.shape)
        out = sdpa(q, keys, v, dtype=dtype)[..., :n_classes].mean(2)
    else:
        onehot = (yi[..., None] == jnp.arange(n_classes)).astype(jnp.float32)  # (B, N, T)
        s = jnp.einsum("bmhd,bnhd->bhmn", q.astype(jnp.float32), keys.astype(jnp.float32)) / math.sqrt(D)
        a = jax.nn.softmax(s, axis=-1)
        out = jnp.einsum("bhmn,bnt->bmt", a, onehot) / n_heads  # mean over heads
    logits = jnp.log(jnp.maximum(out, 1e-5) + 3e-5)
    return jnp.nan_to_num(jnp.swapaxes(logits, 0, 1), nan=0.0)


# ---------------------------------------------------------------- forward ----

def query_logits(p, ctx, x_MBC, *, n_classes: int, dtype=None,
                 n_dist_blocks=3, n_agg_blocks=3, n_cls=8, n_icl=None):
    """Cached-context forward for test rows `x_MBC` (M, 1, C) -> logits (M, 1, n_classes).

    `x_MBC` is the tensor TabPFN passes to the model (after its CPU/GPU
    preprocessing). `dtype=jnp.bfloat16` runs the big matmuls/attention in bf16
    with fp32 accumulation; None keeps everything fp32.
    """
    x = jnp.swapaxes(x_MBC, 0, 1)  # (B, M, C)
    B, M, C = x.shape
    xs, nan_ind, ecdf = preprocess(x, ctx)
    emb = x_embed(p, group_features(xs, nan_ind, ecdf))  # (B, M, C, E)
    E = emb.shape[-1]
    h = jnp.swapaxes(emb, 1, 2).reshape(B * C, M, E)
    for i in range(n_dist_blocks):
        h = dist_block(p, i, h, ctx["dist_k"][i], ctx["dist_v"][i], dtype=dtype)
    h = jnp.swapaxes(h.reshape(B, C, M, E), 1, 2)  # (B, M, C, E)
    cls = jnp.broadcast_to(p["column_aggregator.cls_tokens"], (B, M, n_cls, E))
    h = jnp.concatenate([cls, h], axis=2)
    for j in range(n_agg_blocks - 1):
        h = agg_self_block(p, j, h, dtype=dtype)
    h = agg_readout_block(p, n_agg_blocks - 1, h, n_cls, dtype=dtype)
    h = rms_norm(h, p["column_aggregator.out_ln.weight"]).reshape(B, M, n_cls * E)
    n_icl = n_icl or ctx["icl_k"].shape[0]
    for i in range(n_icl):
        h = icl_block(p, i, h, ctx["icl_k"][i], ctx["icl_v"][i], dtype=dtype)
    h = rms_norm(h, p["output_norm.weight"])
    return decoder(p, h, ctx["decoder_keys"], ctx["y"], n_classes, dtype=dtype)


def make_predict(n_classes: int, class_permutation=None, dtype=None):
    """jitted (params, ctx, x) -> (logits (M, n_classes), proba (M, n_classes)).

    Probabilities follow TabPFNClassifier.forward for 1 estimator: reorder the
    logits by the member's class permutation, then softmax (temperature 1).
    """
    perm = None if class_permutation is None else jnp.asarray(np.asarray(class_permutation))

    @jax.jit
    def predict(p, ctx, x):
        logits = query_logits(p, ctx, x, n_classes=n_classes, dtype=dtype)[:, 0]
        if perm is not None:
            logits = logits[:, perm]
        return logits, jax.nn.softmax(logits, axis=-1)

    return predict


def check_param_coverage(p, ctx, x, n_classes):
    """Trace once with a tracking dict; assert every weight is used or context-only."""
    tp = _Tracker(p)
    jax.eval_shape(lambda c, xx: query_logits(tp, c, xx, n_classes=n_classes), ctx, x)
    # prepare_context reads the dist-embedder K/V weights
    for i in range(3):
        pre = f"feature_distribution_embedder.layers.{i}.cross_attn_block2."
        tp.used |= {pre + "layernorm_kv.weight", pre + "attn.k_projection.weight",
                    pre + "attn.v_projection.weight", pre + "attn.k_norm.weight"}
    unused = sorted(k for k in p if k not in tp.used and not is_context_only(k))
    wrongly_skipped = sorted(k for k in tp.used if is_context_only(k))
    assert not unused, f"weights never read: {unused}"
    assert not wrongly_skipped, f"context-only weights were read: {wrongly_skipped}"
    return len(tp.used), len(p)
