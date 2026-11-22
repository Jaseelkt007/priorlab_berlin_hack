"""Export a fitted TabPFNClassifier's context to an .npz bundle the JAX runtime loads.

Runs in the torch env (.venv). The bundle holds everything a cached query needs that
is not a model weight: the KV cache (ICL K/V, decoder keys, inducing states, scaler
and ECDF context), the train labels, and the fitted per-call preprocessing (constant
column mask, fingerprint salt, column shuffle, soft-clip bounds, class permutation).
It is derived from the training rows, so keep it out of git like the weights.

Supported: 1 estimator, all-numeric features, the 3.5 default preprocessing
(transform "none", no categoricals). Anything else raises instead of silently
producing different numbers.
"""
from __future__ import annotations

import numpy as np
import torch


def _np(t):
    t = t.detach().cpu()
    return t.float().numpy() if t.dtype.is_floating_point else t.numpy()


def _kv_float(entry):
    """Dense fp32 K/V from a cache entry, dequantising int8/fp8 entries."""
    k, v = entry.key.float(), entry.value.float()
    if getattr(entry, "key_scale", None) is not None:
        k, v = k * entry.key_scale.float(), v * entry.value_scale.float()
    return _np(k), _np(v)


def bundle_from_classifier(clf) -> dict[str, np.ndarray]:
    from tabpfn.preprocessing.steps.encode_categorical_features_step import EncodeCategoricalFeaturesStep
    from tabpfn.preprocessing.steps.remove_constant_features_step import RemoveConstantFeaturesStep
    from tabpfn.preprocessing.steps.reshape_feature_distribution_step import ReshapeFeatureDistributionsStep
    from tabpfn.preprocessing.torch.steps import (TorchAddFingerprintFeaturesStep, TorchShuffleFeaturesStep,
                                                  TorchSoftClipOutliersStep)

    ex = clf.executor_
    assert len(ex.ensemble_members) == 1 and len(ex.kv_caches) == 1, "export supports n_estimators=1"
    mem = ex.ensemble_members[0]

    cpu = [s for s, _ in mem.cpu_preprocessor.steps]
    assert [type(s) for s in cpu] == [RemoveConstantFeaturesStep, ReshapeFeatureDistributionsStep,
                                      EncodeCategoricalFeaturesStep], f"unsupported CPU steps {cpu}"
    assert cpu[1].transformer_ is None and cpu[1].data_is_unchanged_, "only transform 'none' is supported"
    assert cpu[2].categorical_transformer_ is None, "categorical features are not supported"

    gpu = mem.gpu_preprocessor
    assert [type(s) for s, _ in gpu.steps] == [TorchAddFingerprintFeaturesStep, TorchShuffleFeaturesStep,
                                               TorchSoftClipOutliersStep], f"unsupported GPU steps {gpu.steps}"
    fp_cache, shuffle_cache, clip_cache = gpu.fitted_cache
    n_cols = int(np.sum(cpu[0].sel_)) + 1  # + fingerprint column
    clip_lower, clip_upper = _np(clip_cache["lower"]).ravel(), _np(clip_cache["upper"]).ravel()
    # The soft clip runs on the NUMERICAL columns of the shuffled schema; with an
    # all-numeric table that is every column, which is what the bound width says.
    assert clip_lower.shape == (n_cols,), f"soft clip covers {clip_lower.shape} of {n_cols} columns"

    cache = ex.kv_caches[0]
    b = {
        "sel": np.asarray(cpu[0].sel_, dtype=bool),
        "fingerprint_salt": np.array(int(fp_cache["n_cells"].item()), dtype=np.int64),
        "permutation": _np(shuffle_cache["permutation"]).astype(np.int64),
        "clip_lower": clip_lower,
        "clip_upper": clip_upper,
        "class_permutation": np.asarray(mem.config.class_permutation, dtype=np.int64),
        "n_classes": np.array(clf.n_classes_, dtype=np.int64),
        "y": np.asarray(mem.y_train, dtype=np.float32).reshape(-1),
        "scaler_mean": _np(cache.scaler_cache["mean"]).reshape(-1),
        "scaler_std": _np(cache.scaler_cache["std"]).reshape(-1),
        "ecdf_context": _np(cache.ecdf_context),
        "decoder_keys": _np(cache.decoder_keys),
    }
    assert clf.softmax_temperature_ == 1.0 and not clf.balance_probabilities, "only T=1, no balancing"
    assert getattr(clf, "downsample_correction_weights_", None) is None
    for i, h in enumerate(cache.inducing_hidden):
        b[f"inducing_hidden_{i}"] = _np(h)
    for i, e in sorted(cache.kv.items()):
        b[f"kv_{i}_key"], b[f"kv_{i}_value"] = _kv_float(e)
    return b


def export_bundle(clf, path) -> dict[str, np.ndarray]:
    b = bundle_from_classifier(clf)
    np.savez(path, **b)
    return b
