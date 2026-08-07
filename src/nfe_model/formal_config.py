from __future__ import annotations

import math
from typing import Any, Mapping

from .data_v2 import ELEMENT_FEATURE_DIM, GLOBAL_FEATURE_DIM, REGRESSION_TARGETS


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"formal predictor config requires mapping section {name!r}")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return parsed


def _finite_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return parsed


def validate_formal_config(config: Mapping[str, Any]) -> None:
    """Reject numerically or physically inconsistent formal predictor settings.

    The validator intentionally targets the audited predictor/ablation/benchmark
    contract rather than every experimental configuration the repository may
    contain. Formal runs should fail before cache construction or GPU work when
    graph, model, target, or optimization semantics are contradictory.
    """

    data = _section(config, "data")
    model = _section(config, "model")
    training = _section(config, "training")
    loss = _section(config, "loss")
    inference = _section(config, "inference")

    radius = _finite_float(data.get("radius"), "data.radius")
    if radius <= 0:
        raise ValueError("data.radius must be > 0")
    _positive_int(data.get("max_neighbors"), "data.max_neighbors")
    skip_fraction = _finite_float(
        data.get("max_cache_skip_fraction", 0.01), "data.max_cache_skip_fraction"
    )
    if not 0.0 <= skip_fraction < 1.0:
        raise ValueError("data.max_cache_skip_fraction must be in [0, 1)")

    cutoff = _finite_float(model.get("cutoff", radius), "model.cutoff")
    if cutoff <= 0:
        raise ValueError("model.cutoff must be > 0")
    if not math.isclose(radius, cutoff, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "formal predictor requires data.radius == model.cutoff so graph and message "
            f"budgets are identical; got radius={radius} cutoff={cutoff}"
        )

    for key in ("hidden_dim", "vector_dim", "num_layers", "num_rbf"):
        _positive_int(model.get(key), f"model.{key}")
    dropout = _finite_float(model.get("dropout", 0.0), "model.dropout")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    max_atomic_number = _positive_int(
        model.get("max_atomic_number", 118), "model.max_atomic_number"
    )
    if max_atomic_number < 118:
        raise ValueError(
            "formal predictor requires max_atomic_number >= 118; smaller vocabularies can "
            "silently alias or reject chemically valid OOD elements"
        )
    if int(model.get("element_features", ELEMENT_FEATURE_DIM)) != int(ELEMENT_FEATURE_DIM):
        raise ValueError(
            "model.element_features disagrees with the cached elemental descriptor width: "
            f"{model.get('element_features')} != {ELEMENT_FEATURE_DIM}"
        )
    if int(model.get("global_features", GLOBAL_FEATURE_DIM)) != int(GLOBAL_FEATURE_DIM):
        raise ValueError(
            "model.global_features disagrees with the audited global descriptor width: "
            f"{model.get('global_features')} != {GLOBAL_FEATURE_DIM}"
        )
    if "num_regression_targets" in model and int(model["num_regression_targets"]) != len(
        REGRESSION_TARGETS
    ):
        raise ValueError(
            "model.num_regression_targets disagrees with the audited target contract: "
            f"{model['num_regression_targets']} != {len(REGRESSION_TARGETS)}"
        )

    epochs = _positive_int(training.get("epochs"), "training.epochs")
    pretrain_epochs = int(training.get("pretrain_epochs", 0))
    if not 0 <= pretrain_epochs <= epochs:
        raise ValueError("training.pretrain_epochs must be between 0 and training.epochs")
    _positive_int(training.get("batch_size_per_gpu"), "training.batch_size_per_gpu")
    grad_accum = _positive_int(training.get("grad_accum_steps", 1), "training.grad_accum_steps")
    if grad_accum != 1:
        raise ValueError(
            "audited predictor currently requires training.grad_accum_steps == 1; "
            "the historical core does not normalize the final partial accumulation window safely"
        )
    learning_rate = _finite_float(training.get("learning_rate"), "training.learning_rate")
    minimum_lr = _finite_float(
        training.get("min_learning_rate", 0.0), "training.min_learning_rate"
    )
    if learning_rate <= 0 or minimum_lr < 0 or minimum_lr > learning_rate:
        raise ValueError(
            "training learning rates require 0 <= min_learning_rate <= learning_rate and learning_rate > 0"
        )
    weight_decay = _finite_float(training.get("weight_decay", 0.0), "training.weight_decay")
    if weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0")
    warmup_epochs = int(training.get("warmup_epochs", 0))
    if not 0 <= warmup_epochs <= epochs:
        raise ValueError("training.warmup_epochs must be between 0 and training.epochs")
    _positive_int(
        training.get("early_stopping_patience", 1), "training.early_stopping_patience"
    )
    grad_clip = _finite_float(training.get("grad_clip", 0.0), "training.grad_clip")
    if grad_clip <= 0:
        raise ValueError("training.grad_clip must be > 0")

    for key in (
        "class_weight",
        "score_weight",
        "auxiliary_weight",
        "masked_atom_weight",
        "denoise_weight",
    ):
        value = _finite_float(loss.get(key, 0.0), f"loss.{key}")
        if value < 0:
            raise ValueError(f"loss.{key} must be >= 0")
    label_smoothing = _finite_float(loss.get("label_smoothing", 0.0), "loss.label_smoothing")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("loss.label_smoothing must be in [0, 1)")

    mask_probability = _finite_float(loss.get("mask_probability", 0.0), "loss.mask_probability")
    if not 0.0 <= mask_probability < 1.0:
        raise ValueError("loss.mask_probability must be in [0, 1)")
    if float(loss.get("masked_atom_weight", 0.0)) > 0 and mask_probability <= 0:
        raise ValueError(
            "loss.mask_probability must be > 0 when the masked-atom objective is enabled"
        )

    noise_min = _finite_float(
        loss.get("coordinate_noise_min_A", 0.0), "loss.coordinate_noise_min_A"
    )
    noise_max = _finite_float(
        loss.get("coordinate_noise_max_A", 0.0), "loss.coordinate_noise_max_A"
    )
    if float(loss.get("denoise_weight", 0.0)) > 0:
        if noise_min <= 0 or noise_max < noise_min:
            raise ValueError(
                "coordinate denoising requires 0 < coordinate_noise_min_A <= coordinate_noise_max_A"
            )
    elif noise_min < 0 or noise_max < 0:
        raise ValueError("coordinate-noise magnitudes cannot be negative")

    _positive_int(inference.get("mc_samples", 1), "inference.mc_samples")
    confidence = _finite_float(
        inference.get("confidence_level", 0.90), "inference.confidence_level"
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("inference.confidence_level must be strictly between 0 and 1")
