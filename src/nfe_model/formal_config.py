from __future__ import annotations

import inspect
import math
from typing import Any, Mapping

from .data_v2 import ELEMENT_FEATURE_DIM, GLOBAL_FEATURE_DIM, REGRESSION_TARGETS
from .model import PeriodicNFEModel


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"formal predictor config requires mapping section {name!r}")
    return value


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return parsed


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite; got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return parsed


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a YAML/JSON boolean; got {value!r}")
    return bool(value)


def validate_formal_config(config: Mapping[str, Any]) -> None:
    """Reject numerically or physically inconsistent formal predictor settings.

    The validator targets the audited predictor/ablation/benchmark contract.
    Formal runs should fail before cache construction or GPU work when graph,
    model, target, optimization, or inference semantics are contradictory.

    ``pretrain_epochs`` and ``warmup_epochs`` may exceed ``epochs``. That is
    useful for deliberately truncated smoke runs: the short run simply remains
    inside the early schedule window rather than mutating the formal schedule.
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
    _nonnegative_int(data.get("num_workers", 0), "data.num_workers")
    if "pin_memory" in data:
        _boolean(data["pin_memory"], "data.pin_memory")
    if "rebuild_cache" in data:
        _boolean(data["rebuild_cache"], "data.rebuild_cache")

    allowed_model_keys = {
        key
        for key in inspect.signature(PeriodicNFEModel.__init__).parameters
        if key != "self"
    }
    unknown_model_keys = sorted(set(model) - allowed_model_keys)
    if unknown_model_keys:
        raise ValueError(f"formal model config has unsupported keys: {unknown_model_keys}")
    if "num_regression_targets" in model:
        raise ValueError(
            "model.num_regression_targets must be omitted: the formal trainer derives it "
            "from the audited regression-target contract"
        )

    if "cutoff" not in model:
        raise ValueError(
            "formal model.cutoff must be explicit; relying on the model constructor default "
            "could disagree with data.radius"
        )
    cutoff = _finite_float(model["cutoff"], "model.cutoff")
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
    if "num_classes" in model and int(model["num_classes"]) != 3:
        raise ValueError("formal NFE classification requires model.num_classes == 3")

    _positive_int(training.get("epochs"), "training.epochs")
    _nonnegative_int(training.get("pretrain_epochs", 0), "training.pretrain_epochs")
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
    _nonnegative_int(training.get("warmup_epochs", 0), "training.warmup_epochs")
    _positive_int(
        training.get("early_stopping_patience", 1), "training.early_stopping_patience"
    )
    grad_clip = _finite_float(training.get("grad_clip", 0.0), "training.grad_clip")
    if grad_clip <= 0:
        raise ValueError("training.grad_clip must be > 0")
    _positive_int(training.get("log_every", 1), "training.log_every")
    if "amp" in training:
        _boolean(training["amp"], "training.amp")
    if "compile" in training:
        _boolean(training["compile"], "training.compile")

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
    _positive_int(inference.get("embedding_bank_size", 1), "inference.embedding_bank_size")
