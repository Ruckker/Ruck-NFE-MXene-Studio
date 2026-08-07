from __future__ import annotations

from typing import Any

import torch

from nfe_model.provenance_v2 import canonical_sha256


def _resolved_device_type(args) -> str:
    explicit = getattr(args, "_resolved_device_type", None)
    if explicit:
        return str(explicit)
    requested = str(getattr(args, "device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested).type


def common_neural_training_protocol(args, data) -> dict[str, Any]:
    """Training/capacity budget shared by controlled, matched, and official neural baselines.

    These are pure supervised architecture comparisons, so their entire
    class+score objective has unit weight from epoch zero. The 0.25x early
    supervised factor belongs only to causal ablations of the full SSL system;
    copying it into external baselines would impose an SSL-specific optimization
    artifact on models that have no SSL objective.
    """

    device_type = _resolved_device_type(args)
    effective_amp = bool((not args.no_amp) and device_type == "cuda")
    return {
        "supervision": "NFE class + NFE pseudo-score only",
        "class_loss": "weighted cross_entropy",
        "score_loss": "smooth_l1_beta_0.5",
        "score_weight": 1.5,
        "optimizer": "AdamW",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "min_learning_rate": float(args.min_learning_rate),
        "warmup_epochs": int(args.warmup_epochs),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "gradient_clip_norm": 5.0,
        "label_smoothing": float(args.label_smoothing),
        "nominal_hidden_dim": int(args.hidden_dim),
        "nominal_layers": int(args.layers),
        "supervised_schedule": "constant_1.0",
        "supervised_factor": 1.0,
        "scheduler": "linear_warmup_cosine",
        "requested_amp": not bool(args.no_amp),
        "resolved_device_type": device_type,
        "effective_amp": effective_amp,
        "graph_radius_A": float(data.config["data"]["radius"]),
        "max_neighbors_soft_cap": int(data.config["data"]["max_neighbors"]),
        "auxiliary_regression": False,
        "masked_atom": False,
        "coordinate_denoising": False,
    }


def common_neural_training_protocol_sha256(args, data) -> str:
    return canonical_sha256(common_neural_training_protocol(args, data))


def neural_model_protocol(name: str, args, data, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Model-specific baseline protocol used to check repeated seeds of one model."""
    result = {
        "common_training_protocol": common_neural_training_protocol(args, data),
        "model": str(name),
    }
    if hasattr(args, "dropout"):
        result["dropout"] = float(args.dropout)
    if extra:
        result["extra"] = extra
    return result


def neural_model_protocol_sha256(
    name: str, args, data, *, extra: dict[str, Any] | None = None
) -> str:
    return canonical_sha256(neural_model_protocol(name, args, data, extra=extra))
