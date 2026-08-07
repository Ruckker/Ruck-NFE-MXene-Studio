from __future__ import annotations

from typing import Any

from nfe_model.provenance_v2 import canonical_sha256


def common_neural_training_protocol(args, data) -> dict[str, Any]:
    """Training budget shared by controlled, matched, and official neural baselines."""
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
        "early_supervised_epochs": int(data.config.get("training", {}).get("pretrain_epochs", 0)),
        "early_supervised_factor": 0.25,
        "scheduler": "linear_warmup_cosine",
        "amp": not bool(args.no_amp),
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
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
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
