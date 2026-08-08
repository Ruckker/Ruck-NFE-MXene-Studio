from __future__ import annotations

import argparse
import functools
import math
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from .ablation import AblationPeriodicNFEModel
from .data_v2 import (
    REGRESSION_TARGETS as BASE_REGRESSION_TARGETS,
    TargetSpec,
    torch_load_compat,
)
from .formal_config import validate_formal_config
from .provenance_v2 import assert_matching_experiment_protocol
from .train_audit_v2 import install_audit_patches
from .utils import load_config


ABLATIONS = (
    "full",
    "no_vector",
    "no_global",
    "no_masked_pretrain",
    "no_denoise",
    "no_self_supervision",
    "no_auxiliary_regression",
    "matched_supervision",
    "classification_only",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train controlled ablations of the periodic NFE predictor."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def _absolute_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base = config_path.resolve().parent
    for key in ("table", "root", "cache"):
        path = Path(config["data"][key])
        if not path.is_absolute():
            path = base / path
        config["data"][key] = str(path.resolve())
    return config


def _target_specs_for(ablation: str) -> tuple[TargetSpec, ...]:
    if ablation in {"no_auxiliary_regression", "matched_supervision"}:
        return tuple(
            TargetSpec(spec.name, spec.transform, main=(index == 0))
            for index, spec in enumerate(BASE_REGRESSION_TARGETS)
        )
    if ablation == "classification_only":
        return tuple(
            TargetSpec(spec.name, spec.transform, main=False)
            for spec in BASE_REGRESSION_TARGETS
        )
    return BASE_REGRESSION_TARGETS


def _classification_selection_score(metrics: dict[str, float]) -> float:
    def finite(key: str, fallback: float) -> float:
        value = float(metrics.get(key, fallback))
        return value if math.isfinite(value) else fallback

    macro_f1 = finite("macro_f1", 0.0)
    macro_auc = finite("macro_roc_auc", 0.5)
    calibration = max(0.0, 1.0 - finite("ece", 1.0))
    return 0.55 * macro_f1 + 0.35 * macro_auc + 0.10 * calibration


def _active_target_heteroscedastic_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve positive-weight scaling while excluding disabled targets."""
    squared = (mean - target) ** 2
    loss = 0.5 * torch.exp(-log_variance) * squared + 0.5 * log_variance
    effective_mask = mask.to(loss.dtype)
    if target_weights is not None:
        weights = target_weights.view(1, -1)
        loss = loss * weights
        effective_mask = effective_mask * (weights > 0).to(loss.dtype)
    if sample_weights is not None:
        effective_mask = effective_mask * sample_weights.view(-1, 1)
    denominator = effective_mask.sum()
    if float(denominator.detach()) <= 0:
        return mean.sum() * 0.0
    return torch.sum(loss * effective_mask) / denominator


def _make_corrupt_structure(*, enable_masking: bool, enable_denoising: bool):
    def corrupt_structure(
        batch: dict[str, Any],
        *,
        mask_probability: float,
        noise_min: float,
        noise_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_original = batch["z"]
        if enable_masking:
            mask = torch.rand(z_original.shape[0], device=z_original.device) < float(
                mask_probability
            )
            if not torch.any(mask):
                mask[torch.randint(0, z_original.shape[0], (1,), device=z_original.device)] = True
            z_corrupted = z_original.clone()
            z_corrupted[mask] = 0
        else:
            mask = torch.zeros_like(z_original, dtype=torch.bool)
            z_corrupted = z_original

        if enable_denoising:
            if float(noise_min) <= 0 or float(noise_max) < float(noise_min):
                raise ValueError("invalid coordinate-noise range for denoising ablation")
            n_graphs = batch["lattice"].shape[0]
            log_sigma = torch.empty(n_graphs, device=z_original.device).uniform_(
                math.log(float(noise_min)), math.log(float(noise_max))
            )
            sigma = torch.exp(log_sigma)
            noise_cart = torch.randn_like(batch["frac_pos"]) * sigma[
                batch["batch"]
            ].unsqueeze(-1)
            inverse_lattice = torch.linalg.inv(batch["lattice"])
            noise_fractional = torch.einsum(
                "ni,nij->nj", noise_cart, inverse_lattice[batch["batch"]]
            )
            noisy_fractional = batch["frac_pos"] + noise_fractional
            denoise_target = -noise_cart
        else:
            noisy_fractional = batch["frac_pos"]
            denoise_target = torch.zeros_like(batch["frac_pos"])
        return z_corrupted, noisy_fractional, mask, denoise_target

    return corrupt_structure


def prepare_ablation(
    config: dict[str, Any], ablation: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = dict(config)
    config["data"] = dict(config["data"])
    config["model"] = dict(config["model"])
    config["training"] = dict(config["training"])
    config["loss"] = dict(config["loss"])
    config["inference"] = dict(config["inference"])

    behavior = {
        "use_vector_features": True,
        "use_global_features": True,
        "enable_masking": True,
        "enable_denoising": True,
        "target_specs": _target_specs_for(ablation),
        "classification_selection": False,
    }

    if ablation == "no_vector":
        behavior["use_vector_features"] = False
        behavior["enable_denoising"] = False
        config["loss"]["denoise_weight"] = 0.0
    elif ablation == "no_global":
        behavior["use_global_features"] = False
    elif ablation == "no_masked_pretrain":
        behavior["enable_masking"] = False
        config["loss"]["masked_atom_weight"] = 0.0
    elif ablation == "no_denoise":
        behavior["enable_denoising"] = False
        config["loss"]["denoise_weight"] = 0.0
    elif ablation == "no_self_supervision":
        behavior["enable_masking"] = False
        behavior["enable_denoising"] = False
        config["loss"]["masked_atom_weight"] = 0.0
        config["loss"]["denoise_weight"] = 0.0
    elif ablation == "no_auxiliary_regression":
        config["loss"]["auxiliary_weight"] = 0.0
    elif ablation == "matched_supervision":
        behavior["enable_masking"] = False
        behavior["enable_denoising"] = False
        config["loss"]["auxiliary_weight"] = 0.0
        config["loss"]["masked_atom_weight"] = 0.0
        config["loss"]["denoise_weight"] = 0.0
    elif ablation == "classification_only":
        behavior["enable_masking"] = False
        behavior["enable_denoising"] = False
        behavior["classification_selection"] = True
        config["loss"]["score_weight"] = 0.0
        config["loss"]["auxiliary_weight"] = 0.0
        config["loss"]["masked_atom_weight"] = 0.0
        config["loss"]["denoise_weight"] = 0.0

    if ablation == "classification_only":
        target_policy = "classification_only"
    elif ablation == "matched_supervision":
        target_policy = "class_score_only_no_ssl"
    elif ablation == "no_auxiliary_regression":
        target_policy = "class_score_only_with_ssl"
    else:
        target_policy = "full_multitask"

    config["ablation"] = {
        "name": ablation,
        "use_vector_features": behavior["use_vector_features"],
        "use_global_features": behavior["use_global_features"],
        "enable_masking": behavior["enable_masking"],
        "enable_denoising": behavior["enable_denoising"],
        "target_policy": target_policy,
        "ssl_policy": (
            "none"
            if not behavior["enable_masking"] and not behavior["enable_denoising"]
            else "partial"
            if not behavior["enable_masking"] or not behavior["enable_denoising"]
            else "full"
        ),
        "supervised_weight_schedule": "retained_from_full",
        "supervised_weight_schedule_epochs": int(config["training"].get("pretrain_epochs", 0)),
        "capacity_preserving_representation_ablation": ablation in {"no_vector", "no_global"},
    }
    return config, behavior


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = _absolute_config_paths(load_config(config_path), config_path)
    config, behavior = prepare_ablation(config, args.ablation)
    project_root = Path(__file__).resolve().parents[2]

    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size_per_gpu"] = int(args.batch_size)
    if args.patience is not None:
        config["training"]["early_stopping_patience"] = int(args.patience)

    checkpoint_dir = (
        Path(args.checkpoint_dir).resolve()
        if args.checkpoint_dir
        else project_root
        / "runs"
        / "ablations"
        / args.ablation
        / f"seed_{int(config['seed'])}"
    )
    config["training"]["checkpoint_dir"] = str(checkpoint_dir)
    validate_formal_config(config)

    if args.resume:
        resume_path = Path(args.resume).resolve()
        checkpoint = torch_load_compat(resume_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"resume checkpoint is not a mapping: {resume_path}")
        assert_matching_experiment_protocol(checkpoint, config)

    from . import train_core as train_module

    install_audit_patches(train_module)
    original_model = train_module.PeriodicNFEModel
    original_targets = train_module.REGRESSION_TARGETS
    original_corrupt = train_module.corrupt_structure
    original_selection = train_module.selection_score
    original_heteroscedastic = train_module.heteroscedastic_loss
    runtime_config: Path | None = None
    try:
        train_module.REGRESSION_TARGETS = behavior["target_specs"]
        train_module.heteroscedastic_loss = _active_target_heteroscedastic_loss
        train_module.corrupt_structure = _make_corrupt_structure(
            enable_masking=bool(behavior["enable_masking"]),
            enable_denoising=bool(behavior["enable_denoising"]),
        )
        if behavior["classification_selection"]:
            train_module.selection_score = _classification_selection_score
        if not behavior["use_vector_features"] or not behavior["use_global_features"]:
            train_module.PeriodicNFEModel = functools.partial(
                AblationPeriodicNFEModel,
                use_vector_features=bool(behavior["use_vector_features"]),
                use_global_features=bool(behavior["use_global_features"]),
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"nfe_{args.ablation}_",
            encoding="utf-8",
            delete=False,
        ) as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
            runtime_config = Path(handle.name)
        worker_args = ["--config", str(runtime_config)]
        if args.resume:
            worker_args.extend(["--resume", str(Path(args.resume).resolve())])
        if args.rebuild_cache:
            worker_args.append("--rebuild-cache")
        return train_module.main(worker_args)
    finally:
        train_module.PeriodicNFEModel = original_model
        train_module.REGRESSION_TARGETS = original_targets
        train_module.corrupt_structure = original_corrupt
        train_module.selection_score = original_selection
        train_module.heteroscedastic_loss = original_heteroscedastic
        if runtime_config is not None:
            runtime_config.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
