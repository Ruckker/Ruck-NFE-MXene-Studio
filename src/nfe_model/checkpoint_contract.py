from __future__ import annotations

from typing import Any, Mapping

import torch

from .data_v2 import REGRESSION_TARGETS
from .provenance_v2 import NORMALIZER_SCHEMA, tensor_mapping_sha256


CHECKPOINT_NORMALIZER_KEYS = (
    "target_median",
    "target_scale",
    "global_median",
    "global_scale",
)


def assert_checkpoint_internal_contract(checkpoint: Mapping[str, Any]) -> None:
    """Verify that checkpoint payload tensors agree with its own audited metadata."""

    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint has no provenance mapping")
    normalizers = checkpoint.get("normalizers")
    if not isinstance(normalizers, Mapping):
        raise ValueError("checkpoint has no train-fitted normalizers mapping")
    missing = [key for key in CHECKPOINT_NORMALIZER_KEYS if key not in normalizers]
    if missing:
        raise ValueError(f"checkpoint normalizers are missing keys: {missing}")
    selected = {key: normalizers[key] for key in CHECKPOINT_NORMALIZER_KEYS}
    for key, value in selected.items():
        if not torch.is_tensor(value):
            raise TypeError(f"checkpoint normalizer {key} is not a tensor")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"checkpoint normalizer {key} contains non-finite values")
    if torch.any(selected["target_scale"] <= 0) or torch.any(selected["global_scale"] <= 0):
        raise ValueError("checkpoint normalizer scales must be strictly positive")
    observed_normalizer_hash = tensor_mapping_sha256(
        selected, schema=NORMALIZER_SCHEMA
    )
    expected_normalizer_hash = str(provenance.get("normalizer_sha256", ""))
    if not expected_normalizer_hash or observed_normalizer_hash != expected_normalizer_hash:
        raise ValueError(
            "checkpoint normalizer tensors do not match checkpoint provenance: "
            f"tensors={observed_normalizer_hash} provenance={expected_normalizer_hash or 'missing'}"
        )
    if provenance.get("normalizer_schema") != NORMALIZER_SCHEMA:
        raise ValueError(
            f"checkpoint normalizer schema={provenance.get('normalizer_schema')!r}, "
            f"expected {NORMALIZER_SCHEMA!r}"
        )

    target_count = len(REGRESSION_TARGETS)
    if selected["target_median"].numel() != target_count:
        raise ValueError(
            f"checkpoint target normalizer width={selected['target_median'].numel()} "
            f"but current target contract has {target_count} targets"
        )
    if selected["target_scale"].shape != selected["target_median"].shape:
        raise ValueError("checkpoint target_median/target_scale shapes disagree")
    if selected["global_median"].shape != selected["global_scale"].shape:
        raise ValueError("checkpoint global_median/global_scale shapes disagree")

    model_config = checkpoint.get("base_model_config", checkpoint.get("model_config"))
    if isinstance(model_config, Mapping):
        configured_targets = model_config.get("num_regression_targets")
        if configured_targets is not None and int(configured_targets) != target_count:
            raise ValueError(
                f"checkpoint model expects {configured_targets} regression targets; current contract has {target_count}"
            )
        configured_globals = model_config.get("global_features")
        if configured_globals is not None and int(configured_globals) != int(
            selected["global_median"].numel()
        ):
            raise ValueError(
                "checkpoint model global feature width disagrees with train normalizer width"
            )
