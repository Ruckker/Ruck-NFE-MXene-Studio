from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from . import predict as _predict
from .data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    build_periodic_graph,
    torch_load_compat,
)


_ORIGINAL_LOADER = _predict.load_checkpoint_model
_ENSEMBLE_GRAPH_CONTRACT: tuple[str, str, str, str, str, float, int] | None = None


def guarded_load_checkpoint_model(path: str | Path, device: torch.device):
    """Reject legacy weights and incompatible ensemble graph contracts."""
    global _ENSEMBLE_GRAPH_CONTRACT
    checkpoint = torch_load_compat(path, map_location="cpu")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("cache_schema") != CACHE_SCHEMA:
        raise ValueError(
            "legacy/incompatible predictor checkpoint: cache_schema="
            f"{provenance.get('cache_schema', 'missing')}; expected {CACHE_SCHEMA}."
        )
    if provenance.get("global_feature_schema") != GLOBAL_FEATURE_SCHEMA:
        raise ValueError(
            "legacy/incompatible predictor checkpoint: global_feature_schema="
            f"{provenance.get('global_feature_schema', 'missing')}; expected {GLOBAL_FEATURE_SCHEMA}. "
            "Retrain/re-export with the audited v2 graph cache before production inference."
        )
    if provenance.get("neighbor_policy") != NEIGHBOR_POLICY:
        raise ValueError(
            "legacy/incompatible predictor checkpoint: neighbor_policy="
            f"{provenance.get('neighbor_policy', 'missing')}; expected {NEIGHBOR_POLICY}."
        )
    config = checkpoint.get("config", {}).get("data", {})
    radius = float(config.get("radius", provenance.get("graph_radius_A", -1.0)))
    max_neighbors = int(config.get("max_neighbors", provenance.get("max_neighbors", -1)))
    if radius <= 0 or max_neighbors <= 0:
        raise ValueError("checkpoint is missing a valid graph radius/max_neighbors contract")
    if abs(radius - float(provenance.get("graph_radius_A", radius))) > 1e-12:
        raise ValueError("checkpoint config radius disagrees with checkpoint provenance")
    if max_neighbors != int(provenance.get("max_neighbors", max_neighbors)):
        raise ValueError("checkpoint config max_neighbors disagrees with checkpoint provenance")
    dataset_hash = str(provenance.get("dataset_table_sha256", ""))
    split_hash = str(provenance.get("split_manifest_sha256", ""))
    if not dataset_hash or not split_hash:
        raise ValueError("checkpoint is missing dataset/split provenance for ensemble inference")
    contract = (
        dataset_hash,
        split_hash,
        CACHE_SCHEMA,
        GLOBAL_FEATURE_SCHEMA,
        NEIGHBOR_POLICY,
        radius,
        max_neighbors,
    )
    if _ENSEMBLE_GRAPH_CONTRACT is None:
        _ENSEMBLE_GRAPH_CONTRACT = contract
    elif contract != _ENSEMBLE_GRAPH_CONTRACT:
        raise ValueError(
            "ensemble checkpoints use incompatible graph contracts: "
            f"{contract} != {_ENSEMBLE_GRAPH_CONTRACT}"
        )
    return _ORIGINAL_LOADER(path, device)


def main(argv: Sequence[str] | None = None) -> int:
    global _ENSEMBLE_GRAPH_CONTRACT
    _ENSEMBLE_GRAPH_CONTRACT = None
    original_loader = _predict.load_checkpoint_model
    original_graph = _predict.build_periodic_graph
    try:
        _predict.load_checkpoint_model = guarded_load_checkpoint_model
        _predict.build_periodic_graph = build_periodic_graph
        return _predict.main(argv)
    finally:
        _predict.load_checkpoint_model = original_loader
        _predict.build_periodic_graph = original_graph
