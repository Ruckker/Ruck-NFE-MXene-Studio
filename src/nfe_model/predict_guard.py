from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from . import predict as _predict
from .data_v2 import GLOBAL_FEATURE_SCHEMA, NEIGHBOR_POLICY, build_periodic_graph, torch_load_compat


_ORIGINAL_LOADER = _predict.load_checkpoint_model


def guarded_load_checkpoint_model(path: str | Path, device: torch.device):
    """Reject legacy weights whose graph/global-feature semantics differ from v2."""
    checkpoint = torch_load_compat(path, map_location="cpu")
    provenance = checkpoint.get("provenance", {})
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
    return _ORIGINAL_LOADER(path, device)


def main(argv: Sequence[str] | None = None) -> int:
    original_loader = _predict.load_checkpoint_model
    original_graph = _predict.build_periodic_graph
    try:
        _predict.load_checkpoint_model = guarded_load_checkpoint_model
        _predict.build_periodic_graph = build_periodic_graph
        return _predict.main(argv)
    finally:
        _predict.load_checkpoint_model = original_loader
        _predict.build_periodic_graph = original_graph
