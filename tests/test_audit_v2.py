from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    REGRESSION_TARGETS,
    _shell_complete_local_indices,
    global_invariants,
)
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.train_ablation import prepare_ablation


def _slab() -> Structure:
    return Structure(
        Lattice.hexagonal(3.1, 20.0),
        ["Nb", "C", "Nb", "O", "O"],
        [
            [0.0, 0.0, 0.46],
            [1 / 3, 2 / 3, 0.50],
            [2 / 3, 1 / 3, 0.54],
            [0.0, 0.0, 0.42],
            [0.0, 0.0, 0.58],
        ],
    )


def test_global_features_are_inplane_supercell_invariant() -> None:
    primitive = _slab()
    supercell = primitive.copy()
    supercell.make_supercell([3, 3, 1])
    np.testing.assert_allclose(
        global_invariants(primitive), global_invariants(supercell), rtol=2e-6, atol=2e-6
    )


def test_neighbor_soft_cap_keeps_whole_degenerate_shell() -> None:
    local = np.arange(5, dtype=np.int64)
    distances = np.asarray([1.0, 1.0, 1.0, 1.5, 2.0], dtype=np.float32)
    kept = _shell_complete_local_indices(local, distances, max_neighbors=2)
    assert set(kept.tolist()) == {0, 1, 2}


def test_graph_semantic_schema_is_v2() -> None:
    assert CACHE_SCHEMA == "nfe-mxene-cache-2.0"
    assert GLOBAL_FEATURE_SCHEMA == "intensive-slab-v2"
    assert NEIGHBOR_POLICY == "radius-shell-complete-v2"


def test_metrics_include_ap_ranking_and_score_rank_quality() -> None:
    logits = np.asarray(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 4], [0, 3, 0]], dtype=float
    )
    labels = np.asarray([0, 1, 2, 2, 1], dtype=int)
    metrics = classification_metrics(logits, labels)
    assert "macro_average_precision" in metrics
    assert "high_average_precision" in metrics
    assert "high_enrichment_at_5pct" in metrics
    pred = np.asarray([[0.1], [0.2], [0.4], [0.8]])
    truth = np.asarray([[0.1], [0.3], [0.5], [0.9]])
    reg = regression_metrics(pred, truth, np.ones_like(pred, dtype=bool), ["score"])
    assert "score_spearman" in reg
    assert "score_r2" in reg


def _config() -> dict:
    return {
        "data": {},
        "model": {},
        "training": {"pretrain_epochs": 35},
        "loss": {
            "score_weight": 1.5,
            "auxiliary_weight": 0.45,
            "masked_atom_weight": 0.35,
            "denoise_weight": 0.65,
        },
        "inference": {},
    }


def test_no_self_supervision_keeps_full_supervised_targets() -> None:
    config, behavior = prepare_ablation(_config(), "no_self_supervision")
    assert config["training"]["pretrain_epochs"] == 0
    assert config["loss"]["masked_atom_weight"] == 0
    assert config["loss"]["denoise_weight"] == 0
    assert config["loss"]["auxiliary_weight"] == 0.45
    assert behavior["target_specs"] == REGRESSION_TARGETS


def test_matched_supervision_is_class_score_only_without_ssl() -> None:
    config, behavior = prepare_ablation(_config(), "matched_supervision")
    assert config["training"]["pretrain_epochs"] == 0
    assert config["loss"]["auxiliary_weight"] == 0
    assert config["loss"]["masked_atom_weight"] == 0
    assert config["loss"]["denoise_weight"] == 0
    assert behavior["target_specs"][0].main
    assert all(not spec.main for spec in behavior["target_specs"][1:])
    assert config["ablation"]["target_policy"] == "class_score_only_no_ssl"
