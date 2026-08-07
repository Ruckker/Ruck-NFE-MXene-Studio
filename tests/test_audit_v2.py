from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    REGRESSION_TARGETS,
    STRUCTURE_MANIFEST_SCHEMA,
    _shell_complete_local_indices,
    build_periodic_graph,
    collate_graphs,
    global_invariants,
)
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.model import PeriodicNFEModel
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


def _batch(structure: Structure) -> dict[str, torch.Tensor]:
    graph = build_periodic_graph(structure, radius=5.0, max_neighbors=36)
    graph.update(
        {
            "targets": torch.zeros(len(REGRESSION_TARGETS)),
            "target_mask": torch.ones(len(REGRESSION_TARGETS), dtype=torch.bool),
            "label": 1,
        }
    )
    return collate_graphs([graph])


def _full_model() -> PeriodicNFEModel:
    torch.manual_seed(41)
    return PeriodicNFEModel(
        hidden_dim=40,
        vector_dim=16,
        num_layers=2,
        num_rbf=20,
        cutoff=5.0,
        dropout=0.0,
        num_regression_targets=len(REGRESSION_TARGETS),
    ).eval()


def test_full_model_is_invariant_to_site_reordering_with_v2_graph() -> None:
    structure = _slab()
    reordered = Structure.from_sites(
        [structure[index] for index in [3, 0, 4, 2, 1]], to_unit_cell=True
    )
    model = _full_model()
    with torch.no_grad():
        first = model(_batch(structure))
        second = model(_batch(reordered))
    assert torch.allclose(first["class_logits"], second["class_logits"], atol=2e-4, rtol=2e-4)
    assert torch.allclose(
        first["regression_mean"], second["regression_mean"], atol=2e-4, rtol=2e-4
    )


def test_full_model_is_consistent_under_exact_inplane_supercell_replication() -> None:
    primitive = _slab()
    supercell = primitive.copy()
    supercell.make_supercell([2, 2, 1])
    model = _full_model()
    with torch.no_grad():
        first = model(_batch(primitive))
        second = model(_batch(supercell))
    assert torch.allclose(first["class_logits"], second["class_logits"], atol=5e-4, rtol=5e-4)
    assert torch.allclose(
        first["regression_mean"], second["regression_mean"], atol=5e-4, rtol=5e-4
    )


def test_graph_semantic_schema_is_v2_1() -> None:
    assert CACHE_SCHEMA == "nfe-mxene-cache-2.1"
    assert GLOBAL_FEATURE_SCHEMA == "intensive-slab-v2"
    assert NEIGHBOR_POLICY == "radius-shell-complete-v2"
    assert STRUCTURE_MANIFEST_SCHEMA == "source-bytes-v1"


def test_metrics_include_ap_ranking_and_score_rank_quality() -> None:
    logits = np.asarray(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 4], [0, 3, 0]], dtype=float
    )
    labels = np.asarray([0, 1, 2, 2, 1], dtype=int)
    metrics = classification_metrics(logits, labels)
    assert "macro_average_precision" in metrics
    assert "high_average_precision" in metrics
    assert "high_enrichment_at_5pct" in metrics
    prediction = np.asarray([[0.1], [0.2], [0.4], [0.8]])
    truth = np.asarray([[0.1], [0.3], [0.5], [0.9]])
    regression = regression_metrics(
        prediction, truth, np.ones_like(prediction, dtype=bool), ["score"]
    )
    assert "score_spearman" in regression
    assert "score_r2" in regression


def test_undefined_one_vs_rest_metrics_are_nan_not_fake_chance() -> None:
    logits = np.asarray([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    labels = np.asarray([0, 1])
    metrics = classification_metrics(logits, labels)
    assert np.isnan(metrics["high_roc_auc"])
    assert np.isnan(metrics["high_average_precision"])
    assert np.isnan(metrics["high_recall_at_5pct"])


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


def test_no_self_supervision_keeps_supervised_schedule() -> None:
    config, behavior = prepare_ablation(_config(), "no_self_supervision")
    assert config["training"]["pretrain_epochs"] == 35
    assert config["loss"]["masked_atom_weight"] == 0
    assert config["loss"]["denoise_weight"] == 0
    assert config["loss"]["auxiliary_weight"] == 0.45
    assert behavior["target_specs"] == REGRESSION_TARGETS


def test_matched_supervision_is_class_score_only_without_ssl_and_same_schedule() -> None:
    config, behavior = prepare_ablation(_config(), "matched_supervision")
    assert config["training"]["pretrain_epochs"] == 35
    assert config["loss"]["auxiliary_weight"] == 0
    assert config["loss"]["masked_atom_weight"] == 0
    assert config["loss"]["denoise_weight"] == 0
    assert behavior["target_specs"][0].main
    assert all(not spec.main for spec in behavior["target_specs"][1:])
    assert config["ablation"]["target_policy"] == "class_score_only_no_ssl"
