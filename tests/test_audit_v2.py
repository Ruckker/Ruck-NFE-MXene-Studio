from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from nfe_model.data_contract import DATA_IMPLEMENTATION_SCHEMA, data_implementation_sha256
from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    REGRESSION_TARGETS,
    STRUCTURE_MANIFEST_SCHEMA,
    TARGET_SCHEMA,
    _shell_complete_local_indices,
    assert_disjoint_split_groups,
    build_periodic_graph,
    collate_graphs,
    global_invariants,
    split_indices,
    target_schema_sha256,
)
from nfe_model.formal_data import assert_formal_primary_target_coverage
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.model import PeriodicNFEModel
from nfe_model.provenance_v2 import experiment_protocol_sha256, training_protocol_sha256
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


def _equivalent_inplane_basis(structure: Structure) -> Structure:
    transformed = structure.copy()
    transformed.make_supercell([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    return transformed


def _thicker_vacuum_same_cartesian_slab(structure: Structure) -> Structure:
    matrix = np.asarray(structure.lattice.matrix, dtype=float).copy()
    matrix[2] *= 1.5
    return Structure(
        Lattice(matrix),
        structure.species,
        structure.cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )


def test_global_features_are_inplane_supercell_invariant() -> None:
    primitive = _slab()
    supercell = primitive.copy()
    supercell.make_supercell([3, 3, 1])
    np.testing.assert_allclose(
        global_invariants(primitive), global_invariants(supercell), rtol=2e-6, atol=2e-6
    )


def test_global_features_ignore_equivalent_inplane_basis_choice() -> None:
    primitive = _slab()
    transformed = _equivalent_inplane_basis(primitive)
    assert len(transformed) == len(primitive)
    np.testing.assert_allclose(
        global_invariants(primitive), global_invariants(transformed), rtol=2e-6, atol=2e-6
    )


def test_global_features_ignore_vacuum_thickness_for_same_cartesian_slab() -> None:
    primitive = _slab()
    thicker = _thicker_vacuum_same_cartesian_slab(primitive)
    np.testing.assert_allclose(
        global_invariants(primitive), global_invariants(thicker), rtol=2e-6, atol=2e-6
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


def _assert_same_full_prediction(first_structure: Structure, second_structure: Structure) -> None:
    model = _full_model()
    with torch.no_grad():
        first = model(_batch(first_structure))
        second = model(_batch(second_structure))
    assert torch.allclose(first["class_logits"], second["class_logits"], atol=8e-4, rtol=8e-4)
    assert torch.allclose(
        first["regression_mean"], second["regression_mean"], atol=8e-4, rtol=8e-4
    )


def test_full_model_is_invariant_to_site_reordering_with_v2_graph() -> None:
    structure = _slab()
    reordered = Structure.from_sites(
        [structure[index] for index in [3, 0, 4, 2, 1]], to_unit_cell=True
    )
    _assert_same_full_prediction(structure, reordered)


def test_full_model_is_consistent_under_exact_inplane_supercell_replication() -> None:
    primitive = _slab()
    supercell = primitive.copy()
    supercell.make_supercell([2, 2, 1])
    _assert_same_full_prediction(primitive, supercell)


def test_full_model_ignores_equivalent_inplane_basis_choice() -> None:
    primitive = _slab()
    _assert_same_full_prediction(primitive, _equivalent_inplane_basis(primitive))


def test_full_model_ignores_extra_vacuum_for_same_cartesian_slab() -> None:
    primitive = _slab()
    _assert_same_full_prediction(primitive, _thicker_vacuum_same_cartesian_slab(primitive))


def test_graph_semantic_schema_is_v2_3() -> None:
    assert CACHE_SCHEMA == "nfe-mxene-cache-2.3"
    assert GLOBAL_FEATURE_SCHEMA == "intrinsic-slab-v3"
    assert NEIGHBOR_POLICY == "radius-shell-complete-v2"
    assert STRUCTURE_MANIFEST_SCHEMA == "source-bytes-v1"
    assert TARGET_SCHEMA == "regression-target-specs-v1"
    assert DATA_IMPLEMENTATION_SCHEMA == "data-code-dependencies-v1"
    assert len(target_schema_sha256()) == 64
    assert len(data_implementation_sha256()) == 64


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


def test_screening_metrics_are_invariant_to_tied_score_row_order() -> None:
    labels = np.asarray([2, 0, 1, 0, 2, 1, 0, 1, 2, 0, 1, 0], dtype=int)
    logits = np.zeros((len(labels), 3), dtype=float)
    first = classification_metrics(logits, labels)
    permutation = np.asarray([7, 2, 11, 0, 8, 1, 10, 5, 4, 9, 3, 6])
    second = classification_metrics(logits[permutation], labels[permutation])
    for suffix in ("1pct", "5pct", "10pct"):
        assert first[f"high_precision_at_{suffix}"] == pytest.approx(
            second[f"high_precision_at_{suffix}"]
        )
        assert first[f"high_recall_at_{suffix}"] == pytest.approx(
            second[f"high_recall_at_{suffix}"]
        )
        assert first[f"high_enrichment_at_{suffix}"] == pytest.approx(1.0)


def test_undefined_class_metrics_are_nan_and_excluded_from_macro() -> None:
    logits = np.asarray([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    labels = np.asarray([0, 1])
    metrics = classification_metrics(logits, labels)
    assert np.isnan(metrics["high_precision"])
    assert np.isnan(metrics["high_recall"])
    assert np.isnan(metrics["high_f1"])
    assert np.isnan(metrics["high_roc_auc"])
    assert np.isnan(metrics["high_average_precision"])
    assert np.isnan(metrics["high_precision_at_5pct"])
    assert np.isnan(metrics["high_recall_at_5pct"])
    assert np.isnan(metrics["high_enrichment_at_5pct"])
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_metrics_fail_fast_on_nonfinite_predictions() -> None:
    logits = np.asarray([[2.0, 0.0, 0.0], [0.0, np.nan, 2.0]])
    with pytest.raises(ValueError, match="non-finite"):
        classification_metrics(logits, np.asarray([0, 2]))
    prediction = np.asarray([[0.2], [np.inf]])
    truth = np.asarray([[0.2], [0.7]])
    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics(
            prediction, truth, np.ones_like(prediction, dtype=bool), ["score"]
        )


def _formal_records() -> tuple[list[dict], dict[str, list[int]]]:
    records: list[dict] = []
    splits = {"train": [], "validation": [], "test": []}
    for split_index, split in enumerate(("train", "validation", "test")):
        for label in range(3):
            index = len(records)
            records.append(
                {
                    "id": f"{split}_{label}",
                    "label": label,
                    "targets": torch.tensor([0.2 + 0.2 * label]),
                    "target_mask": torch.tensor([True]),
                }
            )
            splits[split].append(index)
    return records, splits


def test_formal_primary_target_coverage_requires_all_classes_and_scores() -> None:
    records, splits = _formal_records()
    summary = assert_formal_primary_target_coverage(records, splits)
    assert summary["test"]["class_support"] == (1, 1, 1)

    bad_records = [dict(record) for record in records]
    bad_records[splits["validation"][0]]["target_mask"] = torch.tensor([False])
    with pytest.raises(RuntimeError, match="finite NFE_Pseudo_Score"):
        assert_formal_primary_target_coverage(bad_records, splits)


def _config() -> dict:
    return {
        "seed": 2027,
        "data": {"radius": 6.0, "max_neighbors": 36, "max_cache_skip_fraction": 0.01},
        "model": {"hidden_dim": 192, "num_layers": 6},
        "training": {
            "pretrain_epochs": 35,
            "epochs": 220,
            "batch_size_per_gpu": 96,
            "learning_rate": 3e-4,
            "checkpoint_dir": "ignored_for_hash",
        },
        "loss": {
            "score_weight": 1.5,
            "auxiliary_weight": 0.45,
            "masked_atom_weight": 0.35,
            "denoise_weight": 0.65,
        },
        "inference": {"mc_samples": 30},
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


def test_v2_split_contract_rejects_unknown_split_blank_group_and_duplicate_id() -> None:
    valid = [
        {"id": "a", "split": "train", "split_group": "g1"},
        {"id": "b", "split": "validation", "split_group": "g2"},
        {"id": "c", "split": "test", "split_group": "g3"},
    ]
    splits = split_indices(valid)
    assert splits == {"train": [0], "validation": [1], "test": [2]}
    assert_disjoint_split_groups(valid, splits)
    bad_split = [dict(valid[0]), dict(valid[1]), dict(valid[2])]
    bad_split[2]["split"] = "mystery"
    with pytest.raises(ValueError):
        split_indices(bad_split)
    blank_group = [dict(valid[0]), dict(valid[1]), dict(valid[2])]
    blank_group[1]["split_group"] = ""
    with pytest.raises(RuntimeError):
        split_indices(blank_group)
    duplicate = [dict(valid[0]), dict(valid[1]), dict(valid[2])]
    duplicate[2]["id"] = "a"
    with pytest.raises(RuntimeError):
        split_indices(duplicate)


def test_training_protocol_hash_ignores_seed_and_checkpoint_path_only() -> None:
    first = _config()
    second = _config()
    second["seed"] = 2031
    second["training"] = dict(second["training"])
    second["training"]["checkpoint_dir"] = "/different/machine/path"
    assert training_protocol_sha256(first) == training_protocol_sha256(second)
    assert experiment_protocol_sha256(first) != experiment_protocol_sha256(second)


def test_training_protocol_hash_changes_for_real_hyperparameter_change() -> None:
    first = _config()
    second = _config()
    second["training"] = dict(second["training"])
    second["training"]["learning_rate"] = 1e-4
    assert training_protocol_sha256(first) != training_protocol_sha256(second)
    assert experiment_protocol_sha256(first) != experiment_protocol_sha256(second)
