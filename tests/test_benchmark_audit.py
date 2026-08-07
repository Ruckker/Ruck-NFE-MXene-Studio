from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from nfe_model.data_v2 import build_periodic_graph, collate_graphs
from nfe_model.model import PeriodicNFEModel
from nfe_model.provenance_v2 import assert_matching_provenance, split_manifest_sha256
from nfe_model.train_ablation import _active_target_heteroscedastic_loss
from nfe_model.train_audit_v2 import apply_checkpoint_contract, deduplicate_payload
from training.baselines.matched_painn import MatchedPaiNNBaseline


def _provenance(dataset: str = "dataset-A") -> dict:
    return {
        "dataset_table_sha256": dataset,
        "structure_manifest_schema": "source-bytes-v1",
        "structure_manifest_sha256": "structures-A",
        "split_manifest_sha256": "split-A",
        "cache_schema": "nfe-mxene-cache-2.1",
        "global_feature_schema": "intensive-slab-v2",
        "neighbor_policy": "radius-shell-complete-v2",
        "graph_radius_A": 6.0,
        "max_neighbors": 36,
    }


def test_zero_weight_targets_do_not_dilute_ablation_regression_loss() -> None:
    mean = torch.zeros((2, 3), dtype=torch.float32)
    log_variance = torch.zeros_like(mean)
    target = torch.ones_like(mean)
    mask = torch.ones_like(mean, dtype=torch.bool)
    weights = torch.tensor([1.5, 0.0, 0.0])
    samples = torch.tensor([1.0, 0.5])
    multi = _active_target_heteroscedastic_loss(
        mean, log_variance, target, mask, weights, samples
    )
    score_only = _active_target_heteroscedastic_loss(
        mean[:, :1],
        log_variance[:, :1],
        target[:, :1],
        mask[:, :1],
        weights[:1],
        samples,
    )
    assert torch.allclose(multi, score_only, atol=1e-7, rtol=1e-7)


def test_all_zero_target_weights_return_exact_zero() -> None:
    mean = torch.randn(3, 2)
    value = _active_target_heteroscedastic_loss(
        mean,
        torch.zeros_like(mean),
        torch.randn_like(mean),
        torch.ones_like(mean, dtype=torch.bool),
        torch.zeros(2),
        torch.ones(3),
    )
    assert float(value) == pytest.approx(0.0)


def test_distributed_padding_payload_is_deduplicated_by_record_index() -> None:
    payload = {
        "record_indices": np.asarray([4, 7, 9, 4]),
        "labels": np.asarray([0, 1, 2, 0]),
        "logits": np.asarray(
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]
        ),
        "ids": np.asarray(["a", "b", "c", "a"], dtype=object),
    }
    result = deduplicate_payload(payload)
    assert result["record_indices"].tolist() == [4, 7, 9]
    assert result["ids"].tolist() == ["a", "b", "c"]
    assert result["logits"].shape == (3, 3)


def test_split_manifest_hash_changes_when_assignment_changes_but_not_file_path() -> None:
    records = [
        {"id": "a", "split_group": "g1", "file_path": "/machineA/a.vasp"},
        {"id": "b", "split_group": "g2", "file_path": "/machineA/b.vasp"},
    ]
    first = split_manifest_sha256(records, {"train": [0], "validation": [1], "test": []})
    moved = [dict(record) for record in records]
    moved[0]["file_path"] = "/machineB/a.vasp"
    moved[1]["file_path"] = "/machineB/b.vasp"
    assert first == split_manifest_sha256(
        moved, {"train": [0], "validation": [1], "test": []}
    )
    second = split_manifest_sha256(records, {"train": [1], "validation": [0], "test": []})
    assert first != second


def test_checkpoint_provenance_mismatch_is_rejected() -> None:
    current = _provenance()
    observed = _provenance("dataset-B")
    with pytest.raises(ValueError, match="dataset_table_sha256"):
        assert_matching_provenance(observed, current)
    observed = _provenance()
    observed["structure_manifest_sha256"] = "structures-B"
    with pytest.raises(ValueError, match="structure_manifest_sha256"):
        assert_matching_provenance(observed, current)
    with pytest.raises(ValueError, match="no provenance"):
        assert_matching_provenance(None, current)


def test_full_ablation_checkpoint_has_distinct_format_and_round_trips() -> None:
    model = PeriodicNFEModel(
        hidden_dim=32,
        vector_dim=12,
        num_layers=2,
        num_rbf=16,
        cutoff=5.0,
        dropout=0.0,
        num_regression_targets=10,
    )
    provenance = _provenance()
    payload = apply_checkpoint_contract(
        {
            "format": "nfe-mxene-predictor-1.0",
            "model_state": model.state_dict(),
            "model_config": dict(model.config),
        },
        model=model,
        config={"ablation": {"name": "full"}},
        provenance=provenance,
    )
    assert payload["format"] == "nfe-mxene-predictor-ablation-1.0"
    assert payload["ablation_config"]["name"] == "full"
    assert payload["provenance"] == provenance
    assert "use_vector_features" not in payload["base_model_config"]
    restored = PeriodicNFEModel(**payload["base_model_config"])
    restored.load_state_dict(payload["model_state"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def _structure() -> Structure:
    return Structure(
        Lattice.hexagonal(3.1, 18.0),
        ["Ti", "C", "O", "O"],
        [
            [0.0, 0.0, 0.50],
            [1 / 3, 2 / 3, 0.50],
            [0.0, 0.0, 0.43],
            [0.0, 0.0, 0.57],
        ],
    )


def _model() -> MatchedPaiNNBaseline:
    torch.manual_seed(17)
    return MatchedPaiNNBaseline(
        hidden_dim=48,
        num_layers=2,
        num_rbf=16,
        cutoff=5.0,
        dropout=0.0,
    ).eval()


def _batch(structure: Structure) -> dict[str, torch.Tensor]:
    graph = build_periodic_graph(structure, radius=5.0, max_neighbors=96)
    graph.update(
        {
            "targets": torch.zeros(10),
            "target_mask": torch.ones(10, dtype=torch.bool),
            "label": 1,
        }
    )
    return collate_graphs([graph])


def test_matched_painn_is_atom_permutation_invariant_after_graph_remap() -> None:
    batch = _batch(_structure())
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    remapped = dict(batch)
    for key in ("z", "atom_features", "frac_pos", "batch"):
        remapped[key] = batch[key][permutation]
    remapped["edge_index"] = inverse[batch["edge_index"]]
    model = _model()
    with torch.no_grad():
        original = model(batch)
        permuted = model(remapped)
    assert torch.allclose(
        original["class_logits"], permuted["class_logits"], atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(original["score"], permuted["score"], atol=2e-5, rtol=2e-5)


def test_matched_painn_is_consistent_under_exact_supercell_replication() -> None:
    primitive = _structure()
    supercell = primitive.copy()
    supercell.make_supercell([2, 2, 1])
    model = _model()
    with torch.no_grad():
        primitive_output = model(_batch(primitive))
        supercell_output = model(_batch(supercell))
    assert torch.allclose(
        primitive_output["class_logits"],
        supercell_output["class_logits"],
        atol=3e-4,
        rtol=3e-4,
    )
    assert torch.allclose(
        primitive_output["score"], supercell_output["score"], atol=3e-4, rtol=3e-4
    )
