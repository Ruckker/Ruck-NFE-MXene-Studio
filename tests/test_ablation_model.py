from __future__ import annotations

import torch

from nfe_model.ablation import AblationPeriodicNFEModel


def _fake_batch() -> dict[str, torch.Tensor]:
    z = torch.tensor([6, 41, 8, 6, 22, 1], dtype=torch.long)
    atom_features = torch.randn(6, 14)
    frac_pos = torch.tensor(
        [
            [0.10, 0.10, 0.50],
            [0.20, 0.10, 0.52],
            [0.15, 0.20, 0.55],
            [0.10, 0.10, 0.50],
            [0.20, 0.10, 0.52],
            [0.15, 0.20, 0.55],
        ],
        dtype=torch.float32,
    )
    lattice = torch.eye(3).repeat(2, 1, 1) * 10.0
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    edges = []
    for base in (0, 3):
        for destination in range(base, base + 3):
            for source in range(base, base + 3):
                if source != destination:
                    edges.append((source, destination))
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    return {
        "z": z,
        "atom_features": atom_features,
        "frac_pos": frac_pos,
        "lattice": lattice,
        "batch": batch,
        "edge_index": edge_index,
        "edge_shift": torch.zeros(len(edges), 3),
        "global_features": torch.randn(2, 11),
    }


def _make_model(**kwargs) -> AblationPeriodicNFEModel:
    return AblationPeriodicNFEModel(
        hidden_dim=32,
        vector_dim=12,
        num_layers=2,
        num_rbf=16,
        cutoff=6.0,
        dropout=0.0,
        max_atomic_number=118,
        element_features=14,
        global_features=11,
        num_regression_targets=10,
        num_classes=3,
        **kwargs,
    )


def test_no_vector_ablation_has_valid_outputs_and_zero_denoise() -> None:
    model = _make_model(use_vector_features=False, use_global_features=True)
    output = model(_fake_batch())
    assert output["class_logits"].shape == (2, 3)
    assert output["regression_mean"].shape == (2, 10)
    assert output["embedding"].shape == (2, 32)
    assert output["denoise_vector"].shape == (6, 3)
    assert torch.count_nonzero(output["denoise_vector"]) == 0
    assert model.config["use_vector_features"] is False


def test_no_global_ablation_ignores_global_feature_values() -> None:
    model = _make_model(use_vector_features=True, use_global_features=False)
    model.eval()
    first = _fake_batch()
    second = dict(first)
    second["global_features"] = first["global_features"] + 1000.0
    with torch.no_grad():
        output_a = model(first)["class_logits"]
        output_b = model(second)["class_logits"]
    assert torch.allclose(output_a, output_b, atol=1e-6, rtol=1e-6)
    assert model.config["use_global_features"] is False
