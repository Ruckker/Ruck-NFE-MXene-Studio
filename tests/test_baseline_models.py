from __future__ import annotations

import torch

from training.baselines.models import build_model


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
    edge_shift = torch.zeros(len(edges), 3)
    return {
        "z": z,
        "atom_features": atom_features,
        "frac_pos": frac_pos,
        "lattice": lattice,
        "batch": batch,
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "global_features": torch.randn(2, 11),
    }


def test_controlled_baselines_have_common_output_shape() -> None:
    batch = _fake_batch()
    for name in ("cgcnn", "schnet", "alignn", "m3gnet"):
        model = build_model(
            name,
            hidden_dim=32,
            num_layers=2,
            cutoff=6.0,
            dropout=0.0,
        )
        output = model(batch)
        assert output["class_logits"].shape == (2, 3)
        assert output["score"].shape == (2,)
        assert output["embedding"].shape == (2, 32)
        assert model.parameter_count() > 0
