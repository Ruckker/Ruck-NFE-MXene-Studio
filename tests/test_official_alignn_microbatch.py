from __future__ import annotations

import torch

from nfe_model.data_v2 import collate_graphs
from training.baselines.official.run import _model_batches, _split_graph_batch


def _graph(graph_index: int, node_count: int) -> dict[str, object]:
    source = torch.arange(node_count, dtype=torch.long)
    target = torch.roll(source, shifts=-1)
    return {
        "z": torch.arange(1, node_count + 1, dtype=torch.long),
        "atom_features": torch.full((node_count, 3), float(graph_index)),
        "frac_pos": torch.arange(node_count * 3, dtype=torch.float32).reshape(node_count, 3),
        "edge_index": torch.stack((source, target)),
        "edge_shift": torch.zeros((node_count, 3), dtype=torch.float32),
        "lattice": torch.eye(3, dtype=torch.float32) * (graph_index + 1),
        "global_features": torch.tensor([graph_index, graph_index + 0.5]),
        "targets": torch.tensor([float(graph_index)]),
        "target_mask": torch.tensor([True]),
        "label": graph_index % 3,
        "sample_weight": graph_index + 1.0,
        "id": f"graph-{graph_index}",
        "file_path": f"graph-{graph_index}.vasp",
        "elements": ["C"],
    }


def test_split_graph_batch_preserves_graphs_and_reindexes_edges() -> None:
    batch = collate_graphs([_graph(0, 2), _graph(1, 3), _graph(2, 4)])

    chunks = _split_graph_batch(batch, max_graphs=2)

    assert [chunk["lattice"].shape[0] for chunk in chunks] == [2, 1]
    assert [chunk["ids"] for chunk in chunks] == [
        ["graph-0", "graph-1"],
        ["graph-2"],
    ]
    assert torch.equal(torch.cat([chunk["z"] for chunk in chunks]), batch["z"])
    assert torch.equal(
        torch.cat([chunk["sample_weights"] for chunk in chunks]),
        batch["sample_weights"],
    )

    node_offset = 0
    reconstructed_edges = []
    for chunk in chunks:
        node_count = int(chunk["z"].shape[0])
        assert int(chunk["edge_index"].min()) >= 0
        assert int(chunk["edge_index"].max()) < node_count
        assert torch.equal(
            torch.unique(chunk["batch"]),
            torch.arange(chunk["lattice"].shape[0]),
        )
        reconstructed_edges.append(chunk["edge_index"] + node_offset)
        node_offset += node_count
    assert torch.equal(torch.cat(reconstructed_edges, dim=1), batch["edge_index"])


def test_only_alignn_uses_graph_microbatches() -> None:
    batch = collate_graphs([_graph(0, 2), _graph(1, 3), _graph(2, 4)])

    assert _model_batches(batch, "cgcnn_official") == [batch]
    assert len(_model_batches(batch, "alignn_official")) == 1
    assert len(_split_graph_batch(batch, max_graphs=2)) == 2
