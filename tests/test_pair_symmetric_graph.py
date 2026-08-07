from __future__ import annotations

import torch

from nfe_model import pair_symmetric_graph as contract


def _fake_graph() -> dict:
    return {
        "id": "fake",
        "z": torch.tensor([6, 8], dtype=torch.long),
        "atom_features": torch.zeros((2, 14), dtype=torch.float32),
        "frac_pos": torch.tensor([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5]], dtype=torch.float32),
        "lattice": torch.eye(3, dtype=torch.float32) * 10.0,
        # Deliberately only 1 -> 0. Pair-symmetric wrapper must add 0 -> 1.
        "edge_index": torch.tensor([[1], [0]], dtype=torch.long),
        "edge_shift": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "global_features": torch.zeros(11),
        "elements": [6, 8],
    }


def _keys(graph: dict) -> set[tuple[int, int, int, int, int]]:
    source, destination = graph["edge_index"]
    shift = graph["edge_shift"]
    return {
        (
            int(source[index]),
            int(destination[index]),
            int(round(float(shift[index, 0]))),
            int(round(float(shift[index, 1]))),
            int(round(float(shift[index, 2]))),
        )
        for index in range(source.numel())
    }


def test_pair_symmetric_wrapper_adds_missing_reverse_edge(monkeypatch) -> None:
    monkeypatch.setattr(contract, "_ORIGINAL_BUILD_PERIODIC_GRAPH", lambda *args, **kwargs: _fake_graph())
    graph = contract.pair_symmetric_periodic_graph(None, radius=3.0, max_neighbors=1)
    assert _keys(graph) == {(1, 0, 0, 0, 0), (0, 1, 0, 0, 0)}
    assert graph["neighbor_policy"] == contract.PAIR_NEIGHBOR_POLICY


def test_pair_symmetric_wrapper_is_idempotent_for_existing_reverse(monkeypatch) -> None:
    fake = _fake_graph()
    fake["edge_index"] = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
    fake["edge_shift"] = torch.zeros((2, 3), dtype=torch.float32)
    monkeypatch.setattr(contract, "_ORIGINAL_BUILD_PERIODIC_GRAPH", lambda *args, **kwargs: fake)
    graph = contract.pair_symmetric_periodic_graph(None, radius=3.0, max_neighbors=1)
    assert graph["edge_index"].shape[1] == 2
    assert _keys(graph) == {(1, 0, 0, 0, 0), (0, 1, 0, 0, 0)}


def test_pair_symmetric_policy_uses_new_cache_identity() -> None:
    assert contract.PAIR_CACHE_SCHEMA == "nfe-mxene-cache-2.4"
    assert contract.PAIR_NEIGHBOR_POLICY == "radius-shell-complete-pair-symmetric-v3"
