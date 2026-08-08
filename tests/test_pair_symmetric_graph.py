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
        # Deliberately only 1 -> 0. Candidate closure must add 0 -> 1.
        "edge_index": torch.tensor([[1], [0]], dtype=torch.long),
        "edge_shift": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "global_features": torch.zeros(11),
        "elements": [6, 8],
    }


def _fixed_point_fake_graph() -> dict:
    # Four collinear atoms at x={0,1,2,1.5} A in a 10 A box. With a nominal
    # one-neighbor cap, center 0 selects only atom 1 at 1 A. Reversal forces
    # 0 -> 1 into center 1, whose own nearest shell was atom 3 at 0.5 A.
    # Correct fixed-point shell closure must then also include atom 2 -> 1,
    # because atoms 0 and 2 are a degenerate 1 A shell around center 1.
    frac_x = [0.0, 0.1, 0.2, 0.15]
    source: list[int] = []
    destination: list[int] = []
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            distance = abs(frac_x[i] - frac_x[j]) * 10.0
            if distance <= 2.1 + 1e-8:
                source.append(i)
                destination.append(j)
    return {
        "id": "fixed-point",
        "z": torch.tensor([6, 6, 6, 6], dtype=torch.long),
        "atom_features": torch.zeros((4, 14), dtype=torch.float32),
        "frac_pos": torch.tensor([[x, 0.0, 0.5] for x in frac_x], dtype=torch.float32),
        "lattice": torch.eye(3, dtype=torch.float32) * 10.0,
        "edge_index": torch.tensor([source, destination], dtype=torch.long),
        "edge_shift": torch.zeros((len(source), 3), dtype=torch.float32),
        "global_features": torch.zeros(11),
        "elements": [6],
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


def test_pair_closure_recompletes_degenerate_shells_to_fixed_point(monkeypatch) -> None:
    fake = _fixed_point_fake_graph()
    monkeypatch.setattr(contract, "_ORIGINAL_BUILD_PERIODIC_GRAPH", lambda *args, **kwargs: fake)
    graph = contract.pair_symmetric_periodic_graph(None, radius=2.1, max_neighbors=1)
    keys = _keys(graph)

    # Reverse of 1 -> 0 forces 0 -> 1. Once center 1 reaches the 1 A shell,
    # shell completeness requires the degenerate 2 -> 1 edge as well.
    assert (1, 0, 0, 0, 0) in keys
    assert (0, 1, 0, 0, 0) in keys
    assert (2, 1, 0, 0, 0) in keys
    # And its reverse must also be present at the converged fixed point.
    assert (1, 2, 0, 0, 0) in keys

    for source, destination, sx, sy, sz in keys:
        assert (destination, source, -sx, -sy, -sz) in keys


def test_pair_symmetric_policy_uses_new_cache_identity() -> None:
    assert contract.PAIR_CACHE_SCHEMA == "nfe-mxene-cache-2.4.1"
    assert (
        contract.PAIR_NEIGHBOR_POLICY
        == "radius-shell-complete-pair-symmetric-fixedpoint-v4"
    )
    assert contract.PAIR_CONTRACT_NAME == "formal-pair-symmetric-v2"
