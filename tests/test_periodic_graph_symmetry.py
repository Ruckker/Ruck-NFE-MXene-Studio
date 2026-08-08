from __future__ import annotations

import torch
from pymatgen.core import Lattice, Structure

from nfe_model.data_v2 import build_periodic_graph


def _structure() -> Structure:
    return Structure(
        Lattice.hexagonal(3.05, 18.0),
        ["Nb", "C", "Nb", "O", "O"],
        [
            [0.0, 0.0, 0.46],
            [1 / 3, 2 / 3, 0.50],
            [2 / 3, 1 / 3, 0.54],
            [0.0, 0.0, 0.42],
            [0.0, 0.0, 0.58],
        ],
    )


def test_periodic_graph_edges_have_reverse_counterparts_after_shell_soft_cap() -> None:
    graph = build_periodic_graph(_structure(), radius=5.5, max_neighbors=12)
    source, destination = graph["edge_index"]
    shift = graph["edge_shift"]
    edges = {
        (
            int(source[index]),
            int(destination[index]),
            int(round(float(shift[index, 0]))),
            int(round(float(shift[index, 1]))),
            int(round(float(shift[index, 2]))),
        )
        for index in range(source.numel())
    }
    assert edges
    for source_index, destination_index, sx, sy, sz in edges:
        assert (destination_index, source_index, -sx, -sy, -sz) in edges


def test_periodic_edge_geometry_is_reverse_antisymmetric() -> None:
    graph = build_periodic_graph(_structure(), radius=5.5, max_neighbors=12)
    source, destination = graph["edge_index"]
    shift = graph["edge_shift"]
    frac = graph["frac_pos"].double()
    lattice = graph["lattice"].double()
    delta = frac[source] + shift.double() - frac[destination]
    cart = torch.einsum("ei,ij->ej", delta, lattice)
    lookup = {
        (
            int(source[index]),
            int(destination[index]),
            int(round(float(shift[index, 0]))),
            int(round(float(shift[index, 1]))),
            int(round(float(shift[index, 2]))),
        ): cart[index]
        for index in range(source.numel())
    }
    for key, vector in lookup.items():
        s, d, sx, sy, sz = key
        reverse = lookup[(d, s, -sx, -sy, -sz)]
        assert torch.allclose(vector, -reverse, atol=1e-10, rtol=1e-10)
