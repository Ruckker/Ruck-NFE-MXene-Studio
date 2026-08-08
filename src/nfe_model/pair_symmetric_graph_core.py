from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from . import data_contract, data_v2


PAIR_CACHE_SCHEMA = "nfe-mxene-cache-2.4.1"
PAIR_NEIGHBOR_POLICY = "radius-shell-complete-pair-symmetric-fixedpoint-v4"
PAIR_CONTRACT_NAME = "formal-pair-symmetric-v2"
PAIR_DATA_IMPLEMENTATION_SCHEMA = "data-source-code-pair-v2"
_ORIGINAL_BUILD_PERIODIC_GRAPH = data_v2.build_periodic_graph
_BASE_DATA_IMPLEMENTATION_PAYLOAD = data_contract.data_implementation_payload()
_INSTALLED = False
_FULL_GRAPH_NEIGHBOR_CAP = 2_147_483_647


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_data_implementation_payload() -> dict[str, Any]:
    """Fingerprint base tensor code plus the pair/shell fixed-point wrapper itself."""
    source = Path(__file__).resolve()
    return {
        "schema": PAIR_DATA_IMPLEMENTATION_SCHEMA,
        "contract": PAIR_CONTRACT_NAME,
        "base_implementation": _BASE_DATA_IMPLEMENTATION_PAYLOAD,
        "pair_symmetric_graph_sha256": _file_sha256(source),
    }


def pair_data_implementation_sha256() -> str:
    encoded = json.dumps(
        pair_data_implementation_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer_shift(value: torch.Tensor) -> tuple[int, int, int]:
    values = tuple(int(round(float(component))) for component in value.detach().cpu())
    if len(values) != 3:
        raise ValueError("periodic edge shift must have exactly three components")
    return values


def _reverse_key(key: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    source, destination, sx, sy, sz = key
    return destination, source, -sx, -sy, -sz


def _edge_distance(
    key: tuple[int, int, int, int, int],
    frac: torch.Tensor,
    lattice: torch.Tensor,
) -> float:
    source, destination, sx, sy, sz = key
    image = torch.tensor((sx, sy, sz), dtype=torch.float64)
    delta = frac[source] + image - frac[destination]
    cart = torch.einsum("i,ij->j", delta, lattice)
    distance = float(torch.linalg.vector_norm(cart))
    if not math.isfinite(distance) or distance <= 1e-7:
        raise RuntimeError("pair/shell graph contains a non-finite or zero-distance edge")
    return distance


def _full_radius_candidates(
    structure,
    radius: float,
    identifier: str,
) -> tuple[dict[str, Any], dict[tuple[int, int, int, int, int], tuple[float, int]]]:
    """Return every directed radius edge, explicitly closed under reversal.

    The base v2.3 builder is intentionally invoked with an effectively infinite
    neighbor cap. Pair closure must operate on the *full radius candidate set*;
    starting from an already capped graph cannot recover the other members of a
    degenerate shell when a reverse edge later forces that shell to be present.
    """

    graph = _ORIGINAL_BUILD_PERIODIC_GRAPH(
        structure,
        float(radius),
        _FULL_GRAPH_NEIGHBOR_CAP,
        identifier,
    )
    source, destination = graph["edge_index"]
    shifts = graph["edge_shift"]
    if source.ndim != 1 or destination.ndim != 1 or source.shape != destination.shape:
        raise RuntimeError("periodic graph edge_index has an invalid shape")
    if shifts.shape != (source.numel(), 3):
        raise RuntimeError("periodic graph edge_shift shape disagrees with edge count")

    frac = graph["frac_pos"].detach().cpu().double()
    lattice = graph["lattice"].detach().cpu().double()
    candidates: dict[tuple[int, int, int, int, int], tuple[float, int]] = {}
    raw_keys: list[tuple[int, int, int, int, int]] = []
    for index in range(source.numel()):
        sx, sy, sz = _integer_shift(shifts[index])
        key = (int(source[index]), int(destination[index]), sx, sy, sz)
        raw_keys.append(key)

    # Ensure the candidate universe itself is direction-closed. A correctly
    # generated periodic neighbor list already has these counterparts, but
    # synthesizing a missing reverse is safer than making later invariants depend
    # on pymatgen/backend edge enumeration order.
    for key in list(raw_keys):
        raw_keys.append(_reverse_key(key))

    for key in set(raw_keys):
        distance = _edge_distance(key, frac, lattice)
        if distance > float(radius) + 2e-5:
            raise RuntimeError(
                f"full periodic candidate exceeds graph radius for {identifier or 'structure'}: "
                f"distance={distance:.8f} radius={float(radius):.8f}"
            )
        quantized = int(round(distance * 1_000_000.0))
        candidates[key] = (distance, quantized)
    if not candidates:
        raise RuntimeError(f"no radius candidates for {identifier or 'structure'}")
    return graph, candidates


def _initial_shell_complete_selection(
    candidates: dict[tuple[int, int, int, int, int], tuple[float, int]],
    n_atoms: int,
    max_neighbors: int,
) -> set[tuple[int, int, int, int, int]]:
    selected: set[tuple[int, int, int, int, int]] = set()
    by_destination: dict[int, list[tuple[int, tuple[int, int, int, int, int]]]] = {
        atom: [] for atom in range(n_atoms)
    }
    for key, (_, quantized) in candidates.items():
        by_destination[key[1]].append((quantized, key))

    for atom in range(n_atoms):
        local = sorted(by_destination[atom], key=lambda item: (item[0], item[1]))
        if not local:
            raise RuntimeError(f"atom {atom} has no radius neighbors in formal graph")
        if len(local) <= int(max_neighbors):
            cutoff_quantized = local[-1][0]
        else:
            cutoff_quantized = local[int(max_neighbors) - 1][0]
        selected.update(key for quantized, key in local if quantized <= cutoff_quantized)
    return selected


def _fixed_point_pair_shell_closure(
    candidates: dict[tuple[int, int, int, int, int], tuple[float, int]],
    selected: set[tuple[int, int, int, int, int]],
    n_atoms: int,
) -> set[tuple[int, int, int, int, int]]:
    """Simultaneously enforce reverse-edge and contiguous shell completeness.

    Pair closure can force a farther edge into a center whose own kth-shell cap
    originally stopped earlier. Merely adding that single reverse edge would
    create a partial coordination shell. Therefore every iteration does both:

    1. add the reverse of every selected periodic edge;
    2. for each center, include *all* candidate edges no farther than the most
       distant selected shell at that center.

    The set grows monotonically inside a finite full-radius candidate universe,
    so convergence is guaranteed. The realized degree may exceed the nominal
    max_neighbors soft budget; physical representation invariance takes priority.
    """

    by_destination: dict[int, list[tuple[int, tuple[int, int, int, int, int]]]] = {
        atom: [] for atom in range(n_atoms)
    }
    for key, (_, quantized) in candidates.items():
        by_destination[key[1]].append((quantized, key))

    while True:
        before = len(selected)
        for key in tuple(selected):
            reverse = _reverse_key(key)
            if reverse not in candidates:
                raise RuntimeError(f"full periodic candidate set is missing reverse edge {reverse}")
            selected.add(reverse)

        maximum_selected_shell: dict[int, int] = {}
        for key in selected:
            quantized = candidates[key][1]
            destination = key[1]
            maximum_selected_shell[destination] = max(
                maximum_selected_shell.get(destination, -1), quantized
            )
        for destination, cutoff_quantized in maximum_selected_shell.items():
            selected.update(
                key
                for quantized, key in by_destination[destination]
                if quantized <= cutoff_quantized
            )
        if len(selected) == before:
            return selected


def pair_symmetric_periodic_graph(
    structure,
    radius: float,
    max_neighbors: int,
    identifier: str = "",
) -> dict[str, Any]:
    if float(radius) <= 0 or int(max_neighbors) <= 0:
        raise ValueError("formal pair/shell graph requires radius > 0 and max_neighbors > 0")

    graph, candidates = _full_radius_candidates(structure, radius, identifier)
    n_atoms = int(graph["z"].numel())
    selected = _initial_shell_complete_selection(candidates, n_atoms, int(max_neighbors))
    selected = _fixed_point_pair_shell_closure(candidates, selected, n_atoms)

    # Final invariant check: every retained center must contain all candidate
    # shells up to its realized farthest selected shell, and every edge must have
    # its exact periodic reverse.
    max_shell: dict[int, int] = {}
    for key in selected:
        reverse = _reverse_key(key)
        if reverse not in selected:
            raise RuntimeError(f"pair-symmetric fixed point lost reverse edge {reverse}")
        quantized = candidates[key][1]
        max_shell[key[1]] = max(max_shell.get(key[1], -1), quantized)
    for key, (_, quantized) in candidates.items():
        if quantized <= max_shell.get(key[1], -1) and key not in selected:
            raise RuntimeError(
                "pair-symmetric closure produced a partial distance shell; "
                f"missing edge={key} shell={quantized}"
            )

    ordered_keys = sorted(
        selected,
        key=lambda key: (
            key[1],
            candidates[key][1],
            key[0],
            key[2],
            key[3],
            key[4],
        ),
    )
    graph["edge_index"] = torch.tensor(
        [[key[0] for key in ordered_keys], [key[1] for key in ordered_keys]],
        dtype=torch.long,
    )
    graph["edge_shift"] = torch.tensor(
        [[key[2], key[3], key[4]] for key in ordered_keys], dtype=torch.float32
    )
    graph["neighbor_policy"] = PAIR_NEIGHBOR_POLICY
    return graph


def install_pair_symmetric_graph_contract() -> None:
    """Patch data/cache identity before importing formal trainers/runners."""

    global _INSTALLED
    if _INSTALLED:
        return

    data_contract.DATA_IMPLEMENTATION_SCHEMA = PAIR_DATA_IMPLEMENTATION_SCHEMA
    data_contract.data_implementation_payload = pair_data_implementation_payload
    data_contract.data_implementation_sha256 = pair_data_implementation_sha256

    data_v2.CACHE_SCHEMA = PAIR_CACHE_SCHEMA
    data_v2.NEIGHBOR_POLICY = PAIR_NEIGHBOR_POLICY
    data_v2.DATA_IMPLEMENTATION_SCHEMA = PAIR_DATA_IMPLEMENTATION_SCHEMA
    data_v2.data_implementation_sha256 = pair_data_implementation_sha256
    data_v2.build_periodic_graph = pair_symmetric_periodic_graph
    _INSTALLED = True
