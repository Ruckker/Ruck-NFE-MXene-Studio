from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import data_contract, data_v2


PAIR_CACHE_SCHEMA = "nfe-mxene-cache-2.4"
PAIR_NEIGHBOR_POLICY = "radius-shell-complete-pair-symmetric-v3"
PAIR_CONTRACT_NAME = "formal-pair-symmetric-v1"
PAIR_DATA_IMPLEMENTATION_SCHEMA = "data-source-code-pair-v1"
_ORIGINAL_BUILD_PERIODIC_GRAPH = data_v2.build_periodic_graph
_ORIGINAL_DATA_IMPLEMENTATION_PAYLOAD = data_contract.data_implementation_payload
_INSTALLED = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_data_implementation_payload() -> dict[str, Any]:
    """Fingerprint base tensor code plus the v2.4 pair-symmetry wrapper itself."""
    source = Path(__file__).resolve()
    return {
        "schema": PAIR_DATA_IMPLEMENTATION_SCHEMA,
        "contract": PAIR_CONTRACT_NAME,
        "base_implementation": _ORIGINAL_DATA_IMPLEMENTATION_PAYLOAD(),
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


def pair_symmetric_periodic_graph(
    structure,
    radius: float,
    max_neighbors: int,
    identifier: str = "",
) -> dict[str, Any]:
    """Build the shell-complete graph and close every retained edge under reversal.

    The original v2.3 soft cap is applied first. If a directed periodic edge
    `(source -> destination, shift)` survives that physically motivated local
    shell selection, this wrapper guarantees the reverse
    `(destination -> source, -shift)` is also present. The closure may increase
    a center's degree above the nominal cap; the cap is intentionally a soft
    memory budget, while pair symmetry is a graph-semantic invariant.
    """

    graph = _ORIGINAL_BUILD_PERIODIC_GRAPH(
        structure, radius, max_neighbors, identifier
    )
    source, destination = graph["edge_index"]
    shift = graph["edge_shift"]
    if source.ndim != 1 or destination.ndim != 1 or source.shape != destination.shape:
        raise RuntimeError("periodic graph edge_index has an invalid shape")
    if shift.shape != (source.numel(), 3):
        raise RuntimeError("periodic graph edge_shift shape disagrees with edge count")

    entries: dict[tuple[int, int, int, int, int], tuple[int, int, tuple[int, int, int]]] = {}
    for index in range(source.numel()):
        s = int(source[index])
        d = int(destination[index])
        sx, sy, sz = _integer_shift(shift[index])
        key = (s, d, sx, sy, sz)
        entries[key] = (s, d, (sx, sy, sz))
        reverse = (d, s, -sx, -sy, -sz)
        entries.setdefault(reverse, (d, s, (-sx, -sy, -sz)))

    frac = graph["frac_pos"].detach().cpu().double()
    lattice = graph["lattice"].detach().cpu().double()
    sortable = []
    for key, (s, d, image) in entries.items():
        image_tensor = torch.tensor(image, dtype=torch.float64)
        delta = frac[s] + image_tensor - frac[d]
        cart = torch.einsum("i,ij->j", delta, lattice)
        distance = float(torch.linalg.vector_norm(cart))
        if not math.isfinite(distance) or distance <= 1e-7:
            raise RuntimeError(
                f"pair-symmetric graph produced an invalid edge distance for {identifier or 'structure'}"
            )
        if distance > float(radius) + 2e-5:
            raise RuntimeError(
                f"reverse periodic edge exceeds graph radius for {identifier or 'structure'}: "
                f"distance={distance:.8f} radius={float(radius):.8f}"
            )
        quantized = int(round(distance * 1_000_000.0))
        sortable.append((d, quantized, s, image[0], image[1], image[2], key))
    sortable.sort()

    ordered = [entries[item[-1]] for item in sortable]
    graph["edge_index"] = torch.tensor(
        [[item[0] for item in ordered], [item[1] for item in ordered]],
        dtype=torch.long,
    )
    graph["edge_shift"] = torch.tensor(
        [item[2] for item in ordered], dtype=torch.float32
    )
    graph["neighbor_policy"] = PAIR_NEIGHBOR_POLICY
    return graph


def install_pair_symmetric_graph_contract() -> None:
    """Patch data/cache identity before importing formal trainers/runners.

    Canonical v2.4 entrypoints call this before importing modules that capture
    cache/neighbor/data-implementation constants. The separate cache schema/path
    prevents any v2.3 cache from being silently reused or overwritten, while the
    pair wrapper source hash makes production inference sensitive to implementation
    drift even for structures not present in the training cache.
    """

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
