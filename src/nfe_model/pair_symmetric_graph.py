from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import data_v2 as _data_v2
from . import pair_symmetric_graph_core as _core


_ORIGINAL_INSTALL_PAIR_CONTRACT = _core.install_pair_symmetric_graph_contract
# Capture the preserved v2.3 builder itself, not the public data_v2 wrapper.
# The public wrapper is intentionally mutable under contract installation and
# therefore cannot serve as the recursion-safe base implementation.
_BASE_BUILD_PERIODIC_GRAPH = _data_v2._core.build_periodic_graph
_ORIGINAL_BUILD_PERIODIC_GRAPH = _BASE_BUILD_PERIODIC_GRAPH

for _name in dir(_core):
    if not _name.startswith("__") and _name not in {
        "pair_symmetric_periodic_graph",
        "install_pair_symmetric_graph_contract",
        "pair_data_implementation_payload",
        "pair_data_implementation_sha256",
        "_ORIGINAL_BUILD_PERIODIC_GRAPH",
    }:
        globals()[_name] = getattr(_core, _name)


def pair_data_implementation_payload() -> dict[str, Any]:
    """Fingerprint both the pair-graph core and its wrapper installation semantics."""

    root = Path(__file__).resolve().parent
    return {
        "schema": _core.PAIR_DATA_IMPLEMENTATION_SCHEMA,
        "contract": _core.PAIR_CONTRACT_NAME,
        "base_implementation": _core._BASE_DATA_IMPLEMENTATION_PAYLOAD,
        "pair_symmetric_graph_core_sha256": _core._file_sha256(
            root / "pair_symmetric_graph_core.py"
        ),
        "pair_symmetric_graph_wrapper_sha256": _core._file_sha256(
            root / "pair_symmetric_graph.py"
        ),
    }


def pair_data_implementation_sha256() -> str:
    encoded = json.dumps(
        pair_data_implementation_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pair_symmetric_periodic_graph(
    structure,
    radius: float,
    max_neighbors: int,
    identifier: str = "",
):
    """Run fixed-point pair closure on a permanently captured base-v2 graph.

    Tests may monkeypatch this module's ``_ORIGINAL_BUILD_PERIODIC_GRAPH`` to
    inject a candidate graph. Production keeps that variable bound to the
    preserved ``data_v2_core.build_periodic_graph``. The core implementation is
    synchronized only for the duration of this call, preventing the installed
    public graph wrapper from recursively becoming its own candidate builder.
    """

    previous = _core._ORIGINAL_BUILD_PERIODIC_GRAPH
    _core._ORIGINAL_BUILD_PERIODIC_GRAPH = _ORIGINAL_BUILD_PERIODIC_GRAPH
    try:
        return _core.pair_symmetric_periodic_graph(
            structure,
            radius,
            max_neighbors,
            identifier,
        )
    finally:
        _core._ORIGINAL_BUILD_PERIODIC_GRAPH = previous


def install_pair_symmetric_graph_contract() -> None:
    """Install the pair/shell contract in wrapper, preserved v2 core and identity layer."""

    _ORIGINAL_INSTALL_PAIR_CONTRACT()

    from . import data_contract, data_v2

    data_contract.DATA_IMPLEMENTATION_SCHEMA = _core.PAIR_DATA_IMPLEMENTATION_SCHEMA
    data_contract.data_implementation_payload = pair_data_implementation_payload
    data_contract.data_implementation_sha256 = pair_data_implementation_sha256

    implementation = getattr(data_v2, "_core", None)
    if implementation is not None:
        implementation.CACHE_SCHEMA = _core.PAIR_CACHE_SCHEMA
        implementation.NEIGHBOR_POLICY = _core.PAIR_NEIGHBOR_POLICY
        implementation.DATA_IMPLEMENTATION_SCHEMA = _core.PAIR_DATA_IMPLEMENTATION_SCHEMA
        implementation.data_implementation_sha256 = pair_data_implementation_sha256
        implementation.build_periodic_graph = pair_symmetric_periodic_graph

    data_v2.CACHE_SCHEMA = _core.PAIR_CACHE_SCHEMA
    data_v2.NEIGHBOR_POLICY = _core.PAIR_NEIGHBOR_POLICY
    data_v2.DATA_IMPLEMENTATION_SCHEMA = _core.PAIR_DATA_IMPLEMENTATION_SCHEMA
    data_v2.data_implementation_sha256 = pair_data_implementation_sha256
    data_v2.build_periodic_graph = pair_symmetric_periodic_graph

    # Keep core references consistent for downstream modules that captured
    # identity helpers directly from the preserved implementation. The core
    # graph function itself remains unchanged; its base builder is injected by
    # the wrapper above on each call.
    _core.pair_data_implementation_payload = pair_data_implementation_payload
    _core.pair_data_implementation_sha256 = pair_data_implementation_sha256


_core.install_pair_symmetric_graph_contract = install_pair_symmetric_graph_contract


def __getattr__(name: str):
    return getattr(_core, name)
