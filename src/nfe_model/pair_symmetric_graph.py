from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import pair_symmetric_graph_core as _core


_ORIGINAL_INSTALL_PAIR_CONTRACT = _core.install_pair_symmetric_graph_contract

for _name in dir(_core):
    if not _name.startswith("__"):
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
        implementation.build_periodic_graph = _core.pair_symmetric_periodic_graph

    data_v2.CACHE_SCHEMA = _core.PAIR_CACHE_SCHEMA
    data_v2.NEIGHBOR_POLICY = _core.PAIR_NEIGHBOR_POLICY
    data_v2.DATA_IMPLEMENTATION_SCHEMA = _core.PAIR_DATA_IMPLEMENTATION_SCHEMA
    data_v2.data_implementation_sha256 = pair_data_implementation_sha256
    data_v2.build_periodic_graph = _core.pair_symmetric_periodic_graph

    # Keep core references consistent for downstream modules that captured
    # functions directly from the preserved implementation.
    _core.pair_data_implementation_payload = pair_data_implementation_payload
    _core.pair_data_implementation_sha256 = pair_data_implementation_sha256


_core.install_pair_symmetric_graph_contract = install_pair_symmetric_graph_contract


def __getattr__(name: str):
    return getattr(_core, name)
