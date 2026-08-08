from __future__ import annotations

from . import pair_symmetric_graph_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def install_pair_symmetric_graph_contract() -> None:
    """Install the pair/shell contract in both wrapper and preserved v2 core.

    ``data_v2`` is now a thin compatibility wrapper around ``data_v2_core``.
    The original installer correctly patches the public wrapper, but functions
    such as ``build_cache`` resolve their globals in the preserved core module.
    Mirror the same cache/neighbor/implementation/graph identities there so a
    formal cache cannot silently fall back to v2.3 semantics.
    """

    _core.install_pair_symmetric_graph_contract()

    from . import data_v2

    implementation = getattr(data_v2, "_core", None)
    if implementation is not None:
        implementation.CACHE_SCHEMA = _core.PAIR_CACHE_SCHEMA
        implementation.NEIGHBOR_POLICY = _core.PAIR_NEIGHBOR_POLICY
        implementation.DATA_IMPLEMENTATION_SCHEMA = _core.PAIR_DATA_IMPLEMENTATION_SCHEMA
        implementation.data_implementation_sha256 = _core.pair_data_implementation_sha256
        implementation.build_periodic_graph = _core.pair_symmetric_periodic_graph

    # Keep this wrapper's exported constants/functions synchronized too.
    data_v2.CACHE_SCHEMA = _core.PAIR_CACHE_SCHEMA
    data_v2.NEIGHBOR_POLICY = _core.PAIR_NEIGHBOR_POLICY
    data_v2.DATA_IMPLEMENTATION_SCHEMA = _core.PAIR_DATA_IMPLEMENTATION_SCHEMA
    data_v2.data_implementation_sha256 = _core.pair_data_implementation_sha256
    data_v2.build_periodic_graph = _core.pair_symmetric_periodic_graph


# Any caller holding the preserved module object should get the corrected
# installer as well.
_core.install_pair_symmetric_graph_contract = install_pair_symmetric_graph_contract


def __getattr__(name: str):
    return getattr(_core, name)
