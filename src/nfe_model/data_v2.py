from __future__ import annotations

import math

import numpy as np

from . import data_v2_core as _core


# Re-export the preserved implementation, including private audit helpers used
# by the test suite and formal wrappers. Only the slab-global quantile semantic
# is replaced below.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def _replication_invariant_layer_quantiles(
    values: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Quantiles of unique slab-z levels, invariant to exact site replication.

    An in-plane supercell repeats each physical z layer by the same factor.
    ``np.quantile`` over all atoms changes its interpolation ranks when those
    duplicate observations are introduced, even though the slab is physically
    identical. The intrinsic slab descriptor therefore operates on the unique
    normalized z levels (rounded only to suppress numerical duplicate noise).
    """

    values = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(probabilities)
    unique = np.unique(np.round(values, decimals=10))
    if unique.size == 1:
        return np.full_like(probabilities, float(unique[0]))
    return np.quantile(unique, probabilities)


def global_invariants(structure) -> np.ndarray:
    """Eleven intensive slab descriptors invariant to basis, vacuum and replication."""

    lattice = structure.lattice
    n_atoms = max(len(structure), 1)
    area = float(np.linalg.norm(np.cross(lattice.matrix[0], lattice.matrix[1])))
    cell_height = float(lattice.volume / max(area, 1e-12))
    unwrapped, _, _ = _core._unwrap_slab_fractional_z(structure)
    z_cart = unwrapped * cell_height
    if len(z_cart):
        z_cart = z_cart - float(np.min(z_cart))
    slab_thickness = float(np.ptp(z_cart)) if len(z_cart) > 1 else 0.0
    z_mean = float(np.mean(z_cart)) if len(z_cart) else 0.0
    z_std = float(np.std(z_cart)) if len(z_cart) else 0.0
    z_mad = float(np.mean(np.abs(z_cart - z_mean))) if len(z_cart) else 0.0
    if slab_thickness > 1e-12:
        z_normalized = z_cart / slab_thickness
        quantiles = _replication_invariant_layer_quantiles(
            z_normalized,
            np.asarray([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64),
        )
    else:
        quantiles = np.zeros(5, dtype=np.float64)

    atomic_numbers = np.asarray(
        [int(site.specie.Z) for site in structure], dtype=np.float64
    )
    z_atomic_mean = (
        float(np.mean(atomic_numbers)) / 118.0 if len(atomic_numbers) else 0.0
    )
    z_atomic_std = (
        float(np.std(atomic_numbers)) / 118.0 if len(atomic_numbers) else 0.0
    )

    return np.asarray(
        [
            math.log(max(area / n_atoms, 1e-8)),
            math.log1p(max(slab_thickness, 0.0)),
            math.log1p(max(z_std, 0.0)),
            math.log1p(max(z_mad, 0.0)),
            float(quantiles[0]),
            float(quantiles[1]),
            float(quantiles[2]),
            float(quantiles[3]),
            float(quantiles[4]),
            z_atomic_mean,
            z_atomic_std,
        ],
        dtype=np.float32,
    )


# The preserved builder resolves ``global_invariants`` through its defining
# module globals. Patch that one dependency so every cache/build caller—not
# only direct imports from this wrapper—gets the corrected semantics.
_core.global_invariants = global_invariants


def build_periodic_graph(structure, radius: float, max_neighbors: int, identifier: str = ""):
    """Forward to the active preserved graph builder with corrected globals."""

    return _core.build_periodic_graph(structure, radius, max_neighbors, identifier)


def __getattr__(name: str):
    return getattr(_core, name)
