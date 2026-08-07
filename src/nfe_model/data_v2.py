from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from tqdm import tqdm

from . import data as legacy
from .utils import atomic_torch_save

# Re-export target/dataset APIs so formal v2 code can be a drop-in data provider.
TargetSpec = legacy.TargetSpec
REGRESSION_TARGETS = legacy.REGRESSION_TARGETS
LABEL_TO_INDEX = legacy.LABEL_TO_INDEX
INDEX_TO_LABEL = legacy.INDEX_TO_LABEL
ELEMENT_FEATURE_DIM = legacy.ELEMENT_FEATURE_DIM
NFEDataset = legacy.NFEDataset
collate_graphs = legacy.collate_graphs
move_batch = legacy.move_batch
split_indices = legacy.split_indices
assert_disjoint_split_groups = legacy.assert_disjoint_split_groups
robust_normalizers = legacy.robust_normalizers
class_weights = legacy.class_weights
inverse_target = legacy.inverse_target
torch_load_compat = legacy.torch_load_compat
element_features = legacy.element_features
row_targets = legacy.row_targets
table_sha256 = legacy.table_sha256
finite_float = legacy.finite_float

GLOBAL_FEATURE_DIM = 11
CACHE_SCHEMA = "nfe-mxene-cache-2.0"
GLOBAL_FEATURE_SCHEMA = "intensive-slab-v2"
NEIGHBOR_POLICY = "radius-shell-complete-v2"


def _unwrap_slab_fractional_z(structure: Structure) -> tuple[np.ndarray, float, float]:
    z = np.sort(np.mod(np.asarray(structure.frac_coords)[:, 2], 1.0))
    if len(z) <= 1:
        return np.zeros(len(z), dtype=np.float64), 0.0, 1.0
    gaps = np.diff(np.r_[z, z[0] + 1.0])
    gap_index = int(np.argmax(gaps))
    vacuum_fraction = float(gaps[gap_index])
    start = float(z[(gap_index + 1) % len(z)])
    unwrapped = np.mod(z - start, 1.0)
    unwrapped.sort()
    return unwrapped, max(0.0, 1.0 - vacuum_fraction), vacuum_fraction


def slab_fractions(structure: Structure) -> tuple[float, float]:
    _, slab, vacuum = _unwrap_slab_fractional_z(structure)
    return slab, vacuum


def global_invariants(structure: Structure) -> np.ndarray:
    """Eleven intensive slab descriptors invariant to exact in-plane replication."""
    lattice = structure.lattice
    n_atoms = max(len(structure), 1)
    area = float(np.linalg.norm(np.cross(lattice.matrix[0], lattice.matrix[1])))
    cell_height = float(lattice.volume / max(area, 1e-12))
    unwrapped, slab_fraction, vacuum_fraction = _unwrap_slab_fractional_z(structure)
    z_cart = unwrapped * cell_height
    slab_thickness = float(np.ptp(z_cart)) if len(z_cart) > 1 else 0.0
    z_mean = float(np.mean(z_cart)) if len(z_cart) else 0.0
    z_std = float(np.std(z_cart)) if len(z_cart) else 0.0
    z_mad = float(np.mean(np.abs(z_cart - z_mean))) if len(z_cart) else 0.0
    return np.asarray(
        [
            math.log(max(cell_height, 1e-8)),
            math.cos(math.radians(lattice.alpha)),
            math.cos(math.radians(lattice.beta)),
            math.cos(math.radians(lattice.gamma)),
            math.log(max(lattice.volume / n_atoms, 1e-8)),
            math.log(max(area / n_atoms, 1e-8)),
            slab_fraction,
            vacuum_fraction,
            math.log1p(max(slab_thickness, 0.0)),
            math.log1p(max(z_std, 0.0)),
            math.log1p(max(z_mad, 0.0)),
        ],
        dtype=np.float32,
    )


def _shell_complete_local_indices(local: np.ndarray, distances: np.ndarray, max_neighbors: int) -> np.ndarray:
    if local.size == 0:
        return local
    quantized = np.rint(distances[local].astype(np.float64) * 1_000_000.0).astype(np.int64)
    if max_neighbors > 0 and local.size > max_neighbors:
        kth = int(np.partition(quantized, max_neighbors - 1)[max_neighbors - 1])
        mask = quantized <= kth
        local, quantized = local[mask], quantized[mask]
    return local[np.argsort(quantized, kind="mergesort")]


def build_periodic_graph(structure: Structure, radius: float, max_neighbors: int, identifier: str = "") -> dict[str, Any]:
    try:
        center, neighbor, images, distances = structure.get_neighbor_list(r=radius)
    except (TypeError, ValueError):
        center, neighbor, images, distances = legacy.numpy_neighbor_list(structure, radius)
    center = np.asarray(center, dtype=np.int64)
    neighbor = np.asarray(neighbor, dtype=np.int64)
    images = np.asarray(images, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)
    valid = distances > 1e-7
    center, neighbor, images, distances = center[valid], neighbor[valid], images[valid], distances[valid]
    keep: list[int] = []
    for atom in range(len(structure)):
        local = _shell_complete_local_indices(np.where(center == atom)[0], distances, int(max_neighbors))
        keep.extend(int(x) for x in local)
    if not keep:
        raise ValueError(f"no periodic neighbors found for {identifier or 'structure'}")
    keep_array = np.asarray(keep, dtype=np.int64)
    edge_index = np.stack([neighbor[keep_array], center[keep_array]], axis=0)
    atomic_numbers = [int(site.specie.Z) for site in structure]
    return {
        "id": identifier,
        "z": torch.tensor(atomic_numbers, dtype=torch.long),
        "atom_features": torch.tensor([element_features(z) for z in atomic_numbers], dtype=torch.float32),
        "frac_pos": torch.tensor(np.mod(structure.frac_coords, 1.0), dtype=torch.float32),
        "lattice": torch.tensor(structure.lattice.matrix, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_shift": torch.tensor(images[keep_array], dtype=torch.float32),
        "global_features": torch.tensor(global_invariants(structure)),
        "elements": sorted(set(atomic_numbers)),
    }


def build_cache(
    table_path: str | Path,
    root: str | Path,
    cache_path: str | Path,
    *,
    radius: float,
    max_neighbors: int,
) -> dict[str, Any]:
    table_path, root, cache_path = Path(table_path).resolve(), Path(root).resolve(), Path(cache_path).resolve()
    frame = pd.read_csv(table_path)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="building v2 graph cache", unit="structure"):
        identifier = str(row.get("Structure_Name", ""))
        recorded = Path(str(row.get("File_Path", "")))
        candidates = [recorded] if recorded.is_absolute() else [root / recorded]
        candidates.extend([root / "data" / recorded.name, table_path.parent / "data" / recorded.name])
        file_path = next((p for p in candidates if p.is_file()), candidates[0])
        try:
            structure = Structure.from_file(file_path)
            graph = build_periodic_graph(structure, radius, max_neighbors, identifier)
            targets, target_mask, label = row_targets(row)
            graph.update(
                {
                    "file_path": str(file_path),
                    "split": str(row.get("Suggested_Split", "train")).lower(),
                    "split_group": str(row.get("Split_Group", "")),
                    "targets": targets,
                    "target_mask": target_mask,
                    "label": label,
                    "sample_weight": float(np.clip(finite_float(row.get("Data_Quality_Score"), 1.0), 0.25, 1.0)),
                }
            )
            records.append(graph)
        except Exception as exc:
            skipped.append({"id": identifier, "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "schema": CACHE_SCHEMA,
        "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
        "neighbor_policy": NEIGHBOR_POLICY,
        "table_path": str(table_path),
        "table_sha256": table_sha256(table_path),
        "radius": radius,
        "max_neighbors": max_neighbors,
        "target_specs": [spec.__dict__ for spec in REGRESSION_TARGETS],
        "records": records,
        "skipped": skipped,
    }
    atomic_torch_save(payload, cache_path)
    return payload


def load_or_build_cache(
    table_path: str | Path,
    root: str | Path,
    cache_path: str | Path,
    *,
    radius: float,
    max_neighbors: int,
    rebuild: bool = False,
) -> dict[str, Any]:
    table_path, cache_path = Path(table_path).resolve(), Path(cache_path).resolve()
    if cache_path.is_file() and not rebuild:
        cache = torch_load_compat(cache_path)
        compatible = (
            cache.get("schema") == CACHE_SCHEMA
            and cache.get("global_feature_schema") == GLOBAL_FEATURE_SCHEMA
            and cache.get("neighbor_policy") == NEIGHBOR_POLICY
            and cache.get("table_sha256") == table_sha256(table_path)
            and float(cache.get("radius", -1)) == float(radius)
            and int(cache.get("max_neighbors", -1)) == int(max_neighbors)
        )
        if compatible:
            return cache
    return build_cache(table_path, root, cache_path, radius=radius, max_neighbors=max_neighbors)
