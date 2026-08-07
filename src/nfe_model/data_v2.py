from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from tqdm import tqdm

from . import data as legacy
from .utils import atomic_torch_save

TargetSpec = legacy.TargetSpec
REGRESSION_TARGETS = legacy.REGRESSION_TARGETS
LABEL_TO_INDEX = legacy.LABEL_TO_INDEX
INDEX_TO_LABEL = legacy.INDEX_TO_LABEL
ELEMENT_FEATURE_DIM = legacy.ELEMENT_FEATURE_DIM
NFEDataset = legacy.NFEDataset
collate_graphs = legacy.collate_graphs
move_batch = legacy.move_batch
robust_normalizers = legacy.robust_normalizers
class_weights = legacy.class_weights
inverse_target = legacy.inverse_target
torch_load_compat = legacy.torch_load_compat
element_features = legacy.element_features
row_targets = legacy.row_targets
table_sha256 = legacy.table_sha256
finite_float = legacy.finite_float

GLOBAL_FEATURE_DIM = 11
CACHE_SCHEMA = "nfe-mxene-cache-2.2"
GLOBAL_FEATURE_SCHEMA = "intensive-slab-v2"
NEIGHBOR_POLICY = "radius-shell-complete-v2"
STRUCTURE_MANIFEST_SCHEMA = "source-bytes-v1"
TARGET_SCHEMA = "regression-target-specs-v1"


def target_specs_payload() -> list[dict[str, Any]]:
    return [dict(spec.__dict__) for spec in REGRESSION_TARGETS]


def target_schema_sha256() -> str:
    encoded = json.dumps(
        target_specs_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_split(value: Any) -> str:
    split = str(value).strip().lower()
    if split in {"val", "valid"}:
        split = "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unrecognized Suggested_Split value: {value!r}")
    return split


def _validate_table_frame(frame: pd.DataFrame) -> list[str]:
    required = {"Structure_Name", "File_Path", "Suggested_Split", "Split_Group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"formal v2 dataset table is missing columns: {sorted(missing)}")

    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (identifiers == "").any():
        rows = frame.index[identifiers == ""].tolist()[:5]
        raise ValueError(f"formal v2 dataset contains blank Structure_Name values at rows {rows}")
    duplicates = sorted(identifiers[identifiers.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(
            "formal v2 dataset requires unique Structure_Name values; duplicates="
            f"{duplicates[:5]}"
        )

    groups = frame["Split_Group"].fillna("").astype(str).str.strip()
    if (groups == "").any():
        examples = identifiers[groups == ""].tolist()[:5]
        raise ValueError(
            "formal v2 dataset requires non-empty Split_Group for leakage auditing; examples="
            f"{examples}"
        )

    file_paths = frame["File_Path"].fillna("").astype(str).str.strip()
    if (file_paths == "").any():
        examples = identifiers[file_paths == ""].tolist()[:5]
        raise ValueError(f"formal v2 dataset contains blank File_Path values; examples={examples}")

    return [_normalized_split(value) for value in frame["Suggested_Split"].tolist()]


def _validate_record_identities(records: Sequence[Mapping[str, Any]]) -> None:
    identifiers = [str(record.get("id", "")).strip() for record in records]
    if any(not identifier for identifier in identifiers):
        examples = [i for i, identifier in enumerate(identifiers) if not identifier][:5]
        raise RuntimeError(f"formal v2 dataset contains blank Structure_Name values at records {examples}")
    counts: dict[str, int] = {}
    for identifier in identifiers:
        counts[identifier] = counts.get(identifier, 0) + 1
    duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(
            "formal v2 dataset requires unique Structure_Name values; duplicates="
            f"{duplicates[:5]}"
        )
    blank_groups = [
        identifiers[index]
        for index, record in enumerate(records)
        if not str(record.get("split_group", "")).strip()
    ]
    if blank_groups:
        raise RuntimeError(
            "formal v2 dataset requires non-empty Split_Group for leakage auditing; examples="
            f"{blank_groups[:5]}"
        )


def split_indices(records: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    _validate_record_identities(records)
    result = {"train": [], "validation": [], "test": []}
    for index, record in enumerate(records):
        result[_normalized_split(record.get("split", ""))].append(index)
    return result


def assert_disjoint_split_groups(
    records: Sequence[dict[str, Any]], splits: dict[str, Sequence[int]]
) -> None:
    _validate_record_identities(records)
    legacy.assert_disjoint_split_groups(records, splits)
    groups = {
        split: {str(records[index].get("split_group", "")).strip() for index in indices}
        for split, indices in splits.items()
    }
    conflicts: list[str] = []
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                conflicts.append(f"{left}/{right}: {', '.join(sorted(overlap)[:5])}")
    if conflicts:
        raise RuntimeError("Split_Group leakage detected across formal splits: " + "; ".join(conflicts))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_structure_file(recorded: Path, root: Path, table_path: Path) -> Path:
    candidates = [recorded] if recorded.is_absolute() else [root / recorded]
    candidates.extend([root / "data" / recorded.name, table_path.parent / "data" / recorded.name])
    return next((path for path in candidates if path.is_file()), candidates[0])


def structure_manifest_sha256(table_path: str | Path, root: str | Path) -> str:
    table_path = Path(table_path).resolve()
    root = Path(root).resolve()
    frame = pd.read_csv(table_path)
    _validate_table_frame(frame)
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        identifier = str(row["Structure_Name"]).strip()
        recorded = Path(str(row["File_Path"]).strip())
        file_path = _resolve_structure_file(recorded, root, table_path)
        rows.append(
            {
                "id": identifier,
                "sha256": _file_sha256(file_path) if file_path.is_file() else "MISSING",
            }
        )
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


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


def _shell_complete_local_indices(
    local: np.ndarray, distances: np.ndarray, max_neighbors: int
) -> np.ndarray:
    if local.size == 0:
        return local
    quantized = np.rint(distances[local].astype(np.float64) * 1_000_000.0).astype(np.int64)
    if max_neighbors > 0 and local.size > max_neighbors:
        kth = int(np.partition(quantized, max_neighbors - 1)[max_neighbors - 1])
        mask = quantized <= kth
        local, quantized = local[mask], quantized[mask]
    return local[np.argsort(quantized, kind="mergesort")]


def build_periodic_graph(
    structure: Structure, radius: float, max_neighbors: int, identifier: str = ""
) -> dict[str, Any]:
    if float(radius) <= 0 or int(max_neighbors) <= 0:
        raise ValueError("formal v2 graph requires radius > 0 and max_neighbors > 0")
    try:
        center, neighbor, images, distances = structure.get_neighbor_list(r=radius)
    except (TypeError, ValueError):
        center, neighbor, images, distances = legacy.numpy_neighbor_list(structure, radius)
    center = np.asarray(center, dtype=np.int64)
    neighbor = np.asarray(neighbor, dtype=np.int64)
    images = np.asarray(images, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)
    valid = distances > 1e-7
    center, neighbor, images, distances = (
        center[valid], neighbor[valid], images[valid], distances[valid]
    )
    keep: list[int] = []
    for atom in range(len(structure)):
        local = _shell_complete_local_indices(
            np.where(center == atom)[0], distances, int(max_neighbors)
        )
        keep.extend(int(value) for value in local)
    if not keep:
        raise ValueError(f"no periodic neighbors found for {identifier or 'structure'}")
    keep_array = np.asarray(keep, dtype=np.int64)
    edge_index = np.stack([neighbor[keep_array], center[keep_array]], axis=0)
    atomic_numbers = [int(site.specie.Z) for site in structure]
    return {
        "id": identifier,
        "z": torch.tensor(atomic_numbers, dtype=torch.long),
        "atom_features": torch.tensor(
            [element_features(z) for z in atomic_numbers], dtype=torch.float32
        ),
        "frac_pos": torch.tensor(np.mod(structure.frac_coords, 1.0), dtype=torch.float32),
        "lattice": torch.tensor(structure.lattice.matrix, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_shift": torch.tensor(images[keep_array], dtype=torch.float32),
        "global_features": torch.tensor(global_invariants(structure)),
        "elements": sorted(set(atomic_numbers)),
    }


def _validate_cache_target_contract(cache: Mapping[str, Any]) -> bool:
    return (
        cache.get("target_schema") == TARGET_SCHEMA
        and cache.get("target_schema_sha256") == target_schema_sha256()
        and cache.get("target_specs") == target_specs_payload()
    )


def build_cache(
    table_path: str | Path,
    root: str | Path,
    cache_path: str | Path,
    *,
    radius: float,
    max_neighbors: int,
) -> dict[str, Any]:
    if float(radius) <= 0 or int(max_neighbors) <= 0:
        raise ValueError("formal v2 cache requires radius > 0 and max_neighbors > 0")
    table_path = Path(table_path).resolve()
    root = Path(root).resolve()
    cache_path = Path(cache_path).resolve()
    frame = pd.read_csv(table_path)
    normalized_splits = _validate_table_frame(frame)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    for position, (_, row) in enumerate(
        tqdm(frame.iterrows(), total=len(frame), desc="building v2.2 graph cache", unit="structure")
    ):
        identifier = str(row["Structure_Name"]).strip()
        recorded = Path(str(row["File_Path"]).strip())
        file_path = _resolve_structure_file(recorded, root, table_path)
        source_sha256 = _file_sha256(file_path) if file_path.is_file() else "MISSING"
        manifest_rows.append({"id": identifier, "sha256": source_sha256})
        try:
            structure = Structure.from_file(file_path)
            graph = build_periodic_graph(structure, radius, max_neighbors, identifier)
            targets, target_mask, label = row_targets(row)
            graph.update(
                {
                    "file_path": str(file_path),
                    "source_file_sha256": source_sha256,
                    "split": normalized_splits[position],
                    "split_group": str(row["Split_Group"]).strip(),
                    "targets": targets,
                    "target_mask": target_mask,
                    "label": label,
                    "sample_weight": float(
                        np.clip(finite_float(row.get("Data_Quality_Score"), 1.0), 0.25, 1.0)
                    ),
                }
            )
            records.append(graph)
        except Exception as exc:
            skipped.append({"id": identifier, "error": f"{type(exc).__name__}: {exc}"})
    manifest_encoded = json.dumps(
        manifest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = {
        "schema": CACHE_SCHEMA,
        "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
        "neighbor_policy": NEIGHBOR_POLICY,
        "structure_manifest_schema": STRUCTURE_MANIFEST_SCHEMA,
        "structure_manifest_sha256": hashlib.sha256(manifest_encoded).hexdigest(),
        "target_schema": TARGET_SCHEMA,
        "target_schema_sha256": target_schema_sha256(),
        "table_path": str(table_path),
        "table_sha256": table_sha256(table_path),
        "radius": float(radius),
        "max_neighbors": int(max_neighbors),
        "target_specs": target_specs_payload(),
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
    if float(radius) <= 0 or int(max_neighbors) <= 0:
        raise ValueError("formal v2 cache requires radius > 0 and max_neighbors > 0")
    table_path = Path(table_path).resolve()
    root = Path(root).resolve()
    cache_path = Path(cache_path).resolve()
    if cache_path.is_file() and not rebuild:
        cache = torch_load_compat(cache_path)
        current_structure_manifest = structure_manifest_sha256(table_path, root)
        compatible = (
            cache.get("schema") == CACHE_SCHEMA
            and cache.get("global_feature_schema") == GLOBAL_FEATURE_SCHEMA
            and cache.get("neighbor_policy") == NEIGHBOR_POLICY
            and cache.get("structure_manifest_schema") == STRUCTURE_MANIFEST_SCHEMA
            and cache.get("structure_manifest_sha256") == current_structure_manifest
            and _validate_cache_target_contract(cache)
            and cache.get("table_sha256") == table_sha256(table_path)
            and float(cache.get("radius", -1)) == float(radius)
            and int(cache.get("max_neighbors", -1)) == int(max_neighbors)
        )
        if compatible:
            split_indices(cache.get("records", []))
            return cache
    cache = build_cache(table_path, root, cache_path, radius=radius, max_neighbors=max_neighbors)
    split_indices(cache.get("records", []))
    return cache
