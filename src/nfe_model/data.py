# ==============================================================================
# 中文概述：周期晶体图、目标张量、缓存、数据划分与批处理。
# English overview: Periodic crystal graphs, target tensors, cache, splits, and mini-batches.
#
# 中文输入：清洗后的 CSV 行、CIF/POSCAR、图截断半径与模型配置。
# English inputs: Cleaned CSV rows, CIF/POSCAR structures, graph cutoff, and model config.
# 中文输出：PyTorch 图批次、稳健归一化器、类别权重与无泄漏索引。
# English outputs: PyTorch graph batches, robust normalizers, class weights, and leakage-safe indices.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: TargetSpec, transform_target, inverse_target, finite_float, element_features, table_sha256, slab_fractions, global_invariants, numpy_neighbor_list, build_periodic_graph, row_targets, build_cache, torch_load_compat, load_or_build_cache, split_indices, assert_disjoint_split_groups, robust_normalizers, NFEDataset, collate_graphs, move_batch, class_weights
#
# Author: Ruck
# Generated: 2026-07-29 19:18:46 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element, Structure
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import atomic_torch_save


LABEL_TO_INDEX = {"low": 0, "medium": 1, "high": 2}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
ELEMENT_FEATURE_DIM = 14


# 中文：顶层类 `TargetSpec`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `TargetSpec`; review type hints and callers before extending it.
@dataclass(frozen=True)
class TargetSpec:
    name: str
    transform: str = "identity"
    main: bool = False


REGRESSION_TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("NFE_Pseudo_Score", "identity", True),
    TargetSpec("NFE_Energy_Relative_EF_eV", "identity", True),
    TargetSpec("NFE_Atomic_Projection_Total", "identity", True),
    TargetSpec("NFE_Effective_Mass_Geomean_me", "log1p", True),
    TargetSpec("Work_Function_Mean_eV", "identity"),
    TargetSpec("Band_Gap_eV", "log1p"),
    TargetSpec("DOS_at_EF_per_Atom", "log1p"),
    TargetSpec("ELF_Surface_Top_Mean", "identity"),
    TargetSpec("ELF_Surface_Bottom_Mean", "identity"),
    TargetSpec("Charge_Surface_Total_Fraction", "log10eps"),
)


# 中文：顶层接口 `transform_target`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `transform_target`; review type hints and callers before extending it.
def transform_target(value: float, transform: str) -> float:
    if transform == "identity":
        return value
    if transform == "log1p":
        return math.log1p(max(value, 0.0))
    if transform == "log10eps":
        return math.log10(max(value, 1e-12))
    raise ValueError(f"unknown target transform: {transform}")


# 中文：顶层接口 `inverse_target`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `inverse_target`; review type hints and callers before extending it.
def inverse_target(value: np.ndarray | float, transform: str) -> np.ndarray | float:
    if transform == "identity":
        return value
    if transform == "log1p":
        return np.maximum(np.expm1(value), 0.0)
    if transform == "log10eps":
        return np.power(10.0, value)
    raise ValueError(f"unknown target transform: {transform}")


# 中文：顶层接口 `finite_float`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `finite_float`; review type hints and callers before extending it.
def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else default
    except (TypeError, ValueError):
        return default


# 中文：顶层接口 `element_features`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `element_features`; review type hints and callers before extending it.
@lru_cache(maxsize=119)
def element_features(atomic_number: int) -> tuple[float, ...]:
    if atomic_number <= 0:
        return (0.0,) * ELEMENT_FEATURE_DIM
    element = Element.from_Z(int(atomic_number))
    block = str(getattr(element, "block", ""))
    values = [
        atomic_number / 118.0,
        finite_float(element.group) / 18.0,
        finite_float(element.row) / 7.0,
        math.log1p(max(finite_float(element.atomic_mass), 0.0)) / math.log(301.0),
        finite_float(element.X) / 4.0,
        finite_float(element.atomic_radius) / 3.0,
        finite_float(element.average_ionic_radius) / 3.0,
        finite_float(element.ionization_energy) / 25.0,
        float(np.clip(finite_float(element.electron_affinity) / 4.0, -1.0, 2.0)),
        finite_float(element.mendeleev_no) / 103.0,
        float(block == "s"),
        float(block == "p"),
        float(block == "d"),
        float(block == "f"),
    ]
    return tuple(float(value) for value in values)


# 中文：顶层接口 `table_sha256`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `table_sha256`; review type hints and callers before extending it.
def table_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 中文：顶层接口 `slab_fractions`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `slab_fractions`; review type hints and callers before extending it.
def slab_fractions(structure: Structure) -> tuple[float, float]:
    z = np.sort(np.mod([float(site.frac_coords[2]) for site in structure], 1.0))
    if len(z) <= 1:
        return 0.0, 1.0
    gaps = np.diff(np.r_[z, z[0] + 1.0])
    vacuum = float(np.max(gaps))
    return 1.0 - vacuum, vacuum


# 中文：顶层接口 `global_invariants`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `global_invariants`; review type hints and callers before extending it.
def global_invariants(structure: Structure) -> np.ndarray:
    lattice = structure.lattice
    n_atoms = max(len(structure), 1)
    area = float(np.linalg.norm(np.cross(lattice.matrix[0], lattice.matrix[1])))
    slab_fraction, vacuum_fraction = slab_fractions(structure)
    return np.asarray(
        [
            math.log(max(lattice.a, 1e-8)),
            math.log(max(lattice.b, 1e-8)),
            math.log(max(lattice.c, 1e-8)),
            math.cos(math.radians(lattice.alpha)),
            math.cos(math.radians(lattice.beta)),
            math.cos(math.radians(lattice.gamma)),
            math.log(max(lattice.volume / n_atoms, 1e-8)),
            math.log(max(area / n_atoms, 1e-8)),
            slab_fraction,
            vacuum_fraction,
            math.log(float(n_atoms)),
        ],
        dtype=np.float32,
    )


# 中文：顶层接口 `numpy_neighbor_list`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `numpy_neighbor_list`; review type hints and callers before extending it.
def numpy_neighbor_list(
    structure: Structure, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Portable fallback for incompatible pymatgen/NumPy binary builds."""
    fractional = np.asarray(structure.frac_coords, dtype=np.float64)
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    inverse_lattice = np.linalg.inv(lattice)
    limits = np.ceil(
        float(radius) * np.linalg.norm(inverse_lattice, axis=0) + 1.0
    ).astype(int)
    center_parts: list[np.ndarray] = []
    neighbor_parts: list[np.ndarray] = []
    image_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    for i_shift in range(-int(limits[0]), int(limits[0]) + 1):
        for j_shift in range(-int(limits[1]), int(limits[1]) + 1):
            for k_shift in range(-int(limits[2]), int(limits[2]) + 1):
                image = np.asarray([i_shift, j_shift, k_shift], dtype=np.float64)
                # delta[center, neighbor] points from center to its periodic neighbor.
                delta = (
                    fractional[None, :, :]
                    + image[None, None, :]
                    - fractional[:, None, :]
                )
                cartesian = delta @ lattice
                distance = np.linalg.norm(cartesian, axis=-1)
                selected = (distance <= float(radius) + 1e-8) & (distance > 1e-7)
                centers, neighbors = np.where(selected)
                if centers.size:
                    center_parts.append(centers.astype(np.int64))
                    neighbor_parts.append(neighbors.astype(np.int64))
                    image_parts.append(
                        np.repeat(
                            image.astype(np.float32)[None, :],
                            centers.size,
                            axis=0,
                        )
                    )
                    distance_parts.append(distance[selected].astype(np.float32))
    if not center_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    return (
        np.concatenate(center_parts),
        np.concatenate(neighbor_parts),
        np.concatenate(image_parts),
        np.concatenate(distance_parts),
    )


# 中文：顶层接口 `build_periodic_graph`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `build_periodic_graph`; review type hints and callers before extending it.
def build_periodic_graph(
    structure: Structure,
    radius: float,
    max_neighbors: int,
    identifier: str = "",
) -> dict[str, Any]:
    try:
        center, neighbor, images, distances = structure.get_neighbor_list(r=radius)
    except (TypeError, ValueError):
        center, neighbor, images, distances = numpy_neighbor_list(structure, radius)
    center = np.asarray(center, dtype=np.int64)
    neighbor = np.asarray(neighbor, dtype=np.int64)
    images = np.asarray(images, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)
    valid = distances > 1e-7
    center, neighbor, images, distances = (
        center[valid],
        neighbor[valid],
        images[valid],
        distances[valid],
    )
    keep: list[int] = []
    for atom in range(len(structure)):
        local = np.where(center == atom)[0]
        if local.size:
            # Cartesian rotations can perturb otherwise degenerate distances by a
            # few ulps.  Sorting only by the raw float can then select a different
            # subset when max_neighbors cuts through a coordination shell, which
            # breaks rotation invariance.  Quantize distances to one micro-angstrom
            # and use fractional image/species indices as invariant tie breakers.
            local_distances = np.rint(
                distances[local].astype(np.float64) * 1_000_000.0
            ).astype(np.int64)
            order = np.lexsort(
                (
                    images[local, 2],
                    images[local, 1],
                    images[local, 0],
                    neighbor[local],
                    local_distances,
                )
            )
            local = local[order[:max_neighbors]]
            keep.extend(int(x) for x in local)
    if not keep:
        raise ValueError(f"no periodic neighbors found for {identifier or 'structure'}")
    keep_array = np.asarray(keep, dtype=np.int64)
    # Messages flow source(neighbor) -> destination(center).
    edge_index = np.stack([neighbor[keep_array], center[keep_array]], axis=0)
    atomic_numbers = [int(site.specie.Z) for site in structure]
    return {
        "id": identifier,
        "z": torch.tensor(atomic_numbers, dtype=torch.long),
        "atom_features": torch.tensor(
            [element_features(z) for z in atomic_numbers], dtype=torch.float32
        ),
        "frac_pos": torch.tensor(
            np.mod(structure.frac_coords, 1.0), dtype=torch.float32
        ),
        "lattice": torch.tensor(structure.lattice.matrix, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_shift": torch.tensor(images[keep_array], dtype=torch.float32),
        "global_features": torch.tensor(global_invariants(structure)),
        "elements": sorted(set(atomic_numbers)),
    }


# 中文：顶层接口 `row_targets`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `row_targets`; review type hints and callers before extending it.
def row_targets(row: pd.Series) -> tuple[torch.Tensor, torch.Tensor, int]:
    values: list[float] = []
    masks: list[bool] = []
    for spec in REGRESSION_TARGETS:
        value = pd.to_numeric(row.get(spec.name), errors="coerce")
        valid = bool(pd.notna(value) and np.isfinite(float(value)))
        if spec.name == "Work_Function_Mean_eV" and "Work_Function_Reliable" in row:
            reliability = row.get("Work_Function_Reliable")
            if pd.notna(reliability):
                valid = valid and (
                    bool(reliability)
                    if isinstance(reliability, (bool, np.bool_))
                    else str(reliability).strip().lower()
                    in {"1", "true", "yes"}
                )
        if (
            spec.name == "NFE_Effective_Mass_Geomean_me"
            and "NFE_Parabolic_R2_KG" in row
            and "NFE_Parabolic_R2_GM" in row
        ):
            left_r2 = pd.to_numeric(
                row.get("NFE_Parabolic_R2_KG"), errors="coerce"
            )
            right_r2 = pd.to_numeric(
                row.get("NFE_Parabolic_R2_GM"), errors="coerce"
            )
            if pd.notna(left_r2) and pd.notna(right_r2):
                valid = valid and min(float(left_r2), float(right_r2)) >= 0.80
        masks.append(valid)
        values.append(
            transform_target(float(value), spec.transform) if valid else 0.0
        )
    label_text = str(row.get("NFE_Pseudo_Label", "")).strip().lower()
    label = LABEL_TO_INDEX.get(label_text, -1)
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.bool),
        label,
    )


# 中文：顶层接口 `build_cache`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `build_cache`; review type hints and callers before extending it.
def build_cache(
    table_path: str | Path,
    root: str | Path,
    cache_path: str | Path,
    *,
    radius: float,
    max_neighbors: int,
) -> dict[str, Any]:
    table_path = Path(table_path).resolve()
    root = Path(root).resolve()
    cache_path = Path(cache_path).resolve()
    frame = pd.read_csv(table_path)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for _, row in tqdm(
        frame.iterrows(), total=len(frame), desc="building graph cache", unit="structure"
    ):
        identifier = str(row.get("Structure_Name", ""))
        file_path = Path(str(row.get("File_Path", "")))
        candidates = [file_path] if file_path.is_absolute() else [root / file_path]
        # The CSV may have been copied from the extraction server to another host.
        # Prefer the recorded path, then recover from the portable data/ directory.
        candidates.extend(
            [
                root / "data" / file_path.name,
                table_path.parent / "data" / file_path.name,
            ]
        )
        file_path = next((path for path in candidates if path.is_file()), candidates[0])
        try:
            structure = Structure.from_file(file_path)
            graph = build_periodic_graph(
                structure, radius, max_neighbors, identifier=identifier
            )
            targets, target_mask, label = row_targets(row)
            graph.update(
                {
                    "file_path": str(file_path),
                    "split": str(row.get("Suggested_Split", "train")).lower(),
                    "split_group": str(row.get("Split_Group", "")),
                    "targets": targets,
                    "target_mask": target_mask,
                    "label": label,
                    "sample_weight": float(
                        np.clip(
                            finite_float(row.get("Data_Quality_Score"), 1.0),
                            0.25,
                            1.0,
                        )
                    ),
                }
            )
            records.append(graph)
        except Exception as exc:
            skipped.append({"id": identifier, "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "schema": "nfe-mxene-cache-1.0",
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


# 中文：顶层接口 `torch_load_compat`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `torch_load_compat`; review type hints and callers before extending it.
def torch_load_compat(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# 中文：顶层接口 `load_or_build_cache`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `load_or_build_cache`; review type hints and callers before extending it.
def load_or_build_cache(
    table_path: str | Path,
    root: str | Path,
    cache_path: str | Path,
    *,
    radius: float,
    max_neighbors: int,
    rebuild: bool = False,
) -> dict[str, Any]:
    table_path = Path(table_path).resolve()
    cache_path = Path(cache_path).resolve()
    if cache_path.is_file() and not rebuild:
        cache = torch_load_compat(cache_path)
        compatible = (
            cache.get("schema") == "nfe-mxene-cache-1.0"
            and cache.get("table_sha256") == table_sha256(table_path)
            and float(cache.get("radius", -1)) == float(radius)
            and int(cache.get("max_neighbors", -1)) == int(max_neighbors)
        )
        if compatible:
            return cache
    return build_cache(
        table_path,
        root,
        cache_path,
        radius=radius,
        max_neighbors=max_neighbors,
    )


# 中文：顶层接口 `split_indices`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `split_indices`; review type hints and callers before extending it.
def split_indices(records: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    result = {"train": [], "validation": [], "test": []}
    for index, record in enumerate(records):
        split = str(record.get("split", "train")).lower()
        if split in {"val", "valid"}:
            split = "validation"
        if split not in result:
            split = "train"
        result[split].append(index)
    return result


# 中文：顶层接口 `assert_disjoint_split_groups`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `assert_disjoint_split_groups`; review type hints and callers before extending it.
def assert_disjoint_split_groups(
    records: Sequence[dict[str, Any]], splits: dict[str, Sequence[int]]
) -> None:
    groups = {
        split: {
            str(records[index].get("split_group", "")).strip()
            for index in indices
            if str(records[index].get("split_group", "")).strip()
        }
        for split, indices in splits.items()
    }
    conflicts: list[str] = []
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                conflicts.append(f"{left}/{right}: {examples}")
    if conflicts:
        raise RuntimeError(
            "Split_Group leakage detected across dataset splits: "
            + "; ".join(conflicts)
        )


# 中文：顶层接口 `robust_normalizers`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `robust_normalizers`; review type hints and callers before extending it.
def robust_normalizers(
    records: Sequence[dict[str, Any]], train_indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    globals_array = torch.stack([records[i]["global_features"] for i in train_indices])
    global_median = globals_array.median(dim=0).values
    global_q1 = torch.quantile(globals_array, 0.25, dim=0)
    global_q3 = torch.quantile(globals_array, 0.75, dim=0)
    global_scale = ((global_q3 - global_q1) / 1.349).clamp_min(1e-6)

    target_median = torch.zeros(len(REGRESSION_TARGETS), dtype=torch.float32)
    target_scale = torch.ones(len(REGRESSION_TARGETS), dtype=torch.float32)
    for target_index in range(len(REGRESSION_TARGETS)):
        available = [
            records[i]["targets"][target_index]
            for i in train_indices
            if bool(records[i]["target_mask"][target_index])
        ]
        if available:
            values = torch.stack(available)
            median = values.median()
            q1 = torch.quantile(values, 0.25)
            q3 = torch.quantile(values, 0.75)
            target_median[target_index] = median
            target_scale[target_index] = ((q3 - q1) / 1.349).clamp_min(1e-6)
    return {
        "global_median": global_median,
        "global_scale": global_scale,
        "target_median": target_median,
        "target_scale": target_scale,
    }


# 中文：顶层类 `NFEDataset`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `NFEDataset`; review type hints and callers before extending it.
class NFEDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        indices: Sequence[int],
        normalizers: dict[str, torch.Tensor],
    ) -> None:
        self.records = records
        self.indices = list(indices)
        self.normalizers = normalizers

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[self.indices[item]]
        result = dict(record)
        result["global_features"] = torch.clamp(
            (
                record["global_features"] - self.normalizers["global_median"]
            )
            / self.normalizers["global_scale"],
            -8.0,
            8.0,
        )
        result["targets_normalized"] = torch.clamp(
            (record["targets"] - self.normalizers["target_median"])
            / self.normalizers["target_scale"],
            -8.0,
            8.0,
        )
        return result


# 中文：顶层接口 `collate_graphs`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `collate_graphs`; review type hints and callers before extending it.
def collate_graphs(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    node_offset = 0
    z, atom_features_list, frac_pos, edge_index, edge_shift, batch = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    lattices, global_features = [], []
    targets, target_masks, labels, sample_weights = [], [], [], []
    identifiers, file_paths, elements = [], [], []
    for graph_index, item in enumerate(items):
        n_nodes = int(item["z"].shape[0])
        z.append(item["z"])
        atom_features_list.append(item["atom_features"])
        frac_pos.append(item["frac_pos"])
        edge_index.append(item["edge_index"] + node_offset)
        edge_shift.append(item["edge_shift"])
        batch.append(torch.full((n_nodes,), graph_index, dtype=torch.long))
        lattices.append(item["lattice"])
        global_features.append(item["global_features"])
        targets.append(item.get("targets_normalized", item["targets"]))
        target_masks.append(item["target_mask"])
        labels.append(int(item["label"]))
        sample_weights.append(float(item.get("sample_weight", 1.0)))
        identifiers.append(item.get("id", ""))
        file_paths.append(item.get("file_path", ""))
        elements.append(item.get("elements", []))
        node_offset += n_nodes
    return {
        "z": torch.cat(z, dim=0),
        "atom_features": torch.cat(atom_features_list, dim=0),
        "frac_pos": torch.cat(frac_pos, dim=0),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_shift": torch.cat(edge_shift, dim=0),
        "batch": torch.cat(batch, dim=0),
        "lattice": torch.stack(lattices),
        "global_features": torch.stack(global_features),
        "targets": torch.stack(targets),
        "target_mask": torch.stack(target_masks),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
        "ids": identifiers,
        "file_paths": file_paths,
        "elements": elements,
    }


# 中文：顶层接口 `move_batch`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `move_batch`; review type hints and callers before extending it.
def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# 中文：顶层接口 `class_weights`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `class_weights`; review type hints and callers before extending it.
def class_weights(
    records: Sequence[dict[str, Any]], indices: Sequence[int]
) -> torch.Tensor:
    counts = torch.ones(len(LABEL_TO_INDEX), dtype=torch.float32)
    for index in indices:
        label = int(records[index]["label"])
        if label >= 0:
            counts[label] += 1
    weights = torch.sqrt(counts.sum() / counts)
    return weights / weights.mean()
