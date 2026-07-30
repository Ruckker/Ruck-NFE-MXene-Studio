# ==============================================================================
# 中文概述：基础条件流生成器的数据规范化、居中、组分目录与批处理。
# English overview: Base conditional-flow data canonicalization, centering, composition catalog, and batching.
#
# 中文输入：晶体记录、分数条件、原子序数与晶格。
# English inputs: Crystal records, score conditions, atomic numbers, and lattices.
# 中文输出：中心化坐标、晶格参数、生成批次与新颖性参考。
# English outputs: Centered coordinates, lattice parameters, generation batches, and novelty references.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: center_slab_fractional, center_slab_fractional_tensor, slab_center_fractional_z, canonicalize_atoms, lattice_to_params, params_to_lattice, composition_formula, composition_key, parse_composition, prepare_generator_records, lattice_normalizers, CrystalFlowDataset, collate_crystals, composition_catalog, novelty_reference, generation_batch
#
# Author: Ruck
# Generated: 2026-07-29 20:36:56 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from pymatgen.core import Composition, Element
from torch.utils.data import Dataset

from .data import element_features


# 中文：顶层接口 `center_slab_fractional`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `center_slab_fractional`; review type hints and callers before extending it.
def center_slab_fractional(frac_pos: np.ndarray) -> np.ndarray:
    frac = np.mod(np.asarray(frac_pos, dtype=np.float64), 1.0).copy()
    if len(frac) == 0:
        return frac.astype(np.float32)
    for axis in (0, 1):
        angle = 2.0 * math.pi * frac[:, axis]
        mean_angle = math.atan2(np.sin(angle).mean(), np.cos(angle).mean())
        mean_fraction = (mean_angle / (2.0 * math.pi)) % 1.0
        frac[:, axis] = np.mod(frac[:, axis] + 0.5 - mean_fraction, 1.0)

    z_sorted = np.sort(frac[:, 2])
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    largest = int(np.argmax(gaps))
    start = float(z_sorted[(largest + 1) % len(z_sorted)])
    unwrapped = np.mod(frac[:, 2] - start, 1.0)
    slab_thickness = float(np.max(unwrapped) - np.min(unwrapped))
    frac[:, 2] = np.mod(unwrapped + 0.5 - 0.5 * slab_thickness, 1.0)
    return frac.astype(np.float32)


# 中文：顶层接口 `center_slab_fractional_tensor`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `center_slab_fractional_tensor`; review type hints and callers before extending it.
def center_slab_fractional_tensor(
    frac_pos: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """Fix the periodic translation gauge for each slab in a tensor batch."""
    centered = torch.remainder(frac_pos, 1.0).clone()
    if centered.numel() == 0:
        return centered
    n_graphs = int(batch_index.max().item()) + 1
    for graph_index in range(n_graphs):
        nodes = torch.where(batch_index == graph_index)[0]
        if not len(nodes):
            continue
        local = centered[nodes].clone()
        for axis in (0, 1):
            angle = 2.0 * math.pi * local[:, axis]
            mean_angle = torch.atan2(torch.sin(angle).mean(), torch.cos(angle).mean())
            mean_fraction = torch.remainder(
                mean_angle / (2.0 * math.pi), 1.0
            )
            local[:, axis] = torch.remainder(
                local[:, axis] + 0.5 - mean_fraction, 1.0
            )
        z_sorted = torch.sort(local[:, 2]).values
        gaps = torch.cat(
            [z_sorted[1:] - z_sorted[:-1], z_sorted[:1] + 1.0 - z_sorted[-1:]]
        )
        largest = int(torch.argmax(gaps).item())
        start = z_sorted[(largest + 1) % len(z_sorted)]
        unwrapped = torch.remainder(local[:, 2] - start, 1.0)
        slab_center = 0.5 * (unwrapped.min() + unwrapped.max())
        local[:, 2] = torch.remainder(unwrapped + 0.5 - slab_center, 1.0)
        centered[nodes] = local
    return centered


# 中文：顶层接口 `slab_center_fractional_z`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `slab_center_fractional_z`; review type hints and callers before extending it.
def slab_center_fractional_z(frac_pos: np.ndarray) -> float:
    frac = np.mod(np.asarray(frac_pos, dtype=np.float64), 1.0)
    if len(frac) == 0:
        return 0.5
    z_sorted = np.sort(frac[:, 2])
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    largest = int(np.argmax(gaps))
    start = float(z_sorted[(largest + 1) % len(z_sorted)])
    unwrapped = np.mod(frac[:, 2] - start, 1.0)
    local_center = 0.5 * (float(unwrapped.min()) + float(unwrapped.max()))
    return float((start + local_center) % 1.0)


# 中文：顶层接口 `canonicalize_atoms`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `canonicalize_atoms`; review type hints and callers before extending it.
def canonicalize_atoms(
    z: torch.Tensor, frac_pos: torch.Tensor, atom_features_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frac = center_slab_fractional(frac_pos.cpu().numpy())
    atomic_numbers = z.cpu().numpy()
    order = np.lexsort(
        (frac[:, 0], frac[:, 1], frac[:, 2], atomic_numbers)
    )
    order_tensor = torch.tensor(order, dtype=torch.long)
    return (
        z[order_tensor].clone(),
        torch.tensor(frac[order], dtype=torch.float32),
        atom_features_tensor[order_tensor].clone(),
    )


# 中文：顶层接口 `lattice_to_params`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `lattice_to_params`; review type hints and callers before extending it.
def lattice_to_params(lattice: torch.Tensor) -> torch.Tensor:
    gram = lattice @ lattice.transpose(-1, -2)
    identity = torch.eye(3, dtype=lattice.dtype, device=lattice.device)
    cholesky = torch.linalg.cholesky(gram + 1e-7 * identity)
    return torch.stack(
        [
            torch.log(cholesky[..., 0, 0].clamp_min(1e-6)),
            cholesky[..., 1, 0],
            torch.log(cholesky[..., 1, 1].clamp_min(1e-6)),
            cholesky[..., 2, 0],
            cholesky[..., 2, 1],
            torch.log(cholesky[..., 2, 2].clamp_min(1e-6)),
        ],
        dim=-1,
    )


# 中文：顶层接口 `params_to_lattice`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `params_to_lattice`; review type hints and callers before extending it.
def params_to_lattice(params: torch.Tensor) -> torch.Tensor:
    shape = params.shape[:-1] + (3, 3)
    lattice = params.new_zeros(shape)
    lattice[..., 0, 0] = torch.exp(params[..., 0].clamp(-2.0, 5.0))
    lattice[..., 1, 0] = params[..., 1]
    lattice[..., 1, 1] = torch.exp(params[..., 2].clamp(-2.0, 5.0))
    lattice[..., 2, 0] = params[..., 3]
    lattice[..., 2, 1] = params[..., 4]
    lattice[..., 2, 2] = torch.exp(params[..., 5].clamp(-2.0, 5.0))
    return lattice


# 中文：顶层接口 `composition_formula`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `composition_formula`; review type hints and callers before extending it.
def composition_formula(z: Sequence[int]) -> str:
    counts = Counter(int(value) for value in z)
    composition = Composition(
        {Element.from_Z(atomic_number).symbol: count for atomic_number, count in counts.items()}
    )
    return composition.reduced_formula


# 中文：顶层接口 `composition_key`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `composition_key`; review type hints and callers before extending it.
def composition_key(z: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(z))


# 中文：顶层接口 `parse_composition`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_composition`; review type hints and callers before extending it.
def parse_composition(text: str) -> list[int]:
    composition = Composition(text)
    atomic_numbers: list[int] = []
    for element, amount in composition.items():
        rounded = int(round(float(amount)))
        if rounded <= 0 or not math.isclose(float(amount), rounded, abs_tol=1e-6):
            raise ValueError(
                f"composition must contain positive integer atom counts: {text}"
            )
        atomic_numbers.extend([int(element.Z)] * rounded)
    if not atomic_numbers:
        raise ValueError(f"empty composition: {text}")
    return sorted(atomic_numbers)


# 中文：顶层接口 `prepare_generator_records`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `prepare_generator_records`; review type hints and callers before extending it.
def prepare_generator_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        z, frac, descriptors = canonicalize_atoms(
            record["z"], record["frac_pos"], record["atom_features"]
        )
        score = float(record["targets"][0]) if bool(record["target_mask"][0]) else 0.5
        prepared.append(
            {
                "id": record.get("id", ""),
                "split": record.get("split", "train"),
                "split_group": record.get("split_group", ""),
                "z": z,
                "atom_features": descriptors,
                "frac_pos": frac,
                "lattice_params": lattice_to_params(record["lattice"]),
                "label": int(record.get("label", -1)),
                "score": float(np.clip(score, 0.0, 1.0)),
                "sample_weight": float(record.get("sample_weight", 1.0)),
                "formula": composition_formula(z.tolist()),
            }
        )
    return prepared


# 中文：顶层接口 `lattice_normalizers`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `lattice_normalizers`; review type hints and callers before extending it.
def lattice_normalizers(
    records: Sequence[dict[str, Any]], train_indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    values = torch.stack([records[index]["lattice_params"] for index in train_indices])
    median = values.median(dim=0).values
    q1 = torch.quantile(values, 0.25, dim=0)
    q3 = torch.quantile(values, 0.75, dim=0)
    scale = ((q3 - q1) / 1.349).clamp_min(1e-4)
    return {"lattice_median": median, "lattice_scale": scale}


# 中文：顶层类 `CrystalFlowDataset`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `CrystalFlowDataset`; review type hints and callers before extending it.
class CrystalFlowDataset(Dataset[dict[str, Any]]):
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
        record = dict(self.records[self.indices[item]])
        record["lattice_normalized"] = torch.clamp(
            (
                record["lattice_params"] - self.normalizers["lattice_median"]
            )
            / self.normalizers["lattice_scale"],
            -8.0,
            8.0,
        )
        return record


# 中文：顶层接口 `collate_crystals`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `collate_crystals`; review type hints and callers before extending it.
def collate_crystals(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    z, descriptors, target_frac, batch = [], [], [], []
    lattice, labels, scores, weights = [], [], [], []
    identifiers, formulas = [], []
    for graph_index, item in enumerate(items):
        n_atoms = len(item["z"])
        z.append(item["z"])
        descriptors.append(item["atom_features"])
        target_frac.append(item["frac_pos"])
        batch.append(torch.full((n_atoms,), graph_index, dtype=torch.long))
        lattice.append(item["lattice_normalized"])
        labels.append(int(item["label"]))
        scores.append(float(item["score"]))
        weights.append(float(item.get("sample_weight", 1.0)))
        identifiers.append(item.get("id", ""))
        formulas.append(item.get("formula", ""))
    return {
        "z": torch.cat(z),
        "atom_features": torch.cat(descriptors),
        "target_frac": torch.cat(target_frac),
        "batch": torch.cat(batch),
        "target_lattice": torch.stack(lattice),
        "labels": torch.tensor(labels, dtype=torch.long),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "sample_weights": torch.tensor(weights, dtype=torch.float32),
        "ids": identifiers,
        "formulas": formulas,
    }


# 中文：顶层接口 `composition_catalog`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `composition_catalog`; review type hints and callers before extending it.
def composition_catalog(
    records: Sequence[dict[str, Any]], indices: Sequence[int]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for index in indices:
        record = records[index]
        grouped[tuple(int(value) for value in record["z"].tolist())].append(record)
    catalog = []
    for z_values, members in grouped.items():
        scores = np.asarray([member["score"] for member in members])
        labels = [member["label"] for member in members]
        catalog.append(
            {
                "z": list(z_values),
                "formula": composition_formula(z_values),
                "count": len(members),
                "score_mean": float(scores.mean()),
                "score_max": float(scores.max()),
                "low_fraction": float(np.mean(np.asarray(labels) == 0)),
                "medium_fraction": float(np.mean(np.asarray(labels) == 1)),
                "high_fraction": float(np.mean(np.asarray(labels) == 2)),
            }
        )
    return sorted(catalog, key=lambda item: (-item["count"], item["formula"]))


# 中文：顶层接口 `novelty_reference`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `novelty_reference`; review type hints and callers before extending it.
def novelty_reference(
    records: Sequence[dict[str, Any]], indices: Sequence[int]
) -> dict[str, list[dict[str, Any]]]:
    reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index in indices:
        record = records[index]
        z_values = [int(value) for value in record["z"].tolist()]
        lattice = params_to_lattice(record["lattice_params"]).numpy()
        reference[composition_key(z_values)].append(
            {
                "z": z_values,
                "frac_pos": record["frac_pos"].to(torch.float16).tolist(),
                "lattice": torch.tensor(lattice).to(torch.float16).tolist(),
            }
        )
    return dict(reference)


# 中文：顶层接口 `generation_batch`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `generation_batch`; review type hints and callers before extending it.
def generation_batch(
    compositions: Sequence[Sequence[int]], device: torch.device
) -> dict[str, torch.Tensor]:
    z_parts, descriptor_parts, batch_parts = [], [], []
    for graph_index, atomic_numbers in enumerate(compositions):
        z = torch.tensor(sorted(int(value) for value in atomic_numbers), dtype=torch.long)
        descriptors = torch.tensor(
            [element_features(int(value)) for value in z], dtype=torch.float32
        )
        z_parts.append(z)
        descriptor_parts.append(descriptors)
        batch_parts.append(
            torch.full((len(z),), graph_index, dtype=torch.long)
        )
    return {
        "z": torch.cat(z_parts).to(device),
        "atom_features": torch.cat(descriptor_parts).to(device),
        "batch": torch.cat(batch_parts).to(device),
    }
