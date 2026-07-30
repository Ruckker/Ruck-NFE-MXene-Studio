# ==============================================================================
# 中文概述：将 MXene 拆为内核层、表面基团、锚点与层拓扑模板。
# English overview: Represent MXenes as core layers, surface terminations, anchors, and layered topology templates.
#
# 中文输入：已弛豫 MXene、表面几何统计与 NFE 条件。
# English inputs: Relaxed MXenes, surface-geometry statistics, and NFE conditions.
# 中文输出：表面感知训练记录、模板目录、角色掩码与锚点。
# English outputs: Surface-aware records, template catalog, role masks, and anchors.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: _surface_group_types, _reindex_array, _reindex_anchor, prepare_surface_generator_records, SurfaceTemplateDataset, collate_surface_templates, surface_template_catalog
#
# Author: Ruck
# Generated: 2026-07-29 22:30:46 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .generator_data import (
    canonicalize_atoms,
    composition_formula,
    lattice_normalizers,
    lattice_to_params,
    params_to_lattice,
)
from .surface_geometry import analyze_surface_geometry


GROUP_CORE = 0
GROUP_ATOMIC_TERMINATION = 1
GROUP_OH_OXYGEN = 2
GROUP_OH_HYDROGEN = 3
GROUP_SURFACE_HYDROGEN = 4


# 中文：顶层接口 `_surface_group_types`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_surface_group_types`; review type hints and callers before extending it.
def _surface_group_types(
    atomic_numbers: np.ndarray, analysis: Any
) -> np.ndarray:
    group_type = np.full(len(atomic_numbers), GROUP_CORE, dtype=np.int64)
    group_type[analysis.surface_side != 0] = GROUP_ATOMIC_TERMINATION
    oh_members: set[int] = set()
    for bond in analysis.oh_bonds:
        oxygen = int(bond["oxygen_index"])
        hydrogen = int(bond["hydrogen_index"])
        group_type[oxygen] = GROUP_OH_OXYGEN
        group_type[hydrogen] = GROUP_OH_HYDROGEN
        oh_members.update((oxygen, hydrogen))
    for index, atomic_number in enumerate(atomic_numbers):
        if (
            analysis.surface_side[index] != 0
            and int(atomic_number) == 1
            and index not in oh_members
        ):
            group_type[index] = GROUP_SURFACE_HYDROGEN
    return group_type


# 中文：顶层接口 `_reindex_array`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_reindex_array`; review type hints and callers before extending it.
def _reindex_array(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    return np.asarray(values)[order].copy()


# 中文：顶层接口 `_reindex_anchor`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_reindex_anchor`; review type hints and callers before extending it.
def _reindex_anchor(anchor: np.ndarray, order: np.ndarray) -> np.ndarray:
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    reordered = np.full(len(order), -1, dtype=np.int64)
    for new_index, old_index in enumerate(order):
        old_anchor = int(anchor[old_index])
        if old_anchor >= 0:
            reordered[new_index] = int(inverse[old_anchor])
    return reordered


# 中文：顶层接口 `prepare_surface_generator_records`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `prepare_surface_generator_records`; review type hints and callers before extending it.
def prepare_surface_generator_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for record in records:
        z, frac, descriptors = canonicalize_atoms(
            record["z"], record["frac_pos"], record["atom_features"]
        )
        lattice = record["lattice"].cpu()
        analysis = analyze_surface_geometry(
            z.cpu().numpy(), frac.cpu().numpy(), lattice.numpy()
        )
        group_type = _surface_group_types(z.cpu().numpy(), analysis)
        # The relaxed MXene data has one atom per ordered z layer.  Sorting by
        # physical height creates a stable site correspondence across element
        # substitutions without relying on atomic-number ordering.
        order = np.lexsort(
            (
                analysis.centered_frac[:, 1],
                analysis.centered_frac[:, 0],
                analysis.z_cartesian_A,
            )
        )
        z = z[torch.tensor(order, dtype=torch.long)]
        descriptors = descriptors[torch.tensor(order, dtype=torch.long)]
        frac = torch.tensor(
            _reindex_array(analysis.centered_frac, order), dtype=torch.float32
        )
        side = _reindex_array(analysis.surface_side, order).astype(np.int64)
        layers = _reindex_array(analysis.layer_index, order).astype(np.int64)
        group_type = _reindex_array(group_type, order).astype(np.int64)
        anchor = _reindex_anchor(analysis.anchor_index, order)
        coordination = _reindex_array(
            analysis.adsorption_coordination, order
        ).astype(np.int64)
        layer_center = 0.5 * (float(layers.min()) + float(layers.max()))
        layer_scale = max(0.5 * float(layers.max() - layers.min()), 1.0)
        layer_position = (layers.astype(np.float32) - layer_center) / layer_scale
        role_weight = np.ones(len(z), dtype=np.float32)
        role_weight[side != 0] = 2.0
        role_weight[group_type == GROUP_OH_HYDROGEN] = 3.0
        score = (
            float(record["targets"][0])
            if bool(record["target_mask"][0])
            else 0.5
        )
        topology_key = (
            len(z),
            tuple(int(value) for value in side),
            tuple(int(value) for value in group_type),
        )
        prepared.append(
            {
                "id": record.get("id", ""),
                "split": record.get("split", "train"),
                "split_group": record.get("split_group", ""),
                "z": z.clone(),
                "atom_features": descriptors.clone(),
                "frac_pos": frac,
                "lattice_params": lattice_to_params(lattice),
                "surface_side": torch.tensor(side, dtype=torch.long),
                "layer_position": torch.tensor(
                    layer_position, dtype=torch.float32
                ),
                "group_type": torch.tensor(group_type, dtype=torch.long),
                "anchor_index": torch.tensor(anchor, dtype=torch.long),
                "adsorption_coordination": torch.tensor(
                    coordination, dtype=torch.long
                ),
                "role_weight": torch.tensor(role_weight, dtype=torch.float32),
                "topology_key": topology_key,
                "termination_motif": "|".join(
                    group["formula"] for group in analysis.surface_groups
                ),
                "label": int(record.get("label", -1)),
                "score": float(np.clip(score, 0.0, 1.0)),
                "sample_weight": float(record.get("sample_weight", 1.0)),
                "formula": composition_formula(z.tolist()),
                "surface_warnings": list(analysis.warnings),
            }
        )
    return prepared


# 中文：顶层类 `SurfaceTemplateDataset`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `SurfaceTemplateDataset`; review type hints and callers before extending it.
class SurfaceTemplateDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        indices: Sequence[int],
        normalizers: dict[str, torch.Tensor],
    ) -> None:
        self.records = records
        self.indices = list(indices)
        self.normalizers = normalizers
        pools: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index in self.indices:
            pools[records[index]["topology_key"]].append(index)
        self.pools = dict(pools)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        target_index = self.indices[item]
        target = dict(self.records[target_index])
        pool = self.pools[target["topology_key"]]
        if len(pool) > 1:
            template_index = random.choice(pool)
            if template_index == target_index:
                template_index = pool[(pool.index(template_index) + 1) % len(pool)]
        else:
            template_index = target_index
        template = self.records[template_index]
        target["lattice_normalized"] = torch.clamp(
            (
                target["lattice_params"] - self.normalizers["lattice_median"]
            )
            / self.normalizers["lattice_scale"],
            -8.0,
            8.0,
        )
        target["template_frac"] = template["frac_pos"].clone()
        target["template_lattice_normalized"] = torch.clamp(
            (
                template["lattice_params"]
                - self.normalizers["lattice_median"]
            )
            / self.normalizers["lattice_scale"],
            -8.0,
            8.0,
        )
        target["template_id"] = template.get("id", "")
        return target


# 中文：顶层接口 `collate_surface_templates`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `collate_surface_templates`; review type hints and callers before extending it.
def collate_surface_templates(
    items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    tensor_keys = (
        "z",
        "atom_features",
        "frac_pos",
        "template_frac",
        "surface_side",
        "layer_position",
        "group_type",
        "adsorption_coordination",
        "role_weight",
    )
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
    anchors: list[torch.Tensor] = []
    batch: list[torch.Tensor] = []
    lattices: list[torch.Tensor] = []
    template_lattices: list[torch.Tensor] = []
    labels: list[int] = []
    scores: list[float] = []
    weights: list[float] = []
    identifiers: list[str] = []
    template_identifiers: list[str] = []
    formulas: list[str] = []
    offset = 0
    for graph_index, item in enumerate(items):
        n_atoms = len(item["z"])
        for key in tensor_keys:
            parts[key].append(item[key])
        local_anchor = item["anchor_index"].clone()
        valid = local_anchor >= 0
        local_anchor[valid] += offset
        anchors.append(local_anchor)
        batch.append(torch.full((n_atoms,), graph_index, dtype=torch.long))
        lattices.append(item["lattice_normalized"])
        template_lattices.append(item["template_lattice_normalized"])
        labels.append(int(item["label"]))
        scores.append(float(item["score"]))
        weights.append(float(item.get("sample_weight", 1.0)))
        identifiers.append(item.get("id", ""))
        template_identifiers.append(item.get("template_id", ""))
        formulas.append(item.get("formula", ""))
        offset += n_atoms
    return {
        **{key: torch.cat(value) for key, value in parts.items()},
        "target_frac": torch.cat(parts["frac_pos"]),
        "anchor_index": torch.cat(anchors),
        "batch": torch.cat(batch),
        "target_lattice": torch.stack(lattices),
        "template_lattice": torch.stack(template_lattices),
        "labels": torch.tensor(labels, dtype=torch.long),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "sample_weights": torch.tensor(weights, dtype=torch.float32),
        "ids": identifiers,
        "template_ids": template_identifiers,
        "formulas": formulas,
    }


# 中文：顶层接口 `surface_template_catalog`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `surface_template_catalog`; review type hints and callers before extending it.
def surface_template_catalog(
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for index in indices:
        record = records[index]
        lattice = params_to_lattice(record["lattice_params"]).numpy()
        catalog.append(
            {
                "id": record.get("id", ""),
                "z": [int(value) for value in record["z"].tolist()],
                "formula": record["formula"],
                "frac_pos": record["frac_pos"].to(torch.float16).tolist(),
                "lattice": torch.tensor(lattice).to(torch.float16).tolist(),
                "surface_side": record["surface_side"].tolist(),
                "layer_position": record["layer_position"].tolist(),
                "group_type": record["group_type"].tolist(),
                "anchor_index": record["anchor_index"].tolist(),
                "adsorption_coordination": record[
                    "adsorption_coordination"
                ].tolist(),
                "termination_motif": record["termination_motif"],
                "label": int(record["label"]),
                "score": float(record["score"]),
            }
        )
    return catalog


__all__ = [
    "GROUP_ATOMIC_TERMINATION",
    "GROUP_CORE",
    "GROUP_OH_HYDROGEN",
    "GROUP_OH_OXYGEN",
    "GROUP_SURFACE_HYDROGEN",
    "SurfaceTemplateDataset",
    "collate_surface_templates",
    "lattice_normalizers",
    "prepare_surface_generator_records",
    "surface_template_catalog",
]
