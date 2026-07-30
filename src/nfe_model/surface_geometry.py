# ==============================================================================
# 中文概述：分析并验证 MXene 层序、上下端基、三配位 hollow 位点与键长。
# English overview: Analyze and validate MXene layers, top/bottom terminations, threefold hollow sites, and bonds.
#
# 中文输入：Pymatgen Structure 或原子/晶格数组以及训练集几何分位。
# English inputs: Pymatgen Structure or atomic/lattice arrays plus training geometry quantiles.
# 中文输出：表面角色、配位、端基类型、层数、键长与严格错误列表。
# English outputs: Surface roles, coordination, termination types, layer count, bond lengths, and strict errors.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: _is_surface_metal, _atomic_radius, _cluster_layers, _xy_image_shifts, minimum_xy_image_distance, _surface_metal_coordination, _formula, SurfaceGeometry, analyze_surface_geometry, structure_from_arrays, validate_surface_topology
#
# Author: Ruck
# Generated: 2026-07-29 22:47:20 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Element, Lattice, Structure

from .generator_data import center_slab_fractional


CORE = 0
BOTTOM = -1
TOP = 1
ROLE_NAMES = {CORE: "core", BOTTOM: "bottom", TOP: "top"}


# 中文：顶层接口 `_is_surface_metal`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_is_surface_metal`; review type hints and callers before extending it.
def _is_surface_metal(atomic_number: int) -> bool:
    element = Element.from_Z(int(atomic_number))
    return bool(element.is_transition_metal or str(getattr(element, "block", "")) == "f")


# 中文：顶层接口 `_atomic_radius`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_atomic_radius`; review type hints and callers before extending it.
def _atomic_radius(atomic_number: int) -> float:
    element = Element.from_Z(int(atomic_number))
    radius = getattr(element, "atomic_radius", None)
    if radius is None:
        radius = getattr(element, "average_ionic_radius", None)
    try:
        value = float(radius)
    except (TypeError, ValueError):
        value = 1.2
    return value if math.isfinite(value) and value > 0.0 else 1.2


# 中文：顶层接口 `_cluster_layers`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_cluster_layers`; review type hints and callers before extending it.
def _cluster_layers(z_cartesian: np.ndarray, tolerance_A: float) -> np.ndarray:
    order = np.argsort(z_cartesian)
    groups: list[list[int]] = []
    for atom in order:
        atom_index = int(atom)
        if not groups:
            groups.append([atom_index])
            continue
        center = float(np.mean(z_cartesian[groups[-1]]))
        if abs(float(z_cartesian[atom_index]) - center) <= tolerance_A:
            groups[-1].append(atom_index)
        else:
            groups.append([atom_index])
    layer_index = np.zeros(len(z_cartesian), dtype=np.int64)
    for index, group in enumerate(groups):
        layer_index[group] = index
    return layer_index


# 中文：顶层接口 `_xy_image_shifts`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_xy_image_shifts`; review type hints and callers before extending it.
def _xy_image_shifts() -> np.ndarray:
    return np.asarray(
        [
            [i, j, 0]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
        ],
        dtype=np.float64,
    )


XY_SHIFTS = _xy_image_shifts()


# 中文：顶层接口 `minimum_xy_image_distance`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `minimum_xy_image_distance`; review type hints and callers before extending it.
def minimum_xy_image_distance(
    left_fractional: np.ndarray,
    right_fractional: np.ndarray,
    lattice: np.ndarray,
) -> float:
    delta = np.asarray(left_fractional, dtype=np.float64) - np.asarray(
        right_fractional, dtype=np.float64
    )
    candidates = (delta[None, :] + XY_SHIFTS) @ np.asarray(
        lattice, dtype=np.float64
    )
    return float(np.linalg.norm(candidates, axis=1).min())


# 中文：顶层接口 `_surface_metal_coordination`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_surface_metal_coordination`; review type hints and callers before extending it.
def _surface_metal_coordination(
    atom_index: int,
    metal_indices: np.ndarray,
    frac_pos: np.ndarray,
    lattice: np.ndarray,
    *,
    shell_tolerance_A: float,
) -> tuple[int, float]:
    candidates: list[float] = []
    for metal_index in metal_indices:
        delta = frac_pos[int(atom_index)] - frac_pos[int(metal_index)]
        images = (delta[None, :] + XY_SHIFTS) @ lattice
        candidates.extend(float(value) for value in np.linalg.norm(images, axis=1))
    positive = np.asarray([value for value in candidates if value > 1e-7])
    if not len(positive):
        return 0, float("nan")
    nearest = float(positive.min())
    coordination = int(np.sum(positive <= nearest + float(shell_tolerance_A)))
    return coordination, nearest


# 中文：顶层接口 `_formula`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_formula`; review type hints and callers before extending it.
def _formula(indices: Sequence[int], atomic_numbers: np.ndarray) -> str:
    counts = Counter(
        Element.from_Z(int(atomic_numbers[index])).symbol for index in indices
    )
    return "".join(
        symbol + (str(count) if count > 1 else "")
        for symbol, count in sorted(counts.items())
    )


# 中文：顶层类 `SurfaceGeometry`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `SurfaceGeometry`; review type hints and callers before extending it.
@dataclass
class SurfaceGeometry:
    centered_frac: np.ndarray
    z_cartesian_A: np.ndarray
    layer_index: np.ndarray
    surface_side: np.ndarray
    group_index: np.ndarray
    anchor_index: np.ndarray
    adsorption_coordination: np.ndarray
    anchor_distance_A: np.ndarray
    oh_bonds: list[dict[str, Any]]
    surface_groups: list[dict[str, Any]]
    warnings: list[str]

    def as_training_arrays(self) -> dict[str, np.ndarray]:
        return {
            "surface_side": self.surface_side.astype(np.int64),
            "layer_index": self.layer_index.astype(np.int64),
            "group_index": self.group_index.astype(np.int64),
            "anchor_index": self.anchor_index.astype(np.int64),
            "adsorption_coordination": self.adsorption_coordination.astype(np.int64),
            "anchor_distance_A": self.anchor_distance_A.astype(np.float32),
            "z_cartesian_A": self.z_cartesian_A.astype(np.float32),
        }


# 中文：顶层接口 `analyze_surface_geometry`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `analyze_surface_geometry`; review type hints and callers before extending it.
def analyze_surface_geometry(
    atomic_numbers: Sequence[int] | np.ndarray,
    frac_pos: np.ndarray,
    lattice: np.ndarray,
    *,
    layer_tolerance_A: float = 0.25,
    metal_envelope_margin_A: float = 0.20,
    oh_max_distance_A: float = 1.25,
    adsorption_shell_tolerance_A: float = 0.35,
) -> SurfaceGeometry:
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    lattice = np.asarray(lattice, dtype=np.float64)
    centered = center_slab_fractional(np.asarray(frac_pos, dtype=np.float64)).astype(
        np.float64
    )
    if len(atomic_numbers) != len(centered):
        raise ValueError("atomic number and coordinate counts differ")
    if not len(centered):
        raise ValueError("empty structure")

    wrapped_z = (centered[:, 2] - 0.5 + 0.5) % 1.0 - 0.5
    z_cartesian = wrapped_z * float(np.linalg.norm(lattice[2]))
    layer_index = _cluster_layers(z_cartesian, float(layer_tolerance_A))
    metal_mask = np.asarray(
        [_is_surface_metal(value) for value in atomic_numbers], dtype=bool
    )
    metal_indices = np.where(metal_mask)[0]
    surface_side = np.zeros(len(centered), dtype=np.int64)
    warnings: list[str] = []

    if len(metal_indices) >= 2:
        lower_metal = float(z_cartesian[metal_indices].min())
        upper_metal = float(z_cartesian[metal_indices].max())
        surface_side[z_cartesian < lower_metal - metal_envelope_margin_A] = BOTTOM
        surface_side[z_cartesian > upper_metal + metal_envelope_margin_A] = TOP
    else:
        warnings.append("missing_transition_metal_core")
        bottom_layer = int(layer_index.min())
        top_layer = int(layer_index.max())
        surface_side[layer_index == bottom_layer] = BOTTOM
        surface_side[layer_index == top_layer] = TOP

    if not np.any(surface_side == BOTTOM):
        warnings.append("missing_bottom_termination")
    if not np.any(surface_side == TOP):
        warnings.append("missing_top_termination")
    if len(np.unique(layer_index)) < 3:
        warnings.append("fewer_than_three_layers")

    group_index = np.full(len(centered), -1, dtype=np.int64)
    anchor_index = np.full(len(centered), -1, dtype=np.int64)
    adsorption_coordination = np.zeros(len(centered), dtype=np.int64)
    anchor_distance = np.full(len(centered), np.nan, dtype=np.float64)
    oh_bonds: list[dict[str, Any]] = []
    group_counter = 0

    hydrogen = int(Element("H").Z)
    oxygen = int(Element("O").Z)
    for side in (BOTTOM, TOP):
        side_indices = np.where(surface_side == side)[0]
        side_hydrogen = [
            int(index)
            for index in side_indices
            if int(atomic_numbers[index]) == hydrogen
        ]
        side_oxygen = [
            int(index)
            for index in side_indices
            if int(atomic_numbers[index]) == oxygen
        ]
        available_oxygen = set(side_oxygen)
        for h_index in side_hydrogen:
            possible = sorted(
                (
                    minimum_xy_image_distance(
                        centered[h_index], centered[o_index], lattice
                    ),
                    o_index,
                )
                for o_index in available_oxygen
            )
            if possible and possible[0][0] <= oh_max_distance_A:
                distance, o_index = possible[0]
                group_index[[h_index, o_index]] = group_counter
                available_oxygen.remove(o_index)
                oh_bonds.append(
                    {
                        "group_index": group_counter,
                        "side": ROLE_NAMES[side],
                        "hydrogen_index": h_index,
                        "oxygen_index": o_index,
                        "distance_A": float(distance),
                    }
                )
                group_counter += 1
            else:
                warnings.append(f"orphan_surface_hydrogen_{ROLE_NAMES[side]}")

    for atom_index in np.where(surface_side != CORE)[0]:
        if group_index[atom_index] < 0:
            group_index[atom_index] = group_counter
            group_counter += 1

    surface_groups: list[dict[str, Any]] = []
    for local_group in sorted(set(int(value) for value in group_index if value >= 0)):
        members = np.where(group_index == local_group)[0]
        side_values = surface_side[members]
        side = int(side_values[0])
        heavy = [
            int(index)
            for index in members
            if int(atomic_numbers[index]) != hydrogen
        ]
        leader = heavy[0] if heavy else int(members[0])
        same_side_metals = metal_indices[
            np.sign(z_cartesian[metal_indices]) == np.sign(side)
        ]
        anchor_pool = same_side_metals if len(same_side_metals) else metal_indices
        coordination, nearest = _surface_metal_coordination(
            leader,
            anchor_pool,
            centered,
            lattice,
            shell_tolerance_A=adsorption_shell_tolerance_A,
        )
        if len(anchor_pool):
            distances = [
                minimum_xy_image_distance(
                    centered[leader], centered[int(index)], lattice
                )
                for index in anchor_pool
            ]
            best_local = int(np.argmin(distances))
            anchor = int(anchor_pool[best_local])
            anchor_index[members] = anchor
            adsorption_coordination[members] = coordination
            anchor_distance[members] = nearest
            radius_ratio = nearest / max(
                _atomic_radius(int(atomic_numbers[leader]))
                + _atomic_radius(int(atomic_numbers[anchor])),
                1e-6,
            )
            if radius_ratio < 0.65 or radius_ratio > 1.60:
                warnings.append("surface_anchor_radius_ratio_outlier")
        else:
            anchor = -1
            warnings.append("surface_group_without_metal_anchor")
        surface_groups.append(
            {
                "group_index": local_group,
                "side": ROLE_NAMES[side],
                "member_indices": [int(index) for index in members],
                "formula": _formula(members, atomic_numbers),
                "leader_index": leader,
                "leader_symbol": Element.from_Z(int(atomic_numbers[leader])).symbol,
                "anchor_index": anchor,
                "anchor_symbol": (
                    Element.from_Z(int(atomic_numbers[anchor])).symbol
                    if anchor >= 0
                    else ""
                ),
                "anchor_distance_A": float(nearest),
                "adsorption_coordination": coordination,
            }
        )

    return SurfaceGeometry(
        centered_frac=centered.astype(np.float32),
        z_cartesian_A=z_cartesian.astype(np.float32),
        layer_index=layer_index,
        surface_side=surface_side,
        group_index=group_index,
        anchor_index=anchor_index,
        adsorption_coordination=adsorption_coordination,
        anchor_distance_A=anchor_distance.astype(np.float32),
        oh_bonds=oh_bonds,
        surface_groups=surface_groups,
        warnings=sorted(set(warnings)),
    )


# 中文：顶层接口 `structure_from_arrays`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `structure_from_arrays`; review type hints and callers before extending it.
def structure_from_arrays(
    atomic_numbers: Sequence[int] | np.ndarray,
    frac_pos: np.ndarray,
    lattice: np.ndarray,
) -> Structure:
    return Structure(
        Lattice(np.asarray(lattice, dtype=np.float64)),
        [Element.from_Z(int(value)) for value in atomic_numbers],
        np.asarray(frac_pos, dtype=np.float64),
        coords_are_cartesian=False,
    )


# 中文：顶层接口 `validate_surface_topology`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `validate_surface_topology`; review type hints and callers before extending it.
def validate_surface_topology(
    structure: Structure,
    reference_profile: dict[str, Any],
    *,
    bond_quantile_margin_A: float = 0.08,
    oh_margin_A: float = 0.02,
) -> tuple[bool, dict[str, Any]]:
    atomic_numbers = np.asarray(
        [int(site.specie.Z) for site in structure], dtype=np.int64
    )
    analysis = analyze_surface_geometry(
        atomic_numbers,
        np.asarray(structure.frac_coords),
        np.asarray(structure.lattice.matrix),
    )
    reasons = list(analysis.warnings)
    layer_count = int(len(np.unique(analysis.layer_index)))
    allowed_layers = {
        int(value)
        for value in reference_profile.get("layer_count_distribution", {})
    }
    if allowed_layers and layer_count not in allowed_layers:
        reasons.append("unseen_layer_count")
    if len(analysis.surface_groups) != 2:
        reasons.append("surface_group_count_not_two")

    allowed_motifs = set(
        reference_profile.get("termination_motif_top50", {}).keys()
    )
    bottom = [
        group["formula"]
        for group in analysis.surface_groups
        if group["side"] == "bottom"
    ]
    top = [
        group["formula"]
        for group in analysis.surface_groups
        if group["side"] == "top"
    ]
    motif = (
        f"{bottom[0]}|{top[0]}"
        if len(bottom) == 1 and len(top) == 1
        else ""
    )
    if allowed_motifs and motif not in allowed_motifs:
        reasons.append("unseen_termination_motif")

    bad_coordination = [
        group
        for group in analysis.surface_groups
        if int(group["adsorption_coordination"]) != 3
    ]
    if bad_coordination:
        reasons.append("termination_not_threefold_hollow")

    oh_profile = reference_profile.get("oh_bond_length_A", {})
    oh_lower = float(oh_profile.get("q01", 0.94)) - oh_margin_A
    oh_upper = float(oh_profile.get("q99", 1.02)) + oh_margin_A
    oh_lengths = [float(item["distance_A"]) for item in analysis.oh_bonds]
    if any(value < oh_lower or value > oh_upper for value in oh_lengths):
        reasons.append("oh_bond_outside_training_distribution")

    pair_profiles = reference_profile.get("anchor_distance_A_by_pair", {})
    anchor_details: list[dict[str, Any]] = []
    for group in analysis.surface_groups:
        pair = f"{group['leader_symbol']}-{group['anchor_symbol']}"
        distance = float(group["anchor_distance_A"])
        profile = pair_profiles.get(pair)
        inside = True
        if profile and np.isfinite(distance):
            lower = float(profile.get("q01", -np.inf)) - bond_quantile_margin_A
            upper = float(profile.get("q99", np.inf)) + bond_quantile_margin_A
            inside = lower <= distance <= upper
            if not inside:
                reasons.append(f"anchor_distance_outlier_{pair}")
        anchor_details.append(
            {
                "pair": pair,
                "distance_A": distance,
                "inside_training_range": inside,
                "adsorption_coordination": int(
                    group["adsorption_coordination"]
                ),
            }
        )

    unique_reasons = sorted(set(reasons))
    return not unique_reasons, {
        "Surface_Topology_Valid": not unique_reasons,
        "Surface_Topology_Reasons": "|".join(unique_reasons),
        "Surface_Group_Count": int(len(analysis.surface_groups)),
        "Termination_Motif": motif,
        "Surface_Layer_Count": layer_count,
        "OH_Bond_Lengths_A": "|".join(f"{value:.6f}" for value in oh_lengths),
        "Surface_Anchor_Details": anchor_details,
    }
