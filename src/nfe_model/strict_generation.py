# ==============================================================================
# 中文概述：surface generator 严格生成流水线：模板采样、中心化、CHGNet、拓扑/OOD/重复过滤和 NFE 复评。
# English overview: Strict surface generator pipeline: template sampling, centering, CHGNet, topology/OOD/duplicate filters, and NFE rescoring.
#
# 中文输入：目标档位、模板库、两个检查点、几何先验和生成参数。
# English inputs: Target class, template catalog, two checkpoints, geometry priors, and generation parameters.
# 中文输出：目标匹配且几何合理的 CIF、审计表、运行信息和拒绝统计。
# English outputs: Target-matched physically plausible CIFs, audit tables, run info, and rejection statistics.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: load_generator, lattice_from_state, guided_velocity, template_generation_batch, sample_structures, atomic_radius, minimum_image_delta, distance_ratio_statistics, layer_count, repair_close_contacts, center_structure, validate_structure, create_chgnet_relaxer, relax_with_chgnet, template_entry_from_structure, choose_templates, unique_output_directory, rank_candidates, prediction_matches_target, pure_structure_match, safe_structure_match, reference_structures, parse_args, main
#
# Author: Ruck
# Generated: 2026-07-29 22:47:20 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element, Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter

from .data import (
    build_periodic_graph,
    element_features,
    slab_fractions,
    torch_load_compat,
)
from .generator_data import (
    center_slab_fractional,
    center_slab_fractional_tensor,
    composition_formula,
    composition_key,
    lattice_to_params,
    params_to_lattice,
    parse_composition,
    slab_center_fractional_z,
)
from .surface_generator_data import (
    GROUP_ATOMIC_TERMINATION,
    GROUP_CORE,
    GROUP_OH_HYDROGEN,
    GROUP_OH_OXYGEN,
    GROUP_SURFACE_HYDROGEN,
)
from .surface_generator import (
    SurfaceAwareTemplateFlow,
    surface_coordinate_length_scale,
)
from .predict import infer_chunk, load_checkpoint_model
from .surface_geometry import analyze_surface_geometry, validate_surface_topology


# 中文：桌面程序可注册进度回调；命令行训练/生成不注册时保持原有行为。
# English: The desktop app may register a progress callback; CLI use is unchanged.
ProgressCallback = Callable[[str, Optional[float]], None]
_PROGRESS_CALLBACK: ProgressCallback | None = None


# 中文：替换进度回调并返回旧值，便于调用方在 finally 中恢复全局状态。
# English: Replace the callback and return the old value for reliable restoration.
def set_progress_callback(
    callback: ProgressCallback | None,
) -> ProgressCallback | None:
    global _PROGRESS_CALLBACK
    previous = _PROGRESS_CALLBACK
    _PROGRESS_CALLBACK = callback
    return previous


# 中文：向 GUI 报告当前阶段和 0–100 百分比；无 GUI 时此函数为空操作。
# English: Report a stage and 0–100 percentage to the GUI; otherwise do nothing.
def report_progress(message: str, percent: float | None = None) -> None:
    if _PROGRESS_CALLBACK is not None:
        _PROGRESS_CALLBACK(message, percent)


# 中文：顶层接口 `load_generator`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `load_generator`; review type hints and callers before extending it.
def load_generator(
    path: str | Path, device: torch.device
) -> tuple[SurfaceAwareTemplateFlow, dict[str, Any]]:
    checkpoint = torch_load_compat(path, map_location="cpu")
    if checkpoint.get("format") != "nfe-mxene-surface-generator-1.0":
        raise ValueError(f"unsupported generator checkpoint: {path}")
    model = SurfaceAwareTemplateFlow(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


# 中文：顶层接口 `lattice_from_state`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `lattice_from_state`; review type hints and callers before extending it.
def lattice_from_state(
    state: torch.Tensor, normalizers: dict[str, torch.Tensor]
) -> torch.Tensor:
    physical = (
        state * normalizers["lattice_scale"]
        + normalizers["lattice_median"]
    )
    return params_to_lattice(physical)


# 中文：顶层接口 `guided_velocity`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `guided_velocity`; review type hints and callers before extending it.
@torch.no_grad()
def guided_velocity(
    model: SurfaceAwareTemplateFlow,
    batch: dict[str, torch.Tensor],
    frac_pos: torch.Tensor,
    lattice_state: torch.Tensor,
    lattice: torch.Tensor,
    time: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    conditional = model(
        batch,
        frac_pos,
        lattice_state,
        lattice,
        time,
        labels,
        scores,
    )
    if guidance_scale == 1.0:
        return (
            conditional["coordinate_velocity_cart"],
            conditional["lattice_velocity"],
        )
    unconditional = model(
        batch,
        frac_pos,
        lattice_state,
        lattice,
        time,
        torch.full_like(labels, -1),
        torch.zeros_like(scores),
    )
    coordinate = unconditional["coordinate_velocity_cart"] + guidance_scale * (
        conditional["coordinate_velocity_cart"]
        - unconditional["coordinate_velocity_cart"]
    )
    lattice_velocity = unconditional["lattice_velocity"] + guidance_scale * (
        conditional["lattice_velocity"]
        - unconditional["lattice_velocity"]
    )
    return coordinate, lattice_velocity


# 中文：顶层接口 `template_generation_batch`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `template_generation_batch`; review type hints and callers before extending it.
def template_generation_batch(
    templates: Sequence[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    z_parts, feature_parts, batch_parts = [], [], []
    frac_parts, side_parts, layer_parts, group_parts, coordination_parts = (
        [],
        [],
        [],
        [],
        [],
    )
    for graph_index, template in enumerate(templates):
        z = torch.tensor(template["z"], dtype=torch.long)
        z_parts.append(z)
        feature_parts.append(
            torch.tensor(
                [element_features(int(value)) for value in z],
                dtype=torch.float32,
            )
        )
        batch_parts.append(
            torch.full((len(z),), graph_index, dtype=torch.long)
        )
        frac_parts.append(torch.tensor(template["frac_pos"], dtype=torch.float32))
        side_parts.append(torch.tensor(template["surface_side"], dtype=torch.long))
        layer_parts.append(
            torch.tensor(template["layer_position"], dtype=torch.float32)
        )
        group_parts.append(torch.tensor(template["group_type"], dtype=torch.long))
        coordination_parts.append(
            torch.tensor(
                template["adsorption_coordination"], dtype=torch.long
            )
        )
    return {
        "z": torch.cat(z_parts).to(device),
        "atom_features": torch.cat(feature_parts).to(device),
        "batch": torch.cat(batch_parts).to(device),
        "template_frac": torch.cat(frac_parts).to(device),
        "surface_side": torch.cat(side_parts).to(device),
        "layer_position": torch.cat(layer_parts).to(device),
        "group_type": torch.cat(group_parts).to(device),
        "adsorption_coordination": torch.cat(coordination_parts).to(device),
    }


# 中文：顶层接口 `sample_structures`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `sample_structures`; review type hints and callers before extending it.
@torch.no_grad()
def sample_structures(
    model: SurfaceAwareTemplateFlow,
    checkpoint: dict[str, Any],
    templates: Sequence[dict[str, Any]],
    *,
    target_label: int,
    target_score: float,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> list[Structure]:
    batch = template_generation_batch(templates, device)
    n_graphs = len(templates)
    normalizers = {
        key: value.to(device) for key, value in checkpoint["normalizers"].items()
    }
    loss_config = checkpoint["config"]["generator_loss"]
    template_lattice_params = torch.stack(
        [
            lattice_to_params(
                torch.tensor(item["lattice"], dtype=torch.float32)
            )
            for item in templates
        ]
    ).to(device)
    template_lattice_state = (
        template_lattice_params - normalizers["lattice_median"]
    ) / normalizers["lattice_scale"]
    lattice_mask = torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        device=device,
        dtype=template_lattice_state.dtype,
    )
    lattice_state = (
        template_lattice_state
        + torch.randn_like(template_lattice_state)
        * float(loss_config.get("lattice_template_noise", 0.10))
        * lattice_mask
    ).clamp(-8.0, 8.0)
    physical_template_lattice = params_to_lattice(template_lattice_params)
    sigma = torch.full(
        (len(batch["z"]),),
        float(loss_config.get("core_template_noise_A", 0.18)),
        device=device,
    )
    sigma = torch.where(
        batch["surface_side"] != 0,
        torch.full_like(
            sigma, float(loss_config.get("surface_template_noise_A", 0.35))
        ),
        sigma,
    )
    sigma = torch.where(
        batch["group_type"] == GROUP_OH_HYDROGEN,
        torch.full_like(
            sigma, float(loss_config.get("hydrogen_template_noise_A", 0.16))
        ),
        sigma,
    )
    cartesian_noise = torch.randn(
        (len(batch["z"]), 3), device=device
    ) * sigma.unsqueeze(-1)
    fractional_noise = torch.einsum(
        "ni,nij->nj",
        cartesian_noise,
        torch.linalg.inv(physical_template_lattice)[batch["batch"]],
    )
    frac_pos = batch["template_frac"] + fractional_noise
    frac_pos[:, :2] = torch.remainder(frac_pos[:, :2], 1.0)
    frac_pos[:, 2] = frac_pos[:, 2].clamp(0.05, 0.95)
    frac_pos = center_slab_fractional_tensor(frac_pos, batch["batch"])
    labels = torch.full((n_graphs,), target_label, dtype=torch.long, device=device)
    scores = torch.full((n_graphs,), target_score, device=device)
    step_size = 1.0 / max(1, steps)
    for step_index in range(steps):
        time_value = step_index / steps
        next_time_value = (step_index + 1) / steps
        time = torch.full((n_graphs,), time_value, device=device)
        lattice = lattice_from_state(lattice_state, normalizers)
        cart_velocity_1, lattice_velocity_1 = guided_velocity(
            model,
            batch,
            frac_pos,
            lattice_state,
            lattice,
            time,
            labels,
            scores,
            guidance_scale,
        )
        cart_velocity_1 = cart_velocity_1 * surface_coordinate_length_scale(
            lattice, batch["batch"]
        )[batch["batch"]].unsqueeze(-1)
        frac_velocity_1 = torch.einsum(
            "ni,nij->nj",
            cart_velocity_1,
            torch.linalg.inv(lattice)[batch["batch"]],
        )
        proposed_frac = center_slab_fractional_tensor(
            torch.remainder(frac_pos + step_size * frac_velocity_1, 1.0),
            batch["batch"],
        )
        proposed_state = (
            lattice_state
            + step_size * lattice_velocity_1 * lattice_mask
        ).clamp(-8.0, 8.0)

        if step_index + 1 < steps:
            next_time = torch.full((n_graphs,), next_time_value, device=device)
            proposed_lattice = lattice_from_state(proposed_state, normalizers)
            cart_velocity_2, lattice_velocity_2 = guided_velocity(
                model,
                batch,
                proposed_frac,
                proposed_state,
                proposed_lattice,
                next_time,
                labels,
                scores,
                guidance_scale,
            )
            cart_velocity_2 = cart_velocity_2 * surface_coordinate_length_scale(
                proposed_lattice, batch["batch"]
            )[batch["batch"]].unsqueeze(-1)
            frac_velocity_2 = torch.einsum(
                "ni,nij->nj",
                cart_velocity_2,
                torch.linalg.inv(proposed_lattice)[batch["batch"]],
            )
            frac_pos = center_slab_fractional_tensor(
                torch.remainder(
                    frac_pos
                    + 0.5 * step_size * (frac_velocity_1 + frac_velocity_2),
                    1.0,
                ),
                batch["batch"],
            )
            lattice_state = (
                lattice_state
                + 0.5
                * step_size
                * (lattice_velocity_1 + lattice_velocity_2)
                * lattice_mask
            ).clamp(-8.0, 8.0)
        else:
            frac_pos = proposed_frac
            lattice_state = proposed_state

    lattices = lattice_from_state(lattice_state, normalizers).cpu().numpy()
    frac_pos_numpy = frac_pos.cpu().numpy()
    batch_numpy = batch["batch"].cpu().numpy()
    structures = []
    for graph_index, template in enumerate(templates):
        local = center_slab_fractional(
            frac_pos_numpy[batch_numpy == graph_index]
        )
        structures.append(
            Structure(
                Lattice(lattices[graph_index]),
                [Element.from_Z(int(z)) for z in template["z"]],
                local,
                coords_are_cartesian=False,
                to_unit_cell=True,
            )
        )
    return structures


# 中文：顶层接口 `atomic_radius`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `atomic_radius`; review type hints and callers before extending it.
def atomic_radius(atomic_number: int) -> float:
    element = Element.from_Z(int(atomic_number))
    value = element.atomic_radius or element.average_ionic_radius
    try:
        return max(float(value), 0.6)
    except (TypeError, ValueError):
        return 1.2


# 中文：顶层接口 `minimum_image_delta`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `minimum_image_delta`; review type hints and callers before extending it.
def minimum_image_delta(
    delta_fractional: np.ndarray, lattice: np.ndarray
) -> tuple[np.ndarray, float]:
    shifts = np.asarray(
        [
            [i, j, k]
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
            for k in (-1.0, 0.0, 1.0)
        ]
    )
    candidates = (delta_fractional[None, :] + shifts) @ lattice
    distances = np.linalg.norm(candidates, axis=1)
    best = int(np.argmin(distances))
    return candidates[best], float(distances[best])


# 中文：顶层接口 `distance_ratio_statistics`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `distance_ratio_statistics`; review type hints and callers before extending it.
def distance_ratio_statistics(structure: Structure) -> tuple[float, float]:
    if len(structure) <= 1:
        return math.inf, math.inf
    frac = np.asarray(structure.frac_coords)
    lattice = np.asarray(structure.lattice.matrix)
    radii = np.asarray([atomic_radius(site.specie.Z) for site in structure])
    best = math.inf
    nearest = np.full(len(structure), math.inf)
    for left in range(len(structure)):
        for right in range(left + 1, len(structure)):
            delta = frac[left] - frac[right]
            _, distance = minimum_image_delta(delta, lattice)
            ratio = distance / max(radii[left] + radii[right], 1e-6)
            best = min(best, ratio)
            nearest[left] = min(nearest[left], ratio)
            nearest[right] = min(nearest[right], ratio)
    return best, float(np.max(nearest))


# 中文：顶层接口 `layer_count`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `layer_count`; review type hints and callers before extending it.
def layer_count(structure: Structure, tolerance: float = 0.035) -> int:
    z = np.sort(np.mod(np.asarray(structure.frac_coords)[:, 2], 1.0))
    if len(z) <= 1:
        return len(z)
    gaps = np.diff(np.r_[z, z[0] + 1.0])
    start = int(np.argmax(gaps) + 1) % len(z)
    unwrapped = np.sort(np.mod(z - z[start], 1.0))
    return 1 + int(np.sum(np.diff(unwrapped) > tolerance))


# 中文：顶层接口 `repair_close_contacts`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `repair_close_contacts`; review type hints and callers before extending it.
def repair_close_contacts(
    structure: Structure,
    minimum_factor: float,
    steps: int,
) -> Structure:
    frac = np.mod(np.asarray(structure.frac_coords, dtype=np.float64), 1.0)
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    inverse_lattice = np.linalg.inv(lattice)
    radii = np.asarray([atomic_radius(site.specie.Z) for site in structure])
    for iteration in range(steps):
        corrections = np.zeros_like(frac)
        violations = 0
        for left in range(len(structure)):
            for right in range(left + 1, len(structure)):
                delta_frac = frac[left] - frac[right]
                delta_cart, distance = minimum_image_delta(
                    delta_frac, lattice
                )
                minimum = minimum_factor * (radii[left] + radii[right])
                if distance + 1e-7 >= minimum:
                    continue
                violations += 1
                if distance < 1e-8:
                    angle = 2.0 * math.pi * (
                        ((left + 1) * 37 + (right + 1) * 101 + iteration) % 997
                    ) / 997.0
                    direction = np.asarray(
                        [math.cos(angle), math.sin(angle), 0.25]
                    )
                    direction /= np.linalg.norm(direction)
                else:
                    direction = delta_cart / distance
                shift_cart = 0.52 * (minimum - distance) * direction
                shift_frac = shift_cart @ inverse_lattice
                corrections[left] += shift_frac
                corrections[right] -= shift_frac
        frac = np.mod(frac + corrections, 1.0)
        if violations == 0:
            break
    return Structure(
        structure.lattice,
        [site.specie for site in structure],
        frac,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


# 中文：顶层接口 `center_structure`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `center_structure`; review type hints and callers before extending it.
def center_structure(structure: Structure) -> Structure:
    return Structure(
        structure.lattice,
        [site.specie for site in structure],
        center_slab_fractional(np.asarray(structure.frac_coords)),
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


# 中文：顶层接口 `validate_structure`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `validate_structure`; review type hints and callers before extending it.
def validate_structure(
    structure: Structure,
    minimum_factor: float,
    minimum_vacuum: float,
    maximum_slab_thickness: float = 12.0,
    maximum_nearest_ratio: float = 1.85,
    minimum_layers: int = 3,
    maximum_center_offset_A: float = 0.25,
) -> tuple[bool, dict[str, Any]]:
    lengths = np.asarray(structure.lattice.abc)
    angles = np.asarray(structure.lattice.angles)
    volume_per_atom = float(structure.volume / max(len(structure), 1))
    distance_ratio, max_nearest_ratio = distance_ratio_statistics(structure)
    slab_fraction, vacuum_fraction = slab_fractions(structure)
    vacuum_A = float(vacuum_fraction * structure.lattice.c)
    slab_thickness_A = float(slab_fraction * structure.lattice.c)
    slab_center = slab_center_fractional_z(
        np.asarray(structure.frac_coords)
    )
    center_delta = (slab_center - 0.5 + 0.5) % 1.0 - 0.5
    center_offset_A = abs(float(center_delta)) * float(structure.lattice.c)
    layers = layer_count(structure)
    reasons = []
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 1.5) or np.any(lengths > 80):
        reasons.append("invalid_lattice_length")
    if not np.all(np.isfinite(angles)) or np.any(angles < 30) or np.any(angles > 150):
        reasons.append("invalid_lattice_angle")
    if not (2.0 <= volume_per_atom <= 500.0):
        reasons.append("invalid_volume_per_atom")
    if distance_ratio < 0.95 * minimum_factor:
        reasons.append("close_contact")
    if max_nearest_ratio > maximum_nearest_ratio:
        reasons.append("isolated_atom")
    if vacuum_A < minimum_vacuum:
        reasons.append("insufficient_vacuum")
    if not (0.8 <= slab_thickness_A <= maximum_slab_thickness):
        reasons.append("invalid_slab_thickness")
    if layers < minimum_layers:
        reasons.append("too_few_atomic_layers")
    if center_offset_A > maximum_center_offset_A:
        reasons.append("off_center_slab")
    return not reasons, {
        "Valid_Geometry": not reasons,
        "Geometry_Reasons": "|".join(reasons),
        "Min_Distance_Radius_Ratio": distance_ratio,
        "Max_Nearest_Radius_Ratio": max_nearest_ratio,
        "Volume_per_Atom_A3": volume_per_atom,
        "Vacuum_A": vacuum_A,
        "Slab_Thickness_A": slab_thickness_A,
        "Slab_Center_Fractional_Z": slab_center,
        "Slab_Center_Offset_A": center_offset_A,
        "Atomic_Layer_Count": layers,
        "Lattice_a_A": float(lengths[0]),
        "Lattice_b_A": float(lengths[1]),
        "Lattice_c_A": float(lengths[2]),
    }


# 中文：顶层接口 `create_chgnet_relaxer`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `create_chgnet_relaxer`; review type hints and callers before extending it.
def create_chgnet_relaxer(device: torch.device):
    try:
        from chgnet.model import CHGNet, StructOptimizer
    except ImportError as exc:
        raise RuntimeError(
            "CHGNet is not installed; run `python -m pip install -r "
            "requirements-relax.txt` or use --relaxer none"
        ) from exc
    if device.type == "cuda":
        device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        use_device = f"cuda:{device_index}"
    else:
        use_device = "cpu"
    # CHGNet 0.3.x otherwise queries every physical GPU through NVML and can
    # select an ordinal hidden by CUDA_VISIBLE_DEVICES. Load and bind the model
    # explicitly to the logical PyTorch device used by this process.
    model = CHGNet.load(
        use_device=use_device,
        check_cuda_mem=False,
        verbose=False,
    )
    return StructOptimizer(model=model, use_device=use_device)


# 中文：顶层接口 `relax_with_chgnet`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `relax_with_chgnet`; review type hints and callers before extending it.
def relax_with_chgnet(
    structure: Structure,
    relaxer: Any,
    *,
    fmax: float,
    steps: int,
) -> tuple[Structure, float, float]:
    result = relaxer.relax(
        structure,
        fmax=fmax,
        steps=steps,
        relax_cell=False,
        verbose=False,
    )
    energy = float(result["trajectory"].energies[-1])
    try:
        forces = np.asarray(result["trajectory"].forces[-1], dtype=np.float64)
        maximum_force = float(np.linalg.norm(forces, axis=1).max())
    except (AttributeError, IndexError, TypeError, ValueError):
        maximum_force = math.nan
    return result["final_structure"], energy, maximum_force


# 中文：顶层接口 `template_entry_from_structure`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `template_entry_from_structure`; review type hints and callers before extending it.
def template_entry_from_structure(
    structure: Structure,
    identifier: str,
) -> dict[str, Any]:
    z = np.asarray([int(site.specie.Z) for site in structure], dtype=np.int64)
    analysis = analyze_surface_geometry(
        z,
        np.asarray(structure.frac_coords),
        np.asarray(structure.lattice.matrix),
    )
    group_type = np.full(len(z), GROUP_CORE, dtype=np.int64)
    group_type[analysis.surface_side != 0] = GROUP_ATOMIC_TERMINATION
    for bond in analysis.oh_bonds:
        group_type[int(bond["oxygen_index"])] = GROUP_OH_OXYGEN
        group_type[int(bond["hydrogen_index"])] = GROUP_OH_HYDROGEN
    for index, atomic_number in enumerate(z):
        if (
            analysis.surface_side[index] != 0
            and int(atomic_number) == 1
            and group_type[index] != GROUP_OH_HYDROGEN
        ):
            group_type[index] = GROUP_SURFACE_HYDROGEN
    order = np.argsort(analysis.z_cartesian_A)
    layers = analysis.layer_index[order].astype(np.float32)
    center = 0.5 * (float(layers.min()) + float(layers.max()))
    scale = max(0.5 * float(layers.max() - layers.min()), 1.0)
    return {
        "id": identifier,
        "z": [int(value) for value in z[order]],
        "formula": composition_formula(z.tolist()),
        "frac_pos": analysis.centered_frac[order].tolist(),
        "lattice": np.asarray(structure.lattice.matrix).tolist(),
        "surface_side": analysis.surface_side[order].tolist(),
        "layer_position": ((layers - center) / scale).tolist(),
        "group_type": group_type[order].tolist(),
        "adsorption_coordination": analysis.adsorption_coordination[
            order
        ].tolist(),
        "termination_motif": "|".join(
            group["formula"] for group in analysis.surface_groups
        ),
        "label": -1,
        "score": 0.5,
    }


# 中文：顶层接口 `choose_templates`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `choose_templates`; review type hints and callers before extending it.
def choose_templates(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    total: int,
) -> list[dict[str, Any]]:
    if args.template:
        choices = [
            template_entry_from_structure(
                Structure.from_file(path), identifier=str(path)
            )
            for path in args.template
        ]
        return [choices[index % len(choices)] for index in range(total)]
    catalog = checkpoint.get("surface_template_catalog", [])
    if not catalog:
        raise ValueError(
            "surface generator checkpoint has no surface template catalog"
        )
    if args.composition:
        atomic_numbers = sorted(parse_composition(args.composition))
        catalog = [
            item
            for item in catalog
            if sorted(int(value) for value in item["z"]) == atomic_numbers
        ]
        if not catalog:
            raise ValueError(
                "composition is not present in the safe template catalog; "
                "provide --template for a new composition"
            )
    target = float(args.target_score)
    weights = []
    for item in catalog:
        closeness = math.exp(-abs(float(item["score"]) - target) / 0.16)
        class_bonus = 3.0 if int(item.get("label", -1)) == {
            "low": 0,
            "medium": 1,
            "high": 2,
        }[args.target] else 1.0
        weights.append(closeness * class_bonus)
    selected = random.choices(catalog, weights=weights, k=total)
    return [dict(item) for item in selected]


# 中文：顶层接口 `unique_output_directory`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `unique_output_directory`; review type hints and callers before extending it.
def unique_output_directory(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True)
        return base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"run_{timestamp}"
    counter = 1
    while candidate.exists():
        candidate = base / f"run_{timestamp}_{counter:02d}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


# 中文：顶层接口 `rank_candidates`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `rank_candidates`; review type hints and callers before extending it.
def rank_candidates(
    candidates: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    target: str,
    target_score: float,
) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    risk_penalty = {
        "low": 0.0,
        "medium": 0.12,
        "high": 0.35,
        "unknown": 0.18,
    }
    probability_column = {
        "low": "Probability_Low",
        "medium": "Probability_Medium",
        "high": "Probability_High",
    }[target]
    ranked = []
    for candidate, prediction in zip(candidates, predictions):
        score_agreement = 1.0 - abs(
            float(prediction["Predicted_NFE_Score"]) - target_score
        )
        ranking_score = (
            score_agreement
            + 0.35 * float(prediction[probability_column])
            - 0.35 * float(prediction["NFE_Score_Std"])
            - risk_penalty.get(str(prediction["OOD_Risk"]), 0.2)
        )
        ranked.append((ranking_score, candidate, prediction))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


# 中文：顶层接口 `prediction_matches_target`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `prediction_matches_target`; review type hints and callers before extending it.
def prediction_matches_target(
    prediction: dict[str, Any],
    target: str,
    minimum_probability: float,
) -> tuple[bool, float]:
    probability_column = {
        "low": "Probability_Low",
        "medium": "Probability_Medium",
        "high": "Probability_High",
    }[target]
    probability = float(prediction[probability_column])
    matches = bool(
        prediction["Predicted_NFE_Label"] == target
        and probability >= minimum_probability
    )
    return matches, probability


# 中文：顶层接口 `pure_structure_match`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `pure_structure_match`; review type hints and callers before extending it.
def pure_structure_match(
    structure: Structure,
    reference: Structure,
    *,
    lattice_tolerance: float,
    site_tolerance: float,
    angle_tolerance: float,
) -> bool:
    if sorted(site.specie.Z for site in structure) != sorted(
        site.specie.Z for site in reference
    ):
        return False
    scale = (structure.volume / reference.volume) ** (1.0 / 3.0)
    reference_lattice = np.asarray(reference.lattice.matrix) * scale
    structure_lengths = np.asarray(structure.lattice.abc)
    reference_lengths = np.linalg.norm(reference_lattice, axis=1)
    relative = np.abs(structure_lengths - reference_lengths) / np.maximum(
        structure_lengths, reference_lengths
    )
    if np.any(relative > lattice_tolerance):
        return False
    if np.any(
        np.abs(
            np.asarray(structure.lattice.angles)
            - np.asarray(Lattice(reference_lattice).angles)
        )
        > angle_tolerance
    ):
        return False

    z_structure = np.asarray([site.specie.Z for site in structure])
    z_reference = np.asarray([site.specie.Z for site in reference])
    frac_structure = np.asarray(structure.frac_coords)
    frac_reference = np.asarray(reference.frac_coords)
    unique, counts = np.unique(z_structure, return_counts=True)
    anchor_species = unique[int(np.argmin(counts))]
    structure_anchor = np.where(z_structure == anchor_species)[0][0]
    reference_anchors = np.where(z_reference == anchor_species)[0]
    distance_limit = site_tolerance * (
        structure.volume / max(len(structure), 1)
    ) ** (1.0 / 3.0)
    lattice = np.asarray(structure.lattice.matrix)
    for reference_anchor in reference_anchors:
        translation = (
            frac_reference[reference_anchor]
            - frac_structure[structure_anchor]
        )
        shifted = np.mod(frac_structure + translation, 1.0)
        valid_translation = True
        for atomic_number in unique:
            left = np.where(z_structure == atomic_number)[0]
            right = np.where(z_reference == atomic_number)[0]
            count = len(left)
            cost = np.zeros((count, count), dtype=np.float64)
            for left_index, left_atom in enumerate(left):
                for right_index, right_atom in enumerate(right):
                    _, cost[left_index, right_index] = minimum_image_delta(
                        shifted[left_atom] - frac_reference[right_atom],
                        lattice,
                    )
            if count <= 7:
                best_max = min(
                    max(cost[row, column] for row, column in enumerate(permutation))
                    for permutation in itertools.permutations(range(count))
                )
            else:
                available = set(range(count))
                assigned = []
                for row in range(count):
                    column = min(available, key=lambda value: cost[row, value])
                    available.remove(column)
                    assigned.append(cost[row, column])
                best_max = max(assigned)
            if best_max > distance_limit:
                valid_translation = False
                break
        if valid_translation:
            return True
    return False


# 中文：顶层接口 `safe_structure_match`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `safe_structure_match`; review type hints and callers before extending it.
def safe_structure_match(
    matcher: StructureMatcher,
    structure: Structure,
    reference: Structure,
) -> bool:
    try:
        return bool(
            matcher.fit(
                structure,
                reference,
                skip_structure_reduction=True,
            )
        )
    except TypeError:
        try:
            return bool(matcher.fit(structure, reference))
        except Exception:
            pass
    except Exception:
        pass
    return pure_structure_match(
        structure,
        reference,
        lattice_tolerance=float(matcher.ltol),
        site_tolerance=float(matcher.stol),
        angle_tolerance=float(matcher.angle_tol),
    )


# 中文：顶层接口 `reference_structures`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `reference_structures`; review type hints and callers before extending it.
def reference_structures(
    checkpoint: dict[str, Any], structure: Structure
) -> list[Structure]:
    z_values = sorted(int(site.specie.Z) for site in structure)
    entries = checkpoint.get("novelty_reference", {}).get(
        composition_key(z_values), []
    )
    return [
        Structure(
            Lattice(np.asarray(entry["lattice"], dtype=np.float64)),
            [Element.from_Z(int(z)) for z in entry["z"]],
            np.asarray(entry["frac_pos"], dtype=np.float64),
            coords_are_cartesian=False,
            to_unit_cell=True,
        )
        for entry in entries
    ]


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate physically screened CIF structures and predict NFE."
    )
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument(
        "--predictor-checkpoint", action="append", required=True
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--composition", help="formula such as Ti2CO2")
    source.add_argument("--template", action="append", help="POSCAR/CIF; species are reused")
    parser.add_argument("--target", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--target-score", type=float)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--oversample", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument(
        "--min-target-probability",
        type=float,
        default=0.50,
        help="minimum predictor probability required for the requested class",
    )
    parser.add_argument(
        "--allow-target-mismatch",
        action="store_true",
        help="export top-ranked candidates even if the predictor disagrees",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mc-samples", type=int, default=12)
    parser.add_argument("--relaxer", choices=("none", "chgnet"), default="none")
    parser.add_argument("--relax-fmax", type=float, default=0.08)
    parser.add_argument("--relax-steps", type=int, default=250)
    parser.add_argument(
        "--relax-pool-size",
        type=int,
        help=(
            "number of predictor-ranked candidates sent to the expensive relaxer; "
            "defaults to max(2*num, num+10)"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow-training-match",
        action="store_true",
        help="allow candidates matching a training structure",
    )
    parser.add_argument("--output", default="generated_cifs")
    return parser.parse_args(argv)


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_progress("正在检查生成参数…", 1.0)
    if args.num <= 0:
        raise ValueError("--num must be positive")
    if args.target_score is None:
        args.target_score = {
            "low": 0.25,
            "medium": 0.58,
            "high": 0.85,
        }[args.target]
    if not 0.0 <= args.target_score <= 1.0:
        raise ValueError("--target-score must be between 0 and 1")
    if not 0.0 <= args.min_target_probability <= 1.0:
        raise ValueError("--min-target-probability must be between 0 and 1")
    if args.relax_pool_size is not None and args.relax_pool_size <= 0:
        raise ValueError("--relax-pool-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    report_progress("正在加载表面生成器…", 3.0)
    generator, generator_checkpoint = load_generator(
        args.generator_checkpoint, device
    )
    report_progress("正在加载并初始化 NFE 预测器…", 6.0)
    predictors = [
        load_checkpoint_model(path, device)
        for path in args.predictor_checkpoint
    ]
    generation_config = generator_checkpoint["config"]["generation"]
    oversample = (
        args.oversample
        if args.oversample is not None
        else int(generation_config["oversample_factor"])
    )
    steps = (
        args.steps
        if args.steps is not None
        else int(generation_config["sampling_steps"])
    )
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(generation_config["guidance_scale"])
    )
    profile_path = Path(generation_config["surface_profile"])
    if not profile_path.is_absolute():
        profile_path = (
            Path(args.generator_checkpoint).resolve().parent.parent
            / profile_path
        )
    reference_profile = json.loads(
        profile_path.read_text(encoding="utf-8")
    )
    total_candidates = args.num * max(1, oversample)
    report_progress(
        f"正在选择可信模板，共准备 {total_candidates} 个原始候选…",
        9.0,
    )
    templates = choose_templates(
        args, generator_checkpoint, total_candidates
    )
    label_index = {"low": 0, "medium": 1, "high": 2}[args.target]

    raw_structures: list[Structure] = []
    sampling_starts = list(
        range(0, total_candidates, args.batch_size)
    )
    for batch_index, start in enumerate(sampling_starts, start=1):
        raw_structures.extend(
            sample_structures(
                generator,
                generator_checkpoint,
                templates[start : start + args.batch_size],
                target_label=label_index,
                target_score=args.target_score,
                steps=steps,
                guidance_scale=guidance_scale,
                device=device,
            )
        )
        report_progress(
            f"生成模型采样 {batch_index}/{len(sampling_starts)}："
            f"已得到 {len(raw_structures)}/{total_candidates} 个结构",
            10.0 + 15.0 * batch_index / max(1, len(sampling_starts)),
        )

    minimum_factor = float(generation_config["minimum_distance_factor"])
    repair_steps = int(generation_config["repair_steps"])
    minimum_vacuum = float(generation_config["minimum_vacuum_A"])
    maximum_slab_thickness = float(
        generation_config["maximum_slab_thickness_A"]
    )
    maximum_nearest_ratio = float(
        generation_config["maximum_nearest_radius_ratio"]
    )
    minimum_layers = int(generation_config["minimum_atomic_layers"])
    candidates: list[dict[str, Any]] = []
    rejected = 0
    rejection_reasons: Counter[str] = Counter()
    for candidate_index, structure in enumerate(raw_structures):
        repaired = center_structure(
            repair_close_contacts(
                structure, minimum_factor, repair_steps
            )
        )
        geometry_valid, geometry = validate_structure(
            repaired,
            minimum_factor,
            minimum_vacuum,
            maximum_slab_thickness,
            maximum_nearest_ratio,
            minimum_layers,
        )
        surface_valid, surface_metrics = validate_surface_topology(
            repaired, reference_profile
        )
        geometry.update(surface_metrics)
        if not (geometry_valid and surface_valid):
            rejected += 1
            reasons = [
                reason
                for reason in str(geometry["Geometry_Reasons"]).split("|")
                if reason
            ]
            if reasons:
                rejection_reasons.update(reasons)
            elif not geometry_valid:
                rejection_reasons["geometry_validation_failed"] += 1
            surface_reasons = [
                reason
                for reason in str(
                    surface_metrics["Surface_Topology_Reasons"]
                ).split("|")
                if reason
            ]
            rejection_reasons.update(
                f"surface_{reason}" for reason in surface_reasons
            )
            continue
        candidates.append(
            {
                "candidate_index": candidate_index,
                "structure": repaired,
                "geometry": geometry,
                "relaxation_energy_eV": math.nan,
                "chgnet_max_force_eV_A": math.nan,
            }
        )
        if (
            candidate_index == len(raw_structures) - 1
            or (candidate_index + 1) % max(1, len(raw_structures) // 20) == 0
        ):
            report_progress(
                f"几何与表面拓扑筛选 {candidate_index + 1}/"
                f"{len(raw_structures)}",
                25.0
                + 10.0
                * (candidate_index + 1)
                / max(1, len(raw_structures)),
            )
    if not candidates:
        raise RuntimeError(
            "all generated structures failed geometry validation; train longer, "
            "increase --oversample, or reduce --guidance-scale"
        )

    predictor_config = predictors[0][1]["config"]
    radius = float(predictor_config["data"]["radius"])
    max_neighbors = int(predictor_config["data"]["max_neighbors"])
    valid_candidates = []
    graphs = []
    for graph_index, candidate in enumerate(candidates, start=1):
        try:
            graph = build_periodic_graph(
                candidate["structure"],
                radius,
                max_neighbors,
                identifier=f"candidate_{candidate['candidate_index']:05d}",
            )
            graphs.append(graph)
            valid_candidates.append(candidate)
        except Exception:
            rejected += 1
            rejection_reasons["graph_build_error"] += 1
        if (
            graph_index == len(candidates)
            or graph_index % max(1, len(candidates) // 10) == 0
        ):
            report_progress(
                f"构建周期晶体图 {graph_index}/{len(candidates)}",
                35.0 + 3.0 * graph_index / max(1, len(candidates)),
            )
    if not graphs:
        raise RuntimeError("no geometry-valid candidate could be converted to a periodic graph")
    predictions = []
    prediction_starts = list(range(0, len(graphs), args.batch_size))
    for batch_index, start in enumerate(prediction_starts, start=1):
        predictions.extend(
            infer_chunk(
                graphs[start : start + args.batch_size],
                predictors,
                device,
                args.mc_samples,
            )
        )
        report_progress(
            f"初筛 NFE 预测 {batch_index}/{len(prediction_starts)}",
            38.0
            + 7.0 * batch_index / max(1, len(prediction_starts)),
        )
    ranked = rank_candidates(
        valid_candidates, predictions, args.target, args.target_score
    )
    if args.relaxer == "chgnet":
        # Relax only a predictor-screened pool, then score the relaxed geometry again.
        report_progress("正在初始化 CHGNet 预弛豫器…", 46.0)
        relaxer = create_chgnet_relaxer(device)
        requested_pool_size = (
            args.relax_pool_size
            if args.relax_pool_size is not None
            else max(args.num * 2, args.num + 10)
        )
        pool_size = min(len(ranked), requested_pool_size)
        relaxed_candidates = []
        for relax_index, (_, candidate, _) in enumerate(
            ranked[:pool_size], start=1
        ):
            report_progress(
                f"CHGNet 固定晶胞预弛豫 {relax_index}/{pool_size}",
                47.0
                + 35.0 * (relax_index - 1) / max(1, pool_size),
            )
            try:
                relaxed, energy, maximum_force = relax_with_chgnet(
                    candidate["structure"],
                    relaxer,
                    fmax=args.relax_fmax,
                    steps=args.relax_steps,
                )
                relaxed = center_structure(relaxed)
            except Exception:
                rejected += 1
                rejection_reasons["chgnet_relaxation_error"] += 1
                continue
            geometry_valid, geometry = validate_structure(
                relaxed,
                minimum_factor,
                minimum_vacuum,
                maximum_slab_thickness,
                maximum_nearest_ratio,
                minimum_layers,
            )
            surface_valid, surface_metrics = validate_surface_topology(
                relaxed, reference_profile
            )
            geometry.update(surface_metrics)
            if not (geometry_valid and surface_valid):
                rejected += 1
                reasons = [
                    reason
                    for reason in str(geometry["Geometry_Reasons"]).split("|")
                    if reason
                ]
                if reasons:
                    rejection_reasons.update(
                        f"post_relax_{reason}" for reason in reasons
                    )
                elif not geometry_valid:
                    rejection_reasons[
                        "post_relax_geometry_validation_failed"
                    ] += 1
                surface_reasons = [
                    reason
                    for reason in str(
                        surface_metrics["Surface_Topology_Reasons"]
                    ).split("|")
                    if reason
                ]
                rejection_reasons.update(
                    f"post_relax_surface_{reason}"
                    for reason in surface_reasons
                )
                continue
            if (
                np.isfinite(maximum_force)
                and maximum_force > 1.05 * args.relax_fmax
            ):
                rejected += 1
                rejection_reasons["post_relax_force_above_threshold"] += 1
                continue
            updated = dict(candidate)
            updated["structure"] = relaxed
            updated["geometry"] = geometry
            updated["relaxation_energy_eV"] = energy
            updated["chgnet_max_force_eV_A"] = maximum_force
            relaxed_candidates.append(updated)
            report_progress(
                f"CHGNet 预弛豫 {relax_index}/{pool_size} 完成，"
                f"最大力 {maximum_force:.4f} eV/Å",
                47.0 + 35.0 * relax_index / max(1, pool_size),
            )
        if not relaxed_candidates:
            raise RuntimeError("all CHGNet-relaxed candidates failed")
        relaxed_graphs = [
            build_periodic_graph(
                candidate["structure"],
                radius,
                max_neighbors,
                identifier=f"relaxed_{candidate['candidate_index']:05d}",
            )
            for candidate in relaxed_candidates
        ]
        relaxed_predictions = []
        relaxed_prediction_starts = list(
            range(0, len(relaxed_graphs), args.batch_size)
        )
        for batch_index, start in enumerate(
            relaxed_prediction_starts, start=1
        ):
            relaxed_predictions.extend(
                infer_chunk(
                    relaxed_graphs[start : start + args.batch_size],
                    predictors,
                    device,
                    args.mc_samples,
                )
            )
            report_progress(
                f"弛豫后 NFE 独立复评 {batch_index}/"
                f"{len(relaxed_prediction_starts)}",
                84.0
                + 8.0
                * batch_index
                / max(1, len(relaxed_prediction_starts)),
            )
        ranked = rank_candidates(
            relaxed_candidates,
            relaxed_predictions,
            args.target,
            args.target_score,
        )
    else:
        report_progress("已跳过 CHGNet，准备严格目标筛选…", 92.0)
    report_progress("正在检查目标匹配、训练集重复与候选互重…", 94.0)
    matcher = StructureMatcher(
        ltol=0.15,
        stol=0.20,
        angle_tol=4.0,
        primitive_cell=False,
        scale=True,
        attempt_supercell=False,
    )
    selected = []
    accepted_structures: list[Structure] = []
    target_probabilities = [
        float(prediction.get(f"Probability_{args.target.capitalize()}", 0.0))
        for _, _, prediction in ranked
    ]
    predicted_label_counts = Counter(
        str(prediction.get("Predicted_NFE_Label", "unknown"))
        for _, _, prediction in ranked
    )
    for item in ranked:
        _, candidate, prediction = item
        target_matches, target_probability = prediction_matches_target(
            prediction,
            args.target,
            args.min_target_probability,
        )
        candidate["target_matches"] = target_matches
        candidate["target_probability"] = target_probability
        if not target_matches and not args.allow_target_mismatch:
            rejected += 1
            if prediction["Predicted_NFE_Label"] != args.target:
                rejection_reasons["target_label_mismatch"] += 1
            else:
                rejection_reasons["target_probability_below_threshold"] += 1
            continue
        structure = candidate["structure"]
        matches_training = any(
            safe_structure_match(matcher, structure, reference)
            for reference in reference_structures(
                generator_checkpoint, structure
            )
        )
        if matches_training and not args.allow_training_match:
            rejected += 1
            rejection_reasons["matches_training_structure"] += 1
            continue
        if any(
            safe_structure_match(matcher, structure, accepted)
            for accepted in accepted_structures
        ):
            rejected += 1
            rejection_reasons["duplicate_generated_structure"] += 1
            continue
        candidate["matches_training_structure"] = matches_training
        selected.append(item)
        accepted_structures.append(structure)
        if len(selected) >= args.num:
            break
    report_progress(
        f"严格筛选完成：接受 {len(selected)}/{args.num} 个候选",
        97.0,
    )
    output_dir = unique_output_directory(Path(args.output).resolve())
    rows = []
    for rank, (ranking_score, candidate, prediction) in enumerate(selected, start=1):
        report_progress(
            f"正在导出 CIF {rank}/{len(selected)}",
            97.0 + 2.0 * rank / max(1, len(selected)),
        )
        structure = candidate["structure"]
        formula = composition_formula(
            [int(site.specie.Z) for site in structure]
        )
        safe_formula = re.sub(r"[^A-Za-z0-9_.-]+", "_", formula)
        filename = (
            f"rank_{rank:03d}_{safe_formula}_"
            f"nfe_{float(prediction['Predicted_NFE_Score']):.3f}.cif"
        )
        path = output_dir / filename
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
        CifWriter(structure, symprec=None).write_file(path, mode="wt")
        row = {
            "Rank": rank,
            "CIF_Path": str(path),
            "Formula": formula,
            "Generator_Ranking_Score": ranking_score,
            "Target_Label": args.target,
            "Target_Score": args.target_score,
            "Target_Probability": candidate["target_probability"],
            "Target_Matched": candidate["target_matches"],
            "Relaxer": args.relaxer,
            "Matches_Training_Structure": candidate.get(
                "matches_training_structure", False
            ),
            "Relaxation_Energy_eV": candidate["relaxation_energy_eV"],
            "CHGNet_Max_Force_eV_A": candidate[
                "chgnet_max_force_eV_A"
            ],
            **candidate["geometry"],
            **prediction,
        }
        rows.append(row)
    summary_path = output_dir / "generation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    run_info = {
        "output_directory": str(output_dir),
        "requested": args.num,
        "generated_raw": len(raw_structures),
        "geometry_valid": len(valid_candidates),
        "relaxation_pool_size": (
            pool_size if args.relaxer == "chgnet" else 0
        ),
        "post_relaxation_valid": len(ranked),
        "rejected": rejected,
        # A candidate can fail more than one geometry rule, so the sum of these
        # diagnostic counts can be larger than `rejected`.
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "exported": len(rows),
        "target_label": args.target,
        "target_score": args.target_score,
        "min_target_probability": args.min_target_probability,
        "prediction_diagnostics": {
            "evaluated": len(ranked),
            "predicted_label_counts": dict(
                sorted(predicted_label_counts.items())
            ),
            "target_probability_min": (
                min(target_probabilities) if target_probabilities else None
            ),
            "target_probability_median": (
                float(np.median(target_probabilities))
                if target_probabilities
                else None
            ),
            "target_probability_max": (
                max(target_probabilities) if target_probabilities else None
            ),
        },
        "allow_target_mismatch": args.allow_target_mismatch,
        "sampling_steps": steps,
        "guidance_scale": guidance_scale,
        "relaxer": args.relaxer,
        "surface_profile": str(profile_path),
        "summary": str(summary_path),
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_progress(
        f"生成流水线完成：导出 {len(rows)} 个严格候选",
        100.0,
    )
    print(json.dumps(run_info, ensure_ascii=False, indent=2))
    if rows:
        display_columns = [
            "Rank",
            "Formula",
            "Predicted_NFE_Label",
            "Probability_Low",
            "Probability_Medium",
            "Probability_High",
            "Predicted_NFE_Score",
            "NFE_Score_Std",
            "OOD_Risk",
            "CIF_Path",
        ]
        print(pd.DataFrame(rows)[display_columns].to_string(index=False))
    elif not args.allow_target_mismatch:
        print(
            "No candidate satisfied both the requested class and minimum "
            "target probability. Increase --oversample, adjust the requested "
            "composition, or explicitly use --allow-target-mismatch.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
