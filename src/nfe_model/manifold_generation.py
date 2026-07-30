# ==============================================================================
# 中文概述：manifold generator 流形投影与训练集未见金属组合替换，减少层塌缩并提升新颖性。
# English overview: manifold projection and unseen metal-pair substitution to reduce layer collapse and improve novelty.
#
# 中文输入：surface generator 样本、模板角色、目标骨架和训练集组合目录。
# English inputs: surface generator samples, template roles, target framework, and training-combination catalog.
# 中文输出：投影回物理模板流形且可由用户指定核心/内层金属的候选。
# English outputs: Candidates projected to the physical template manifold with user-selected core/inner metals.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: _element_radius, _substitution_weight, mutate_template_to_unseen, _limit_xy_displacement, project_structure_to_template_manifold, parse_args, choose_templates, sample_structures, _enrich_run_info, main
#
# Author: Ruck
# Generated: 2026-07-30 01:12:13 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Element, Lattice, Structure

from . import strict_generation
from .surface_generator_data import GROUP_OH_HYDROGEN, GROUP_OH_OXYGEN


_BASE_PARSE_ARGS = strict_generation.parse_args
_BASE_CHOOSE_TEMPLATES = strict_generation.choose_templates
_BASE_SAMPLE_STRUCTURES = strict_generation.sample_structures
_OPTIONS: argparse.Namespace | None = None

_MXENE_METALS = tuple(
    int(Element(symbol).Z)
    for symbol in ("Sc", "Ti", "V", "Cr", "Y", "Zr", "Nb", "Mo", "Hf", "Ta", "W")
)


# 中文：顶层接口 `_element_radius`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_element_radius`; review type hints and callers before extending it.
def _element_radius(atomic_number: int) -> float:
    element = Element.from_Z(int(atomic_number))
    radius = element.atomic_radius or element.average_ionic_radius
    try:
        return max(float(radius), 0.6)
    except (TypeError, ValueError):
        return 1.2


# 中文：顶层接口 `_substitution_weight`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_substitution_weight`; review type hints and callers before extending it.
def _substitution_weight(old_z: int, new_z: int) -> float:
    old = Element.from_Z(int(old_z))
    new = Element.from_Z(int(new_z))
    radius_delta = abs(_element_radius(old_z) - _element_radius(new_z))
    old_group = int(old.group or 7)
    new_group = int(new.group or 7)
    group_delta = abs(old_group - new_group)
    period_delta = abs(int(old.row) - int(new.row))
    return math.exp(
        -radius_delta / 0.22 - group_delta / 4.0 - period_delta / 2.0
    )


# 中文：顶层接口 `mutate_template_to_unseen`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `mutate_template_to_unseen`; review type hints and callers before extending it.
def mutate_template_to_unseen(
    template: dict[str, Any],
    forbidden_composition_keys: set[str],
) -> dict[str, Any]:
    """Create a chemically conservative metal substitution absent from training."""

    original_z = [int(value) for value in template["z"]]
    metal_indices = [
        index
        for index, (atomic_number, group_type, side) in enumerate(
            zip(
                original_z,
                template["group_type"],
                template["surface_side"],
            )
        )
        if int(group_type) == 0
        and int(side) == 0
        and int(atomic_number) in _MXENE_METALS
    ]
    if not metal_indices:
        raise ValueError("template contains no recognized MXene core metal")

    choices: list[tuple[list[int], str, float]] = []
    for atom_index in metal_indices:
        old_z = original_z[atom_index]
        for new_z in _MXENE_METALS:
            if new_z == old_z:
                continue
            candidate_z = list(original_z)
            candidate_z[atom_index] = int(new_z)
            key = strict_generation.composition_key(candidate_z)
            if key in forbidden_composition_keys:
                continue
            description = (
                f"{Element.from_Z(old_z).symbol}"
                f"{atom_index}->{Element.from_Z(new_z).symbol}"
            )
            choices.append(
                (
                    candidate_z,
                    description,
                    _substitution_weight(old_z, new_z),
                )
            )

    # A two-site substitution is a conservative fallback when every one-site
    # composition is already represented in the training set.
    if not choices and len(metal_indices) >= 2:
        left, right = metal_indices[:2]
        for new_left in _MXENE_METALS:
            for new_right in _MXENE_METALS:
                if (
                    new_left == original_z[left]
                    and new_right == original_z[right]
                ):
                    continue
                candidate_z = list(original_z)
                candidate_z[left] = int(new_left)
                candidate_z[right] = int(new_right)
                key = strict_generation.composition_key(candidate_z)
                if key in forbidden_composition_keys:
                    continue
                description = (
                    f"{Element.from_Z(original_z[left]).symbol}{left}"
                    f"->{Element.from_Z(new_left).symbol};"
                    f"{Element.from_Z(original_z[right]).symbol}{right}"
                    f"->{Element.from_Z(new_right).symbol}"
                )
                choices.append(
                    (
                        candidate_z,
                        description,
                        _substitution_weight(original_z[left], new_left)
                        * _substitution_weight(original_z[right], new_right),
                    )
                )
    if not choices:
        raise RuntimeError(
            "could not create a metal-substituted composition absent from training"
        )

    candidate_z, description, _ = random.choices(
        choices,
        weights=[max(item[2], 1e-8) for item in choices],
        k=1,
    )[0]
    result = dict(template)
    result["z"] = candidate_z
    result["source_template_id"] = str(template.get("id", "unknown"))
    result["mutation"] = description
    result["id"] = f"{result['source_template_id']}::manifold::{description}"
    result["formula"] = strict_generation.composition_formula(candidate_z)
    return result


# 中文：顶层接口 `_limit_xy_displacement`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_limit_xy_displacement`; review type hints and callers before extending it.
def _limit_xy_displacement(
    delta_fractional: np.ndarray,
    lattice: np.ndarray,
    maximum_A: float,
) -> np.ndarray:
    delta = np.asarray(delta_fractional, dtype=np.float64).copy()
    delta[2] = 0.0
    distance = float(np.linalg.norm(delta @ lattice))
    if distance > maximum_A > 0.0:
        delta *= maximum_A / distance
    return delta


# 中文：顶层接口 `project_structure_to_template_manifold`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `project_structure_to_template_manifold`; review type hints and callers before extending it.
def project_structure_to_template_manifold(
    structure: Structure,
    template: dict[str, Any],
    *,
    maximum_lattice_strain: float = 0.05,
    core_xy_A: float = 0.18,
    core_z_A: float = 0.15,
    surface_xy_A: float = 0.10,
    surface_z_A: float = 0.15,
    hydrogen_xy_A: float = 0.08,
    hydrogen_z_A: float = 0.12,
    oh_bond_A: float = 0.9772,
) -> Structure:
    """Project a sampled slab onto the trusted layer/surface topology."""

    template_lattice = np.asarray(template["lattice"], dtype=np.float64)
    sampled_lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    projected_lattice = template_lattice.copy()
    for axis in (0, 1):
        template_length = float(np.linalg.norm(template_lattice[axis]))
        sampled_length = float(np.linalg.norm(sampled_lattice[axis]))
        scale = sampled_length / max(template_length, 1e-8)
        scale = float(
            np.clip(
                scale,
                1.0 - maximum_lattice_strain,
                1.0 + maximum_lattice_strain,
            )
        )
        projected_lattice[axis] *= scale
    # The vacuum direction is a representation choice, not a generated
    # material degree of freedom. Preserve it exactly from the relaxed template.
    projected_lattice[2] = template_lattice[2]
    inverse_lattice = np.linalg.inv(projected_lattice)

    sampled_frac = strict_generation.center_slab_fractional(
        np.asarray(structure.frac_coords, dtype=np.float64)
    )
    template_frac = strict_generation.center_slab_fractional(
        np.asarray(template["frac_pos"], dtype=np.float64)
    )
    if len(sampled_frac) != len(template_frac):
        raise ValueError("sample/template atom counts differ")
    delta = sampled_frac - template_frac
    delta = (delta + 0.5) % 1.0 - 0.5
    group_type = np.asarray(template["group_type"], dtype=np.int64)
    surface_side = np.asarray(template["surface_side"], dtype=np.int64)
    corrected = template_frac.copy()

    for atom_index in range(len(corrected)):
        if int(group_type[atom_index]) == GROUP_OH_HYDROGEN:
            maximum_xy, maximum_z = hydrogen_xy_A, hydrogen_z_A
        elif int(surface_side[atom_index]) != 0:
            maximum_xy, maximum_z = surface_xy_A, surface_z_A
        else:
            maximum_xy, maximum_z = core_xy_A, core_z_A
        xy_delta = _limit_xy_displacement(
            np.asarray([delta[atom_index, 0], delta[atom_index, 1], 0.0]),
            projected_lattice,
            maximum_xy,
        )
        c_length = float(np.linalg.norm(projected_lattice[2]))
        z_delta = float(
            np.clip(
                delta[atom_index, 2],
                -maximum_z / max(c_length, 1e-8),
                maximum_z / max(c_length, 1e-8),
            )
        )
        corrected[atom_index] += xy_delta
        corrected[atom_index, 2] += z_delta

    # Preserve each OH group as a covalent unit with the observed relaxed-data
    # median bond length. The strict topology validator still makes the final call.
    for side in (-1, 1):
        hydrogens = np.where(
            (surface_side == side) & (group_type == GROUP_OH_HYDROGEN)
        )[0]
        oxygens = np.where(
            (surface_side == side) & (group_type == GROUP_OH_OXYGEN)
        )[0]
        for hydrogen_index, oxygen_index in zip(hydrogens, oxygens):
            vector_frac = (
                template_frac[int(hydrogen_index)]
                - template_frac[int(oxygen_index)]
            )
            vector_frac = (vector_frac + 0.5) % 1.0 - 0.5
            vector_cart = vector_frac @ projected_lattice
            length = float(np.linalg.norm(vector_cart))
            if length <= 1e-8:
                continue
            target_cart = vector_cart * (float(oh_bond_A) / length)
            corrected[int(hydrogen_index)] = (
                corrected[int(oxygen_index)] + target_cart @ inverse_lattice
            )

    corrected[:, :2] = np.mod(corrected[:, :2], 1.0)
    corrected = strict_generation.center_slab_fractional(corrected)
    return Structure(
        Lattice(projected_lattice),
        [site.specie for site in structure],
        corrected,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    global _OPTIONS
    extension = argparse.ArgumentParser(add_help=False)
    extension.add_argument(
        "--composition-mode",
        choices=("catalog", "unseen-metal-substitution"),
        default="unseen-metal-substitution",
    )
    extension.add_argument("--disable-manifold-projection", action="store_true")
    extension.add_argument("--maximum-lattice-strain", type=float, default=0.05)
    extension.add_argument("--core-xy-limit-A", type=float, default=0.18)
    extension.add_argument("--core-z-limit-A", type=float, default=0.15)
    extension.add_argument("--surface-xy-limit-A", type=float, default=0.10)
    extension.add_argument("--surface-z-limit-A", type=float, default=0.15)
    extension.add_argument("--hydrogen-xy-limit-A", type=float, default=0.08)
    extension.add_argument("--hydrogen-z-limit-A", type=float, default=0.12)
    extension.add_argument("--oh-bond-A", type=float, default=0.9772)
    extra, remaining = extension.parse_known_args(argv)
    args = _BASE_PARSE_ARGS(remaining)
    for key, value in vars(extra).items():
        setattr(args, key, value)
    _OPTIONS = args
    return args


# 中文：顶层接口 `choose_templates`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `choose_templates`; review type hints and callers before extending it.
def choose_templates(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    total: int,
) -> list[dict[str, Any]]:
    templates = _BASE_CHOOSE_TEMPLATES(args, checkpoint, total)
    if args.composition_mode == "catalog":
        return templates
    training_keys = set(checkpoint.get("novelty_reference", {}))
    used_keys = set(training_keys)
    mutated = []
    for template in templates:
        try:
            item = mutate_template_to_unseen(template, used_keys)
        except RuntimeError:
            # Composition uniqueness within this generation batch is desirable
            # but secondary to remaining outside the complete training set.
            item = mutate_template_to_unseen(template, training_keys)
        mutated.append(item)
        used_keys.add(strict_generation.composition_key(item["z"]))
    return mutated


# 中文：顶层接口 `sample_structures`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `sample_structures`; review type hints and callers before extending it.
def sample_structures(*args: Any, **kwargs: Any) -> list[Structure]:
    structures = _BASE_SAMPLE_STRUCTURES(*args, **kwargs)
    templates = args[2]
    if _OPTIONS is None or _OPTIONS.disable_manifold_projection:
        return structures
    return [
        project_structure_to_template_manifold(
            structure,
            template,
            maximum_lattice_strain=_OPTIONS.maximum_lattice_strain,
            core_xy_A=_OPTIONS.core_xy_limit_A,
            core_z_A=_OPTIONS.core_z_limit_A,
            surface_xy_A=_OPTIONS.surface_xy_limit_A,
            surface_z_A=_OPTIONS.surface_z_limit_A,
            hydrogen_xy_A=_OPTIONS.hydrogen_xy_limit_A,
            hydrogen_z_A=_OPTIONS.hydrogen_z_limit_A,
            oh_bond_A=_OPTIONS.oh_bond_A,
        )
        for structure, template in zip(structures, templates)
    ]


# 中文：顶层接口 `_enrich_run_info`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_enrich_run_info`; review type hints and callers before extending it.
def _enrich_run_info() -> None:
    if _OPTIONS is None:
        return
    base = Path(_OPTIONS.output).resolve()
    candidates = [base / "run_info.json"]
    if base.exists():
        candidates.extend(base.glob("run_*/run_info.json"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    path = max(existing, key=lambda item: item.stat().st_mtime_ns)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generator_variant"] = "manifold-unseen-composition-1.0"
    payload["composition_mode"] = _OPTIONS.composition_mode
    payload["manifold_projection"] = not _OPTIONS.disable_manifold_projection
    payload["manifold_limits_A"] = {
        "core_xy": _OPTIONS.core_xy_limit_A,
        "core_z": _OPTIONS.core_z_limit_A,
        "surface_xy": _OPTIONS.surface_xy_limit_A,
        "surface_z": _OPTIONS.surface_z_limit_A,
        "hydrogen_xy": _OPTIONS.hydrogen_xy_limit_A,
        "hydrogen_z": _OPTIONS.hydrogen_z_limit_A,
        "oh_bond": _OPTIONS.oh_bond_A,
    }
    payload["maximum_lattice_strain"] = _OPTIONS.maximum_lattice_strain
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main(argv: Sequence[str] | None = None) -> int:
    original_parse = strict_generation.parse_args
    original_choose = strict_generation.choose_templates
    original_sample = strict_generation.sample_structures
    strict_generation.parse_args = parse_args
    strict_generation.choose_templates = choose_templates
    strict_generation.sample_structures = sample_structures
    try:
        result = strict_generation.main(argv)
        _enrich_run_info()
        return result
    finally:
        strict_generation.parse_args = original_parse
        strict_generation.choose_templates = original_choose
        strict_generation.sample_structures = original_sample


if __name__ == "__main__":
    raise SystemExit(main())
