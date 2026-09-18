# ==============================================================================
# 中文概述：把任意表示的二维 slab 规范化为预测器训练时的表示：原胞、面内约化晶格、
#           固定真空轴长度、slab 居中、原子排序。
# English overview: Canonicalize any slab representation to the one the predictor was trained
#                   on: primitive cell, reduced in-plane lattice, fixed vacuum axis length,
#                   centered slab, sorted atoms.
#
# 中文输入：pymatgen Structure（CIF/POSCAR 解析结果）。
# English inputs: A pymatgen Structure parsed from CIF/POSCAR.
# 中文输出：规范化后的 Structure；同一材料的任何超胞、晶格设定、真空厚度、平移或
#           原子顺序都映射到同一个结构，因此得到同一个周期图。
# English outputs: A canonical Structure; supercells, lattice settings, vacuum thicknesses,
#                  translations and atom orderings of the same material map to one structure
#                  and therefore to one periodic graph.
#
# 关键约束 / Key invariants:
# - 训练集全部为 c = 30 Å、γ = 120° 的六方 1×1 胞，预测器的 11 维全局特征中 log c 与三个
#   cos 角在训练集内方差为零；不经规范化的输入会把这些特征推到 ±8 的饱和值。
#   All training cells have c = 30 Å and γ = 120°; four of the eleven global features have zero
#   variance, so an un-canonicalized input saturates them at ±8.
# - 只改变表示，不改变原子的笛卡尔相对几何。
#   Only the representation changes; relative Cartesian geometry is preserved.
# - 主要接口 / Main APIs: DEFAULT_VACUUM_LENGTH_A, reduce_in_plane_basis, vacuum_axis, canonicalize_slab
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import math

import numpy as np
from pymatgen.core import Lattice, Structure

from .compat import ensure_pymatgen_int64_compatibility

# Windows pymatgen wheels reject int32 pbc arrays inside get_primitive_structure,
# Lattice.get_points_in_sphere and StructureMatcher; patch once on import.
ensure_pymatgen_int64_compatibility()

DEFAULT_VACUUM_LENGTH_A = 30.0
_COORD_QUANTUM = 1e-3


# 中文：Gauss/Lagrange 二维格约化，返回最短基并采用钝角约定（六方格 γ = 120°）。
# English: Gauss/Lagrange reduction of a 2D lattice; shortest basis with the obtuse convention.
def reduce_in_plane_basis(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    for _ in range(200):
        if np.dot(a, a) > np.dot(b, b):
            a, b = b, a
        mu = int(round(float(np.dot(a, b) / np.dot(a, a))))
        if mu == 0:
            break
        b = b - mu * a
    if np.dot(a, b) > 1e-8 * max(np.dot(a, a), 1e-12):
        b = -b
    return a, b


# 中文：选择真空轴：分数坐标最大空隙所在的轴；并列时取最长轴。
# English: Choose the vacuum axis: the one with the largest fractional gap; ties go to the longest axis.
def vacuum_axis(structure: Structure) -> int:
    frac = np.mod(np.asarray(structure.frac_coords, dtype=np.float64), 1.0)
    best_axis, best_gap = 2, -1.0
    lengths = np.asarray(structure.lattice.abc, dtype=np.float64)
    for axis in range(3):
        values = np.sort(frac[:, axis])
        if len(values) <= 1:
            gap = 1.0
        else:
            gap = float(np.max(np.diff(np.r_[values, values[0] + 1.0])))
        if gap > best_gap + 1e-9 or (abs(gap - best_gap) <= 1e-9 and lengths[axis] > lengths[best_axis]):
            best_axis, best_gap = axis, gap
    return best_axis


def _center_slab(frac: np.ndarray) -> np.ndarray:
    frac = np.mod(np.asarray(frac, dtype=np.float64), 1.0).copy()
    if len(frac) == 0:
        return frac
    for axis in (0, 1):
        angle = 2.0 * math.pi * frac[:, axis]
        mean_angle = math.atan2(float(np.sin(angle).mean()), float(np.cos(angle).mean()))
        mean_fraction = (mean_angle / (2.0 * math.pi)) % 1.0
        frac[:, axis] = np.mod(frac[:, axis] + 0.5 - mean_fraction, 1.0)
    z_sorted = np.sort(frac[:, 2])
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    start = float(z_sorted[(int(np.argmax(gaps)) + 1) % len(z_sorted)])
    unwrapped = np.mod(frac[:, 2] - start, 1.0)
    center = 0.5 * (float(unwrapped.min()) + float(unwrapped.max()))
    frac[:, 2] = np.mod(unwrapped + 0.5 - center, 1.0)
    return frac


# 中文：顶层接口 `canonicalize_slab`；同一材料的不同表示映射到同一结构。
# English: Top-level function `canonicalize_slab`; different representations of one material map to one structure.
def canonicalize_slab(
    structure: Structure,
    *,
    vacuum_length_A: float = DEFAULT_VACUUM_LENGTH_A,
    primitive_tolerance_A: float = 0.1,
    reduce_to_primitive: bool = True,
) -> Structure:
    """Return the canonical representation used for graph construction.

    Steps: (1) reduce to the primitive cell so in-plane supercells collapse;
    (2) move the vacuum axis to the third position; (3) reduce the in-plane
    basis to the shortest pair with the obtuse-angle convention, which turns
    any hexagonal setting into gamma = 120 degrees; (4) replace the vacuum
    vector by the slab normal with a fixed length so vacuum thickness stops
    being an input; (5) center the slab at z = 0.5 and fix the in-plane
    translation gauge; (6) sort atoms by species and position.
    """

    working = structure.copy()
    if reduce_to_primitive and len(working) > 1:
        try:
            candidate = working.get_primitive_structure(tolerance=primitive_tolerance_A)
            if len(candidate) < len(working):
                working = candidate
        except Exception:  # pragma: no cover - pymatgen internal failures
            pass
    # Niggli reduction makes every lattice vector as short as possible, so a
    # tilted vacuum vector (c + a) or an in-plane vector carrying a vacuum
    # component (b + c) cannot survive into the in-plane reduction below.
    try:
        working = working.get_reduced_structure(reduction_algo="niggli")
    except Exception:  # pragma: no cover - degenerate lattices
        pass

    matrix = np.asarray(working.lattice.matrix, dtype=np.float64)
    axis = vacuum_axis(working)
    in_plane = [index for index in range(3) if index != axis]
    a, b = reduce_in_plane_basis(matrix[in_plane[0]], matrix[in_plane[1]])
    normal = np.cross(a, b)
    normal_length = float(np.linalg.norm(normal))
    if normal_length < 1e-8:
        raise ValueError("degenerate in-plane lattice")
    normal /= normal_length
    c = normal * float(vacuum_length_A)
    lattice = Lattice(np.stack([a, b, c]))
    cartesian = np.asarray(working.cart_coords, dtype=np.float64)
    species = [site.specie for site in working]
    frac = lattice.get_fractional_coords(cartesian)
    frac = _center_slab(frac)

    atomic_numbers = np.asarray([int(specie.Z) for specie in species], dtype=np.int64)
    quantized = np.round(frac / _COORD_QUANTUM).astype(np.int64)
    order = np.lexsort((quantized[:, 1], quantized[:, 0], quantized[:, 2], atomic_numbers))
    return Structure(
        lattice,
        [species[index] for index in order],
        frac[order],
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


__all__ = ["DEFAULT_VACUUM_LENGTH_A", "canonicalize_slab", "reduce_in_plane_basis", "vacuum_axis"]
