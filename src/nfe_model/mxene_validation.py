# ==============================================================================
# 中文概述：判断一个结构是否为 MXene 片层，供 Windows 程序在预测前拒绝非 MXene 输入。
#           判据依次为：元素组成、二维周期与真空层、M(n+1)X(n) 化学计量、M/X 交替的内核层序、
#           端基位于内核外侧并与表面金属成键。
# English overview: Decide whether a structure is an MXene slab so the Windows application can reject
#           non-MXene inputs before prediction. Checks, in order: elements, 2D periodicity with a vacuum
#           gap, M(n+1)X(n) stoichiometry, an alternating M/X core, and terminations outside the core
#           bonded to surface metals.
#
# 中文输入：pymatgen Structure，或 CIF/POSCAR 文件路径。
# English inputs: A pymatgen Structure, or a CIF/POSCAR file path.
# 中文输出：MXeneCheck：是否合法、中文原因、n、元素、真空与厚度、是否落在预测器训练域内。
# English outputs: MXeneCheck with validity, Chinese reasons, n, elements, vacuum and thickness,
#                  and a training-domain flag with notes.
#
# 关键约束 / Key invariants:
# - 只判定结构类型，不评价能量、稳定性或 NFE 性质；阈值集中在模块常量中，可按课题组约定修改。
#   Only the structure type is judged, not energy, stability or NFE character; thresholds are
#   module constants that can be edited to group conventions.
# - 训练集结构必须全部判为合法（training/audits/audit_mxene_inputs.py 全量核对）。
#   Every training structure must pass (checked in full by training/audits/audit_mxene_inputs.py).
# - 合法但超出训练域（n ≠ 1、未见金属或端基、单面无端基等）的 MXene 仍然合法，只给出提示。
#   MXenes outside the training domain stay valid and only receive domain notes.
# - 主要接口 / Main APIs: MXeneCheck, check_mxene_structure, check_mxene_file
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from pymatgen.core import Structure

from .compat import ensure_pymatgen_int64_compatibility
from .surface_geometry import _cluster_layers

# Windows pymatgen wheels need int64 pbc arrays for neighbor searches.
ensure_pymatgen_int64_compatibility()

# 中文：MXene 的 M 取早期过渡金属；训练集使用其中 11 种。
# English: M is an early transition metal; the training set uses 11 of them.
MXENE_M_ELEMENTS = frozenset({"Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn"})
MXENE_X_ELEMENTS = frozenset({"C", "N"})
TERMINATION_ELEMENTS = frozenset({"O", "H", "F", "Cl", "Br", "I", "S", "Se", "Te"})
TRAINING_M_ELEMENTS = frozenset({"Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"})
TRAINING_TERMINATIONS = frozenset({"OH", "F", "Cl", "Br", "I", "S", "Se"})

# 中文：阈值依据训练集 15,206 个 DFT 弛豫结构的极值留出余量：真空最小 22.3 Å；C/N 距外侧金属面
#       最近 0.30 Å（W/Mo/Cr 氮化物明显偏心）；端基距外侧金属面最近 0.91 Å；两层金属最小间距 1.52 Å；
#       金属到最近 C/N 最远 2.87 Å；端基到最近金属最远 3.27 Å；O–H 最长 1.01 Å。
# English: thresholds leave margin around the extremes of the 15,206 DFT-relaxed training structures:
#          vacuum >= 22.3 A; C/N as close as 0.30 A to an outer metal plane (off-centre W/Mo/Cr nitrides);
#          terminations >= 0.91 A outside the metal planes; metal planes >= 1.52 A apart; metal-to-nearest-C/N
#          <= 2.87 A; termination-to-nearest-metal <= 3.27 A; O-H <= 1.01 A.
MIN_VACUUM_A = 6.0
LAYER_TOLERANCE_A = 0.45
X_INSIDE_MARGIN_A = 0.1
TERMINATION_OUTSIDE_MARGIN_A = 0.3
X_METAL_CUTOFF_A = 3.2
MIN_X_METAL_NEIGHBORS = 3
TERMINATION_METAL_CUTOFF_A = 3.8
OH_CUTOFF_A = 1.25
MAX_N = 4
MAX_READ_ERROR_CHARS = 160


# 中文：单个结构的 MXene 判定结果。/ English: MXene decision for one structure.
@dataclass(frozen=True)
class MXeneCheck:
    valid: bool
    reasons: tuple[str, ...] = ()
    formula: str = ""
    n: int | None = None
    metals: tuple[str, ...] = ()
    cores: tuple[str, ...] = ()
    bottom_terminations: tuple[str, ...] = ()
    top_terminations: tuple[str, ...] = ()
    vacuum_A: float = math.nan
    thickness_A: float = math.nan
    in_training_domain: bool = False
    domain_notes: tuple[str, ...] = ()

    @property
    def message(self) -> str:
        return "；".join(self.reasons)


def _invalid(formula: str, *reasons: str, vacuum_A: float = math.nan, thickness_A: float = math.nan) -> MXeneCheck:
    return MXeneCheck(
        valid=False,
        reasons=tuple(reasons),
        formula=formula,
        vacuum_A=vacuum_A,
        thickness_A=thickness_A,
    )


def _symbol(site: object) -> str:
    specie = getattr(site, "specie")
    return str(getattr(specie, "symbol", specie))


# 中文：每个晶格轴方向上的晶胞高度（体积除以另外两轴张成的面积）。
# English: Cell height along each axis (volume divided by the area spanned by the other two axes).
def _axis_heights(matrix: np.ndarray) -> np.ndarray:
    volume = abs(float(np.linalg.det(matrix)))
    heights = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        j, k = [index for index in range(3) if index != axis]
        area = float(np.linalg.norm(np.cross(matrix[j], matrix[k])))
        heights[axis] = volume / area if area > 1e-12 else 0.0
    return heights


# 中文：分数坐标上的最大周期空隙，以及空隙之后片层开始的位置。
# English: Largest periodic gap in fractional coordinates and where the slab starts after it.
def _largest_gap(fractions: np.ndarray) -> tuple[float, float]:
    values = np.sort(np.mod(np.asarray(fractions, dtype=np.float64), 1.0))
    if len(values) == 1:
        return 1.0, float(values[0])
    gaps = np.diff(np.r_[values, values[0] + 1.0])
    index = int(np.argmax(gaps))
    return float(gaps[index]), float(values[(index + 1) % len(values)])


def _termination_types(symbols: Sequence[str], indices: Sequence[int]) -> tuple[str, ...]:
    counts = Counter(symbols[index] for index in indices)
    types: set[str] = set()
    for symbol, count in counts.items():
        if symbol == "H":
            continue
        if symbol == "O":
            hydroxyl = min(count, counts.get("H", 0))
            if hydroxyl:
                types.add("OH")
            if count > hydroxyl:
                types.add("O")
        else:
            types.add(symbol)
    return tuple(sorted(types))


def _count_neighbors(centers: np.ndarray, mask: np.ndarray, n_atoms: int) -> np.ndarray:
    selected = np.asarray(centers[mask], dtype=np.int64)
    return np.bincount(selected, minlength=n_atoms)


# 中文：顶层接口 `check_mxene_structure`；按层级给出第一处不满足 MXene 定义的原因。
# English: Top-level function `check_mxene_structure`; report the first level at which the MXene definition fails.
def check_mxene_structure(structure: Structure) -> MXeneCheck:
    n_atoms = len(structure)
    formula = structure.composition.reduced_formula if n_atoms else ""
    if not structure.is_ordered:
        return _invalid(formula, "含部分占位的无序位点，无法判定为具体的 MXene 结构")

    # 1. 元素 / elements
    symbols = [_symbol(site) for site in structure]
    present = set(symbols)
    reasons: list[str] = []
    foreign = sorted(present - MXENE_M_ELEMENTS - MXENE_X_ELEMENTS - TERMINATION_ELEMENTS)
    if foreign:
        reasons.append("含有 MXene 以外的元素：" + "、".join(foreign))
    metal_index = np.asarray([i for i, s in enumerate(symbols) if s in MXENE_M_ELEMENTS], dtype=np.int64)
    core_index = np.asarray([i for i, s in enumerate(symbols) if s in MXENE_X_ELEMENTS], dtype=np.int64)
    if metal_index.size == 0:
        reasons.append("没有 MXene 的过渡金属 M（如 Ti、V、Nb、Mo）")
    if core_index.size == 0:
        reasons.append("没有 C 或 N，不是碳化物或氮化物 MXene")
    if reasons:
        return _invalid(formula, *reasons)
    if n_atoms < 3:
        return _invalid(formula, f"只有 {n_atoms} 个原子，不足以构成 MXene 片层")

    # 2. 二维周期与真空层 / 2D periodicity and vacuum
    matrix = np.asarray(structure.lattice.matrix, dtype=np.float64)
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    heights = _axis_heights(matrix)
    gaps_A: list[float] = []
    starts: list[float] = []
    for axis in range(3):
        gap, start = _largest_gap(frac[:, axis])
        gaps_A.append(gap * float(heights[axis]))
        starts.append(start)
    vacuum_axes = [axis for axis in range(3) if gaps_A[axis] >= MIN_VACUUM_A]
    if not vacuum_axes:
        return _invalid(
            formula,
            f"没有真空层：各方向最大原子间空隙只有 {max(gaps_A):.1f} Å（至少需要 {MIN_VACUUM_A:.0f} Å），"
            "看起来是体相晶体而不是二维片层",
        )
    if len(vacuum_axes) > 1:
        return _invalid(
            formula,
            f"在 {len(vacuum_axes)} 个方向上都有真空，不是二维周期片层（可能是分子、团簇或纳米带）",
        )
    axis = vacuum_axes[0]
    shifted = frac[:, axis] - starts[axis]
    shifted = shifted - np.floor(shifted + 1e-9)
    z = shifted * float(heights[axis])
    vacuum_A = float(gaps_A[axis])
    thickness_A = float(z.max() - z.min())

    # 3. 化学计量 / stoichiometry
    n_metal = int(metal_index.size)
    n_core = int(core_index.size)
    n_value = next((n for n in range(1, MAX_N + 1) if n_metal * n == n_core * (n + 1)), None)
    if n_value is None:
        return _invalid(
            formula,
            f"金属 M 与 C/N 的原子数之比为 {n_metal}:{n_core}，不符合 M(n+1)X(n)（n = 1–4，如 M₂X、M₃X₂、M₄X₃）",
            vacuum_A=vacuum_A,
            thickness_A=thickness_A,
        )

    # 4. 内核层序 / core layer sequence
    z_metal = z[metal_index]
    z_core = z[core_index]
    low, high = float(z_metal.min()), float(z_metal.max())
    if np.any(z_core <= low + X_INSIDE_MARGIN_A) or np.any(z_core >= high - X_INSIDE_MARGIN_A):
        return _invalid(
            formula,
            "C/N 原子没有全部夹在金属层之间，不是 M/X 交替的 MXene 内核",
            vacuum_A=vacuum_A,
            thickness_A=thickness_A,
        )
    # 金属与 C/N 分别聚层，再比较层的平均高度；弛豫后偏心的 C/N 不会被并入金属层。
    # Cluster metal and C/N atoms separately, then order the layer means, so an off-centre
    # relaxed C/N atom is never merged into a metal plane.
    layer_means: list[tuple[float, str]] = []
    for kind, indices in (("M", metal_index), ("X", core_index)):
        heights_of_kind = z[indices]
        layer_of = _cluster_layers(heights_of_kind, LAYER_TOLERANCE_A)
        for layer in np.unique(layer_of):
            layer_means.append((float(heights_of_kind[layer_of == layer].mean()), kind))
    observed = [kind for _height, kind in sorted(layer_means)]
    expected = ["M", "X"] * n_value + ["M"]
    if observed != expected:
        return _invalid(
            formula,
            f"内核层序为 {'-'.join(observed)}，M{n_value + 1}X{n_value} 型 MXene 应为 {'-'.join(expected)}",
            vacuum_A=vacuum_A,
            thickness_A=thickness_A,
        )

    termination_index = np.asarray([i for i, s in enumerate(symbols) if s in TERMINATION_ELEMENTS], dtype=np.int64)
    if termination_index.size:
        z_termination = z[termination_index]
        inside = int(
            np.sum(
                (z_termination > low - TERMINATION_OUTSIDE_MARGIN_A)
                & (z_termination < high + TERMINATION_OUTSIDE_MARGIN_A)
            )
        )
        if inside:
            return _invalid(
                formula,
                f"{inside} 个端基原子位于金属层之间或与表面金属层几乎同高，不是表面端基",
                vacuum_A=vacuum_A,
                thickness_A=thickness_A,
            )

    # 5. 成键 / bonding
    cutoff = max(X_METAL_CUTOFF_A, TERMINATION_METAL_CUTOFF_A, OH_CUTOFF_A)
    centers, points, _images, distances = structure.get_neighbor_list(cutoff)
    centers = np.asarray(centers, dtype=np.int64)
    points = np.asarray(points, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    is_metal = np.zeros(n_atoms, dtype=bool)
    is_metal[metal_index] = True
    is_core = np.zeros(n_atoms, dtype=bool)
    is_core[core_index] = True
    is_oxygen = np.asarray([symbol == "O" for symbol in symbols], dtype=bool)

    core_metal_count = _count_neighbors(centers, (distances <= X_METAL_CUTOFF_A) & is_metal[points], n_atoms)
    metal_core_count = _count_neighbors(centers, (distances <= X_METAL_CUTOFF_A) & is_core[points], n_atoms)
    termination_metal_count = _count_neighbors(
        centers, (distances <= TERMINATION_METAL_CUTOFF_A) & is_metal[points], n_atoms
    )
    hydrogen_oxygen_count = _count_neighbors(centers, (distances <= OH_CUTOFF_A) & is_oxygen[points], n_atoms)

    weak_core = int(sum(1 for atom in core_index if core_metal_count[atom] < MIN_X_METAL_NEIGHBORS))
    if weak_core:
        reasons.append(
            f"{weak_core} 个 C/N 原子在 {X_METAL_CUTOFF_A:.1f} Å 内的金属近邻少于 {MIN_X_METAL_NEIGHBORS} 个，"
            "不是 MXene 的八面体内核"
        )
    loose_metal = int(sum(1 for atom in metal_index if metal_core_count[atom] == 0))
    if loose_metal:
        reasons.append(f"{loose_metal} 个金属原子没有与 C/N 成键（可能是吸附金属或金属团簇）")
    hydrogens = [int(atom) for atom in termination_index if symbols[int(atom)] == "H"]
    bad_hydrogen = int(sum(1 for atom in hydrogens if hydrogen_oxygen_count[atom] == 0))
    if bad_hydrogen:
        reasons.append(f"{bad_hydrogen} 个 H 原子不属于 OH 端基")
    heavy_terminations = [int(atom) for atom in termination_index if symbols[int(atom)] != "H"]
    floating = int(sum(1 for atom in heavy_terminations if termination_metal_count[atom] == 0))
    if floating:
        reasons.append(f"{floating} 个端基原子没有与表面金属成键（可能是游离分子或吸附物）")
    if reasons:
        return _invalid(formula, *reasons, vacuum_A=vacuum_A, thickness_A=thickness_A)

    # 6. 训练域提示 / training-domain notes (valid either way)
    bottom = [int(atom) for atom in termination_index if z[int(atom)] < low]
    top = [int(atom) for atom in termination_index if z[int(atom)] > high]
    bottom_types = _termination_types(symbols, bottom)
    top_types = _termination_types(symbols, top)
    metals = tuple(sorted({symbols[int(atom)] for atom in metal_index}))
    cores = tuple(sorted({symbols[int(atom)] for atom in core_index}))
    notes: list[str] = []
    if n_value != 1:
        notes.append(f"n = {n_value}（M{n_value + 1}X{n_value}），训练集只有 M₂X")
    unseen_metals = sorted(set(metals) - TRAINING_M_ELEMENTS)
    if unseen_metals:
        notes.append("金属 " + "、".join(unseen_metals) + " 不在训练集中")
    if len(cores) > 1:
        notes.append("同时含 C 和 N，训练集每个结构只有一种")
    for side, types in (("底面", bottom_types), ("顶面", top_types)):
        if not types:
            notes.append(f"{side}没有端基")
        elif len(types) > 1:
            notes.append(f"{side}为混合端基 " + "、".join(types))
        elif types[0] not in TRAINING_TERMINATIONS:
            notes.append(f"{side}端基 {types[0]} 不在训练集中")
    return MXeneCheck(
        valid=True,
        reasons=(),
        formula=formula,
        n=n_value,
        metals=metals,
        cores=cores,
        bottom_terminations=bottom_types,
        top_terminations=top_types,
        vacuum_A=vacuum_A,
        thickness_A=thickness_A,
        in_training_domain=not notes,
        domain_notes=tuple(notes),
    )


# 中文：顶层接口 `check_mxene_file`；读取失败也作为不合法结果返回，而不是抛出异常。
# English: Top-level function `check_mxene_file`; read failures are returned as invalid results instead of raised.
def check_mxene_file(path: str | Path) -> MXeneCheck:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(str(path))
    except Exception as exc:  # noqa: BLE001 - any parser failure means the input is not a readable structure
        detail = f"{type(exc).__name__}: {exc}".strip()
        if len(detail) > MAX_READ_ERROR_CHARS:
            detail = detail[: MAX_READ_ERROR_CHARS - 1] + "…"
        return MXeneCheck(valid=False, reasons=(f"无法读取为晶体结构（{detail}）",))
    return check_mxene_structure(structure)


__all__ = [
    "MXENE_M_ELEMENTS",
    "MXENE_X_ELEMENTS",
    "MXeneCheck",
    "TERMINATION_ELEMENTS",
    "TRAINING_M_ELEMENTS",
    "TRAINING_TERMINATIONS",
    "check_mxene_file",
    "check_mxene_structure",
]
