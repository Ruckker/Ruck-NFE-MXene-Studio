# ==============================================================================
# 中文概述：三维预览的场景模型：元素配色与半径、片层法向与层识别、面内超胞重复、共价半径判定的
#           化学键、跨边界的周期镜像原子与晶胞边。与界面无关，可离屏测试。
# English overview: Scene model of the 3D preview: element colors and radii, slab normal and layer
#           detection, in-plane supercell repetition, covalent-radius bonds, periodic image atoms across
#           the cell boundary, and cell edges. GUI-independent and testable off screen.
#
# 中文输入：pymatgen Structure、面内重复次数、是否显示周期镜像。
# English inputs: A pymatgen Structure, in-plane repetition counts, and whether to show periodic images.
# 中文输出：DisplayModel（显示用原子、键、层、晶胞边与晶格信息）；兼容旧接口的 StructureScene。
# English outputs: DisplayModel (display atoms, bonds, layers, cell edges, lattice info) and the legacy StructureScene.
#
# 关键约束 / Key invariants:
# - 片层沿法向保持连续，不会被晶胞边界切开；面内坐标折回 [0, 1)。
#   The slab stays contiguous along its normal; in-plane coordinates are wrapped into [0, 1).
# - 键判据沿用 1.0 预览：d ≤ min(1.24 (r_i + r_j) + 0.12, 3.65) Å。
#   Bond rule kept from the 1.0 preview: d <= min(1.24 (r_i + r_j) + 0.12, 3.65) A.
# - 主要接口 / Main APIs: element_color, covalent_radius, DisplayModel, build_display_model,
#   StructureScene, build_structure_scene
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, Structure

# 中文：柔和但可辨认的元素配色（金属偏冷色，端基偏暖色）。
# English: Soft but recognizable element colors (cool metals, warm terminations).
ELEMENT_COLORS = {
    "H": "#F4F6F8",
    "C": "#56606B",
    "N": "#3F72E0",
    "O": "#E2453C",
    "F": "#86C25A",
    "Cl": "#2EB88A",
    "Br": "#B5573C",
    "I": "#8E4FB0",
    "S": "#EDBE2F",
    "Se": "#F08A24",
    "Te": "#C9762A",
    "Sc": "#C9CED4",
    "Ti": "#9FAAB6",
    "V": "#8C95AB",
    "Cr": "#7D93C9",
    "Mn": "#9C7CC6",
    "Y": "#74CBD3",
    "Zr": "#5FBDB5",
    "Nb": "#57A9BF",
    "Mo": "#4C9DA6",
    "Hf": "#5AAFE8",
    "Ta": "#4E92E6",
    "W": "#3C79C8",
}
BOND_SCALE = 1.24
BOND_PADDING_A = 0.12
BOND_MAX_A = 3.65
BOND_MIN_A = 0.25
LAYER_TOLERANCE_A = 0.45
MAX_BOND_ATOMS = 1500
SLAB_BOX_PADDING_A = 1.0


# 中文：元素颜色（#RRGGBB）。/ English: Element color as #RRGGBB.
def element_color(symbol: str) -> str:
    if symbol in ELEMENT_COLORS:
        return ELEMENT_COLORS[symbol]
    try:
        number = int(Element(symbol).Z)
    except Exception:  # noqa: BLE001 - dummy species
        number = sum(ord(char) for char in symbol)
    red, green, blue = colorsys.hsv_to_rgb((number * 0.618033988749895) % 1.0, 0.45, 0.85)
    return f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


# 中文：共价半径（Å）。/ English: Covalent radius in Angstrom.
def covalent_radius(symbol: str) -> float:
    value = CovalentRadius.radius.get(symbol)
    if value is None:
        try:
            value = Element(symbol).atomic_radius
        except Exception:  # noqa: BLE001
            value = None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.2
    return number if math.isfinite(number) and number > 0 else 1.2


def _symbol(site: object) -> str:
    specie = getattr(site, "specie")
    return str(getattr(specie, "symbol", specie))


def _axis_heights(lattice: np.ndarray) -> np.ndarray:
    volume = abs(float(np.linalg.det(lattice)))
    heights = np.ones(3)
    for axis in range(3):
        j, k = [index for index in range(3) if index != axis]
        area = float(np.linalg.norm(np.cross(lattice[j], lattice[k])))
        heights[axis] = volume / area if area > 1e-12 else 1.0
    return heights


def _largest_gap(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(np.mod(values, 1.0))
    if len(ordered) == 1:
        return 1.0, float(ordered[0])
    gaps = np.diff(np.r_[ordered, ordered[0] + 1.0])
    index = int(np.argmax(gaps))
    return float(gaps[index]), float(ordered[(index + 1) % len(ordered)])


def _cluster(values: np.ndarray, tolerance: float) -> np.ndarray:
    order = np.argsort(values)
    labels = np.zeros(len(values), dtype=np.int64)
    current, members = 0, []
    for index in order:
        if members and abs(float(values[index]) - float(np.mean(values[members]))) > tolerance:
            current += 1
            members = []
        members.append(int(index))
        labels[index] = current
    return labels


def _box_edges(origin: np.ndarray, vectors: np.ndarray, kinds: tuple[str, str, str], other: str) -> list[tuple[np.ndarray, np.ndarray, str]]:
    edges = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                start = origin + i * vectors[0] + j * vectors[1] + k * vectors[2]
                for axis, step in enumerate((i, j, k)):
                    if step:
                        continue
                    kind = kinds[axis] if (i, j, k) == (0, 0, 0) else other
                    edges.append((start, start + vectors[axis], kind))
    return edges


# 中文：显示用场景。/ English: Scene prepared for display.
@dataclass(frozen=True)
class DisplayModel:
    positions: np.ndarray
    symbols: tuple[str, ...]
    base_index: np.ndarray
    cell_shift: np.ndarray
    is_image: np.ndarray
    bonds: np.ndarray
    lattice: np.ndarray
    normal_axis: int
    normal: np.ndarray
    heights: np.ndarray
    layer_index: np.ndarray
    layer_count: int
    cell_edges: tuple[tuple[np.ndarray, np.ndarray, str], ...]
    full_cell_edges: tuple[tuple[np.ndarray, np.ndarray, str], ...]
    repeat: tuple[int, int]
    formula: str
    lattice_abc: tuple[float, float, float]
    lattice_angles: tuple[float, float, float]
    thickness_A: float
    vacuum_A: float
    base_count: int
    bonds_skipped: bool = False

    @property
    def center(self) -> np.ndarray:
        visible = self.positions[~self.is_image] if np.any(~self.is_image) else self.positions
        return (visible.min(axis=0) + visible.max(axis=0)) / 2.0


def _base_bonds(positions: np.ndarray, symbols: tuple[str, ...], lattice: np.ndarray) -> list[tuple[int, int, tuple[int, int, int]]]:
    n_atoms = len(positions)
    radii = np.asarray([covalent_radius(symbol) for symbol in symbols])
    cutoff = np.minimum(BOND_SCALE * (radii[:, None] + radii[None, :]) + BOND_PADDING_A, BOND_MAX_A)
    ranges = [np.arange(-int(math.ceil(BOND_MAX_A / h)), int(math.ceil(BOND_MAX_A / h)) + 1) for h in _axis_heights(lattice)]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = grid @ lattice
    result: list[tuple[int, int, tuple[int, int, int]]] = []
    for image, offset in zip(grid, offsets):
        delta = positions[None, :, :] + offset[None, None, :] - positions[:, None, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        mask = (distance > BOND_MIN_A) & (distance <= cutoff)
        for first, second in zip(*np.nonzero(mask)):
            result.append((int(first), int(second), (int(image[0]), int(image[1]), int(image[2]))))
    return result


# 中文：顶层接口 `build_display_model`；把结构转成显示用原子、键与晶胞边。
# English: Top-level function `build_display_model`; convert a structure into display atoms, bonds and cell edges.
def build_display_model(structure: Structure, repeat: tuple[int, int] = (1, 1), show_images: bool = True) -> DisplayModel:
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    symbols = tuple(_symbol(site) for site in structure)
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    heights_axis = _axis_heights(lattice)
    gaps = []
    for axis in range(3):
        gap, start = _largest_gap(frac[:, axis]) if len(frac) else (1.0, 0.0)
        gaps.append((gap * heights_axis[axis], start))
    normal_axis = int(np.argmax([value for value, _start in gaps]))
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    normal = np.cross(lattice[plane_axes[0]], lattice[plane_axes[1]])
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    if float(np.dot(normal, lattice[normal_axis])) < 0:
        normal = -normal

    display_frac = frac.copy()
    for axis in plane_axes:
        display_frac[:, axis] = np.mod(display_frac[:, axis], 1.0)
    start = gaps[normal_axis][1]
    along = frac[:, normal_axis] - start
    along = along - np.floor(along + 1e-9) + start
    if len(along):
        along -= math.floor(float(np.mean(along)))
    display_frac[:, normal_axis] = along
    base_positions = display_frac @ lattice

    heights_all = base_positions @ normal
    base_heights = heights_all - (heights_all.min() if len(heights_all) else 0.0)
    base_layers = _cluster(base_heights, LAYER_TOLERANCE_A) if len(base_heights) else np.zeros(0, dtype=np.int64)
    thickness = float(base_heights.max()) if len(base_heights) else 0.0
    cell_height = float(abs(np.dot(lattice[normal_axis], normal)))

    n1, n2 = max(1, int(repeat[0])), max(1, int(repeat[1]))
    index_of: dict[tuple[int, int, int, int], int] = {}
    positions, owners, shifts, images = [], [], [], []

    def add(owner: int, shift: tuple[int, int, int], image: bool) -> int:
        key = (owner, *shift)
        if key not in index_of:
            index_of[key] = len(positions)
            positions.append(base_positions[owner] + np.asarray(shift, dtype=np.float64) @ lattice)
            owners.append(owner)
            shifts.append(shift)
            images.append(image)
        return index_of[key]

    for i in range(n1):
        for j in range(n2):
            shift = [0, 0, 0]
            shift[plane_axes[0]], shift[plane_axes[1]] = i, j
            for owner in range(len(symbols)):
                add(owner, tuple(shift), False)

    bonds: set[tuple[int, int]] = set()
    skipped = len(symbols) > MAX_BOND_ATOMS
    if not skipped and len(symbols):
        base_bonds = _base_bonds(base_positions, symbols, lattice)
        for key, first_index in list(index_of.items()):
            owner, shift = key[0], np.asarray(key[1:], dtype=np.int64)
            for first, second, image in base_bonds:
                if first != owner:
                    continue
                target = tuple(int(value) for value in shift + np.asarray(image))
                inside = target[normal_axis] == 0 and 0 <= target[plane_axes[0]] < n1 and 0 <= target[plane_axes[1]] < n2
                if inside:
                    second_index = add(second, target, False)
                elif show_images:
                    second_index = add(second, target, True)
                else:
                    continue
                bonds.add((min(first_index, second_index), max(first_index, second_index)))

    full_edges = _box_edges(np.zeros(3), lattice, ("a", "b", "c"), "edge")
    # 中文：默认晶胞框只包住片层（上下各留 SLAB_BOX_PADDING_A），避免 30 Å 真空把片层压成一条线。
    # English: the default cell box only encloses the slab (padded), so a 30 A vacuum does not flatten the view.
    normal_component = float(np.dot(lattice[normal_axis], normal))
    bottom_level = float(heights_all.min()) - SLAB_BOX_PADDING_A if len(heights_all) else 0.0
    top_level = float(heights_all.max()) + SLAB_BOX_PADDING_A if len(heights_all) else normal_component
    prism = lattice.copy()
    prism[normal_axis] = lattice[normal_axis] * ((top_level - bottom_level) / normal_component)
    prism_origin = lattice[normal_axis] * (bottom_level / normal_component)
    edges = _box_edges(prism_origin, prism, ("a", "b", "c"), "edge")
    if n1 > 1 or n2 > 1:
        for box, origin, target in ((lattice, np.zeros(3), full_edges), (prism, prism_origin, edges)):
            block = box.copy()
            block[plane_axes[0]] *= n1
            block[plane_axes[1]] *= n2
            target += [(s, e, "super") for s, e, _kind in _box_edges(origin, block, ("super",) * 3, "super")]

    owners_array = np.asarray(owners, dtype=np.int64)
    abc = tuple(float(value) for value in structure.lattice.abc)
    angles = tuple(float(value) for value in structure.lattice.angles)
    return DisplayModel(
        positions=np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        symbols=tuple(symbols[owner] for owner in owners),
        base_index=owners_array,
        cell_shift=np.asarray(shifts, dtype=np.int64).reshape(-1, 3),
        is_image=np.asarray(images, dtype=bool),
        bonds=np.asarray(sorted(bonds), dtype=np.int64).reshape(-1, 2),
        lattice=lattice,
        normal_axis=normal_axis,
        normal=normal,
        heights=base_heights[owners_array] if len(owners_array) else np.zeros(0),
        layer_index=base_layers[owners_array] if len(owners_array) else np.zeros(0, dtype=np.int64),
        layer_count=int(base_layers.max() + 1) if len(base_layers) else 0,
        cell_edges=tuple(edges),
        full_cell_edges=tuple(full_edges),
        repeat=(n1, n2),
        formula=structure.composition.reduced_formula if len(structure) else "",
        lattice_abc=abc,
        lattice_angles=angles,
        thickness_A=thickness,
        vacuum_A=max(cell_height - thickness, 0.0),
        base_count=len(symbols),
        bonds_skipped=skipped,
    )


# 中文：兼容 1.x 接口的场景。/ English: Scene kept for the 1.x interface.
@dataclass(frozen=True)
class StructureScene:
    positions: np.ndarray
    symbols: tuple[str, ...]
    colors: tuple[str, ...]
    marker_sizes: np.ndarray
    bonds: tuple[tuple[np.ndarray, np.ndarray], ...]
    ghost_positions: np.ndarray
    ghost_symbols: tuple[str, ...]
    ghost_colors: tuple[str, ...]
    ghost_marker_sizes: np.ndarray
    cell_segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    atom_limits: tuple[np.ndarray, np.ndarray]
    full_limits: tuple[np.ndarray, np.ndarray]


def _padded_limits(points: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    minimum, maximum = np.min(points, axis=0), np.max(points, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = np.maximum(padding, span * 0.08)
    return minimum - margin, maximum + margin


# 中文：顶层接口 `build_structure_scene`；旧接口，供自检与脚本使用。
# English: Top-level function `build_structure_scene`; legacy interface used by the self-test and scripts.
def build_structure_scene(structure: Structure) -> StructureScene:
    model = build_display_model(structure, (1, 1), True)
    base = ~model.is_image
    sizes = np.square(np.clip([covalent_radius(symbol) for symbol in model.symbols], 0.35, 1.9) * 12.0)
    bonds = tuple((model.positions[i].copy(), model.positions[j].copy()) for i, j in model.bonds)
    cell = tuple((start, end) for start, end, kind in model.full_cell_edges if kind != "super")
    corners = np.asarray([point for segment in cell for point in segment])
    bond_points = np.asarray([point for segment in bonds for point in segment]).reshape(-1, 3)
    atoms = model.positions[base]
    return StructureScene(
        positions=atoms,
        symbols=tuple(symbol for symbol, keep in zip(model.symbols, base) if keep),
        colors=tuple(element_color(symbol) for symbol, keep in zip(model.symbols, base) if keep),
        marker_sizes=np.asarray(sizes)[base],
        bonds=bonds,
        ghost_positions=model.positions[model.is_image].reshape(-1, 3),
        ghost_symbols=tuple(symbol for symbol, image in zip(model.symbols, model.is_image) if image),
        ghost_colors=tuple(element_color(symbol) for symbol, image in zip(model.symbols, model.is_image) if image),
        ghost_marker_sizes=np.asarray(sizes)[model.is_image] * 0.68,
        cell_segments=cell,
        atom_limits=_padded_limits(np.vstack([atoms, bond_points]) if len(bond_points) else atoms, 1.2),
        full_limits=_padded_limits(np.vstack([atoms, corners]), 0.5),
    )


__all__ = [
    "DisplayModel",
    "ELEMENT_COLORS",
    "StructureScene",
    "build_display_model",
    "build_structure_scene",
    "covalent_radius",
    "element_color",
    "hex_to_rgb",
]
