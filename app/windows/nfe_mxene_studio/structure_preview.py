# ==============================================================================
# 中文概述：类 VESTA 的 Matplotlib 三维晶体预览，显示原子、周期键、幽灵像和晶胞。
# English overview: VESTA-like Matplotlib 3D preview for atoms, periodic bonds, ghost images, and the unit cell.
#
# 中文输入：Pymatgen Structure 与预览显示选项。
# English inputs: A Pymatgen Structure and preview display options.
# 中文输出：可旋转/缩放/重置的场景、元素图例、键和晶胞边。
# English outputs: A rotatable/zoomable/resettable scene, element legend, bonds, and cell edges.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: StructureScene, element_color, covalent_radius, _cell_segments, _padded_limits, build_structure_scene, StructurePreview3D
#
# Author: Ruck
# Generated: 2026-07-30 08:27:27 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, Structure


CPK_COLORS = {
    "H": "#FFFFFF",
    "C": "#4C4C4C",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
    "S": "#FFFF30",
    "Se": "#FFA100",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
}


# 中文：顶层类 `StructureScene`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `StructureScene`; review type hints and callers before extending it.
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


# 中文：顶层接口 `element_color`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `element_color`; review type hints and callers before extending it.
def element_color(symbol: str) -> str:
    if symbol in CPK_COLORS:
        return CPK_COLORS[symbol]
    hue = (int(Element(symbol).Z) * 0.618033988749895) % 1.0
    import colorsys

    red, green, blue = colorsys.hsv_to_rgb(hue, 0.52, 0.88)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


# 中文：顶层接口 `covalent_radius`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `covalent_radius`; review type hints and callers before extending it.
def covalent_radius(symbol: str) -> float:
    value = CovalentRadius.radius.get(symbol)
    if value is None:
        value = Element(symbol).atomic_radius
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.2


# 中文：顶层接口 `_cell_segments`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_cell_segments`; review type hints and callers before extending it.
def _cell_segments(lattice: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    vertices: dict[tuple[int, int, int], np.ndarray] = {}
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                vertices[(i, j, k)] = (
                    i * lattice[0] + j * lattice[1] + k * lattice[2]
                )
    segments = []
    for vertex, start in vertices.items():
        for axis in range(3):
            neighbor = list(vertex)
            if neighbor[axis] == 1:
                continue
            neighbor[axis] = 1
            segments.append((start, vertices[tuple(neighbor)]))
    return tuple(segments)


# 中文：顶层接口 `_padded_limits`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `_padded_limits`; review type hints and callers before extending it.
def _padded_limits(points: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    return minimum - np.maximum(padding, span * 0.08), maximum + np.maximum(
        padding, span * 0.08
    )


# 中文：顶层接口 `build_structure_scene`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `build_structure_scene`; review type hints and callers before extending it.
def build_structure_scene(structure: Structure) -> StructureScene:
    positions = np.asarray(structure.cart_coords, dtype=np.float64)
    symbols = tuple(site.specie.symbol for site in structure)
    colors = tuple(element_color(symbol) for symbol in symbols)
    radii = np.asarray([covalent_radius(symbol) for symbol in symbols])
    marker_sizes = np.square(np.clip(radii, 0.35, 1.9) * 12.0)

    bonds: list[tuple[np.ndarray, np.ndarray]] = []
    ghost_records: dict[tuple[str, float, float, float], tuple[np.ndarray, int]] = {}
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    for first in range(len(structure)):
        for second in range(first + 1, len(structure)):
            distance, image = structure[first].distance_and_image(structure[second])
            cutoff = 1.24 * (radii[first] + radii[second]) + 0.12
            if 0.25 < float(distance) <= min(float(cutoff), 3.65):
                start = positions[first]
                end = positions[second] + np.asarray(image, dtype=float) @ lattice
                bonds.append((start.copy(), end.copy()))
                if np.any(np.asarray(image, dtype=int) != 0):
                    key = (
                        symbols[second],
                        *np.round(end, decimals=5).tolist(),
                    )
                    ghost_records[key] = (end.copy(), second)
                    reverse_end = (
                        positions[first]
                        - np.asarray(image, dtype=float) @ lattice
                    )
                    bonds.append(
                        (positions[second].copy(), reverse_end.copy())
                    )
                    reverse_key = (
                        symbols[first],
                        *np.round(reverse_end, decimals=5).tolist(),
                    )
                    ghost_records[reverse_key] = (reverse_end.copy(), first)

    ghost_positions = np.asarray(
        [record[0] for record in ghost_records.values()],
        dtype=np.float64,
    )
    if not len(ghost_positions):
        ghost_positions = np.empty((0, 3), dtype=np.float64)
    ghost_indices = [record[1] for record in ghost_records.values()]
    ghost_symbols = tuple(symbols[index] for index in ghost_indices)
    ghost_colors = tuple(colors[index] for index in ghost_indices)
    ghost_marker_sizes = np.asarray(
        [marker_sizes[index] * 0.68 for index in ghost_indices],
        dtype=np.float64,
    )

    cell_segments = _cell_segments(lattice)
    cell_vertices = np.asarray(
        [point for segment in cell_segments for point in segment],
        dtype=np.float64,
    )
    bond_points = np.asarray(
        [point for segment in bonds for point in segment],
        dtype=np.float64,
    )
    atom_points = (
        np.vstack([positions, bond_points]) if bond_points.size else positions
    )
    atom_limits = _padded_limits(atom_points, 1.2)
    full_limits = _padded_limits(np.vstack([positions, cell_vertices]), 0.5)
    return StructureScene(
        positions=positions,
        symbols=symbols,
        colors=colors,
        marker_sizes=marker_sizes,
        bonds=tuple(bonds),
        ghost_positions=ghost_positions,
        ghost_symbols=ghost_symbols,
        ghost_colors=ghost_colors,
        ghost_marker_sizes=ghost_marker_sizes,
        cell_segments=cell_segments,
        atom_limits=atom_limits,
        full_limits=full_limits,
    )


# 中文：顶层类 `StructurePreview3D`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `StructurePreview3D`; review type hints and callers before extending it.
class StructurePreview3D(ttk.Frame):
    """Interactive Tk crystal viewer backed by Matplotlib's 3D canvas."""

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(0, 4))
        self.info_var = tk.StringVar(value="尚未选择结构")
        ttk.Label(controls, textvariable=self.info_var).pack(side="left")
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="元素标记",
            variable=self.show_labels_var,
            command=self.redraw,
        ).pack(side="right", padx=(8, 0))
        self.full_cell_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="完整晶胞",
            variable=self.full_cell_var,
            command=self.redraw,
        ).pack(side="right", padx=(8, 0))
        ttk.Button(
            controls,
            text="复位视角",
            command=self.reset_view,
        ).pack(side="right", padx=(8, 0))
        ttk.Label(
            controls,
            text="左键旋转 · 滚轮缩放",
            style="Hint.TLabel",
        ).pack(side="right", padx=(8, 8))

        self.figure = Figure(figsize=(7.2, 3.2), dpi=100, facecolor="#fbfcfe")
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.cjk_font = FontProperties(
            family=["Microsoft YaHei UI", "Microsoft YaHei", "SimHei"],
            size=10,
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.structure: Structure | None = None
        self.scene: StructureScene | None = None
        self.display_name = ""
        self._draw_empty("导入或生成结构后将在此显示3D模型")

    def _draw_empty(self, message: str) -> None:
        self.axes.clear()
        self.axes.set_facecolor("#fbfcfe")
        self.axes.text2D(
            0.5,
            0.5,
            message,
            transform=self.axes.transAxes,
            ha="center",
            va="center",
            color="#6b7785",
            fontproperties=self.cjk_font,
        )
        self.axes.set_axis_off()
        self.canvas.draw_idle()

    def clear(self) -> None:
        self.structure = None
        self.scene = None
        self.display_name = ""
        self.info_var.set("尚未选择结构")
        self._draw_empty("导入或生成结构后将在此显示3D模型")

    def show_error(self, message: str) -> None:
        self.structure = None
        self.scene = None
        self.info_var.set("结构预览失败")
        self._draw_empty(message)

    def set_path(self, path: str | Path) -> None:
        candidate = Path(path).expanduser().resolve()
        try:
            structure = Structure.from_file(candidate)
        except Exception as exc:
            self.show_error(f"无法读取 {candidate.name}\n{type(exc).__name__}: {exc}")
            return
        self.set_structure(structure, candidate.name)

    def set_structure(self, structure: Structure, display_name: str = "") -> None:
        self.structure = structure.copy()
        self.scene = build_structure_scene(self.structure)
        self.display_name = display_name
        formula = self.structure.composition.reduced_formula
        self.info_var.set(
            f"{display_name or formula} · {formula} · {len(self.structure)} 原子"
        )
        self.redraw(reset_view=True)

    def redraw(self, reset_view: bool = False) -> None:
        if self.structure is None or self.scene is None:
            return
        old_elev = float(self.axes.elev)
        old_azim = float(self.axes.azim)
        self.axes.clear()
        self.axes.set_facecolor("#fbfcfe")
        try:
            self.axes.set_proj_type("persp", focal_length=0.9)
        except TypeError:
            self.axes.set_proj_type("persp")

        for start, end in self.scene.cell_segments:
            self.axes.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color="#8291a3",
                linewidth=0.8,
                alpha=0.55,
                linestyle="--",
                zorder=1,
            )
        for start, end in self.scene.bonds:
            midpoint = (start + end) / 2
            self.axes.plot(
                [start[0], midpoint[0]],
                [start[1], midpoint[1]],
                [start[2], midpoint[2]],
                color="#8b96a3",
                linewidth=2.0,
                alpha=0.78,
                zorder=2,
            )
            self.axes.plot(
                [midpoint[0], end[0]],
                [midpoint[1], end[1]],
                [midpoint[2], end[2]],
                color="#65717f",
                linewidth=2.0,
                alpha=0.78,
                zorder=2,
            )
        if len(self.scene.ghost_positions):
            self.axes.scatter(
                self.scene.ghost_positions[:, 0],
                self.scene.ghost_positions[:, 1],
                self.scene.ghost_positions[:, 2],
                s=self.scene.ghost_marker_sizes,
                c=self.scene.ghost_colors,
                edgecolors="#65717f",
                linewidths=0.55,
                depthshade=True,
                alpha=0.38,
                zorder=3,
            )
        self.axes.scatter(
            self.scene.positions[:, 0],
            self.scene.positions[:, 1],
            self.scene.positions[:, 2],
            s=self.scene.marker_sizes,
            c=self.scene.colors,
            edgecolors="#26323d",
            linewidths=0.65,
            depthshade=True,
            alpha=1.0,
            zorder=5,
        )
        if self.show_labels_var.get():
            for index, (position, symbol) in enumerate(
                zip(self.scene.positions, self.scene.symbols), start=1
            ):
                self.axes.text(
                    position[0],
                    position[1],
                    position[2],
                    f" {symbol}{index}",
                    fontsize=7.5,
                    color="#18222c",
                    zorder=6,
                )

        limits = (
            self.scene.full_limits
            if self.full_cell_var.get()
            else self.scene.atom_limits
        )
        self._set_limits(*limits)
        unique_symbols = tuple(dict.fromkeys(self.scene.symbols))
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label=symbol,
                markerfacecolor=element_color(symbol),
                markeredgecolor="#26323d",
                markersize=7,
            )
            for symbol in unique_symbols
        ]
        if handles:
            self.axes.legend(
                handles=handles,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.99),
                framealpha=0.82,
                fontsize=7,
                ncols=min(4, len(handles)),
            )
        self.axes.set_xlabel("x (Å)", fontsize=8, labelpad=2)
        self.axes.set_ylabel("y (Å)", fontsize=8, labelpad=2)
        self.axes.set_zlabel("z (Å)", fontsize=8, labelpad=2)
        self.axes.tick_params(labelsize=7, pad=0)
        self.axes.grid(False)
        if reset_view:
            self.axes.view_init(elev=20, azim=38)
        else:
            self.axes.view_init(elev=old_elev, azim=old_azim)
        self.figure.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.99)
        self.canvas.draw_idle()

    def _set_limits(self, minimum: np.ndarray, maximum: np.ndarray) -> None:
        center = (minimum + maximum) / 2
        span = np.maximum(maximum - minimum, 1.0)
        half = span / 2
        self.axes.set_xlim(center[0] - half[0], center[0] + half[0])
        self.axes.set_ylim(center[1] - half[1], center[1] + half[1])
        self.axes.set_zlim(center[2] - half[2], center[2] + half[2])
        try:
            self.axes.set_box_aspect(span)
        except AttributeError:
            pass

    def reset_view(self) -> None:
        if self.scene is not None:
            self.redraw(reset_view=True)

    def _on_scroll(self, event: Any) -> None:
        if self.scene is None or event.inaxes is not self.axes:
            return
        scale = 0.84 if event.button == "up" else 1.19
        for getter, setter in (
            (self.axes.get_xlim3d, self.axes.set_xlim3d),
            (self.axes.get_ylim3d, self.axes.set_ylim3d),
            (self.axes.get_zlim3d, self.axes.set_zlim3d),
        ):
            lower, upper = getter()
            center = (lower + upper) / 2
            half = (upper - lower) * scale / 2
            setter(center - half, center + half)
        self.canvas.draw_idle()
