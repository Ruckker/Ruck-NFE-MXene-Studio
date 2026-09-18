# ==============================================================================
# 中文概述：三维预览的软件渲染器（NumPy + Pillow）：相机与视角预设、带高光的球体与圆柱键、按深度遮挡、
#           深度雾化、选中高亮、距离/键角/二面角测量线、坐标轴指示与信息框、原子拾取。与 Tk 无关。
# English overview: Software renderer of the 3D preview (NumPy + Pillow): camera and view presets, shaded
#           spheres and cylinder bonds, depth-sorted occlusion, depth fog, selection halos, distance/angle/
#           dihedral measurements, axis triad and info box, and atom picking. Independent of Tk.
#
# 中文输入：DisplayModel、Camera、RenderOptions 与输出尺寸。
# English inputs: DisplayModel, Camera, RenderOptions and the output size.
# 中文输出：RenderResult（PIL 图像与每个原子的屏幕位置、半径、深度）。
# English outputs: RenderResult (PIL image plus per-atom screen position, radius and depth).
#
# 关键约束 / Key invariants:
# - 相机坐标：x 向右、y 向上、z 指向观察者；深度越大越靠近观察者。
#   Camera frame: x right, y up, z toward the viewer; larger depth is nearer.
# - "final" 质量 2 倍超采样后缩小，拖动时用 "draft" 质量保证流畅。
#   "final" quality renders at 2x and downsamples; "draft" keeps dragging fluid.
# - 主要接口 / Main APIs: Camera, RenderOptions, render_scene, view_rotation, fit_camera, pick_atom,
#   describe_atom, measurement_lines
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .structure_scene import DisplayModel, covalent_radius, element_color, hex_to_rgb

THEMES = {
    "light": {"top": (249, 251, 254), "bottom": (223, 230, 239), "cell": (120, 134, 152), "text": (30, 40, 52), "panel": (255, 255, 255), "halo": (255, 166, 0), "measure": (18, 104, 196)},
    "dark": {"top": (40, 47, 60), "bottom": (15, 18, 25), "cell": (150, 163, 180), "text": (232, 237, 244), "panel": (26, 31, 41), "halo": (255, 196, 64), "measure": (110, 188, 255)},
    "white": {"top": (255, 255, 255), "bottom": (255, 255, 255), "cell": (118, 126, 138), "text": (22, 26, 32), "panel": (255, 255, 255), "halo": (255, 150, 0), "measure": (0, 100, 200)},
}
AXIS_COLORS = {"a": (214, 66, 54), "b": (40, 156, 72), "c": (48, 108, 226)}
LAYER_COLORS = ("#E2453C", "#5AAFE8", "#56606B", "#9C7CC6", "#EDBE2F", "#2EB88A", "#F08A24", "#4E92E6")
MODES = ("ball", "space", "stick", "wire")
COLOR_SCHEMES = ("element", "layer", "height")
LIGHT = np.asarray([-0.42, 0.58, 0.70]) / np.linalg.norm([-0.42, 0.58, 0.70])
HALF = (LIGHT + np.asarray([0.0, 0.0, 1.0])) / np.linalg.norm(LIGHT + np.asarray([0.0, 0.0, 1.0]))


@dataclass
class Camera:
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 40.0
    pan: np.ndarray = field(default_factory=lambda: np.zeros(2))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    perspective: bool = False

    def copy(self) -> "Camera":
        return Camera(self.rotation.copy(), float(self.scale), self.pan.copy(), self.center.copy(), self.perspective)


@dataclass
class RenderOptions:
    mode: str = "ball"
    color_scheme: str = "element"
    theme: str = "light"
    atom_scale: float = 1.0
    bond_scale: float = 1.0
    show_cell: bool = True
    full_cell: bool = False
    show_labels: bool = False
    show_info: bool = True
    show_axes: bool = True
    fog: bool = True
    highlight_symbol: Optional[str] = None
    selection: tuple[int, ...] = ()
    title: str = ""
    empty_message: str = "拖入或选择 CIF / POSCAR 后在此显示三维结构"


@dataclass
class RenderResult:
    image: Image.Image
    screen_xy: np.ndarray
    screen_r: np.ndarray
    depth: np.ndarray


def axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=np.float64) / max(float(np.linalg.norm(axis)), 1e-12)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray([[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def look_rotation(toward_viewer: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = np.asarray(toward_viewer, dtype=np.float64)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    x = np.cross(up, z)
    if float(np.linalg.norm(x)) < 1e-8:
        x = np.cross([1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0], z)
    x = x / float(np.linalg.norm(x))
    return np.stack([x, np.cross(z, x), z])


# 中文：视角预设：iso 等轴测、top 俯视、side_a / side_b 沿面内晶轴侧视（法向朝上）。
# English: View presets: iso, top, and side views along the in-plane axes with the slab normal up.
def view_rotation(model: DisplayModel, name: str = "iso") -> np.ndarray:
    normal = model.normal
    plane = [axis for axis in range(3) if axis != model.normal_axis]
    first = model.lattice[plane[0]] / max(float(np.linalg.norm(model.lattice[plane[0]])), 1e-12)
    second = model.lattice[plane[1]] / max(float(np.linalg.norm(model.lattice[plane[1]])), 1e-12)
    across = np.cross(normal, first)
    if name == "top":
        return look_rotation(normal, across)
    if name == "side_a":
        return look_rotation(-first, normal)
    if name == "side_b":
        return look_rotation(-second, normal)
    azimuth, elevation = math.radians(32.0), math.radians(24.0)
    side = -first * math.cos(azimuth) - across * math.sin(azimuth)
    return look_rotation(side * math.cos(elevation) + normal * math.sin(elevation), normal)


def atom_radii(model: DisplayModel, options: RenderOptions) -> np.ndarray:
    radii = []
    for symbol, image in zip(model.symbols, model.is_image):
        rc = covalent_radius(symbol)
        if options.mode == "space":
            value = 0.92 * rc * options.atom_scale
        elif options.mode == "stick":
            value = 0.17 * options.bond_scale
        elif options.mode == "wire":
            value = 0.08
        else:
            value = min(max(0.30 * rc + 0.17, 0.27), 0.86) * options.atom_scale
        radii.append(value * (0.9 if image and options.mode in ("ball", "space") else 1.0))
    return np.asarray(radii, dtype=np.float64)


def atom_colors(model: DisplayModel, options: RenderOptions) -> list[tuple[int, int, int]]:
    if options.color_scheme == "layer":
        return [hex_to_rgb(LAYER_COLORS[int(layer) % len(LAYER_COLORS)]) for layer in model.layer_index]
    if options.color_scheme == "height":
        span = max(float(model.heights.max()) if len(model.heights) else 1.0, 1e-6)
        colors = []
        for height in model.heights:
            t = float(height) / span
            colors.append(tuple(int(round(a + (b - a) * t)) for a, b in zip((58, 96, 206), (232, 84, 60))))
        return colors
    return [hex_to_rgb(element_color(symbol)) for symbol in model.symbols]


def cell_edges_for(model: DisplayModel, options: RenderOptions) -> tuple:
    return model.full_cell_edges if options.full_cell else model.cell_edges


def fit_camera(model: DisplayModel, camera: Camera, width: int, height: int, options: RenderOptions, margin: float = 28.0) -> None:
    if model is None or not len(model.positions):
        return
    camera.center = model.center.copy()
    points = [model.positions]
    radii = [atom_radii(model, options)]
    if options.show_cell:
        corners = np.asarray([point for start, end, _kind in cell_edges_for(model, options) for point in (start, end)])
        points.append(corners)
        radii.append(np.zeros(len(corners)))
    cam = (np.vstack(points) - camera.center) @ camera.rotation.T
    r = np.concatenate(radii)
    x0, x1 = float(np.min(cam[:, 0] - r)), float(np.max(cam[:, 0] + r))
    y0, y1 = float(np.min(cam[:, 1] - r)), float(np.max(cam[:, 1] + r))
    top = margin + (84.0 if options.show_info and height > 260 else 0.0)
    usable_w, usable_h = max(width - 2 * margin, 40.0), max(height - top - margin, 40.0)
    camera.scale = float(np.clip(min(usable_w / max(x1 - x0, 0.5), usable_h / max(y1 - y0, 0.5)), 2.0, 600.0))
    # 屏幕中心相对可用区域中心的偏移（px，向上为正）。/ Offset of the usable area's centre from the screen centre (px, up positive).
    shift_up = -(top - margin) / 2.0
    camera.pan = np.asarray([-(x0 + x1) / 2.0 * camera.scale, -(y0 + y1) / 2.0 * camera.scale + shift_up])


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    fonts = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
    names = ("msyhbd.ttc", "segoeuib.ttf", "arialbd.ttf") if bold else ("msyh.ttc", "segoeui.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(os.path.join(fonts, name), size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


@lru_cache(maxsize=8)
def _background(width: int, height: int, theme: str) -> Image.Image:
    top, bottom = np.asarray(THEMES[theme]["top"], np.float32), np.asarray(THEMES[theme]["bottom"], np.float32)
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    column = top[None, None, :] * (1 - t) + bottom[None, None, :] * t
    return Image.fromarray(np.repeat(column, width, axis=1).astype(np.uint8))


@lru_cache(maxsize=1024)
def _sphere(color: tuple[int, int, int], radius: int, fog: int, alpha: int, background: tuple[int, int, int], outline: bool) -> Image.Image:
    size = 2 * radius + 4
    center = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    nx, ny = (x - center) / radius, -(y - center) / radius
    distance = np.sqrt(nx * nx + ny * ny)
    nz = np.sqrt(np.clip(1.0 - distance * distance, 0.0, 1.0))
    diffuse = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0.0, 1.0)
    specular = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0.0, 1.0) ** 42
    rgb = np.asarray(color, np.float32)[None, None, :] * (0.26 + 0.80 * diffuse)[..., None]
    rgb = rgb * (0.78 + 0.22 * np.clip(nz * 1.5, 0.0, 1.0))[..., None] + 255.0 * 0.42 * specular[..., None]
    rgb = rgb * (1 - fog / 10.0) + np.asarray(background, np.float32) * (fog / 10.0)
    if outline:
        rgb = rgb * (0.58 + 0.42 * np.clip((1.0 - distance) * radius / 1.6, 0.0, 1.0))[..., None]
    coverage = np.clip((1.0 - distance) * radius + 0.5, 0.0, 1.0) * (alpha / 10.0)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), coverage * 255.0]).astype(np.uint8))


def _mix(color: Sequence[float], background: Sequence[float], amount: float) -> tuple[int, int, int]:
    return tuple(int(round(c * (1 - amount) + b * amount)) for c, b in zip(color, background))


def _cylinder(draw: ImageDraw.ImageDraw, p0: np.ndarray, p1: np.ndarray, half_width: float, color: Sequence[int], fog: float, background: Sequence[int], strips: int) -> None:
    delta = p1 - p0
    length = float(np.hypot(delta[0], delta[1]))
    if length < 0.5 or half_width < 0.3:
        return
    perpendicular = np.asarray([-delta[1], delta[0]]) / length
    light_along = perpendicular[0] * LIGHT[0] - perpendicular[1] * LIGHT[1]
    strips = max(1, strips if half_width >= 2.5 else 1)
    for k in range(strips):
        t0, t1 = -1.0 + 2.0 * k / strips, -1.0 + 2.0 * (k + 1) / strips
        tc = (t0 + t1) / 2.0
        lambert = max(0.0, tc * light_along + math.sqrt(max(0.0, 1.0 - tc * tc)) * LIGHT[2])
        shade = [min(255.0, c * (0.30 + 0.78 * lambert) + 255.0 * 0.30 * lambert ** 24) for c in color]
        fill = _mix(shade, background, fog)
        a, b = perpendicular * (t0 * half_width), perpendicular * (t1 * half_width)
        draw.polygon([tuple(p0 + a), tuple(p0 + b), tuple(p1 + b), tuple(p1 + a)], fill=fill)


def _panel(image: Image.Image, box: tuple[int, int, int, int], color: Sequence[int], opacity: int, radius: int) -> None:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=opacity)
    image.paste(Image.new("RGB", image.size, tuple(color)), (0, 0), mask)


# 中文：顶层接口 `render_scene`；按深度绘制晶胞边、键与原子，再叠加选中、测量、坐标轴与信息框。
# English: Top-level function `render_scene`; draw cell edges, bonds and atoms by depth, then selections, measurements, axes and info.
def render_scene(model: Optional[DisplayModel], camera: Camera, options: RenderOptions, width: int, height: int, quality: str = "final", pixel_ratio: float = 1.0) -> RenderResult:
    out_w, out_h = max(int(width * pixel_ratio), 16), max(int(height * pixel_ratio), 16)
    # 超大画面（高 DPI 或导出）不再额外超采样，避免单帧过慢。/ Skip supersampling for very large outputs.
    ss = 2 if quality == "final" and out_w * out_h <= 1_600_000 else 1
    factor = ss * pixel_ratio
    big_w, big_h = out_w * ss, out_h * ss
    theme = THEMES.get(options.theme, THEMES["light"])
    image = _background(big_w, big_h, options.theme if options.theme in THEMES else "light").copy()
    draw = ImageDraw.Draw(image)
    background = tuple(int((a + b) / 2) for a, b in zip(theme["top"], theme["bottom"]))
    empty = np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    if model is None or not len(model.positions):
        font = _font(int(15 * factor))
        cx, cy = big_w / 2, big_h / 2
        for k in range(6):
            angle = math.pi / 3 * k
            x0, y0 = cx + 26 * factor * math.cos(angle), cy - 34 * factor + 26 * factor * math.sin(angle)
            x1, y1 = cx + 26 * factor * math.cos(angle + math.pi / 3), cy - 34 * factor + 26 * factor * math.sin(angle + math.pi / 3)
            draw.line([(x0, y0), (x1, y1)], fill=theme["cell"], width=max(1, int(2 * factor)))
            draw.ellipse([x0 - 5 * factor, y0 - 5 * factor, x0 + 5 * factor, y0 + 5 * factor], fill=theme["cell"])
        draw.text((cx, cy + 16 * factor), options.empty_message, font=font, fill=theme["cell"], anchor="mm")
        final = image.resize((out_w, out_h), Image.LANCZOS) if ss > 1 else image
        return RenderResult(final, *empty)

    cam = (model.positions - camera.center) @ camera.rotation.T
    extent = max(float(np.ptp(cam, axis=0).max()), 1.0)
    distance = extent * 2.6 + 12.0
    perspective = (distance / np.maximum(distance - cam[:, 2], 1e-3)) if camera.perspective else np.ones(len(cam))
    px = camera.scale * factor
    sx = big_w / 2 + camera.pan[0] * factor + cam[:, 0] * px * perspective
    sy = big_h / 2 - camera.pan[1] * factor - cam[:, 1] * px * perspective
    radii = atom_radii(model, options)
    sr = radii * px * perspective
    depth = cam[:, 2]
    z0, z1 = float(depth.min()), float(depth.max())
    fog_t = (z1 - depth) / (z1 - z0) if z1 - z0 > 1e-6 else np.zeros(len(depth))
    fog_amount = 0.40 * fog_t if options.fog else np.zeros(len(depth))
    colors = atom_colors(model, options)
    strips = 5 if quality == "final" else 3

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = (points - camera.center) @ camera.rotation.T
        f = (distance / np.maximum(distance - c[:, 2], 1e-3)) if camera.perspective else np.ones(len(c))
        return np.stack([big_w / 2 + camera.pan[0] * factor + c[:, 0] * px * f, big_h / 2 - camera.pan[1] * factor - c[:, 1] * px * f], axis=1), c[:, 2]

    items: list[tuple[float, int, object]] = [(float(depth[i]), 2, i) for i in range(len(depth))]
    if options.mode != "space":
        for first, second in model.bonds:
            pa, pb = model.positions[first], model.positions[second]
            span = float(np.linalg.norm(pb - pa))
            if span < 1e-6:
                continue
            unit = (pb - pa) / span
            middle = (pa + pb) / 2.0
            for atom, start in ((first, pa + unit * radii[first] * 0.8), (second, pb - unit * radii[second] * 0.8)):
                screen, z = project(np.stack([start, middle]))
                items.append((float(z.mean()) - 1e-4, 1, (atom, screen[0], screen[1], float(z.mean()))))
    if options.show_cell:
        for start, end, kind in cell_edges_for(model, options):
            pieces = np.linspace(0.0, 1.0, 9)
            points = start[None, :] + (end - start)[None, :] * pieces[:, None]
            screen, z = project(points)
            for k in range(8):
                items.append((float((z[k] + z[k + 1]) / 2) - 2e-4, 0, (screen[k], screen[k + 1], kind, float((z[k] + z[k + 1]) / 2))))
    items.sort(key=lambda item: (item[0], item[1]))

    def dim_for(atom: int) -> tuple[float, int]:
        fog = float(fog_amount[atom])
        alpha = 10
        if model.is_image[atom]:
            fog, alpha = max(fog, 0.45), 7
        if options.highlight_symbol and model.symbols[atom] != options.highlight_symbol:
            fog, alpha = max(fog, 0.72), 5
        return fog, alpha

    selection = [index for index in options.selection if 0 <= index < len(depth)]
    for _depth, kind, payload in items:
        if kind == 0:
            p0, p1, edge_kind, z = payload
            base = AXIS_COLORS.get(edge_kind, theme["cell"])
            fog = 0.35 * ((z1 - z) / (z1 - z0) if z1 - z0 > 1e-6 else 0.0) + (0.45 if edge_kind == "super" else 0.0)
            width_px = max(1, int(round((2.2 if edge_kind in AXIS_COLORS else 1.4) * factor)))
            draw.line([tuple(p0), tuple(p1)], fill=_mix(base, background, min(fog, 0.8)), width=width_px)
        elif kind == 1:
            atom, p0, p1, _z = payload
            fog, _alpha = dim_for(atom)
            if options.mode == "wire":
                draw.line([tuple(p0), tuple(p1)], fill=_mix(colors[atom], background, fog), width=max(1, int(round(2.4 * factor))))
            else:
                half = (0.11 if options.mode == "ball" else 0.17) * options.bond_scale * px * float(perspective[atom])
                _cylinder(draw, p0, p1, half, colors[atom], fog, background, strips)
        else:
            atom = payload
            fog, alpha = dim_for(atom)
            radius = max(2, int(round(sr[atom])))
            if options.mode == "wire":
                r = max(2.0, 2.2 * factor)
                draw.ellipse([sx[atom] - r, sy[atom] - r, sx[atom] + r, sy[atom] + r], fill=_mix(colors[atom], background, fog))
            else:
                sprite = _sphere(tuple(colors[atom]), radius, int(round(fog * 10)), alpha, background, options.mode != "stick")
                image.paste(sprite, (int(round(sx[atom] - radius - 1.5)), int(round(sy[atom] - radius - 1.5))), sprite)
            if atom in selection:
                ring = sr[atom] + 3.5 * factor
                draw.ellipse([sx[atom] - ring, sy[atom] - ring, sx[atom] + ring, sy[atom] + ring], outline=theme["halo"], width=max(2, int(round(2.6 * factor))))

    if options.show_labels:
        font_cache: dict[int, ImageFont.ImageFont] = {}
        for atom in np.argsort(depth):
            if model.is_image[atom]:
                continue
            size = int(np.clip(sr[atom] * 0.95, 9 * factor, 22 * factor))
            font = font_cache.setdefault(size, _font(size, True))
            luminance = 0.299 * colors[atom][0] + 0.587 * colors[atom][1] + 0.114 * colors[atom][2]
            ink, stroke = ((22, 28, 36), (255, 255, 255)) if luminance > 150 else ((255, 255, 255), (20, 24, 30))
            draw.text((sx[atom], sy[atom]), model.symbols[atom], font=font, fill=ink, anchor="mm", stroke_width=max(1, int(factor)), stroke_fill=stroke)

    if len(selection) >= 2:
        label_font = _font(int(12 * factor), True)
        for first, second in zip(selection, selection[1:]):
            a, b = np.asarray([sx[first], sy[first]]), np.asarray([sx[second], sy[second]])
            length = float(np.linalg.norm(b - a))
            dashes = max(int(length / (7 * factor)), 1)
            for k in range(0, dashes, 2):
                draw.line([tuple(a + (b - a) * k / dashes), tuple(a + (b - a) * min(k + 1, dashes) / dashes)], fill=theme["measure"], width=max(2, int(round(2 * factor))))
            text = f"{np.linalg.norm(model.positions[second] - model.positions[first]):.3f} Å"
            mid = (a + b) / 2.0
            box = draw.textbbox((mid[0], mid[1]), text, font=label_font, anchor="mm")
            _panel(image, (box[0] - 5 * factor, box[1] - 3 * factor, box[2] + 5 * factor, box[3] + 3 * factor), theme["panel"], 225, int(6 * factor))
            draw = ImageDraw.Draw(image)
            draw.text((mid[0], mid[1]), text, font=label_font, fill=theme["measure"], anchor="mm")
        for k, atom in enumerate(selection, start=1):
            bx, by, br = sx[atom] + sr[atom] * 0.72, sy[atom] - sr[atom] * 0.72, 8 * factor
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=theme["halo"])
            draw.text((bx, by), str(k), font=_font(int(10 * factor), True), fill=(255, 255, 255), anchor="mm")

    if options.show_axes:
        origin = np.asarray([54 * factor, big_h - 50 * factor])
        vectors = [(name, model.lattice[index] / max(float(np.linalg.norm(model.lattice[index])), 1e-12)) for index, name in enumerate("abc")]
        oriented = sorted(((float((camera.rotation @ vector)[2]), name, camera.rotation @ vector) for name, vector in vectors))
        axis_font = _font(int(12 * factor), True)
        for _z, name, vector in oriented:
            tip = origin + np.asarray([vector[0], -vector[1]]) * 28 * factor
            draw.line([tuple(origin), tuple(tip)], fill=AXIS_COLORS[name], width=max(2, int(round(3 * factor))))
            draw.ellipse([tip[0] - 3 * factor, tip[1] - 3 * factor, tip[0] + 3 * factor, tip[1] + 3 * factor], fill=AXIS_COLORS[name])
            label = origin + np.asarray([vector[0], -vector[1]]) * 40 * factor
            draw.text(tuple(label), name, font=axis_font, fill=AXIS_COLORS[name], anchor="mm")
        draw.ellipse([origin[0] - 3 * factor, origin[1] - 3 * factor, origin[0] + 3 * factor, origin[1] + 3 * factor], fill=theme["cell"])

    if options.show_info:
        a, b, c = model.lattice_abc
        lines = [
            (options.title or model.formula, True),
            (f"{model.formula} · {model.base_count} 原子" + (f" · 显示 {model.repeat[0]}×{model.repeat[1]}" if model.repeat != (1, 1) else ""), False),
            (f"a {a:.3f}   b {b:.3f}   c {c:.3f} Å   γ {model.lattice_angles[2]:.1f}°", False),
            (f"厚度 {model.thickness_A:.2f} Å · 真空 {model.vacuum_A:.2f} Å · {model.layer_count} 层", False),
        ]
        fonts = [_font(int(13 * factor), True)] + [_font(int(11 * factor))] * 3
        x0, y0, gap = 12 * factor, 10 * factor, 4 * factor
        boxes = [draw.textbbox((0, 0), text, font=font) for (text, _bold), font in zip(lines, fonts)]
        panel_w = max(box[2] - box[0] for box in boxes) + 20 * factor
        panel_h = sum(box[3] - box[1] + gap for box in boxes) + 14 * factor
        _panel(image, (x0, y0, x0 + panel_w, y0 + panel_h), theme["panel"], 200 if options.theme != "white" else 235, int(8 * factor))
        draw = ImageDraw.Draw(image)
        cursor = y0 + 8 * factor
        for (text, bold), font, box in zip(lines, fonts, boxes):
            draw.text((x0 + 10 * factor, cursor), text, font=font, fill=theme["text"] if bold else _mix(theme["text"], theme["panel"], 0.25))
            cursor += box[3] - box[1] + gap

    final = image.resize((out_w, out_h), Image.LANCZOS) if ss > 1 else image
    scale = ss
    return RenderResult(final, np.stack([sx, sy], axis=1) / scale / pixel_ratio, sr / scale / pixel_ratio, depth)


# 中文：返回屏幕坐标下最靠近观察者的原子。/ English: Return the nearest atom under a screen point.
def pick_atom(result: Optional[RenderResult], x: float, y: float) -> Optional[int]:
    if result is None or not len(result.depth):
        return None
    d2 = (result.screen_xy[:, 0] - x) ** 2 + (result.screen_xy[:, 1] - y) ** 2
    hits = np.nonzero(d2 <= np.maximum(result.screen_r + 2.0, 6.0) ** 2)[0]
    if not len(hits):
        return None
    return int(hits[np.argmax(result.depth[hits])])


def atom_name(model: DisplayModel, index: int) -> str:
    return f"{model.symbols[index]}{int(model.base_index[index]) + 1}"


def describe_atom(model: DisplayModel, index: int) -> str:
    x, y, z = model.positions[index]
    lines = [
        f"{atom_name(model, index)}（{model.symbols[index]}）" + ("，周期镜像" if model.is_image[index] else ""),
        f"笛卡尔坐标 ({x:.3f}, {y:.3f}, {z:.3f}) Å",
        f"距片层底面 {float(model.heights[index]):.3f} Å · 第 {int(model.layer_index[index]) + 1}/{model.layer_count} 层",
    ]
    neighbors = []
    for first, second in model.bonds:
        if index in (first, second):
            other = int(second if first == index else first)
            neighbors.append((float(np.linalg.norm(model.positions[other] - model.positions[index])), atom_name(model, other)))
    if neighbors:
        lines.append(f"配位 {len(neighbors)}：" + "，".join(f"{name} {dist:.3f} Å" for dist, name in sorted(neighbors)[:8]))
    return "\n".join(lines)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    cosine = float(np.dot(u, v) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / max(float(np.linalg.norm(b1)), 1e-12)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(float(np.dot(np.cross(b1, v), w)), float(np.dot(v, w))))


def measurement_lines(model: DisplayModel, selection: Sequence[int]) -> list[str]:
    atoms = [index for index in selection if 0 <= index < len(model.positions)]
    p = [model.positions[index] for index in atoms]
    names = [atom_name(model, index) for index in atoms]
    lines = [f"距离 {names[k]}–{names[k + 1]}：{np.linalg.norm(p[k + 1] - p[k]):.3f} Å" for k in range(len(atoms) - 1)]
    if len(atoms) >= 3:
        lines.append(f"键角 {names[0]}–{names[1]}–{names[2]}：{_angle(p[0], p[1], p[2]):.2f}°")
    if len(atoms) == 4:
        lines.append(f"二面角 {'–'.join(names)}：{_dihedral(*p):.2f}°")
    return lines


__all__ = [
    "Camera",
    "COLOR_SCHEMES",
    "MODES",
    "RenderOptions",
    "RenderResult",
    "THEMES",
    "atom_name",
    "axis_angle",
    "describe_atom",
    "fit_camera",
    "measurement_lines",
    "pick_atom",
    "render_scene",
    "view_rotation",
]
