# ==============================================================================
# 中文概述：Windows 程序的交互式三维结构预览组件：工具栏（视角预设、显示模式、着色、超胞、晶胞/标签/镜像、
#           主题与导出）、软件渲染画布（旋转、平移、缩放、拾取、测量、自动旋转）与侧栏（元素图例、选中信息）。
# English overview: Interactive 3D structure preview widget of the Windows application: toolbar (view presets,
#           display mode, coloring, supercell, cell/labels/images, theme, export), a software-rendered canvas
#           (rotate, pan, zoom, pick, measure, auto-rotate) and a side panel (element legend, selection info).
#
# 中文输入：结构文件路径或 pymatgen Structure；鼠标与键盘操作。
# English inputs: A structure file path or pymatgen Structure; mouse and keyboard interaction.
# 中文输出：Tk 画布上的三维视图、选中原子与测量信息、导出的 PNG。
# English outputs: The 3D view on a Tk canvas, selection and measurement info, exported PNG images.
#
# 关键约束 / Key invariants:
# - 对外接口与 1.x 相同：set_path、set_structure、clear、show_error、reset_view、redraw，
#   并继续导出 build_structure_scene、element_color、covalent_radius。
#   The public interface matches 1.x and still exports build_structure_scene, element_color, covalent_radius.
# - 拖动时用 draft 质量，停止 180 ms 后自动补一帧 final 质量。
#   Dragging renders in draft quality; a final frame follows 180 ms after the last interaction.
# - 主要接口 / Main APIs: StructurePreview3D
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk
from pymatgen.core import Structure

from .structure_render import (
    THEMES,
    Camera,
    RenderOptions,
    axis_angle,
    describe_atom,
    fit_camera,
    measurement_lines,
    pick_atom,
    render_scene,
    view_rotation,
)
from .structure_scene import (
    StructureScene,
    build_display_model,
    build_structure_scene,
    covalent_radius,
    element_color,
)

MODE_NAMES = {"球棍": "ball", "空间填充": "space", "棍状": "stick", "线框": "wire"}
COLOR_NAMES = {"元素": "element", "按层": "layer", "按高度": "height"}
THEME_NAMES = {"浅色": "light", "深色": "dark", "白底": "white"}
HINT_TEXT = (
    "左键拖动旋转 · Shift+左键绕视线转 · 右键或中键拖动平移 · 滚轮缩放\n"
    "单击原子查看信息，依次单击 2–4 个原子测距离、键角、二面角 · Ctrl+单击追加\n"
    "双击原子居中 · 双击空白复位 · 右键菜单 · 快捷键 1–4 视角、R 复位、F 适配、L 标签、空格自动旋转、Esc 清除"
)
ROTATE_RAD_PER_PX = 0.0085
MAX_SELECTION = 4


# 中文：交互式三维结构预览。/ English: Interactive 3D structure preview.
class StructurePreview3D(ttk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.structure: Optional[Structure] = None
        self.model = None
        self.display_name = ""
        self.camera = Camera()
        self.result = None
        self.selection: list[int] = []
        self.highlight_symbol: Optional[str] = None
        self.last_export: Optional[Path] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._draft_job: Optional[str] = None
        self._final_job: Optional[str] = None
        self._spin_job: Optional[str] = None
        self._hover_job: Optional[str] = None
        self._drag: Optional[dict[str, Any]] = None
        self._size = (0, 0)
        self._needs_fit = True
        self._message = "拖入或选择 CIF / POSCAR 后在此显示三维结构"

        self.info_var = tk.StringVar(value="尚未选择结构")
        self.mode_var = tk.StringVar(value="球棍")
        self.color_var = tk.StringVar(value="元素")
        self.theme_var = tk.StringVar(value="浅色")
        self.repeat_a_var = tk.StringVar(value="3")
        self.repeat_b_var = tk.StringVar(value="3")
        self.cell_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=False)
        self.images_var = tk.BooleanVar(value=False)
        self.full_cell_var = tk.BooleanVar(value=False)
        self.perspective_var = tk.BooleanVar(value=False)
        self.fog_var = tk.BooleanVar(value=True)
        self.spin_var = tk.BooleanVar(value=False)
        self.overlay_var = tk.BooleanVar(value=True)
        self.axes_var = tk.BooleanVar(value=True)
        self.panel_var = tk.BooleanVar(value=True)
        self.atom_scale = 1.0
        self.bond_scale = 1.0

        self._build_toolbar()
        self._build_body()
        self._build_context_menu()
        self._update_panel()

    # ------------------------------------------------------------------ layout
    def _build_toolbar(self) -> None:
        self.toolbar = ttk.Frame(self)
        self.toolbar.pack(fill="x", pady=(0, 4))
        primary = self.toolbar_primary = ttk.Frame(self.toolbar)
        secondary = self.toolbar_secondary = ttk.Frame(self.toolbar)
        for text, name in (("等轴测", "iso"), ("俯视", "top"), ("侧视 a", "side_a"), ("侧视 b", "side_b")):
            ttk.Button(primary, text=text, width=6, style="Toolbutton", command=lambda n=name: self.set_view(n)).pack(side="left")
        ttk.Separator(primary, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(primary, text="复位", width=4, style="Toolbutton", command=self.reset_view).pack(side="left")
        ttk.Button(primary, text="适配", width=4, style="Toolbutton", command=self.fit_view).pack(side="left")
        ttk.Separator(primary, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(primary, text="模式").pack(side="left")
        mode = ttk.Combobox(primary, textvariable=self.mode_var, values=list(MODE_NAMES), state="readonly", width=7)
        mode.pack(side="left", padx=(3, 8))
        mode.bind("<<ComboboxSelected>>", lambda _e: self._option_changed(refit=True))
        ttk.Label(primary, text="着色").pack(side="left")
        color = ttk.Combobox(primary, textvariable=self.color_var, values=list(COLOR_NAMES), state="readonly", width=6)
        color.pack(side="left", padx=(3, 0))
        color.bind("<<ComboboxSelected>>", lambda _e: self._option_changed())
        ttk.Label(secondary, text="超胞").pack(side="left")
        for variable in (self.repeat_a_var, self.repeat_b_var):
            spin = ttk.Spinbox(secondary, from_=1, to=6, width=2, textvariable=variable, command=self._repeat_changed, state="readonly")
            spin.pack(side="left", padx=(3, 0))
            if variable is self.repeat_a_var:
                ttk.Label(secondary, text="×").pack(side="left", padx=(2, 0))
        ttk.Separator(secondary, orient="vertical").pack(side="left", fill="y", padx=6)
        for text, variable, command in (
            ("晶胞", self.cell_var, lambda: self._option_changed(refit=True)),
            ("标签", self.show_labels_var, self._option_changed),
            ("镜像", self.images_var, self._repeat_changed),
        ):
            ttk.Checkbutton(secondary, text=text, variable=variable, command=command).pack(side="left", padx=(0, 6))
        self.display_menu_button = ttk.Menubutton(secondary, text="显示 ▾")
        self.display_menu_button.pack(side="left", padx=(2, 6))
        self.display_menu_button["menu"] = self._build_display_menu(self.display_menu_button)
        ttk.Button(secondary, text="导出图片", command=self.export_image).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(secondary, text="面板", variable=self.panel_var, command=self._toggle_panel, style="Toolbutton").pack(side="left")
        self._wide_toolbar: Optional[bool] = None
        self._panel_user_set = False
        self._layout_toolbar(True)
        self.bind("<Configure>", self._on_widget_resize)

    def _layout_toolbar(self, wide: bool) -> None:
        if wide == self._wide_toolbar:
            return
        self._wide_toolbar = wide
        self.toolbar_primary.grid(row=0, column=0, sticky="w")
        if wide:
            self.toolbar_secondary.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=0)
        else:
            self.toolbar_secondary.grid(row=1, column=0, sticky="w", padx=0, pady=(4, 0))

    # 中文：窄面板（例如生成页右侧）自动把工具栏折成两行并收起侧栏。
    # English: Narrow hosts (such as the generation tab) wrap the toolbar onto two rows and hide the side panel.
    def _on_widget_resize(self, event: Any) -> None:
        if event.widget is not self:
            return
        width = int(event.width)
        needed = self.toolbar_primary.winfo_reqwidth() + self.toolbar_secondary.winfo_reqwidth() + 16
        self._layout_toolbar(width >= needed)
        if not self._panel_user_set:
            show = width >= 960
            if show != bool(self.panel_var.get()):
                self.panel_var.set(show)
                self._apply_panel()

    def _build_display_menu(self, master: tk.Widget) -> tk.Menu:
        menu = tk.Menu(master, tearoff=False)
        for text, variable, command in (
            ("完整晶胞（含真空）", self.full_cell_var, lambda: self._option_changed(refit=True)),
            ("透视投影", self.perspective_var, self._option_changed),
            ("深度雾化", self.fog_var, self._option_changed),
            ("自动旋转", self.spin_var, self._toggle_spin),
            ("信息框", self.overlay_var, self._option_changed),
            ("坐标轴", self.axes_var, self._option_changed),
        ):
            menu.add_checkbutton(label=text, variable=variable, command=command)
        menu.add_separator()
        for name in THEME_NAMES:
            menu.add_radiobutton(label=f"{name}背景", value=name, variable=self.theme_var, command=self._option_changed)
        menu.add_separator()
        menu.add_command(label="原子放大", accelerator="Ctrl+=", command=lambda: self.scale_atoms(1.15))
        menu.add_command(label="原子缩小", accelerator="Ctrl+-", command=lambda: self.scale_atoms(1 / 1.15))
        menu.add_command(label="键加粗", command=lambda: self.scale_bonds(1.2))
        menu.add_command(label="键变细", command=lambda: self.scale_bonds(1 / 1.2))
        menu.add_command(label="恢复大小", command=self.reset_sizes)
        return menu

    def _build_body(self) -> None:
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, highlightthickness=0, bd=0, takefocus=1, background=self._theme_hex("top"), cursor="fleur")
        self.canvas.pack(side="left", fill="both", expand=True)
        self._image_item = self.canvas.create_image(0, 0, anchor="nw")
        self._status_item = self.canvas.create_text(0, 0, anchor="se", text="", font=("Microsoft YaHei UI", 9), fill="#445566")

        self.panel = ttk.Frame(body, width=250, padding=(8, 0, 0, 0))
        self.panel.pack(side="right", fill="y", before=self.canvas)
        self.panel.pack_propagate(False)
        ttk.Label(self.panel, text="元素 · 点击高亮", font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w")
        self.legend = ttk.Frame(self.panel)
        self.legend.pack(fill="x", pady=(2, 6))
        ttk.Label(self.panel, text="选中与测量", font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w")
        self.info_text = tk.Text(self.panel, height=7, wrap="word", relief="flat", font=("Microsoft YaHei UI", 9), background="#f6f8fb", padx=6, pady=4)
        self.info_text.pack(fill="both", expand=True, pady=(2, 4))
        row = ttk.Frame(self.panel)
        row.pack(fill="x")
        ttk.Button(row, text="清除选择", command=self.clear_selection).pack(side="left")
        ttk.Button(row, text="复制", command=self.copy_info).pack(side="left", padx=(6, 0))
        ttk.Label(self.panel, text=HINT_TEXT, style="Hint.TLabel", wraplength=238, justify="left").pack(anchor="w", pady=(6, 0))

        canvas = self.canvas
        canvas.bind("<Configure>", self._on_resize)
        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<Double-Button-1>", self._on_double)
        for button in ("2", "3"):
            canvas.bind(f"<ButtonPress-{button}>", self._on_press)
            canvas.bind(f"<B{button}-Motion>", self._on_drag)
            canvas.bind(f"<ButtonRelease-{button}>", self._on_release)
        canvas.bind("<MouseWheel>", self._on_wheel)
        canvas.bind("<Motion>", self._on_hover)
        canvas.bind("<Leave>", lambda _e: self._set_status(""))
        for key, handler in (
            ("<Left>", lambda e: self.rotate(0.0, -math.radians(10))),
            ("<Right>", lambda e: self.rotate(0.0, math.radians(10))),
            ("<Up>", lambda e: self.rotate(-math.radians(10), 0.0)),
            ("<Down>", lambda e: self.rotate(math.radians(10), 0.0)),
            ("<plus>", lambda e: self.zoom(1.15)),
            ("<equal>", lambda e: self.zoom(1.15)),
            ("<KP_Add>", lambda e: self.zoom(1.15)),
            ("<minus>", lambda e: self.zoom(1 / 1.15)),
            ("<KP_Subtract>", lambda e: self.zoom(1 / 1.15)),
            ("<Control-equal>", lambda e: self.scale_atoms(1.15)),
            ("<Control-minus>", lambda e: self.scale_atoms(1 / 1.15)),
            ("<Key-r>", lambda e: self.reset_view()),
            ("<Key-f>", lambda e: self.fit_view()),
            ("<Key-l>", lambda e: self._toggle_var(self.show_labels_var)),
            ("<space>", lambda e: self._toggle_var(self.spin_var, self._toggle_spin)),
            ("<Escape>", lambda e: self.clear_selection()),
            ("<Key-1>", lambda e: self.set_view("iso")),
            ("<Key-2>", lambda e: self.set_view("top")),
            ("<Key-3>", lambda e: self.set_view("side_a")),
            ("<Key-4>", lambda e: self.set_view("side_b")),
        ):
            canvas.bind(key, handler)

    def _build_context_menu(self) -> None:
        menu = tk.Menu(self, tearoff=False)
        for text, name in (("等轴测", "iso"), ("俯视", "top"), ("侧视 a", "side_a"), ("侧视 b", "side_b")):
            menu.add_command(label=f"视角：{text}", command=lambda n=name: self.set_view(n))
        menu.add_command(label="复位视角", command=self.reset_view)
        menu.add_command(label="适配窗口", command=self.fit_view)
        menu.add_separator()
        menu.add_checkbutton(label="元素标签", variable=self.show_labels_var, command=self._option_changed)
        menu.add_checkbutton(label="晶胞", variable=self.cell_var, command=lambda: self._option_changed(refit=True))
        menu.add_checkbutton(label="周期镜像", variable=self.images_var, command=self._repeat_changed)
        menu.add_checkbutton(label="自动旋转", variable=self.spin_var, command=self._toggle_spin)
        menu.add_separator()
        menu.add_command(label="清除选择", command=self.clear_selection)
        menu.add_command(label="复制选中信息", command=self.copy_info)
        menu.add_command(label="导出图片…", command=self.export_image)
        self.context_menu = menu

    # ------------------------------------------------------------------ public API
    def destroy(self) -> None:
        for name in ("_draft_job", "_final_job", "_spin_job", "_hover_job"):
            job = getattr(self, name, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        super().destroy()

    def clear(self) -> None:
        self._stop_spin()
        self.structure, self.model, self.display_name = None, None, ""
        self.selection.clear()
        self.highlight_symbol = None
        self._message = "拖入或选择 CIF / POSCAR 后在此显示三维结构"
        self.info_var.set("尚未选择结构")
        self._update_panel()
        self.request_render("final")

    def show_error(self, message: str) -> None:
        self._stop_spin()
        self.structure, self.model = None, None
        self.selection.clear()
        self._message = message
        self.info_var.set("结构预览失败")
        self._update_panel()
        self.request_render("final")

    def set_path(self, path: str | Path) -> None:
        candidate = Path(path).expanduser().resolve()
        try:
            structure = Structure.from_file(candidate)
        except Exception as exc:  # noqa: BLE001
            self.show_error(f"无法读取 {candidate.name}：{type(exc).__name__}")
            return
        self.set_structure(structure, candidate.name)

    def set_structure(self, structure: Structure, display_name: str = "") -> None:
        self.structure = structure.copy()
        self.display_name = display_name
        self.selection.clear()
        self.highlight_symbol = None
        formula = self.structure.composition.reduced_formula
        self.info_var.set(f"{display_name or formula} · {formula} · {len(self.structure)} 原子")
        self._rebuild_model()
        self.camera.rotation = view_rotation(self.model, "iso")
        self._needs_fit = True
        self._update_panel()
        self.request_render("final")

    def redraw(self, reset_view: bool = False) -> None:
        if reset_view:
            self.reset_view()
        else:
            self.request_render("final")

    def reset_view(self) -> None:
        if self.model is not None:
            self.camera.rotation = view_rotation(self.model, "iso")
            self.camera.perspective = bool(self.perspective_var.get())
            self._fit()
        self.request_render("final")

    def set_view(self, name: str) -> None:
        if self.model is None:
            return
        self.camera.rotation = view_rotation(self.model, name)
        self._fit()
        self.request_render("final")

    def fit_view(self) -> None:
        self._fit()
        self.request_render("final")

    def rotate(self, about_x: float, about_y: float, about_z: float = 0.0) -> None:
        if self.model is None:
            return
        delta = axis_angle((0.0, 0.0, 1.0), about_z) @ axis_angle((1.0, 0.0, 0.0), about_x) @ axis_angle((0.0, 1.0, 0.0), about_y)
        self.camera.rotation = delta @ self.camera.rotation
        self.request_render("draft")

    def zoom(self, factor: float, x: Optional[float] = None, y: Optional[float] = None) -> None:
        if self.model is None:
            return
        width, height = self._canvas_size()
        old = self.camera.scale
        new = float(np.clip(old * factor, 2.0, 900.0))
        if x is not None and y is not None:
            point = np.asarray([x - width / 2.0, height / 2.0 - y])
            self.camera.pan = point - (point - self.camera.pan) * (new / old)
        else:
            self.camera.pan = self.camera.pan * (new / old)
        self.camera.scale = new
        self.request_render("draft")

    def scale_atoms(self, factor: float) -> None:
        self.atom_scale = float(np.clip(self.atom_scale * factor, 0.4, 2.5))
        self.request_render("final")

    def scale_bonds(self, factor: float) -> None:
        self.bond_scale = float(np.clip(self.bond_scale * factor, 0.4, 3.0))
        self.request_render("final")

    def reset_sizes(self) -> None:
        self.atom_scale, self.bond_scale = 1.0, 1.0
        self.request_render("final")

    def clear_selection(self) -> None:
        self.selection.clear()
        self._update_panel()
        self.request_render("final")

    def copy_info(self) -> None:
        text = self.info_text.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("已复制到剪贴板")

    def export_image(self) -> None:
        if self.model is None:
            messagebox.showinfo("导出图片", "请先选择一个结构。")
            return
        stem = Path(self.display_name).stem or self.model.formula or "structure"
        target = filedialog.asksaveasfilename(title="导出三维结构图片", defaultextension=".png", initialfile=f"{stem}_3d.png", filetypes=(("PNG 图片", "*.png"),))
        if not target:
            return
        try:
            self.save_image(target, pixel_ratio=2.0)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", f"{type(exc).__name__}: {exc}")
            return
        self._set_status(f"已导出 {Path(target).name}")

    def save_image(self, target: str | Path, pixel_ratio: float = 2.0) -> Path:
        width, height = self._canvas_size()
        result = render_scene(self.model, self.camera, self._render_options(), width, height, "final", pixel_ratio)
        path = Path(target)
        result.image.save(path)
        self.last_export = path
        return path

    # ------------------------------------------------------------------ state
    def _render_options(self) -> RenderOptions:
        return RenderOptions(
            mode=MODE_NAMES.get(self.mode_var.get(), "ball"),
            color_scheme=COLOR_NAMES.get(self.color_var.get(), "element"),
            theme=THEME_NAMES.get(self.theme_var.get(), "light"),
            atom_scale=self.atom_scale,
            bond_scale=self.bond_scale,
            show_cell=bool(self.cell_var.get()),
            full_cell=bool(self.full_cell_var.get()),
            show_labels=bool(self.show_labels_var.get()),
            show_info=bool(self.overlay_var.get()),
            show_axes=bool(self.axes_var.get()),
            fog=bool(self.fog_var.get()),
            highlight_symbol=self.highlight_symbol,
            selection=tuple(self.selection),
            title=self.display_name,
            empty_message=self._message,
        )

    def _theme_hex(self, key: str) -> str:
        color = THEMES[THEME_NAMES.get(self.theme_var.get(), "light")][key]
        return "#%02x%02x%02x" % tuple(color)

    def _canvas_size(self) -> tuple[int, int]:
        width, height = int(self.canvas.winfo_width()), int(self.canvas.winfo_height())
        if width <= 1 or height <= 1:  # not mapped yet: use the requested size
            width, height = int(self.canvas.winfo_reqwidth()), int(self.canvas.winfo_reqheight())
        return max(width, 60), max(height, 60)

    def _rebuild_model(self) -> None:
        if self.structure is None:
            return
        try:
            repeat = (int(self.repeat_a_var.get()), int(self.repeat_b_var.get()))
        except ValueError:
            repeat = (1, 1)
        self.model = build_display_model(self.structure, repeat, bool(self.images_var.get()))
        self.selection = [index for index in self.selection if index < len(self.model.positions)]

    def _fit(self) -> None:
        if self.model is None:
            return
        width, height = self._canvas_size()
        self.camera.perspective = bool(self.perspective_var.get())
        fit_camera(self.model, self.camera, width, height, self._render_options())
        self._needs_fit = False

    def _option_changed(self, refit: bool = False) -> None:
        self.camera.perspective = bool(self.perspective_var.get())
        self.canvas.configure(background=self._theme_hex("top"))
        self.canvas.itemconfigure(self._status_item, fill=self._theme_hex("text"))
        if refit:
            self._fit()
        self.request_render("final")

    def _repeat_changed(self) -> None:
        if self.structure is None:
            return
        self._rebuild_model()
        self.selection.clear()
        self._fit()
        self._update_panel()
        self.request_render("final")

    def _toggle_var(self, variable: tk.BooleanVar, command: Any = None) -> None:
        variable.set(not variable.get())
        (command or self._option_changed)()

    def _toggle_panel(self) -> None:
        self._panel_user_set = True
        self._apply_panel()

    def _apply_panel(self) -> None:
        if self.panel_var.get():
            self.panel.pack(side="right", fill="y", before=self.canvas)
        else:
            self.panel.pack_forget()
        self._needs_fit = self._needs_fit or self.model is not None
        self.request_render("final")

    def _update_panel(self) -> None:
        for child in self.legend.winfo_children():
            child.destroy()
        if self.structure is not None:
            counts = Counter(str(getattr(site.specie, "symbol", site.specie)) for site in self.structure)
            for symbol, count in sorted(counts.items(), key=lambda item: -covalent_radius(item[0])):
                row = tk.Frame(self.legend, background="#dfe9f5" if symbol == self.highlight_symbol else self.legend.winfo_toplevel().cget("background"), cursor="hand2")
                row.pack(fill="x", pady=1)
                swatch = tk.Canvas(row, width=16, height=16, highlightthickness=0, background=row.cget("background"))
                swatch.create_oval(2, 2, 14, 14, fill=element_color(symbol), outline="#51606f")
                swatch.pack(side="left", padx=(2, 6))
                label = tk.Label(row, text=f"{symbol}   ×{count}   r = {covalent_radius(symbol):.2f} Å", background=row.cget("background"), font=("Microsoft YaHei UI", 9), anchor="w")
                label.pack(side="left", fill="x")
                for widget in (row, swatch, label):
                    widget.bind("<Button-1>", lambda _e, s=symbol: self._toggle_highlight(s))
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        if self.model is None:
            self.info_text.insert("end", "尚未选择结构。")
        elif not self.selection:
            self.info_text.insert("end", "单击原子查看坐标、所在层与配位；依次单击 2–4 个原子测量距离、键角和二面角。")
        else:
            blocks = [describe_atom(self.model, self.selection[-1])]
            if len(self.selection) >= 2:
                blocks.append("\n".join(measurement_lines(self.model, self.selection)))
            self.info_text.insert("end", "\n\n".join(blocks))
        self.info_text.configure(state="disabled")

    def _toggle_highlight(self, symbol: str) -> None:
        self.highlight_symbol = None if self.highlight_symbol == symbol else symbol
        self._update_panel()
        self.request_render("final")

    # ------------------------------------------------------------------ rendering
    def request_render(self, quality: str = "final") -> None:
        if quality == "draft":
            if self._draft_job is None:
                self._draft_job = self.after_idle(self._render_draft)
            if self._final_job is not None:
                self.after_cancel(self._final_job)
            self._final_job = self.after(180, self._render_final)
        else:
            if self._final_job is not None:
                self.after_cancel(self._final_job)
            self._final_job = self.after(15, self._render_final)

    def _render_draft(self) -> None:
        self._draft_job = None
        self.render_now("draft")

    def _render_final(self) -> None:
        self._final_job = None
        self.render_now("final")

    def render_now(self, quality: str = "final") -> None:
        width, height = self._canvas_size()
        if self._needs_fit and self.model is not None and width > 80 and height > 80:
            self._fit()
        self.result = render_scene(self.model, self.camera, self._render_options(), width, height, quality)
        self._photo = ImageTk.PhotoImage(self.result.image)
        self.canvas.itemconfigure(self._image_item, image=self._photo)
        self.canvas.coords(self._status_item, width - 10, height - 8)
        self.canvas.tag_raise(self._status_item)

    # ------------------------------------------------------------------ interaction
    def _on_resize(self, event: Any) -> None:
        size = (int(event.width), int(event.height))
        if size == self._size:
            return
        if self._size == (0, 0) or self._needs_fit:
            self._needs_fit = True
        self._size = size
        self.request_render("final")

    def _on_press(self, event: Any) -> None:
        self.canvas.focus_set()
        self._stop_spin()
        button = int(getattr(event, "num", 1) or 1)
        mode = "rotate" if button == 1 else "pan"
        if button == 1 and int(getattr(event, "state", 0)) & 0x0001:
            mode = "roll"
        self._drag = {"x": event.x, "y": event.y, "x0": event.x, "y0": event.y, "moved": False, "mode": mode, "button": button}

    def _on_drag(self, event: Any) -> None:
        if self._drag is None or self.model is None:
            return
        dx, dy = event.x - self._drag["x"], event.y - self._drag["y"]
        self._drag["x"], self._drag["y"] = event.x, event.y
        if abs(event.x - self._drag["x0"]) + abs(event.y - self._drag["y0"]) > 3:
            self._drag["moved"] = True
        if self._drag["mode"] == "rotate":
            self.rotate(dy * ROTATE_RAD_PER_PX, dx * ROTATE_RAD_PER_PX)
        elif self._drag["mode"] == "roll":
            self.rotate(0.0, 0.0, -dx * ROTATE_RAD_PER_PX)
        else:
            self.camera.pan = self.camera.pan + np.asarray([dx, -dy], dtype=np.float64)
            self.request_render("draft")

    def _on_release(self, event: Any) -> None:
        drag, self._drag = self._drag, None
        if drag is None:
            return
        if not drag["moved"]:
            if drag["button"] == 1:
                self._click_atom(event)
            elif drag["button"] == 3:
                self.context_menu.tk_popup(event.x_root, event.y_root)
                return
        self.request_render("final")

    def _click_atom(self, event: Any) -> None:
        atom = pick_atom(self.result, event.x, event.y) if self.model is not None else None
        additive = bool(int(getattr(event, "state", 0)) & 0x0004)
        if atom is None:
            if not additive:
                self.selection.clear()
        elif atom in self.selection:
            self.selection.remove(atom)
        else:
            if not additive and len(self.selection) >= MAX_SELECTION:
                self.selection.clear()
            self.selection.append(atom)
            del self.selection[:-MAX_SELECTION]
        self._update_panel()

    def _on_double(self, event: Any) -> None:
        atom = pick_atom(self.result, event.x, event.y) if self.model is not None else None
        if atom is None:
            self.reset_view()
            return
        self.camera.center = self.model.positions[atom].copy()
        self.camera.pan = np.zeros(2)
        self.request_render("final")

    def _on_wheel(self, event: Any) -> None:
        delta = float(getattr(event, "delta", 0.0))
        if delta:
            self.zoom(1.15 ** (delta / 120.0), event.x, event.y)

    def _on_hover(self, event: Any) -> None:
        if self._drag is not None or self._hover_job is not None:
            return
        self._hover_job = self.after(40, lambda x=event.x, y=event.y: self._hover(x, y))

    def _hover(self, x: int, y: int) -> None:
        self._hover_job = None
        atom = pick_atom(self.result, x, y) if self.model is not None else None
        if atom is None:
            self.canvas.configure(cursor="fleur")
            self._set_status("")
            return
        self.canvas.configure(cursor="hand2")
        name = f"{self.model.symbols[atom]}{int(self.model.base_index[atom]) + 1}"
        self._set_status(f"{name} · 距底面 {float(self.model.heights[atom]):.2f} Å · 第 {int(self.model.layer_index[atom]) + 1} 层")

    def _set_status(self, text: str) -> None:
        self.canvas.itemconfigure(self._status_item, text=text)

    def _toggle_spin(self) -> None:
        if self.spin_var.get():
            if self._spin_job is None:
                self._spin_job = self.after(33, self._spin_step)
        else:
            self._stop_spin()

    def _stop_spin(self) -> None:
        if self._spin_job is not None:
            self.after_cancel(self._spin_job)
            self._spin_job = None
            self.request_render("final")
        self.spin_var.set(False)

    def _spin_step(self) -> None:
        self._spin_job = None
        if not self.spin_var.get() or self.model is None:
            return
        self.camera.rotation = self.camera.rotation @ axis_angle(self.model.normal, math.radians(1.6))
        self.render_now("draft")
        self._spin_job = self.after(33, self._spin_step)


__all__ = ["StructurePreview3D", "StructureScene", "build_structure_scene", "covalent_radius", "element_color"]
