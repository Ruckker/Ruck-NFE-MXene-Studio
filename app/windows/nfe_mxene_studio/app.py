# ==============================================================================
# 中文概述：Windows 桌面 GUI：拖放/批选、批量预测、骨架条件生成和交互式三维预览。
# English overview: Windows desktop GUI for drag/drop and batch prediction, framework-conditioned generation, and interactive 3D preview.
#
# 中文输入：用户选择的 CIF/POSCAR、NFE 档位、核心元素、内层金属与生成选项。
# English inputs: User-selected CIF/POSCAR files, NFE class, core element, inner metals, and generation options.
# 中文输出：结果表、CIF/POSCAR 文件、可旋转结构视图和运行日志。
# English outputs: Result tables, CIF/POSCAR files, rotatable structure view, and run logs.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: NFEMXeneApp, enable_windows_dpi_awareness, run_frozen_self_test, main
#
# Author: Ruck
# Generated: 2026-07-30 08:19:57 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from pymatgen.core import Structure

# 中文：PyInstaller 源码包使用 `windows_app`；GitHub 分层源码使用相对导入。
# English: PyInstaller uses `windows_app`; the layered GitHub source uses relative imports.
try:
    from windows_app.backend import (
        CORE_ELEMENTS,
        MXENE_METALS,
        NFEEngine,
        application_root,
        collect_structure_files,
        device_description,
    )
    from windows_app.structure_preview import StructurePreview3D, build_structure_scene
except ModuleNotFoundError:
    from .backend import (
        CORE_ELEMENTS,
        MXENE_METALS,
        NFEEngine,
        application_root,
        collect_structure_files,
        device_description,
    )
    from .structure_preview import StructurePreview3D, build_structure_scene


LABEL_ZH = {"low": "低", "medium": "中", "high": "高"}
LABEL_EN = {value: key for key, value in LABEL_ZH.items()}


# 中文：顶层类 `NFEMXeneApp`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `NFEMXeneApp`; review type hints and callers before extending it.
class NFEMXeneApp(TkinterDnD.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NFE MXene Studio 1.0")
        self.geometry("1480x920")
        self.minsize(1180, 760)
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.engine: NFEEngine | None = None
        self.input_files: list[Path] = []
        self.prediction_rows: list[dict[str, Any]] = []
        self.generation_rows: list[dict[str, Any]] = []
        self.last_generation_directory: Path | None = None
        self.busy_count = 0

        self._configure_style()
        self._build_header()
        self._build_tabs()
        self._build_status_bar()
        self.after(150, self._initialize_engine)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        available = style.theme_names()
        if "vista" in available:
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 19, "bold"))
        style.configure(
            "Subtitle.TLabel",
            foreground="#4a5568",
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Hint.TLabel",
            foreground="#5e6b78",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Treeview", rowheight=28)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_header(self) -> None:
        header = ttk.Frame(self, padding=(20, 16, 20, 8))
        header.pack(fill="x")
        title_area = ttk.Frame(header)
        title_area.pack(side="left", fill="x", expand=True)
        ttk.Label(
            title_area, text="NFE MXene Studio", style="Title.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            title_area,
            text="批量预测近自由电子态 · 指定MXene骨架生成CIF/POSCAR",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        self.device_var = tk.StringVar(value="正在检测计算设备…")
        ttk.Label(
            header,
            textvariable=self.device_var,
            padding=(12, 7),
        ).pack(side="right")

    def _build_tabs(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=18, pady=(4, 10))
        self.predict_tab = ttk.Frame(notebook, padding=12)
        self.generate_tab = ttk.Frame(notebook, padding=12)
        self.about_tab = ttk.Frame(notebook, padding=20)
        notebook.add(self.predict_tab, text="  批量NFE预测  ")
        notebook.add(self.generate_tab, text="  MXene结构生成  ")
        notebook.add(self.about_tab, text="  模型说明  ")
        self._build_predict_tab()
        self._build_generate_tab()
        self._build_about_tab()

    def _build_predict_tab(self) -> None:
        vertical = ttk.Panedwindow(self.predict_tab, orient="vertical")
        vertical.pack(fill="both", expand=True)
        pane = ttk.Panedwindow(vertical, orient="horizontal")
        preview_panel = ttk.LabelFrame(
            vertical, text="输入结构3D预览", padding=8
        )
        vertical.add(pane, weight=3)
        vertical.add(preview_panel, weight=2)

        left = ttk.LabelFrame(pane, text="输入结构", padding=12)
        right = ttk.LabelFrame(pane, text="预测结果", padding=12)
        pane.add(left, weight=2)
        pane.add(right, weight=3)

        drop_frame = tk.Frame(
            left,
            bg="#edf4fb",
            highlightbackground="#8aa9c7",
            highlightthickness=1,
            height=100,
        )
        drop_frame.pack(fill="x")
        drop_frame.pack_propagate(False)
        drop_label = tk.Label(
            drop_frame,
            text="拖动 CIF / POSCAR 到这里\n支持多个文件和文件夹",
            bg="#edf4fb",
            fg="#2f5f88",
            font=("Microsoft YaHei UI", 11, "bold"),
            justify="center",
        )
        drop_label.pack(expand=True)
        for widget in (drop_frame, drop_label):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(10, 8))
        ttk.Button(
            buttons, text="选择文件", command=self._select_files
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            buttons, text="选择文件夹", command=self._select_folder
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            buttons, text="清空", command=self._clear_inputs
        ).pack(side="right")

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        self.input_list = tk.Listbox(
            list_frame,
            selectmode="extended",
            activestyle="none",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#cbd5df",
        )
        list_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.input_list.yview
        )
        self.input_list.configure(yscrollcommand=list_scroll.set)
        self.input_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self.input_list.bind("<<ListboxSelect>>", self._input_list_selected)

        options = ttk.Frame(left)
        options.pack(fill="x", pady=(10, 0))
        ttk.Label(options, text="MC采样：").pack(side="left")
        self.mc_samples_var = tk.IntVar(value=20)
        ttk.Spinbox(
            options,
            from_=5,
            to=50,
            increment=5,
            textvariable=self.mc_samples_var,
            width=6,
        ).pack(side="left")
        self.predict_button = ttk.Button(
            options,
            text="开始批量预测",
            style="Primary.TButton",
            command=self._start_prediction,
        )
        self.predict_button.pack(side="right")

        result_toolbar = ttk.Frame(right)
        result_toolbar.pack(fill="x", pady=(0, 8))
        self.predict_summary_var = tk.StringVar(value="尚未预测")
        ttk.Label(
            result_toolbar, textvariable=self.predict_summary_var
        ).pack(side="left")
        self.export_button = ttk.Button(
            result_toolbar,
            text="导出CSV",
            command=self._export_predictions,
            state="disabled",
        )
        self.export_button.pack(side="right")

        columns = (
            "file",
            "formula",
            "label",
            "low",
            "medium",
            "high",
            "score",
            "std",
            "ood",
            "status",
        )
        widths = {
            "file": 190,
            "formula": 90,
            "label": 55,
            "low": 70,
            "medium": 70,
            "high": 70,
            "score": 75,
            "std": 70,
            "ood": 65,
            "status": 90,
        }
        headings = {
            "file": "文件",
            "formula": "化学式",
            "label": "NFE",
            "low": "P(低)",
            "medium": "P(中)",
            "high": "P(高)",
            "score": "NFE值",
            "std": "不确定度",
            "ood": "OOD",
            "status": "状态",
        }
        self.predict_tree = self._treeview(
            right, columns, headings, widths
        )
        self.predict_tree.bind(
            "<<TreeviewSelect>>", self._prediction_tree_selected
        )

        preview_selector = ttk.Frame(preview_panel)
        preview_selector.pack(fill="x", pady=(0, 5))
        ttk.Label(preview_selector, text="预览文件：").pack(side="left")
        self.input_preview_var = tk.StringVar()
        self.input_preview_combo = ttk.Combobox(
            preview_selector,
            textvariable=self.input_preview_var,
            state="readonly",
            width=58,
        )
        self.input_preview_combo.pack(side="left", fill="x", expand=True)
        self.input_preview_combo.bind(
            "<<ComboboxSelected>>", self._input_preview_selected
        )
        self.input_preview = StructurePreview3D(preview_panel)
        self.input_preview.pack(fill="both", expand=True)

    def _build_generate_tab(self) -> None:
        content = ttk.Frame(self.generate_tab)
        content.pack(fill="both", expand=True)
        form = ttk.LabelFrame(
            content,
            text="生成条件",
            padding=(18, 14),
        )
        form.pack(fill="x")

        ttk.Label(form, text="下层金属").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=6
        )
        self.bottom_metal_var = tk.StringVar(value="Nb")
        ttk.Combobox(
            form,
            textvariable=self.bottom_metal_var,
            values=MXENE_METALS,
            state="readonly",
            width=9,
        ).grid(row=0, column=1, sticky="w", padx=(0, 24), pady=6)

        ttk.Label(form, text="内核元素").grid(
            row=0, column=2, sticky="w", padx=(0, 8), pady=6
        )
        self.core_var = tk.StringVar(value="C")
        ttk.Combobox(
            form,
            textvariable=self.core_var,
            values=CORE_ELEMENTS,
            state="readonly",
            width=9,
        ).grid(row=0, column=3, sticky="w", padx=(0, 24), pady=6)

        ttk.Label(form, text="上层金属").grid(
            row=0, column=4, sticky="w", padx=(0, 8), pady=6
        )
        self.top_metal_var = tk.StringVar(value="V")
        ttk.Combobox(
            form,
            textvariable=self.top_metal_var,
            values=MXENE_METALS,
            state="readonly",
            width=9,
        ).grid(row=0, column=5, sticky="w", padx=(0, 24), pady=6)

        ttk.Label(form, text="目标NFE").grid(
            row=0, column=6, sticky="w", padx=(0, 8), pady=6
        )
        self.target_var = tk.StringVar(value="高")
        ttk.Combobox(
            form,
            textvariable=self.target_var,
            values=("低", "中", "高"),
            state="readonly",
            width=9,
        ).grid(row=0, column=7, sticky="w", padx=(0, 24), pady=6)

        ttk.Label(form, text="候选数量").grid(
            row=0, column=8, sticky="w", padx=(0, 8), pady=6
        )
        self.number_var = tk.IntVar(value=2)
        ttk.Spinbox(
            form,
            from_=1,
            to=5,
            textvariable=self.number_var,
            width=6,
        ).grid(row=0, column=9, sticky="w", pady=6)

        ttk.Label(form, text="输出目录").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=8
        )
        default_output = Path.home() / "Documents" / "NFE_MXene_Output"
        self.output_var = tk.StringVar(value=str(default_output))
        ttk.Entry(
            form, textvariable=self.output_var
        ).grid(
            row=1,
            column=1,
            columnspan=8,
            sticky="ew",
            padx=(0, 8),
            pady=8,
        )
        ttk.Button(
            form, text="浏览…", command=self._select_output_directory
        ).grid(row=1, column=9, sticky="e", pady=8)
        form.columnconfigure(7, weight=1)

        ttk.Label(
            form,
            text=(
                "端基由模型根据目标NFE自动选择，界面不提供端基指定；"
                "输出同时包含CIF与POSCAR，并经过CHGNet固定晶胞预弛豫。"
            ),
            style="Hint.TLabel",
        ).grid(
            row=2,
            column=0,
            columnspan=9,
            sticky="w",
            pady=(6, 2),
        )
        self.generate_button = ttk.Button(
            form,
            text="生成并严格筛选",
            style="Primary.TButton",
            command=self._start_generation,
        )
        self.generate_button.grid(
            row=2, column=9, sticky="e", pady=(4, 0)
        )

        result_area = ttk.Panedwindow(content, orient="horizontal")
        result_area.pack(fill="both", expand=True, pady=(12, 0))
        result = ttk.LabelFrame(
            result_area,
            text="生成结果",
            padding=12,
        )
        generated_preview_panel = ttk.LabelFrame(
            result_area,
            text="生成结构3D预览",
            padding=8,
        )
        result_area.add(result, weight=3)
        result_area.add(generated_preview_panel, weight=2)
        toolbar = ttk.Frame(result)
        toolbar.pack(fill="x", pady=(0, 8))
        self.generation_summary_var = tk.StringVar(value="尚未生成")
        ttk.Label(
            toolbar, textvariable=self.generation_summary_var
        ).pack(side="left")
        self.open_output_button = ttk.Button(
            toolbar,
            text="打开输出目录",
            state="disabled",
            command=self._open_generation_directory,
        )
        self.open_output_button.pack(side="right")

        # 中文：独立的确定型进度条展示生成模型内部阶段，不与底部忙碌动画混用。
        # English: A determinate bar exposes internal generation stages separately.
        generation_progress_frame = ttk.Frame(result)
        generation_progress_frame.pack(fill="x", pady=(0, 8))
        self.generation_progress_var = tk.DoubleVar(value=0.0)
        self.generation_progress_text_var = tk.StringVar(
            value="等待生成 · 0%"
        )
        self.generation_progress = ttk.Progressbar(
            generation_progress_frame,
            mode="determinate",
            maximum=100.0,
            variable=self.generation_progress_var,
        )
        self.generation_progress.pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        ttk.Label(
            generation_progress_frame,
            textvariable=self.generation_progress_text_var,
            width=48,
            anchor="w",
        ).pack(side="right")

        columns = (
            "rank",
            "formula",
            "target",
            "label",
            "low",
            "medium",
            "high",
            "score",
            "std",
            "ood",
            "force",
            "termination",
            "cif",
            "poscar",
        )
        widths = {
            "rank": 45,
            "formula": 90,
            "target": 55,
            "label": 55,
            "low": 65,
            "medium": 65,
            "high": 65,
            "score": 70,
            "std": 70,
            "ood": 55,
            "force": 75,
            "termination": 80,
            "cif": 180,
            "poscar": 180,
        }
        headings = {
            "rank": "序号",
            "formula": "化学式",
            "target": "目标",
            "label": "预测",
            "low": "P(低)",
            "medium": "P(中)",
            "high": "P(高)",
            "score": "NFE值",
            "std": "不确定度",
            "ood": "OOD",
            "force": "最大力",
            "termination": "自动端基",
            "cif": "CIF",
            "poscar": "POSCAR",
        }
        self.generate_tree = self._treeview(
            result, columns, headings, widths
        )
        self.generate_tree.bind(
            "<<TreeviewSelect>>", self._generation_tree_selected
        )

        generated_selector = ttk.Frame(generated_preview_panel)
        generated_selector.pack(fill="x", pady=(0, 5))
        ttk.Label(generated_selector, text="预览候选：").pack(side="left")
        self.generated_preview_var = tk.StringVar()
        self.generated_preview_combo = ttk.Combobox(
            generated_selector,
            textvariable=self.generated_preview_var,
            state="readonly",
            width=38,
        )
        self.generated_preview_combo.pack(side="left", fill="x", expand=True)
        self.generated_preview_combo.bind(
            "<<ComboboxSelected>>", self._generated_preview_selected
        )
        self.generated_preview = StructurePreview3D(generated_preview_panel)
        self.generated_preview.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(
            content, text="运行信息", padding=8
        )
        log_frame.pack(fill="x", pady=(10, 0))
        self.log_text = tk.Text(
            log_frame,
            height=5,
            wrap="word",
            bg="#f7f9fb",
            borderwidth=0,
            state="disabled",
        )
        self.log_text.pack(fill="x")

    def _build_about_tab(self) -> None:
        text = (
            "功能\n\n"
            "1. 批量预测：支持CIF、POSCAR、CONTCAR及.vasp文件，可拖放、"
            "多选或导入整个文件夹；同时输出低/中/高三类概率、连续NFE值、"
            "不确定度和OOD风险。导入后可用下拉框切换结构，并在3D窗口中"
            "自由旋转、缩放和查看原子、化学键与晶胞。\n\n"
            "2. 条件生成：用户仅指定上下层金属、C/N内核与目标NFE档位。"
            "端基由生成模型自动选择，候选通过几何、表面拓扑、独立NFE预测、"
            "CHGNet固定晶胞预弛豫和重复结构筛选后，同时写出CIF与POSCAR；"
            "每个生成候选也可在相同的3D窗口中交互检查。\n\n"
            "模型边界\n\n"
            "本程序给出机器学习筛选结果。生成结构已尽可能在VASP之前完成"
            "几何与力学预筛选，但NFE强度和最终稳定性仍需DFT弛豫、静态计算、"
            "能带/电荷密度等结果确认。OOD为medium或high的预测应谨慎使用。"
        )
        label = ttk.Label(
            self.about_tab,
            text=text,
            justify="left",
            wraplength=920,
        )
        label.pack(anchor="nw")

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self, padding=(18, 5, 18, 8))
        bar.pack(fill="x")
        self.status_var = tk.StringVar(value="正在初始化…")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(
            bar, mode="indeterminate", length=180
        )
        self.progress.pack(side="right")

    def _treeview(
        self,
        parent: tk.Widget,
        columns: tuple[str, ...],
        headings: dict[str, str],
        widths: dict[str, int],
    ) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        x_scroll = ttk.Scrollbar(
            frame, orient="horizontal", command=tree.xview
        )
        y_scroll = ttk.Scrollbar(
            frame, orient="vertical", command=tree.yview
        )
        tree.configure(
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
        )
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(
                column,
                width=widths[column],
                minwidth=45,
                stretch=column in {"file", "cif", "poscar"},
                anchor="w" if column in {"file", "cif", "poscar"} else "center",
            )
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _initialize_engine(self) -> None:
        try:
            self.engine = NFEEngine()
            self.device_var.set(device_description(self.engine.device))
            self.status_var.set("模型资源已就绪；模型将在首次使用时加载。")
        except Exception as exc:
            self.status_var.set("初始化失败")
            messagebox.showerror("初始化失败", str(exc))

    def _on_drop(self, event: Any) -> str:
        values = self.tk.splitlist(event.data)
        self._add_inputs(values)
        return event.action

    def _select_files(self) -> None:
        values = filedialog.askopenfilenames(
            title="选择CIF或POSCAR结构",
            filetypes=(
                ("晶体结构", "*.cif *.vasp *.poscar *.contcar"),
                ("所有文件", "*.*"),
            ),
        )
        self._add_inputs(values)

    def _select_folder(self) -> None:
        value = filedialog.askdirectory(title="选择包含结构的文件夹")
        if value:
            self._add_inputs((value,))

    def _add_inputs(self, values: Any) -> None:
        combined = collect_structure_files([*self.input_files, *values])
        self.input_files = combined
        self.input_list.delete(0, "end")
        for path in self.input_files:
            self.input_list.insert("end", str(path))
        preview_values = [
            f"{index + 1:03d} · {path.name}"
            for index, path in enumerate(self.input_files)
        ]
        self.input_preview_combo.configure(values=preview_values)
        if self.input_files:
            self.input_preview_combo.current(0)
            self.input_list.selection_clear(0, "end")
            self.input_list.selection_set(0)
            self.input_list.see(0)
            self._show_input_preview(0)
        else:
            self.input_preview_var.set("")
            self.input_preview.clear()
        self.status_var.set(f"已加入 {len(self.input_files)} 个结构文件。")

    def _clear_inputs(self) -> None:
        self.input_files.clear()
        self.input_list.delete(0, "end")
        self.input_preview_combo.configure(values=())
        self.input_preview_var.set("")
        self.input_preview.clear()
        self.status_var.set("输入列表已清空。")

    def _show_input_preview(self, index: int) -> None:
        if not 0 <= index < len(self.input_files):
            return
        self.input_preview_combo.current(index)
        self.input_preview.set_path(self.input_files[index])
        self.status_var.set(f"正在预览：{self.input_files[index].name}")

    def _input_preview_selected(self, _event: Any = None) -> None:
        index = self.input_preview_combo.current()
        if index >= 0:
            self.input_list.selection_clear(0, "end")
            self.input_list.selection_set(index)
            self.input_list.see(index)
            self._show_input_preview(index)

    def _input_list_selected(self, _event: Any = None) -> None:
        selected = self.input_list.curselection()
        if selected:
            self._show_input_preview(int(selected[0]))

    def _prediction_tree_selected(self, _event: Any = None) -> None:
        selected = self.predict_tree.selection()
        if not selected:
            return
        index = self.predict_tree.index(selected[0])
        if not 0 <= index < len(self.prediction_rows):
            return
        path = Path(str(self.prediction_rows[index].get("Input_File", "")))
        try:
            input_index = self.input_files.index(path.resolve())
        except ValueError:
            self.input_preview.set_path(path)
        else:
            self.input_list.selection_clear(0, "end")
            self.input_list.selection_set(input_index)
            self.input_list.see(input_index)
            self._show_input_preview(input_index)

    def _start_prediction(self) -> None:
        if not self.engine:
            messagebox.showwarning("尚未就绪", "模型引擎仍在初始化")
            return
        if not self.input_files:
            messagebox.showwarning("没有输入", "请拖入或选择CIF/POSCAR文件")
            return
        try:
            mc_samples = int(self.mc_samples_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "MC采样必须为整数")
            return
        self._set_busy(True)
        self._run_thread(
            lambda: self.engine.predict_files(
                self.input_files,
                mc_samples=mc_samples,
                progress=self._thread_progress,
            ),
            self._prediction_done,
            self._task_failed,
        )

    def _prediction_done(self, rows: list[dict[str, Any]]) -> None:
        self._set_busy(False)
        self.prediction_rows = rows
        for item in self.predict_tree.get_children():
            self.predict_tree.delete(item)
        success = 0
        for row in rows:
            ok = row.get("Status") == "成功"
            success += int(ok)
            label = LABEL_ZH.get(str(row.get("Predicted_NFE_Label")), "")
            self.predict_tree.insert(
                "",
                "end",
                values=(
                    Path(row["Input_File"]).name,
                    row.get("Formula", ""),
                    label,
                    self._fmt(row.get("Probability_Low")),
                    self._fmt(row.get("Probability_Medium")),
                    self._fmt(row.get("Probability_High")),
                    self._fmt(row.get("Predicted_NFE_Score")),
                    self._fmt(row.get("NFE_Score_Std")),
                    row.get("OOD_Risk", ""),
                    row.get("Status", ""),
                ),
            )
        self.predict_summary_var.set(
            f"完成：成功 {success}，失败 {len(rows) - success}"
        )
        self.export_button.configure(state="normal")
        self.status_var.set("批量NFE预测完成。")

    def _export_predictions(self) -> None:
        if not self.prediction_rows:
            return
        path = filedialog.asksaveasfilename(
            title="保存预测结果",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"),),
            initialfile="nfe_predictions.csv",
        )
        if not path:
            return
        pd.DataFrame(self.prediction_rows).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        self.status_var.set(f"预测结果已保存：{path}")

    def _select_output_directory(self) -> None:
        value = filedialog.askdirectory(title="选择生成结构输出目录")
        if value:
            self.output_var.set(value)

    def _start_generation(self) -> None:
        if not self.engine:
            messagebox.showwarning("尚未就绪", "模型引擎仍在初始化")
            return
        output = self.output_var.get().strip()
        if not output:
            messagebox.showerror("参数错误", "请选择输出目录")
            return
        try:
            number = int(self.number_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "候选数量必须为整数")
            return
        target = LABEL_EN[self.target_var.get()]
        self._append_log(
            f"请求：{self.bottom_metal_var.get()}-"
            f"{self.core_var.get()}-{self.top_metal_var.get()}，"
            f"NFE={self.target_var.get()}，数量={number}"
        )
        self.generation_progress_var.set(0.0)
        self.generation_progress_text_var.set("准备生成 · 0%")
        self.generation_summary_var.set("生成任务运行中，请查看下方进度")
        self._set_busy(True)
        self._run_thread(
            lambda: self.engine.generate_skeleton(
                bottom_metal=self.bottom_metal_var.get(),
                core_element=self.core_var.get(),
                top_metal=self.top_metal_var.get(),
                target=target,
                number=number,
                output_parent=output,
                progress=self._thread_generation_progress,
            ),
            self._generation_done,
            self._generation_failed,
        )

    def _generation_done(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self.generation_progress_var.set(100.0)
        self.generation_progress_text_var.set("全部完成 · 100%")
        rows = payload["rows"]
        self.generation_rows = rows
        self.last_generation_directory = Path(payload["output_directory"])
        for item in self.generate_tree.get_children():
            self.generate_tree.delete(item)
        for row in rows:
            self.generate_tree.insert(
                "",
                "end",
                values=(
                    row.get("Rank", ""),
                    row.get("Formula", ""),
                    LABEL_ZH.get(
                        str(row.get("Requested_Target_Label", "")), ""
                    ),
                    LABEL_ZH.get(
                        str(row.get("Predicted_NFE_Label", "")), ""
                    ),
                    self._fmt(row.get("Probability_Low")),
                    self._fmt(row.get("Probability_Medium")),
                    self._fmt(row.get("Probability_High")),
                    self._fmt(row.get("Predicted_NFE_Score")),
                    self._fmt(row.get("NFE_Score_Std")),
                    row.get("OOD_Risk", ""),
                    self._fmt(row.get("CHGNet_Max_Force_eV_A"), 4),
                    row.get("Termination_Motif", ""),
                    Path(str(row.get("CIF_Path", ""))).name,
                    Path(str(row.get("POSCAR_Path", ""))).name,
                ),
            )
        preview_values = [
            f"{int(row.get('Rank', index + 1)):03d} · "
            f"{row.get('Formula', '')} · "
            f"NFE {self._fmt(row.get('Predicted_NFE_Score'))}"
            for index, row in enumerate(rows)
        ]
        self.generated_preview_combo.configure(values=preview_values)
        if rows:
            self.generated_preview_combo.current(0)
            first_item = self.generate_tree.get_children()[0]
            self.generate_tree.selection_set(first_item)
            self.generate_tree.focus(first_item)
            self.generate_tree.see(first_item)
            self._show_generated_preview(0)
        else:
            self.generated_preview_var.set("")
            self.generated_preview.clear()
        self.generation_summary_var.set(
            f"已导出 {len(rows)} 组严格候选；端基由模型自动选择"
        )
        self.open_output_button.configure(state="normal")
        self.status_var.set("MXene结构生成与筛选完成。")
        self._append_log(f"输出目录：{payload['output_directory']}")
        messagebox.showinfo(
            "生成完成",
            f"已生成 {len(rows)} 组CIF和POSCAR：\n"
            f"{payload['output_directory']}",
        )

    def _show_generated_preview(self, index: int) -> None:
        if not 0 <= index < len(self.generation_rows):
            return
        self.generated_preview_combo.current(index)
        path = Path(str(self.generation_rows[index].get("CIF_Path", "")))
        self.generated_preview.set_path(path)
        self.status_var.set(f"正在预览生成结构：{path.name}")

    def _generated_preview_selected(self, _event: Any = None) -> None:
        index = self.generated_preview_combo.current()
        if index < 0:
            return
        items = self.generate_tree.get_children()
        if index < len(items):
            self.generate_tree.selection_set(items[index])
            self.generate_tree.focus(items[index])
            self.generate_tree.see(items[index])
        self._show_generated_preview(index)

    def _generation_tree_selected(self, _event: Any = None) -> None:
        selected = self.generate_tree.selection()
        if selected:
            self._show_generated_preview(
                self.generate_tree.index(selected[0])
            )

    def _open_generation_directory(self) -> None:
        if self.last_generation_directory and self.last_generation_directory.exists():
            os.startfile(self.last_generation_directory)

    def _thread_progress(
        self, message: str, percent: float | None = None
    ) -> None:
        self.after(0, lambda: self._progress_message(message))

    def _thread_generation_progress(
        self, message: str, percent: float | None = None
    ) -> None:
        self.after(
            0,
            lambda: self._generation_progress_message(message, percent),
        )

    def _generation_progress_message(
        self, message: str, percent: float | None
    ) -> None:
        if percent is not None:
            clipped = min(100.0, max(0.0, float(percent)))
            self.generation_progress_var.set(clipped)
            self.generation_progress_text_var.set(
                f"{message} · {clipped:.0f}%"
            )
        else:
            self.generation_progress_text_var.set(message)
        self._progress_message(message)

    def _progress_message(self, message: str) -> None:
        self.status_var.set(message)
        self._append_log(message)

    def _append_log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _run_thread(
        self,
        function: Callable[[], Any],
        on_success: Callable[[Any], None],
        on_error: Callable[[BaseException], None],
    ) -> None:
        def worker() -> None:
            try:
                result = function()
            except BaseException as exc:
                self.after(0, lambda: on_error(exc))
            else:
                self.after(0, lambda: on_success(result))

        threading.Thread(target=worker, daemon=True).start()

    def _task_failed(self, exc: BaseException) -> None:
        self._set_busy(False)
        detail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self._append_log(detail)
        self.status_var.set("任务失败；请查看运行信息。")
        messagebox.showerror("任务失败", str(exc))

    def _generation_failed(self, exc: BaseException) -> None:
        self.generation_progress_text_var.set("生成失败 · 请查看运行信息")
        self.generation_summary_var.set("生成失败，已保留日志和失败记录")
        self._task_failed(exc)

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.busy_count += 1
        else:
            self.busy_count = max(0, self.busy_count - 1)
        state = "disabled" if self.busy_count else "normal"
        self.predict_button.configure(state=state)
        self.generate_button.configure(state=state)
        if self.busy_count:
            self.progress.start(12)
        else:
            self.progress.stop()

    @staticmethod
    def _fmt(value: Any, digits: int = 3) -> str:
        try:
            number = float(value)
            if number != number:
                return ""
            return f"{number:.{digits}f}"
        except (TypeError, ValueError):
            return ""


# 中文：顶层接口 `enable_windows_dpi_awareness`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `enable_windows_dpi_awareness`; review type hints and callers before extending it.
def enable_windows_dpi_awareness() -> None:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


# 中文：顶层接口 `run_frozen_self_test`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `run_frozen_self_test`; review type hints and callers before extending it.
def run_frozen_self_test(output_file: str | Path) -> int:
    """Exercise lazy predictor and generator imports inside a frozen build."""

    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = application_root() / "samples"
    result: dict[str, Any] = {
        "status": "running",
        "application_root": str(application_root()),
        "device": device_description(),
    }
    try:
        engine = NFEEngine()
        result["predictions"] = engine.predict_files(
            [
                samples / "sample_low_ScTaCSeBr.cif",
                samples / "sample_medium_TiNbCSeCl.cif",
                samples / "sample_high_ZrTiHSNO.cif",
            ],
            mc_samples=3,
        )
        preview_structure = Structure.from_file(
            samples / "sample_medium_TiNbCSeCl.cif"
        )
        preview_scene = build_structure_scene(preview_structure)
        result["preview"] = {
            "atoms": len(preview_scene.positions),
            "bonds": len(preview_scene.bonds),
            "cell_edges": len(preview_scene.cell_segments),
            "symbols": list(preview_scene.symbols),
        }
        result["generation"] = engine.generate_skeleton(
            bottom_metal="Sc",
            core_element="C",
            top_metal="Ta",
            target="low",
            number=1,
            output_parent=output_path.parent / "generated",
            oversample=32,
            mc_samples=3,
            sampling_steps=40,
            relax_steps=150,
        )
        result["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        exit_code = 1
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return exit_code


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main() -> int:
    if "--self-test-output" in sys.argv:
        index = sys.argv.index("--self-test-output")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--self-test-output requires a JSON path")
        return run_frozen_self_test(sys.argv[index + 1])
    enable_windows_dpi_awareness()
    app = NFEMXeneApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
