# ==============================================================================
# 中文概述：Windows 推理后端，封装模型加载、批量预测、条件生成、兼容补丁和文件导出。
# English overview: Windows inference backend wrapping model loading, batch prediction, conditional generation, compatibility patches, and export.
#
# 中文输入：本地模型/资源、结构文件与 GUI 请求。
# English inputs: Bundled models/resources, structure files, and GUI requests.
# 中文输出：结构级预测、严格生成候选、进度回调与 CIF/POSCAR。
# English outputs: Structure-level predictions, strict generated candidates, progress callbacks, and CIF/POSCAR.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: application_root, is_structure_file, collect_structure_files, ModelPaths, auto_device, device_description, ensure_windows_neighbor_list_compatibility, ensure_windows_chgnet_graph_compatibility, NFEEngine
#
# Author: Ruck
# Generated: 2026-07-30 07:34:34 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import contextlib
import json
import math
import random
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element, Structure
from pymatgen.io.vasp import Poscar

from nfe_model.data import build_periodic_graph, torch_load_compat
from nfe_model import manifold_generation
from nfe_model import strict_generation
from nfe_model.predict import infer_chunk, load_checkpoint_model


ProgressCallback = Callable[[str, Optional[float]], None]

MXENE_METALS = (
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Hf",
    "Ta",
    "W",
)
CORE_ELEMENTS = ("C", "N")
TARGET_TO_SCORE = {"low": 0.25, "medium": 0.58, "high": 0.85}
TARGET_TO_INDEX = {"low": 0, "medium": 1, "high": 2}
SUPPORTED_SUFFIXES = {".cif", ".vasp", ".poscar", ".contcar"}


# 中文：把单次严格生成的 0–100 进度映射到最多两次尝试的总进度。
# English: Map one strict run's 0–100 progress into a two-attempt total range.
def scale_generation_progress(attempt: int, percent: float | None) -> float | None:
    if percent is None:
        return None
    lower, upper = ((3.0, 49.0), (52.0, 96.0))[min(max(attempt, 0), 1)]
    clipped = min(100.0, max(0.0, float(percent)))
    return lower + (upper - lower) * clipped / 100.0


# 中文：顶层接口 `application_root`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `application_root`; review type hints and callers before extending it.
def application_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parent


# 中文：顶层接口 `is_structure_file`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `is_structure_file`; review type hints and callers before extending it.
def is_structure_file(path: str | Path) -> bool:
    candidate = Path(path)
    basename = candidate.name.upper()
    return bool(
        candidate.is_file()
        and (
            candidate.suffix.lower() in SUPPORTED_SUFFIXES
            or basename.startswith("POSCAR")
            or basename.startswith("CONTCAR")
        )
    )


# 中文：顶层接口 `collect_structure_files`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `collect_structure_files`; review type hints and callers before extending it.
def collect_structure_files(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        candidates: Iterable[Path]
        if path.is_dir():
            candidates = path.rglob("*")
        else:
            candidates = (path,)
        for candidate in candidates:
            if not is_structure_file(candidate):
                continue
            key = str(candidate).casefold()
            if key not in seen:
                seen.add(key)
                result.append(candidate)
    return result


# 中文：顶层类 `ModelPaths`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `ModelPaths`; review type hints and callers before extending it.
@dataclass(frozen=True)
class ModelPaths:
    predictor: Path
    generator: Path
    surface_profile: Path

    @classmethod
    def bundled(cls) -> "ModelPaths":
        root = application_root()
        return cls(
            predictor=root / "models" / "nfe_predictor.pt",
            generator=root / "models" / "mxene_generator.pt",
            surface_profile=root / "resources" / "surface_geometry_summary.json",
        )

    def validate(self) -> None:
        missing = [
            str(path)
            for path in (self.predictor, self.generator, self.surface_profile)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "缺少模型资源：\n" + "\n".join(missing)
            )


# 中文：顶层接口 `auto_device`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `auto_device`; review type hints and callers before extending it.
def auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 中文：顶层接口 `device_description`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `device_description`; review type hints and callers before extending it.
def device_description(device: torch.device | None = None) -> str:
    selected = device or auto_device()
    if selected.type == "cuda" and torch.cuda.is_available():
        index = selected.index if selected.index is not None else 0
        return f"GPU · {torch.cuda.get_device_name(index)}"
    return "CPU"


# 中文：顶层接口 `ensure_windows_neighbor_list_compatibility`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `ensure_windows_neighbor_list_compatibility`; review type hints and callers before extending it.
def ensure_windows_neighbor_list_compatibility() -> None:
    """Use explicit int64 PBC flags for pymatgen's Windows Cython extension."""

    if sys.platform != "win32" or getattr(
        Structure.get_neighbor_list, "_nfe_windows_int64", False
    ):
        return
    try:
        from pymatgen.optimization.neighbors import find_points_in_spheres
    except ImportError:
        return

    def get_neighbor_list_int64(
        structure: Structure,
        r: float,
        sites: Sequence[Any] | None = None,
        numerical_tol: float = 1e-8,
        exclude_self: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        selected_sites = structure.sites if sites is None else sites
        site_coords = np.ascontiguousarray(
            [site.coords for site in selected_sites], dtype=np.float64
        )
        cart_coords = np.ascontiguousarray(
            structure.cart_coords, dtype=np.float64
        )
        lattice_matrix = np.ascontiguousarray(
            structure.lattice.matrix, dtype=np.float64
        )
        # `dtype=int` is int32 on 64-bit Windows but the compiled extension
        # expects `const int64_t`; Linux did not expose this platform mismatch.
        pbc = np.ascontiguousarray(structure.pbc, dtype=np.int64)
        center, neighbor, images, distances = find_points_in_spheres(
            cart_coords,
            site_coords,
            r=r,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        keep = np.ones(len(center), dtype=bool)
        if exclude_self:
            keep = ~(
                (center == neighbor) & (distances <= numerical_tol)
            )
        return (
            center[keep],
            neighbor[keep],
            images[keep],
            distances[keep],
        )

    get_neighbor_list_int64._nfe_windows_int64 = True  # type: ignore[attr-defined]
    Structure.get_neighbor_list = get_neighbor_list_int64  # type: ignore[method-assign]


# 中文：顶层接口 `ensure_windows_chgnet_graph_compatibility`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `ensure_windows_chgnet_graph_compatibility`; review type hints and callers before extending it.
def ensure_windows_chgnet_graph_compatibility() -> None:
    """Use CHGNet's supported pure-Python graph builder on Windows.

    The CHGNet 0.3.x Windows wheel compiles ``long`` as 32 bit, while the
    neighbor-list extension correctly returns 64-bit indices.  Its optional
    fast Cython graph builder therefore rejects otherwise valid structures.
    The bundled legacy builder is API-compatible and only affects the small
    CHGNet pre-relaxation batches used by this desktop application.
    """

    if sys.platform != "win32" or getattr(
        strict_generation.create_chgnet_relaxer, "_nfe_windows_legacy_graph", False
    ):
        return
    original_factory = strict_generation.create_chgnet_relaxer

    def create_windows_compatible_relaxer(device: torch.device) -> Any:
        relaxer = original_factory(device)
        converter = relaxer.calculator.model.graph_converter
        converter.create_graph = converter._create_graph_legacy
        converter.algorithm = "legacy"
        return relaxer

    create_windows_compatible_relaxer._nfe_windows_legacy_graph = True  # type: ignore[attr-defined]
    strict_generation.create_chgnet_relaxer = create_windows_compatible_relaxer


# 中文：顶层类 `NFEEngine`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `NFEEngine`; review type hints and callers before extending it.
class NFEEngine:
    def __init__(
        self,
        model_paths: ModelPaths | None = None,
        *,
        device: str = "auto",
    ) -> None:
        ensure_windows_neighbor_list_compatibility()
        ensure_windows_chgnet_graph_compatibility()
        self.paths = model_paths or ModelPaths.bundled()
        self.paths.validate()
        self.device = auto_device() if device == "auto" else torch.device(device)
        self._predictors: list[tuple[Any, dict[str, Any]]] | None = None
        self._predictor_lock = threading.RLock()
        self._generation_lock = threading.Lock()

    def _load_predictors(
        self, progress: ProgressCallback | None = None
    ) -> list[tuple[Any, dict[str, Any]]]:
        with self._predictor_lock:
            if self._predictors is None:
                if progress:
                    progress(f"正在加载NFE预测器（{device_description(self.device)}）…")
                self._predictors = [
                    load_checkpoint_model(self.paths.predictor, self.device)
                ]
            return self._predictors

    def predict_files(
        self,
        paths: Sequence[str | Path],
        *,
        mc_samples: int = 20,
        batch_size: int = 64,
        progress: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        files = collect_structure_files(paths)
        if not files:
            raise ValueError("没有找到可读取的CIF或POSCAR文件")
        predictors = self._load_predictors(progress)
        config = predictors[0][1]["config"]
        radius = float(config["data"]["radius"])
        max_neighbors = int(config["data"]["max_neighbors"])
        graphs: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, path in enumerate(files, start=1):
            if progress:
                progress(f"解析结构 {index}/{len(files)}：{path.name}")
            try:
                structure = Structure.from_file(path)
                graph = build_periodic_graph(
                    structure,
                    radius,
                    max_neighbors,
                    identifier=path.stem or path.name,
                )
                graph["file_path"] = str(path)
                graphs.append(graph)
                metadata.append(
                    {
                        "Input_File": str(path),
                        "Formula": structure.composition.reduced_formula,
                        "Atom_Count": len(structure),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "Input_File": str(path),
                        "Formula": "",
                        "Atom_Count": 0,
                        "Status": "读取失败",
                        "Error": f"{type(exc).__name__}: {exc}",
                    }
                )
        rows: list[dict[str, Any]] = []
        for start in range(0, len(graphs), batch_size):
            stop = min(start + batch_size, len(graphs))
            if progress:
                progress(
                    f"预测 {start + 1}–{stop}/{len(graphs)}（MC={mc_samples}）…"
                )
            predictions = infer_chunk(
                graphs[start:stop],
                predictors,
                self.device,
                int(mc_samples),
            )
            for offset, prediction in enumerate(predictions):
                item = {
                    **metadata[start + offset],
                    **prediction,
                    "Status": "成功",
                    "Error": "",
                }
                rows.append(item)
        rows.extend(errors)
        by_path = {str(path): index for index, path in enumerate(files)}
        rows.sort(key=lambda item: by_path.get(item["Input_File"], len(files)))
        if progress:
            progress(
                f"预测完成：成功 {len(graphs)}，失败 {len(errors)}。"
            )
        return rows

    @staticmethod
    def _skeleton_template(
        template: dict[str, Any],
        *,
        bottom_metal: str,
        core_element: str,
        top_metal: str,
    ) -> dict[str, Any] | None:
        z_values = [int(value) for value in template["z"]]
        group_type = [int(value) for value in template["group_type"]]
        surface_side = [int(value) for value in template["surface_side"]]
        metal_z = {int(Element(symbol).Z) for symbol in MXENE_METALS}
        metal_indices = [
            index
            for index, (z, group, side) in enumerate(
                zip(z_values, group_type, surface_side)
            )
            if group == 0 and side == 0 and z in metal_z
        ]
        core_indices = [
            index
            for index, (z, group, side) in enumerate(
                zip(z_values, group_type, surface_side)
            )
            if group == 0
            and side == 0
            and Element.from_Z(z).symbol in CORE_ELEMENTS
        ]
        if len(metal_indices) != 2 or len(core_indices) != 1:
            return None
        metal_indices.sort(
            key=lambda index: float(template["layer_position"][index])
        )
        result = dict(template)
        result["z"] = list(z_values)
        result["z"][metal_indices[0]] = int(Element(bottom_metal).Z)
        result["z"][core_indices[0]] = int(Element(core_element).Z)
        result["z"][metal_indices[-1]] = int(Element(top_metal).Z)
        result["source_template_id"] = str(template.get("id", "unknown"))
        result["requested_skeleton"] = (
            f"{bottom_metal}-{core_element}-{top_metal}"
        )
        result["id"] = (
            f"{result['source_template_id']}::windows::"
            f"{result['requested_skeleton']}"
        )
        result["formula"] = strict_generation.composition_formula(result["z"])
        return result

    @classmethod
    def _skeleton_template_selector(
        cls,
        *,
        bottom_metal: str,
        core_element: str,
        top_metal: str,
        target: str,
        state: dict[str, Any],
    ) -> Callable[[Any, dict[str, Any], int], list[dict[str, Any]]]:
        def select(
            args: Any,
            checkpoint: dict[str, Any],
            total: int,
        ) -> list[dict[str, Any]]:
            training_keys = set(checkpoint.get("novelty_reference", {}))
            transformed = []
            for source in checkpoint.get("surface_template_catalog", []):
                candidate = cls._skeleton_template(
                    source,
                    bottom_metal=bottom_metal,
                    core_element=core_element,
                    top_metal=top_metal,
                )
                if candidate is None:
                    continue
                candidate["_unseen_composition"] = (
                    strict_generation.composition_key(candidate["z"]) not in training_keys
                )
                transformed.append(candidate)
            if not transformed:
                raise RuntimeError("检查点中没有可用于该MXene骨架的安全模板")
            unseen = [
                item for item in transformed if item["_unseen_composition"]
            ]
            pool = unseen if unseen else transformed
            state["unseen_only"] = bool(unseen)
            state["allow_training_match"] = not bool(unseen)
            target_score = TARGET_TO_SCORE[target]
            target_index = TARGET_TO_INDEX[target]
            weights = []
            for item in pool:
                closeness = math.exp(
                    -abs(float(item.get("score", 0.5)) - target_score) / 0.16
                )
                class_bonus = (
                    3.0
                    if int(item.get("label", -1)) == target_index
                    else 1.0
                )
                weights.append(max(closeness * class_bonus, 1e-8))
            return [
                dict(item)
                for item in random.choices(
                    pool,
                    weights=weights,
                    k=total,
                )
            ]

        return select

    def generate_skeleton(
        self,
        *,
        bottom_metal: str,
        core_element: str,
        top_metal: str,
        target: str,
        number: int,
        output_parent: str | Path,
        oversample: int = 48,
        mc_samples: int = 20,
        sampling_steps: int = 80,
        relax_steps: int = 250,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if bottom_metal not in MXENE_METALS or top_metal not in MXENE_METALS:
            raise ValueError("金属元素不在模型支持范围内")
        if core_element not in CORE_ELEMENTS:
            raise ValueError("内核元素只能为C或N")
        if target not in TARGET_TO_SCORE:
            raise ValueError("NFE档位必须为low、medium或high")
        if not 1 <= int(number) <= 10:
            raise ValueError("每次生成数量必须在1–10之间")
        output_root = Path(output_parent).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        with self._generation_lock:
            attempt_records = []
            for attempt in range(2):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                skeleton = f"{bottom_metal}{core_element}{top_metal}"
                run_dir = output_root / (
                    f"NFE_{target}_{skeleton}_{timestamp}_try{attempt + 1}"
                )
                if run_dir.exists():
                    raise FileExistsError(
                        f"拒绝覆盖已有生成目录：{run_dir}"
                    )
                log_path = output_root / f"{run_dir.name}.log"
                selector_state: dict[str, Any] = {}
                selector = self._skeleton_template_selector(
                    bottom_metal=bottom_metal,
                    core_element=core_element,
                    top_metal=top_metal,
                    target=target,
                    state=selector_state,
                )
                original_selector = manifold_generation._BASE_CHOOSE_TEMPLATES
                manifold_generation._BASE_CHOOSE_TEMPLATES = selector
                seed = 52000 + attempt * 997 + random.randint(0, 996)
                current_oversample = int(oversample) * (attempt + 1)
                argv = [
                    "--generator-checkpoint",
                    str(self.paths.generator),
                    "--predictor-checkpoint",
                    str(self.paths.predictor),
                    "--composition-mode",
                    "catalog",
                    "--target",
                    target,
                    "--num",
                    str(int(number)),
                    "--oversample",
                    str(current_oversample),
                    "--steps",
                    str(int(sampling_steps)),
                    "--guidance-scale",
                    "2.0",
                    "--min-target-probability",
                    "0.50",
                    "--batch-size",
                    "32",
                    "--mc-samples",
                    str(int(mc_samples)),
                    "--relaxer",
                    "chgnet",
                    "--relax-fmax",
                    "0.05",
                    "--relax-steps",
                    str(int(relax_steps)),
                    "--device",
                    str(self.device),
                    "--seed",
                    str(seed),
                    "--output",
                    str(run_dir),
                ]
                if progress:
                    progress(
                        f"生成尝试 {attempt + 1}/2：{skeleton}，"
                        f"NFE={target}，原始候选={number * current_oversample}…",
                        2.0 if attempt == 0 else 51.0,
                    )
                previous_progress_callback = (
                    strict_generation.set_progress_callback(
                        (
                            lambda message, percent, current_attempt=attempt: progress(
                                f"尝试 {current_attempt + 1}/2 · {message}",
                                scale_generation_progress(
                                    current_attempt, percent
                                ),
                            )
                        )
                        if progress
                        else None
                    )
                )
                try:
                    with log_path.open(
                        "w", encoding="utf-8", errors="replace"
                    ) as log_file, contextlib.redirect_stdout(
                        log_file
                    ), contextlib.redirect_stderr(log_file):
                        return_code = manifold_generation.main(argv)
                except Exception as exc:
                    return_code = 3
                    with log_path.open(
                        "a", encoding="utf-8", errors="replace"
                    ) as log_file:
                        log_file.write(
                            f"\n{type(exc).__name__}: {exc}\n"
                        )
                    attempt_records.append(
                        {
                            "attempt": attempt + 1,
                            "output": str(run_dir),
                            "log": str(log_path),
                            "return_code": return_code,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    strict_generation.set_progress_callback(
                        previous_progress_callback
                    )
                    manifold_generation._BASE_CHOOSE_TEMPLATES = original_selector
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                summary_path = run_dir / "generation_summary.csv"
                try:
                    frame = (
                        pd.read_csv(summary_path)
                        if summary_path.is_file()
                        and summary_path.stat().st_size > 0
                        else pd.DataFrame()
                    )
                except pd.errors.EmptyDataError:
                    frame = pd.DataFrame()
                attempt_records.append(
                    {
                        "attempt": attempt + 1,
                        "output": str(run_dir),
                        "log": str(log_path),
                        "return_code": int(return_code),
                        "exported": int(len(frame)),
                    }
                )
                if return_code == 0 and not frame.empty:
                    if progress:
                        progress("正在写出 POSCAR 与生成元数据…", 98.0)
                    rows = self._write_poscars_and_metadata(
                        frame,
                        run_dir,
                        bottom_metal=bottom_metal,
                        core_element=core_element,
                        top_metal=top_metal,
                        target=target,
                        selector_state=selector_state,
                    )
                    if progress:
                        progress(
                            f"生成完成：导出 {len(rows)} 组CIF和POSCAR。",
                            100.0,
                        )
                    return {
                        "output_directory": str(run_dir),
                        "log": str(log_path),
                        "rows": rows,
                        "attempts": attempt_records,
                    }
                if progress:
                    progress(
                        f"第{attempt + 1}次未得到足够严格候选，"
                        "将提高过采样继续筛选。",
                        50.0 if attempt == 0 else 97.0,
                    )
            raise RuntimeError(
                "两次严格生成均未得到目标候选。"
                "失败记录："
                + json.dumps(attempt_records, ensure_ascii=False)
            )

    @staticmethod
    def _write_poscars_and_metadata(
        frame: pd.DataFrame,
        run_dir: Path,
        *,
        bottom_metal: str,
        core_element: str,
        top_metal: str,
        target: str,
        selector_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        poscar_paths = []
        for _, row in frame.iterrows():
            cif_path = Path(str(row["CIF_Path"])).resolve()
            structure = Structure.from_file(cif_path)
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", cif_path.stem)
            poscar_path = run_dir / f"POSCAR_{safe_stem}"
            if poscar_path.exists():
                raise FileExistsError(
                    f"拒绝覆盖已有POSCAR：{poscar_path}"
                )
            Poscar(structure, sort_structure=False).write_file(poscar_path)
            # Reparse the exported POSCAR before reporting success.
            Structure.from_file(poscar_path)
            poscar_paths.append(str(poscar_path))
        frame = frame.copy()
        frame["POSCAR_Path"] = poscar_paths
        frame["Requested_Bottom_Metal"] = bottom_metal
        frame["Requested_Core_Element"] = core_element
        frame["Requested_Top_Metal"] = top_metal
        frame["Automatic_Termination"] = True
        frame["Requested_Target_Label"] = target
        frame["Used_Unseen_Compositions_Only"] = bool(
            selector_state.get("unseen_only", False)
        )
        summary_path = run_dir / "generation_summary_with_poscar.csv"
        frame.to_csv(summary_path, index=False)
        run_info_path = run_dir / "run_info.json"
        if run_info_path.is_file():
            payload = json.loads(run_info_path.read_text(encoding="utf-8"))
        else:
            payload = {}
        payload.update(
            {
                "windows_app": True,
                "requested_skeleton": {
                    "bottom_metal": bottom_metal,
                    "core_element": core_element,
                    "top_metal": top_metal,
                },
                "automatic_termination": True,
                "used_unseen_compositions_only": bool(
                    selector_state.get("unseen_only", False)
                ),
                "summary_with_poscar": str(summary_path),
            }
        )
        run_info_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return frame.to_dict(orient="records")
