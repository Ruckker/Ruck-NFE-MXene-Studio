#!/usr/bin/env python3
# ==============================================================================
# 中文概述：从静态/能带 VASP 结果提取结构、电子、真空和 NFE 伪标签特征，并隔离脏数据。
# English overview: Extract structural, electronic, vacuum, and NFE pseudo-label features from static/band VASP results and quarantine dirty data.
#
# 中文输入：static_calc 计算目录中的 CONTCAR/OUTCAR/OSZICAR/EIGENVAL/DOSCAR/LOCPOT 等文件。
# English inputs: CONTCAR/OUTCAR/OSZICAR/EIGENVAL/DOSCAR/LOCPOT and related files in static_calc.
# 中文输出：118 列数据表、清洗结构、脏数据样本、审计表和汇总。
# English outputs: A 118-column table, cleaned structures, dirty samples, audit table, and summary.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: finite_or_none, rounded, sigmoid, safe_read_text, canonical_structure_name, parse_name_fields, deterministic_split, unwrap_slab_z, structure_features, parse_outcar, parse_oszicar, EigenData, parse_eigenval, band_edge_features, parse_doscar, read_first_grid_plane_average, profile_mask, profile_window_stats, parse_gamma_projections, fit_parabola, nfe_band_features, inspect_one, unique_destination, write_csv_atomic, process_all, build_parser, main
#
# Author: Ruck
# Generated: 2026-07-29 09:27:33 Asia/Shanghai
# ==============================================================================

"""Build a clean, physics-informed dataset for generative MXene/NFE models.

The script is deliberately non-destructive:

* source calculation directories are never changed;
* structures are copied (never moved) to ``data/`` or ``dirty/``;
* existing destination structures are never overwritten;
* existing CSV outputs are never overwritten unless ``--overwrite-tables`` is
  explicitly supplied (that option replaces tables only, never structures).

The NFE label produced here is a *pseudo-label*.  It combines a weak atomic
projection at Gamma, parabolic in-plane dispersion, an effective mass close to
the free-electron mass, energetic proximity to the Fermi level, and in-plane
isotropy.  It should be calibrated against manually inspected band-decomposed
charge densities before being treated as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Structure


H2_OVER_2ME_EV_A2 = 3.80998212
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
SEVERE_VASP_PATTERNS = (
    "VERY BAD NEWS",
    "BRMIX: very serious problems",
    "ZBRENT: fatal error",
    "internal error in subroutine",
    "ERROR FEXCP",
    "ERROR: charge density could not be read",
    "TOO FEW BANDS",
)


# 中文：顶层接口 `finite_or_none`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `finite_or_none`; review type hints and callers before extending it.
def finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# 中文：顶层接口 `rounded`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `rounded`; review type hints and callers before extending it.
def rounded(value: Any, digits: int = 8) -> float | None:
    value = finite_or_none(value)
    return None if value is None else round(value, digits)


# 中文：顶层接口 `sigmoid`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `sigmoid`; review type hints and callers before extending it.
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# 中文：顶层接口 `safe_read_text`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `safe_read_text`; review type hints and callers before extending it.
def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# 中文：顶层接口 `canonical_structure_name`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `canonical_structure_name`; review type hints and callers before extending it.
def canonical_structure_name(calc_dir: Path) -> str:
    name = calc_dir.name
    return name[5:] if name.startswith("calc_") else name


# 中文：顶层接口 `parse_name_fields`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_name_fields`; review type hints and callers before extending it.
def parse_name_fields(name: str) -> dict[str, Any]:
    tokens = name.split("-")
    result: dict[str, Any] = {
        "Metal_Top": None,
        "Metal_Bottom": None,
        "X_Element": None,
        "Termination_Top": None,
        "Termination_Bottom": None,
        "Stacking_Top": None,
        "Stacking_Bottom": None,
        "Name_Parse_OK": False,
    }
    # Expected: M1-M2-X-Term1-Term2-T_fcc-T_hcp.  The stacking labels
    # contain underscores, not hyphens, so they remain single tokens.
    if (
        len(tokens) == 7
        and tokens[5] in {"T_fcc", "T_hcp", "T_top"}
        and tokens[6] in {"T_fcc", "T_hcp", "T_top"}
    ):
        result.update(
            {
                "Metal_Top": tokens[0],
                "Metal_Bottom": tokens[1],
                "X_Element": tokens[2],
                "Termination_Top": tokens[3],
                "Termination_Bottom": tokens[4],
                "Stacking_Top": tokens[5],
                "Stacking_Bottom": tokens[6],
                "Name_Parse_OK": True,
            }
        )
    return result


# 中文：顶层接口 `deterministic_split`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `deterministic_split`; review type hints and callers before extending it.
def deterministic_split(name_fields: dict[str, Any]) -> tuple[str, str]:
    metals = sorted(
        str(x)
        for x in (name_fields.get("Metal_Top"), name_fields.get("Metal_Bottom"))
        if x
    )
    terms = sorted(
        str(x)
        for x in (
            name_fields.get("Termination_Top"),
            name_fields.get("Termination_Bottom"),
        )
        if x
    )
    group = "|".join(
        [
            "-".join(metals) or "unknown-metals",
            str(name_fields.get("X_Element") or "unknown-X"),
            "-".join(terms) or "unknown-terminations",
        ]
    )
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    split = "train" if bucket < 80 else ("validation" if bucket < 90 else "test")
    return group, split


# 中文：顶层接口 `unwrap_slab_z`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `unwrap_slab_z`; review type hints and callers before extending it.
def unwrap_slab_z(structure: Structure) -> dict[str, float]:
    frac = np.sort(np.mod([float(site.frac_coords[2]) for site in structure], 1.0))
    if not len(frac):
        raise ValueError("structure has no sites")
    if len(frac) == 1:
        gaps = np.array([1.0])
    else:
        gaps = np.diff(np.r_[frac, frac[0] + 1.0])
    gap_index = int(np.argmax(gaps))
    start_index = (gap_index + 1) % len(frac)
    rolled = np.roll(frac, -start_index).astype(float)
    rolled[rolled < rolled[0]] += 1.0
    low = float(rolled[0])
    high = float(rolled[-1])
    c_len = float(structure.lattice.c)
    vacuum_frac = float(gaps[gap_index])
    return {
        "slab_low_frac": low,
        "slab_high_frac": high,
        "vacuum_fraction": vacuum_frac,
        "slab_thickness_A": (1.0 - vacuum_frac) * c_len,
        "vacuum_thickness_A": vacuum_frac * c_len,
    }


# 中文：顶层接口 `structure_features`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `structure_features`; review type hints and callers before extending it.
def structure_features(structure: Structure) -> dict[str, Any]:
    lattice = structure.lattice
    slab = unwrap_slab_z(structure)
    distances = np.array(structure.distance_matrix, dtype=float)
    distances[distances < 1e-12] = np.inf
    min_distance = float(np.min(distances)) if distances.size > 1 else math.inf
    composition = structure.composition
    return {
        "Formula": composition.formula.replace(" ", ""),
        "Reduced_Formula": composition.reduced_formula,
        "Elements": "|".join(sorted(str(e) for e in composition.elements)),
        "N_Elements": len(composition.elements),
        "N_Atoms": len(structure),
        "Lattice_a_A": rounded(lattice.a),
        "Lattice_b_A": rounded(lattice.b),
        "Lattice_c_A": rounded(lattice.c),
        "Lattice_alpha_deg": rounded(lattice.alpha),
        "Lattice_beta_deg": rounded(lattice.beta),
        "Lattice_gamma_deg": rounded(lattice.gamma),
        "Cell_Volume_A3": rounded(lattice.volume),
        "InPlane_Area_A2": rounded(
            np.linalg.norm(np.cross(lattice.matrix[0], lattice.matrix[1]))
        ),
        "Min_Interatomic_Distance_A": rounded(min_distance),
        **{k: rounded(v) for k, v in slab.items()},
    }


# 中文：顶层接口 `parse_outcar`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_outcar`; review type hints and callers before extending it.
def parse_outcar(path: Path) -> dict[str, Any]:
    text = safe_read_text(path)
    energies = re.findall(
        rf"energy\s+without entropy=\s*({FLOAT_RE})", text, flags=re.IGNORECASE
    )
    fermi = re.findall(rf"E-fermi\s*:\s*({FLOAT_RE})", text)
    nelect = re.findall(rf"\bNELECT\s*=\s*({FLOAT_RE})", text)
    nions = re.findall(r"\bNIONS\s*=\s*(\d+)", text)
    nelm = re.findall(r"\bNELM\s*=\s*(\d+)", text)
    severe = [pattern for pattern in SEVERE_VASP_PATTERNS if pattern.lower() in text.lower()]
    return {
        "Total_Energy_eV": finite_or_none(energies[-1]) if energies else None,
        "Fermi_Level_eV": finite_or_none(fermi[-1]) if fermi else None,
        "NELECT": finite_or_none(nelect[-1]) if nelect else None,
        "OUTCAR_NIONS": int(nions[-1]) if nions else None,
        "NELM": int(nelm[-1]) if nelm else None,
        "Run_Complete": (
            "General timing and accounting informations for this job:" in text
        ),
        "Electronic_Converged": (
            "aborting loop because EDIFF is reached" in text
            or "reached required accuracy" in text
        ),
        "Severe_VASP_Errors": "|".join(severe),
        "VASP_Warning_Count": len(re.findall(r"\bWARNING\b", text, flags=re.IGNORECASE)),
    }


# 中文：顶层接口 `parse_oszicar`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_oszicar`; review type hints and callers before extending it.
def parse_oszicar(path: Path) -> dict[str, Any]:
    text = safe_read_text(path)
    mag = re.findall(rf"\bmag=\s*({FLOAT_RE})", text)
    scf_steps = [int(x) for x in re.findall(r"^\s*(?:DAV|RMM):\s*(\d+)", text, re.MULTILINE)]
    return {
        "Total_Mag_muB": finite_or_none(mag[-1]) if mag else None,
        "SCF_Steps": max(scf_steps) if scf_steps else None,
    }


# 中文：顶层类 `EigenData`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `EigenData`; review type hints and callers before extending it.
@dataclass
class EigenData:
    nelect: float
    kpoints: np.ndarray
    weights: np.ndarray
    energies: np.ndarray  # [spin, kpoint, band]
    occupations: np.ndarray


# 中文：顶层接口 `parse_eigenval`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_eigenval`; review type hints and callers before extending it.
def parse_eigenval(path: Path) -> EigenData:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = [next(handle) for _ in range(6)]
        counts = header[5].split()
        if len(counts) < 3:
            raise ValueError("invalid EIGENVAL counts line")
        nelect, nkpoints, nbands = float(counts[0]), int(counts[1]), int(counts[2])
        kpoints: list[list[float]] = []
        weights: list[float] = []
        blocks: list[list[list[float]]] = []
        occ_blocks: list[list[list[float]]] = []
        nspin: int | None = None
        for _ in range(nkpoints):
            line = next(handle, "")
            while line and not line.strip():
                line = next(handle, "")
            kp = line.split()
            if len(kp) < 4:
                raise ValueError("truncated EIGENVAL k-point block")
            kpoints.append([float(kp[0]), float(kp[1]), float(kp[2])])
            weights.append(float(kp[3]))
            e_block: list[list[float]] = []
            o_block: list[list[float]] = []
            for _band in range(nbands):
                fields = next(handle).split()
                if len(fields) >= 5:
                    energies = [float(fields[1]), float(fields[2])]
                    occupations = [float(fields[3]), float(fields[4])]
                elif len(fields) >= 3:
                    energies = [float(fields[1])]
                    occupations = [float(fields[2])]
                else:
                    raise ValueError("truncated EIGENVAL band row")
                if nspin is None:
                    nspin = len(energies)
                e_block.append(energies)
                o_block.append(occupations)
            blocks.append(e_block)
            occ_blocks.append(o_block)
    # Input blocks are [k, band, spin].
    return EigenData(
        nelect=nelect,
        kpoints=np.asarray(kpoints, dtype=float),
        weights=np.asarray(weights, dtype=float),
        energies=np.asarray(blocks, dtype=float).transpose(2, 0, 1),
        occupations=np.asarray(occ_blocks, dtype=float).transpose(2, 0, 1),
    )


# 中文：顶层接口 `band_edge_features`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `band_edge_features`; review type hints and callers before extending it.
def band_edge_features(data: EigenData, efermi: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    gaps: list[float] = []
    metallic = False
    for spin in range(data.energies.shape[0]):
        energies = data.energies[spin]
        occ = data.occupations[spin]
        occupied = energies[occ > 0.5]
        empty = energies[occ <= 0.5]
        vbm = float(np.max(occupied)) if occupied.size else math.nan
        cbm = float(np.min(empty)) if empty.size else math.nan
        gap = max(0.0, cbm - vbm) if np.isfinite(vbm) and np.isfinite(cbm) else math.nan
        partial = bool(np.any((occ > 0.05) & (occ < 0.95)))
        if partial or (np.isfinite(gap) and gap < 1e-4):
            metallic = True
        gaps.append(gap)
        suffix = "Up" if spin == 0 else "Down"
        result[f"VBM_{suffix}_eV"] = rounded(vbm)
        result[f"CBM_{suffix}_eV"] = rounded(cbm)
        result[f"Band_Gap_{suffix}_eV"] = rounded(gap)
    finite_gaps = [x for x in gaps if np.isfinite(x)]
    result["Band_Gap_eV"] = rounded(min(finite_gaps)) if finite_gaps else None
    result["Is_Metal"] = metallic
    result["VBM_Relative_EF_eV"] = rounded(
        max(
            x
            for x in (
                result.get("VBM_Up_eV"),
                result.get("VBM_Down_eV"),
            )
            if x is not None
        )
        - efermi
    )
    result["CBM_Relative_EF_eV"] = rounded(
        min(
            x
            for x in (
                result.get("CBM_Up_eV"),
                result.get("CBM_Down_eV"),
            )
            if x is not None
        )
        - efermi
    )
    return result


# 中文：顶层接口 `parse_doscar`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_doscar`; review type hints and callers before extending it.
def parse_doscar(path: Path, n_atoms: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(5):
            next(handle)
        header = next(handle).split()
        if len(header) < 4:
            raise ValueError("invalid DOSCAR header")
        nedos = int(float(header[2]))
        efermi_dos = float(header[3])
        rows = []
        for _ in range(nedos):
            fields = next(handle).split()
            if len(fields) >= 3:
                rows.append([float(x) for x in fields])
    values = np.asarray(rows, dtype=float)
    if len(values) != nedos:
        raise ValueError("truncated DOSCAR total DOS")
    order = np.argsort(values[:, 0])
    energy = values[order, 0]
    values = values[order]
    dos_up = float(np.interp(efermi_dos, energy, values[:, 1]))
    dos_down = (
        float(np.interp(efermi_dos, energy, values[:, 2]))
        if values.shape[1] >= 5
        else dos_up
    )
    total = dos_up + dos_down if values.shape[1] >= 5 else dos_up
    polarization = (
        (dos_up - dos_down) / (dos_up + dos_down)
        if abs(dos_up + dos_down) > 1e-12
        else 0.0
    )
    return {
        "DOSCAR_Fermi_Level_eV": rounded(efermi_dos),
        "DOS_at_EF_States_per_eV": rounded(total),
        "DOS_at_EF_per_Atom": rounded(total / max(n_atoms, 1)),
        "DOS_at_EF_Up": rounded(dos_up),
        "DOS_at_EF_Down": rounded(dos_down),
        "DOS_Spin_Polarization_at_EF": rounded(polarization),
    }


# 中文：顶层接口 `read_first_grid_plane_average`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `read_first_grid_plane_average`; review type hints and callers before extending it.
def read_first_grid_plane_average(path: Path, n_atoms: int) -> np.ndarray:
    """Read only the first volumetric dataset and return its z-plane average.

    VASP writes x fastest, then y, then z.  Reading only the first dataset is
    substantially faster and lighter than materializing spin-difference grids.
    """

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        # POSCAR-compatible header embedded in CHGCAR/LOCPOT/ELFCAR.
        next(handle)
        next(handle)
        for _ in range(3):
            next(handle)
        next(handle)  # element symbols
        next(handle)  # atom counts
        line = next(handle)
        if line.strip().lower().startswith("s"):
            line = next(handle)
        # line now contains Direct/Cartesian
        for _ in range(n_atoms):
            next(handle)

        dims: tuple[int, int, int] | None = None
        for line in handle:
            fields = line.split()
            if len(fields) == 3:
                try:
                    candidate = tuple(int(x) for x in fields)
                except ValueError:
                    continue
                if all(x > 0 for x in candidate):
                    dims = candidate
                    break
        if dims is None:
            raise ValueError(f"grid dimensions not found in {path.name}")
        nx, ny, nz = dims
        plane_size = nx * ny
        expected = plane_size * nz
        plane_sum = np.zeros(nz, dtype=np.float64)
        count = 0
        for line in handle:
            for token in line.split():
                if count >= expected:
                    break
                try:
                    value = float(token)
                except ValueError as exc:
                    raise ValueError(f"non-numeric grid value in {path.name}") from exc
                plane_sum[count // plane_size] += value
                count += 1
            if count >= expected:
                break
        if count != expected:
            raise ValueError(
                f"truncated {path.name} grid: expected {expected}, read {count}"
            )
    return plane_sum / float(plane_size)


# 中文：顶层接口 `profile_mask`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `profile_mask`; review type hints and callers before extending it.
def profile_mask(
    n_grid: int, slab_low: float, lower_unwrapped: float, upper_unwrapped: float
) -> tuple[np.ndarray, np.ndarray]:
    frac = np.arange(n_grid, dtype=float) / n_grid
    unwrapped = frac.copy()
    unwrapped[unwrapped < slab_low] += 1.0
    mask = (unwrapped >= lower_unwrapped) & (unwrapped <= upper_unwrapped)
    return mask, unwrapped


# 中文：顶层接口 `profile_window_stats`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `profile_window_stats`; review type hints and callers before extending it.
def profile_window_stats(
    profile: np.ndarray,
    slab: dict[str, float],
    c_len: float,
    *,
    charge_profile: bool = False,
) -> dict[str, Any]:
    low = slab["slab_low_frac"]
    high = slab["slab_high_frac"]
    vacuum_a = slab["vacuum_thickness_A"]
    if vacuum_a <= 2.0:
        raise ValueError("vacuum is too small for surface windows")

    near1 = min(1.5, 0.12 * vacuum_a)
    near2 = min(4.5, 0.38 * vacuum_a)
    far1 = min(4.0, 0.25 * vacuum_a)
    far2 = min(8.0, 0.46 * vacuum_a)
    if near2 <= near1:
        near2 = near1 + 0.5
    if far2 <= far1:
        far2 = far1 + 0.5

    top_near, z_unwrapped = profile_mask(
        len(profile), low, high + near1 / c_len, high + near2 / c_len
    )
    bottom_near, _ = profile_mask(
        len(profile), low, low + 1.0 - near2 / c_len, low + 1.0 - near1 / c_len
    )
    top_far, _ = profile_mask(
        len(profile), low, high + far1 / c_len, high + far2 / c_len
    )
    bottom_far, _ = profile_mask(
        len(profile), low, low + 1.0 - far2 / c_len, low + 1.0 - far1 / c_len
    )
    vacuum_mid = 0.5 * (high + low + 1.0)
    deep, _ = profile_mask(
        len(profile),
        low,
        vacuum_mid - min(1.0, 0.08 * vacuum_a) / c_len,
        vacuum_mid + min(1.0, 0.08 * vacuum_a) / c_len,
    )

    def stats(mask: np.ndarray) -> tuple[float, float]:
        if not np.any(mask):
            return math.nan, math.nan
        vals = profile[mask]
        return float(np.mean(vals)), float(np.max(vals))

    def slope(mask: np.ndarray) -> float:
        if np.count_nonzero(mask) < 3:
            return math.nan
        x = z_unwrapped[mask] * c_len
        y = profile[mask]
        return float(np.polyfit(x, y, 1)[0])

    top_near_mean, top_near_max = stats(top_near)
    bottom_near_mean, bottom_near_max = stats(bottom_near)
    top_far_mean, top_far_max = stats(top_far)
    bottom_far_mean, bottom_far_max = stats(bottom_far)
    deep_mean, deep_max = stats(deep)

    result = {
        "top_near_mean": top_near_mean,
        "top_near_max": top_near_max,
        "bottom_near_mean": bottom_near_mean,
        "bottom_near_max": bottom_near_max,
        "top_far_mean": top_far_mean,
        "top_far_max": top_far_max,
        "bottom_far_mean": bottom_far_mean,
        "bottom_far_max": bottom_far_max,
        "deep_vacuum_mean": deep_mean,
        "deep_vacuum_max": deep_max,
        "top_far_slope_per_A": slope(top_far),
        "bottom_far_slope_per_A": slope(bottom_far),
    }
    if charge_profile:
        positive = np.clip(profile, 0.0, None)
        total = float(np.sum(positive))
        result.update(
            {
                "top_surface_fraction": (
                    float(np.sum(positive[top_near]) / total) if total > 0 else math.nan
                ),
                "bottom_surface_fraction": (
                    float(np.sum(positive[bottom_near]) / total)
                    if total > 0
                    else math.nan
                ),
                "deep_vacuum_fraction": (
                    float(np.sum(positive[deep]) / total) if total > 0 else math.nan
                ),
            }
        )
    return result


# 中文：顶层接口 `parse_gamma_projections`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_gamma_projections`; review type hints and callers before extending it.
def parse_gamma_projections(
    path: Path, gamma_kpoints_one_based: set[int]
) -> dict[tuple[int, int, int], dict[str, float]]:
    projections: dict[tuple[int, int, int], dict[str, float]] = {}
    spin = 0
    kpoint = 0
    band = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.lower().startswith("spin component"):
                try:
                    spin = int(stripped.split()[-1]) - 1
                except ValueError:
                    spin += 1
            elif stripped.startswith("k-point"):
                match = re.match(r"k-point\s+(\d+)", stripped)
                if match:
                    kpoint = int(match.group(1))
            elif stripped.startswith("band"):
                match = re.match(r"band\s+(\d+)", stripped)
                if match:
                    band = int(match.group(1)) - 1
            elif stripped.startswith("tot") and kpoint in gamma_kpoints_one_based:
                fields = stripped.split()[1:]
                try:
                    values = [float(x) for x in fields]
                except ValueError:
                    continue
                if len(values) >= 4:
                    projections[(spin, kpoint, band)] = {
                        "s": values[0],
                        "p": float(sum(values[1:4])),
                        "d": float(sum(values[4:-1])) if len(values) > 5 else 0.0,
                        "total": values[-1],
                    }
    return projections


# 中文：顶层接口 `fit_parabola`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `fit_parabola`; review type hints and callers before extending it.
def fit_parabola(
    kpoints_frac: np.ndarray,
    energies: np.ndarray,
    reciprocal_matrix: np.ndarray,
) -> dict[str, float]:
    gamma = kpoints_frac[-1] if np.linalg.norm(kpoints_frac[-1]) < np.linalg.norm(kpoints_frac[0]) else kpoints_frac[0]
    cart = np.dot(kpoints_frac - gamma, reciprocal_matrix)
    distance = np.linalg.norm(cart, axis=1)
    x = distance**2
    if len(np.unique(np.round(x, 12))) < 3:
        raise ValueError("insufficient unique k-points for parabolic fit")
    coeff = np.polyfit(x, energies, 1)
    predicted = np.polyval(coeff, x)
    ss_res = float(np.sum((energies - predicted) ** 2))
    ss_tot = float(np.sum((energies - np.mean(energies)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-16 else 1.0
    alpha = float(coeff[0])
    mass = H2_OVER_2ME_EV_A2 / alpha if alpha > 1e-10 else math.nan
    return {"alpha": alpha, "mass": mass, "r2": r2, "rmse": math.sqrt(ss_res / len(x))}


# 中文：顶层接口 `nfe_band_features`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `nfe_band_features`; review type hints and callers before extending it.
def nfe_band_features(
    band_data: EigenData,
    projections: dict[tuple[int, int, int], dict[str, float]],
    structure: Structure,
    efermi: float,
    fit_points: int = 12,
) -> dict[str, Any]:
    nk = band_data.energies.shape[1]
    if nk < 15 or nk % 3 != 0:
        raise ValueError(f"unexpected band path k-point count: {nk}")
    segment = nk // 3
    gamma_left = 2 * segment - 1
    gamma_right = 2 * segment
    left_indices = np.arange(max(segment, gamma_left - fit_points + 1), gamma_left + 1)
    right_indices = np.arange(gamma_right, min(nk, gamma_right + fit_points))
    reciprocal = np.asarray(structure.lattice.reciprocal_lattice.matrix, dtype=float)

    candidates: list[dict[str, Any]] = []
    for spin in range(band_data.energies.shape[0]):
        for band in range(band_data.energies.shape[2]):
            energy_gamma = float(
                np.mean(
                    [
                        band_data.energies[spin, gamma_left, band],
                        band_data.energies[spin, gamma_right, band],
                    ]
                )
            )
            relative = energy_gamma - efermi
            if relative < -1.5 or relative > 3.0:
                continue
            proj_values = [
                projections.get((spin, gamma_left + 1, band)),
                projections.get((spin, gamma_right + 1, band)),
            ]
            proj_values = [x for x in proj_values if x is not None]
            if not proj_values:
                continue
            projection = {
                key: float(np.mean([x[key] for x in proj_values]))
                for key in ("s", "p", "d", "total")
            }
            left = fit_parabola(
                band_data.kpoints[left_indices],
                band_data.energies[spin, left_indices, band],
                reciprocal,
            )
            # Right branch has Gamma first; reverse so fit_parabola sees either endpoint.
            right = fit_parabola(
                band_data.kpoints[right_indices],
                band_data.energies[spin, right_indices, band],
                reciprocal,
            )
            masses = [left["mass"], right["mass"]]
            positive_masses = all(np.isfinite(m) and m > 0 for m in masses)
            mean_mass = math.sqrt(masses[0] * masses[1]) if positive_masses else math.nan
            anisotropy = (
                max(masses) / min(masses) if positive_masses and min(masses) > 0 else math.nan
            )
            projection_score = float(np.clip((0.65 - projection["total"]) / 0.65, 0, 1))
            parabola_score = float(np.clip(min(left["r2"], right["r2"]), 0, 1))
            energy_score = math.exp(-abs(relative) / 1.0)
            mass_score = (
                math.exp(-abs(math.log(mean_mass))) if positive_masses else 0.0
            )
            isotropy_score = (
                math.exp(-abs(math.log(anisotropy))) if positive_masses else 0.0
            )
            curvature_gate = 1.0 if positive_masses else 0.0
            score = curvature_gate * (
                0.32 * projection_score
                + 0.24 * parabola_score
                + 0.20 * energy_score
                + 0.14 * mass_score
                + 0.10 * isotropy_score
            )
            occupation = float(
                np.mean(
                    [
                        band_data.occupations[spin, gamma_left, band],
                        band_data.occupations[spin, gamma_right, band],
                    ]
                )
            )
            candidates.append(
                {
                    "score": score,
                    "spin": spin,
                    "band": band,
                    "energy": energy_gamma,
                    "relative": relative,
                    "occupation": occupation,
                    "projection": projection,
                    "left": left,
                    "right": right,
                    "mean_mass": mean_mass,
                    "anisotropy": anisotropy,
                    "projection_score": projection_score,
                    "parabola_score": parabola_score,
                    "energy_score": energy_score,
                    "mass_score": mass_score,
                    "isotropy_score": isotropy_score,
                }
            )
    if not candidates:
        raise ValueError("no NFE candidate band in the configured energy window")
    best = max(candidates, key=lambda x: x["score"])
    score = float(best["score"])
    if score >= 0.70 and abs(best["relative"]) <= 1.0:
        label = "high"
    elif score >= 0.48:
        label = "medium"
    else:
        label = "low"
    p = best["projection"]
    return {
        "NFE_Pseudo_Score": rounded(score),
        "NFE_Pseudo_Label": label,
        "NFE_Label_Is_Ground_Truth": False,
        "NFE_Candidate_Spin": "up" if best["spin"] == 0 else "down",
        "NFE_Candidate_Band_Index": best["band"] + 1,
        "NFE_Energy_at_Gamma_eV": rounded(best["energy"]),
        "NFE_Energy_Relative_EF_eV": rounded(best["relative"]),
        "NFE_Occupation_at_Gamma": rounded(best["occupation"]),
        "NFE_Atomic_Projection_Total": rounded(p["total"]),
        "NFE_Atomic_Projection_s": rounded(p["s"]),
        "NFE_Atomic_Projection_p": rounded(p["p"]),
        "NFE_Atomic_Projection_d": rounded(p["d"]),
        "NFE_Effective_Mass_KG_me": rounded(best["left"]["mass"]),
        "NFE_Effective_Mass_GM_me": rounded(best["right"]["mass"]),
        "NFE_Effective_Mass_Geomean_me": rounded(best["mean_mass"]),
        "NFE_Mass_Anisotropy": rounded(best["anisotropy"]),
        "NFE_Parabolic_R2_KG": rounded(best["left"]["r2"]),
        "NFE_Parabolic_R2_GM": rounded(best["right"]["r2"]),
        "NFE_Parabolic_RMSE_KG_eV": rounded(best["left"]["rmse"]),
        "NFE_Parabolic_RMSE_GM_eV": rounded(best["right"]["rmse"]),
        "NFE_Score_Projection_Component": rounded(best["projection_score"]),
        "NFE_Score_Parabola_Component": rounded(best["parabola_score"]),
        "NFE_Score_Energy_Component": rounded(best["energy_score"]),
        "NFE_Score_Mass_Component": rounded(best["mass_score"]),
        "NFE_Score_Isotropy_Component": rounded(best["isotropy_score"]),
        "NFE_Candidate_Count": len(candidates),
    }


# 中文：顶层接口 `inspect_one`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `inspect_one`; review type hints and callers before extending it.
def inspect_one(task: tuple[str, bool]) -> dict[str, Any]:
    calc_dir = Path(task[0])
    with_grid_features = task[1]
    name = canonical_structure_name(calc_dir)
    band_dir = calc_dir / "Band"
    hard_reasons: list[str] = []
    warnings: list[str] = []
    row: dict[str, Any] = {
        "Structure_Name": name,
        "Source_Directory": str(calc_dir),
    }
    row.update(parse_name_fields(name))
    split_group, split = deterministic_split(row)
    row["Split_Group"] = split_group
    row["Suggested_Split"] = split

    required = [
        calc_dir / "CONTCAR",
        calc_dir / "OUTCAR",
        calc_dir / "OSZICAR",
        calc_dir / "EIGENVAL",
        calc_dir / "DOSCAR",
        calc_dir / "LOCPOT",
        calc_dir / "ELFCAR",
        calc_dir / "CHGCAR",
        band_dir / "OUTCAR",
        band_dir / "EIGENVAL",
        band_dir / "PROCAR",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            hard_reasons.append(f"missing_or_empty:{path.relative_to(calc_dir)}")

    structure_path = calc_dir / "CONTCAR"
    if not structure_path.is_file() or structure_path.stat().st_size <= 0:
        structure_path = calc_dir / "POSCAR"
    row["_structure_source"] = str(structure_path) if structure_path.is_file() else ""

    try:
        structure = Structure.from_file(structure_path)
        row.update(structure_features(structure))
    except Exception as exc:
        hard_reasons.append(f"structure_parse:{type(exc).__name__}:{exc}")
        return {
            "status": "dirty",
            "row": row,
            "hard_reasons": hard_reasons,
            "warnings": warnings,
        }

    min_distance = row.get("Min_Interatomic_Distance_A")
    if min_distance is not None and min_distance < 0.65:
        hard_reasons.append(f"unphysical_min_distance:{min_distance}")
    if row.get("vacuum_thickness_A") is not None and row["vacuum_thickness_A"] < 10.0:
        hard_reasons.append(f"insufficient_vacuum:{row['vacuum_thickness_A']}")

    static_out: dict[str, Any] = {}
    if (calc_dir / "OUTCAR").is_file():
        try:
            static_out = parse_outcar(calc_dir / "OUTCAR")
            row.update(static_out)
        except Exception as exc:
            hard_reasons.append(f"static_outcar_parse:{type(exc).__name__}:{exc}")
    if not static_out.get("Run_Complete", False):
        hard_reasons.append("static_run_incomplete")
    if not static_out.get("Electronic_Converged", False):
        hard_reasons.append("static_scf_not_converged")
    if static_out.get("Severe_VASP_Errors"):
        hard_reasons.append(f"static_vasp_error:{static_out['Severe_VASP_Errors']}")

    if (calc_dir / "OSZICAR").is_file():
        try:
            row.update(parse_oszicar(calc_dir / "OSZICAR"))
        except Exception as exc:
            warnings.append(f"oszicar_parse:{type(exc).__name__}:{exc}")
    if row.get("Total_Energy_eV") is not None:
        row["Energy_per_Atom_eV"] = rounded(
            row["Total_Energy_eV"] / max(int(row["N_Atoms"]), 1)
        )
    else:
        hard_reasons.append("missing_total_energy")
    efermi = finite_or_none(row.get("Fermi_Level_eV"))
    if efermi is None:
        hard_reasons.append("missing_fermi_level")

    if (band_dir / "OUTCAR").is_file():
        try:
            band_out = parse_outcar(band_dir / "OUTCAR")
            row.update(
                {
                    "Band_Run_Complete": band_out["Run_Complete"],
                    "Band_Electronic_Converged": band_out["Electronic_Converged"],
                    "Band_Severe_VASP_Errors": band_out["Severe_VASP_Errors"],
                    "Band_VASP_Warning_Count": band_out["VASP_Warning_Count"],
                }
            )
            if not band_out["Run_Complete"]:
                hard_reasons.append("band_run_incomplete")
            if not band_out["Electronic_Converged"]:
                hard_reasons.append("band_scf_not_converged")
            if band_out["Severe_VASP_Errors"]:
                hard_reasons.append(f"band_vasp_error:{band_out['Severe_VASP_Errors']}")
        except Exception as exc:
            hard_reasons.append(f"band_outcar_parse:{type(exc).__name__}:{exc}")

    static_eigen: EigenData | None = None
    if (calc_dir / "EIGENVAL").is_file() and efermi is not None:
        try:
            static_eigen = parse_eigenval(calc_dir / "EIGENVAL")
            row.update(band_edge_features(static_eigen, efermi))
            row["Static_NKPoints"] = static_eigen.energies.shape[1]
            row["Static_NBands"] = static_eigen.energies.shape[2]
            row["N_Spin_Channels"] = static_eigen.energies.shape[0]
        except Exception as exc:
            hard_reasons.append(f"static_eigenval_parse:{type(exc).__name__}:{exc}")

    if (calc_dir / "DOSCAR").is_file():
        try:
            row.update(parse_doscar(calc_dir / "DOSCAR", int(row["N_Atoms"])))
        except Exception as exc:
            hard_reasons.append(f"doscar_parse:{type(exc).__name__}:{exc}")

    if (
        (band_dir / "EIGENVAL").is_file()
        and (band_dir / "PROCAR").is_file()
        and efermi is not None
    ):
        try:
            band_data = parse_eigenval(band_dir / "EIGENVAL")
            nk = band_data.energies.shape[1]
            if nk % 3 != 0:
                raise ValueError(f"band k-point count {nk} is not divisible by 3")
            segment = nk // 3
            gamma_one_based = {2 * segment, 2 * segment + 1}
            projections = parse_gamma_projections(
                band_dir / "PROCAR", gamma_one_based
            )
            row.update(nfe_band_features(band_data, projections, structure, efermi))
            row["Band_NKPoints"] = nk
            row["Band_NBands"] = band_data.energies.shape[2]
        except Exception as exc:
            hard_reasons.append(f"nfe_band_parse:{type(exc).__name__}:{exc}")

    if with_grid_features and efermi is not None:
        slab = {
            "slab_low_frac": float(row["slab_low_frac"]),
            "slab_high_frac": float(row["slab_high_frac"]),
            "vacuum_thickness_A": float(row["vacuum_thickness_A"]),
        }
        c_len = float(row["Lattice_c_A"])
        try:
            locpot = read_first_grid_plane_average(
                calc_dir / "LOCPOT", int(row["N_Atoms"])
            )
            potential = profile_window_stats(locpot, slab, c_len)
            row.update(
                {
                    "Vacuum_Level_Top_eV": rounded(potential["top_far_mean"]),
                    "Vacuum_Level_Bottom_eV": rounded(potential["bottom_far_mean"]),
                    "Work_Function_Top_eV": rounded(
                        potential["top_far_mean"] - efermi
                    ),
                    "Work_Function_Bottom_eV": rounded(
                        potential["bottom_far_mean"] - efermi
                    ),
                    "Work_Function_Mean_eV": rounded(
                        0.5
                        * (
                            potential["top_far_mean"]
                            + potential["bottom_far_mean"]
                        )
                        - efermi
                    ),
                    "Vacuum_Potential_Asymmetry_eV": rounded(
                        potential["top_far_mean"]
                        - potential["bottom_far_mean"]
                    ),
                    "Vacuum_Field_Top_eV_per_A": rounded(
                        potential["top_far_slope_per_A"]
                    ),
                    "Vacuum_Field_Bottom_eV_per_A": rounded(
                        potential["bottom_far_slope_per_A"]
                    ),
                }
            )
            max_field = max(
                abs(potential["top_far_slope_per_A"]),
                abs(potential["bottom_far_slope_per_A"]),
            )
            reliable = bool(np.isfinite(max_field) and max_field <= 0.02)
            row["Work_Function_Reliable"] = reliable
            if not reliable:
                warnings.append(
                    "work_function_vacuum_not_flat_or_missing_dipole_correction"
                )
        except Exception as exc:
            hard_reasons.append(f"locpot_parse:{type(exc).__name__}:{exc}")

        try:
            elf = read_first_grid_plane_average(
                calc_dir / "ELFCAR", int(row["N_Atoms"])
            )
            elf_stats = profile_window_stats(elf, slab, c_len)
            row.update(
                {
                    "ELF_Surface_Top_Mean": rounded(elf_stats["top_near_mean"]),
                    "ELF_Surface_Top_Max": rounded(elf_stats["top_near_max"]),
                    "ELF_Surface_Bottom_Mean": rounded(
                        elf_stats["bottom_near_mean"]
                    ),
                    "ELF_Surface_Bottom_Max": rounded(
                        elf_stats["bottom_near_max"]
                    ),
                    "ELF_Deep_Vacuum_Mean": rounded(
                        elf_stats["deep_vacuum_mean"]
                    ),
                }
            )
        except Exception as exc:
            hard_reasons.append(f"elfcar_parse:{type(exc).__name__}:{exc}")

        try:
            charge = read_first_grid_plane_average(
                calc_dir / "CHGCAR", int(row["N_Atoms"])
            )
            charge_stats = profile_window_stats(
                charge, slab, c_len, charge_profile=True
            )
            row.update(
                {
                    "Charge_Surface_Top_Fraction": rounded(
                        charge_stats["top_surface_fraction"], 12
                    ),
                    "Charge_Surface_Bottom_Fraction": rounded(
                        charge_stats["bottom_surface_fraction"], 12
                    ),
                    "Charge_Deep_Vacuum_Fraction": rounded(
                        charge_stats["deep_vacuum_fraction"], 12
                    ),
                    "Charge_Surface_Total_Fraction": rounded(
                        charge_stats["top_surface_fraction"]
                        + charge_stats["bottom_surface_fraction"],
                        12,
                    ),
                }
            )
        except Exception as exc:
            hard_reasons.append(f"chgcar_parse:{type(exc).__name__}:{exc}")
    elif not with_grid_features:
        warnings.append("grid_features_skipped")

    # Quality score is deliberately separate from the NFE pseudo-score.
    quality = 1.0
    quality -= min(0.15, 0.01 * int(row.get("VASP_Warning_Count") or 0))
    quality -= min(0.10, 0.01 * int(row.get("Band_VASP_Warning_Count") or 0))
    quality -= 0.10 * len(warnings)
    quality -= 0.35 * len(hard_reasons)
    row["Data_Quality_Score"] = rounded(max(0.0, quality))
    row["Quality_Warnings"] = "|".join(warnings)
    row["Hard_Failure_Reasons"] = "|".join(hard_reasons)
    row["Extraction_Schema_Version"] = "nfe-v1.0"
    row["Extraction_UTC"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return {
        "status": "dirty" if hard_reasons else "clean",
        "row": row,
        "hard_reasons": hard_reasons,
        "warnings": warnings,
    }


# 中文：顶层接口 `unique_destination`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `unique_destination`; review type hints and callers before extending it.
def unique_destination(directory: Path, filename: str, source: Path) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    # Never overwrite.  Reuse only when byte-identical.
    if candidate.is_file() and candidate.read_bytes() == source.read_bytes():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for index in range(1, 10000):
        candidate = directory / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate a unique destination for {filename}")


# 中文：顶层接口 `write_csv_atomic`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `write_csv_atomic`; review type hints and callers before extending it.
def write_csv_atomic(
    path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; use --overwrite-tables to replace CSV tables"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("_") or key in seen:
                continue
            seen.add(key)
            fields.append(key)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    os.replace(temp, path)


# 中文：顶层接口 `process_all`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `process_all`; review type hints and callers before extending it.
def process_all(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    source = (root / args.source).resolve()
    data_dir = (root / args.data_dir).resolve()
    dirty_dir = (root / args.dirty_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    calc_dirs = sorted(
        path for path in source.iterdir() if path.is_dir() and path.name.startswith("calc_")
    )
    if args.limit is not None:
        calc_dirs = calc_dirs[: args.limit]
    if not calc_dirs:
        raise RuntimeError(f"no calculation directories found under {source}")

    print(
        json.dumps(
            {
                "root": str(root),
                "source": str(source),
                "calculations": len(calc_dirs),
                "workers": args.workers,
                "grid_features": not args.skip_grid_features,
                "write_outputs": args.write_outputs,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    tasks = [(str(path), not args.skip_grid_features) for path in calc_dirs]
    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            results.append(inspect_one(task))
            if index % args.progress_every == 0 or index == len(tasks):
                print(f"processed {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(inspect_one, task): task[0] for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    results.append(future.result())
                except Exception as exc:
                    calc_dir = Path(futures[future])
                    results.append(
                        {
                            "status": "dirty",
                            "row": {
                                "Structure_Name": canonical_structure_name(calc_dir),
                                "Source_Directory": str(calc_dir),
                                "_structure_source": str(calc_dir / "POSCAR"),
                                "Hard_Failure_Reasons": (
                                    f"worker_exception:{type(exc).__name__}:{exc}"
                                ),
                                "Extraction_Schema_Version": "nfe-v1.0",
                            },
                            "hard_reasons": [f"worker_exception:{type(exc).__name__}:{exc}"],
                            "warnings": [],
                        }
                    )
                if index % args.progress_every == 0 or index == len(tasks):
                    print(f"processed {index}/{len(tasks)}", flush=True)

    results.sort(key=lambda item: item["row"].get("Structure_Name", ""))
    clean_rows: list[dict[str, Any]] = []
    dirty_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    if args.write_outputs:
        data_dir.mkdir(parents=True, exist_ok=True)
        dirty_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        row = result["row"]
        status = result["status"]
        structure_source = Path(row.get("_structure_source", ""))
        copied_path = ""
        if args.write_outputs and structure_source.is_file():
            destination_dir = data_dir if status == "clean" else dirty_dir
            filename = f"{row['Structure_Name']}.vasp"
            destination = unique_destination(destination_dir, filename, structure_source)
            if not destination.exists():
                shutil.copy2(structure_source, destination)
            copied_path = str(destination.relative_to(root))
        if status == "clean":
            row["File_Path"] = copied_path or f"{args.data_dir}/{row['Structure_Name']}.vasp"
            clean_rows.append(row)
        else:
            row["Dirty_File_Path"] = (
                copied_path or f"{args.dirty_dir}/{row['Structure_Name']}.vasp"
            )
            dirty_rows.append(row)
        audit_rows.append(
            {
                "Structure_Name": row.get("Structure_Name"),
                "Status": status,
                "Hard_Failure_Reasons": row.get("Hard_Failure_Reasons", ""),
                "Quality_Warnings": row.get("Quality_Warnings", ""),
                "Data_Quality_Score": row.get("Data_Quality_Score"),
                "Source_Directory": row.get("Source_Directory"),
                "Copied_Path": copied_path,
            }
        )

    summary = {
        "total": len(results),
        "clean": len(clean_rows),
        "dirty": len(dirty_rows),
        "warnings": sum(bool(x["warnings"]) for x in results),
        "write_outputs": args.write_outputs,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

    if args.write_outputs:
        write_csv_atomic(
            root / args.output,
            clean_rows,
            overwrite=args.overwrite_tables,
        )
        write_csv_atomic(
            root / args.dirty_output,
            dirty_rows,
            overwrite=args.overwrite_tables,
        )
        write_csv_atomic(
            root / args.audit_output,
            audit_rows,
            overwrite=args.overwrite_tables,
        )
        summary_path = root / args.summary_output
        if summary_path.exists() and not args.overwrite_tables:
            raise FileExistsError(
                f"{summary_path} already exists; use --overwrite-tables to replace it"
            )
        temp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, summary_path)
    return 0


# 中文：顶层接口 `build_parser`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `build_parser`; review type hints and callers before extending it.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a clean NFE-focused MXene deep-learning dataset from VASP outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="project root (default: script directory)",
    )
    parser.add_argument("--source", default="static_calc")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dirty-dir", default="dirty")
    parser.add_argument("--output", default="nfe_dataset.csv")
    parser.add_argument("--dirty-output", default="dirty_manifest.csv")
    parser.add_argument("--audit-output", default="extraction_audit.csv")
    parser.add_argument("--summary-output", default="extraction_summary.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--skip-grid-features",
        action="store_true",
        help="skip LOCPOT/ELFCAR/CHGCAR features for a fast structural/electronic audit",
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="copy structures and write tables; without this flag the run is read-only",
    )
    parser.add_argument(
        "--overwrite-tables",
        action="store_true",
        help="replace existing CSV/JSON tables only; structures are still never overwritten",
    )
    return parser


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return process_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
