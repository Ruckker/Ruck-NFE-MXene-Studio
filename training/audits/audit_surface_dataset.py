# ==============================================================================
# 中文概述：提供 NFE MXene 项目中的单一、可复用源码职责。
# English overview: Provide one reusable source-code responsibility in the NFE MXene project.
#
# 中文输入：请结合类型标注、命令行帮助和调用方查看输入。
# English inputs: Read type hints, CLI help, and callers for the expected inputs.
# 中文输出：返回值或生成文件由公开接口和命令行参数定义。
# English outputs: Return values or generated files are defined by public APIs and CLI arguments.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: parse_args, quantiles, main
#
# Author: Ruck
# Generated: 2026-07-29 22:18:09 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Element

from nfe_model.data import INDEX_TO_LABEL, torch_load_compat
from nfe_model.surface_geometry import (
    BOTTOM,
    TOP,
    analyze_surface_geometry,
)
from nfe_model.train import resolve_config_paths
from nfe_model.utils import load_config


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit relaxed slab surface roles and termination geometry."
    )
    parser.add_argument("--config", default="nfe_predictor.yaml")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


# 中文：顶层接口 `quantiles`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `quantiles`; review type hints and callers before extending it.
def quantiles(values: Sequence[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"count": 0}
    points = np.quantile(finite, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(len(finite)),
        "q01": float(points[0]),
        "q05": float(points[1]),
        "q25": float(points[2]),
        "median": float(points[3]),
        "q75": float(points[4]),
        "q95": float(points[5]),
        "q99": float(points[6]),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = resolve_config_paths(load_config(config_path), config_path)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {output}")
    output.mkdir(parents=True)

    cache = torch_load_compat(config["data"]["cache"])
    records = cache["records"]
    structure_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    motif_counts: Counter[str] = Counter()
    layer_counts: Counter[int] = Counter()
    oh_lengths: list[float] = []
    anchor_by_pair: defaultdict[str, list[float]] = defaultdict(list)

    for record in records:
        z = record["z"].cpu().numpy()
        frac = record["frac_pos"].cpu().numpy()
        lattice = record["lattice"].cpu().numpy()
        analysis = analyze_surface_geometry(z, frac, lattice)
        symbols = np.asarray(
            [Element.from_Z(int(value)).symbol for value in z], dtype=object
        )
        bottom = "".join(sorted(symbols[analysis.surface_side == BOTTOM].tolist()))
        top = "".join(sorted(symbols[analysis.surface_side == TOP].tolist()))
        motif = f"{bottom}|{top}"
        motif_counts[motif] += 1
        layer_count = int(len(np.unique(analysis.layer_index)))
        layer_counts[layer_count] += 1
        warning_counts.update(analysis.warnings)
        oh_lengths.extend(float(item["distance_A"]) for item in analysis.oh_bonds)

        for group in analysis.surface_groups:
            pair = f"{group['leader_symbol']}-{group['anchor_symbol']}"
            if group["anchor_symbol"] and np.isfinite(group["anchor_distance_A"]):
                anchor_by_pair[pair].append(float(group["anchor_distance_A"]))
            group_rows.append(
                {
                    "Structure_Name": record.get("id", ""),
                    "Split": record.get("split", ""),
                    "NFE_Label": INDEX_TO_LABEL.get(int(record.get("label", -1)), ""),
                    **group,
                }
            )

        structure_rows.append(
            {
                "Structure_Name": record.get("id", ""),
                "File_Path": record.get("file_path", ""),
                "Split": record.get("split", ""),
                "NFE_Label": INDEX_TO_LABEL.get(int(record.get("label", -1)), ""),
                "N_Atoms": int(len(z)),
                "Layer_Count": layer_count,
                "Bottom_Termination": bottom,
                "Top_Termination": top,
                "Termination_Motif": motif,
                "Surface_Group_Count": int(len(analysis.surface_groups)),
                "OH_Group_Count": int(len(analysis.oh_bonds)),
                "Surface_Warnings": "|".join(analysis.warnings),
                "Has_Surface_Warning": bool(analysis.warnings),
            }
        )

    frame = pd.DataFrame(structure_rows)
    groups = pd.DataFrame(group_rows)
    frame.to_csv(output / "surface_structure_audit.csv", index=False)
    groups.to_csv(output / "surface_group_audit.csv", index=False)
    warning_frame = frame[frame["Has_Surface_Warning"]].copy()
    warning_frame.to_csv(output / "surface_warning_candidates.csv", index=False)

    summary = {
        "schema": "nfe-mxene-surface-audit-1.0",
        "records": int(len(records)),
        "surface_warning_records": int(frame["Has_Surface_Warning"].sum()),
        "warning_counts": dict(warning_counts.most_common()),
        "layer_count_distribution": {
            str(key): int(value) for key, value in sorted(layer_counts.items())
        },
        "termination_motif_top50": dict(motif_counts.most_common(50)),
        "oh_bond_length_A": quantiles(oh_lengths),
        "anchor_distance_A_by_pair": {
            pair: quantiles(values)
            for pair, values in sorted(anchor_by_pair.items())
            if len(values) >= 5
        },
        "notes": [
            "Warnings are audit candidates, not automatic dirty-data labels.",
            "Thresholds for surface-generator losses must use robust training-split quantiles.",
            "Top and bottom terminations are learned independently to preserve Janus slabs.",
        ],
    }
    (output / "surface_geometry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved audit to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
