# ==============================================================================
# 中文概述：比较两个版本的 nfe_dataset.csv（例如 nfe-v1.0 与 nfe-v1.1），量化标签与特征迁移。
# English overview: Compare two versions of nfe_dataset.csv (e.g. nfe-v1.0 vs nfe-v1.1) and quantify label and feature migration.
#
# 中文输入：旧表与新表路径。
# English inputs: Paths to the old and the new table.
# 中文输出：标签迁移矩阵、分数差分位数、按磁性分层的差异、候选自旋分布，JSON + Markdown。
# English outputs: Label migration matrix, score-delta quantiles, magnetic/non-magnetic breakdown, candidate-spin counts, JSON + Markdown.
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LABELS = ("low", "medium", "high")
COMPARE_COLUMNS = (
    "NFE_Pseudo_Score",
    "NFE_Energy_Relative_EF_eV",
    "NFE_Atomic_Projection_Total",
    "NFE_Effective_Mass_Geomean_me",
    "NFE_Score_Projection_Component",
    "NFE_Score_Parabola_Component",
    "NFE_Score_Energy_Component",
    "NFE_Score_Mass_Component",
    "NFE_Score_Isotropy_Component",
    "NFE_Parabolic_R2_KG",
    "NFE_Candidate_Count",
)


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {
        "n": int(len(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "q50": float(np.quantile(values, 0.5)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "max_abs": float(np.max(np.abs(values))),
        "changed_fraction": float(np.mean(np.abs(values) > 1e-9)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two dataset versions row by row.")
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/metadata/label_version_comparison.json"))
    args = parser.parse_args(argv)
    old = pd.read_csv(args.old).set_index("Structure_Name")
    new = pd.read_csv(args.new).set_index("Structure_Name")
    common = old.index.intersection(new.index)
    old, new = old.loc[common], new.loc[common]
    magnetic = new["Total_Mag_muB"].abs().fillna(0.0) > 0.1

    migration = {
        a: {b: int(((old["NFE_Pseudo_Label"] == a) & (new["NFE_Pseudo_Label"] == b)).sum()) for b in LABELS}
        for a in LABELS
    }
    changed = old["NFE_Pseudo_Label"] != new["NFE_Pseudo_Label"]
    band_changed = old["NFE_Candidate_Band_Index"] != new["NFE_Candidate_Band_Index"]
    spin_changed = old["NFE_Candidate_Spin"] != new["NFE_Candidate_Spin"]
    deltas = {
        column: quantiles((new[column] - old[column]).to_numpy(dtype=float))
        for column in COMPARE_COLUMNS
        if column in old and column in new
    }
    by_magnetism = {}
    for name, mask in (("magnetic", magnetic), ("non_magnetic", ~magnetic)):
        by_magnetism[name] = {
            "rows": int(mask.sum()),
            "label_changed": int((changed & mask).sum()),
            "label_changed_fraction": float((changed & mask).sum() / max(1, mask.sum())),
            "candidate_band_changed": int((band_changed & mask).sum()),
            "candidate_spin_changed": int((spin_changed & mask).sum()),
            "score_mean_abs_delta": float((new["NFE_Pseudo_Score"] - old["NFE_Pseudo_Score"]).abs()[mask].mean()),
        }
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "old_table": str(args.old),
        "new_table": str(args.new),
        "old_schema": sorted(set(old["Extraction_Schema_Version"].astype(str))),
        "new_schema": sorted(set(new["Extraction_Schema_Version"].astype(str))),
        "rows_compared": int(len(common)),
        "only_in_old": int(len(old.index.difference(new.index))) if False else int(len(set(pd.read_csv(args.old)["Structure_Name"]) - set(common))),
        "only_in_new": int(len(set(pd.read_csv(args.new)["Structure_Name"]) - set(common))),
        "label_counts_old": old["NFE_Pseudo_Label"].value_counts().to_dict(),
        "label_counts_new": new["NFE_Pseudo_Label"].value_counts().to_dict(),
        "label_migration_old_rows_new_cols": migration,
        "label_changed": int(changed.sum()),
        "label_changed_fraction": float(changed.mean()),
        "candidate_spin_counts_old": old["NFE_Candidate_Spin"].value_counts().to_dict(),
        "candidate_spin_counts_new": new["NFE_Candidate_Spin"].value_counts().to_dict(),
        "candidate_band_changed": int(band_changed.sum()),
        "split_changed": int((old["Suggested_Split"] != new["Suggested_Split"]).sum()),
        "feature_deltas_new_minus_old": deltas,
        "by_magnetism": by_magnetism,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| old \\ new | " + " | ".join(LABELS) + " |", "|---|" + "---:|" * len(LABELS)]
    for a in LABELS:
        lines.append(f"| {a} | " + " | ".join(str(migration[a][b]) for b in LABELS) + " |")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in report.items() if k not in ("feature_deltas_new_minus_old", "label_migration_old_rows_new_cols")}, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
