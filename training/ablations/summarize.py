from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ABLATION_ORDER = {
    "full": 0,
    "no_vector": 1,
    "no_global": 2,
    "no_masked_pretrain": 3,
    "no_denoise": 4,
    "no_self_supervision": 5,
    "no_auxiliary_regression": 6,
    "matched_supervision": 7,
    "classification_only": 8,
}
DISPLAY_NAMES = {
    "full": "Full model",
    "no_vector": "− vector branch (and denoise)",
    "no_global": "− global slab features",
    "no_masked_pretrain": "− masked-atom objective",
    "no_denoise": "− coordinate denoising",
    "no_self_supervision": "− all self-supervision",
    "no_auxiliary_regression": "− auxiliary regression (score only, SSL kept)",
    "matched_supervision": "Class + score only; no SSL",
    "classification_only": "Classification only",
}
PAPER_METRICS = (
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_macro_roc_auc",
    "test_macro_average_precision",
    "test_high_average_precision",
    "test_high_precision_at_5pct",
    "test_high_recall_at_5pct",
    "test_high_enrichment_at_5pct",
    "test_low_f1",
    "test_medium_f1",
    "test_high_f1",
    "test_low_recall",
    "test_high_recall",
    "test_NFE_Pseudo_Score_mae",
    "test_NFE_Pseudo_Score_rmse",
    "test_NFE_Pseudo_Score_spearman",
    "test_NFE_Pseudo_Score_r2",
    "test_calibrated_ece",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate audited NFE predictor ablations.")
    p.add_argument("--runs-root", default="runs/ablations")
    p.add_argument("--output-dir", default="training/ablations/results")
    return p.parse_args(argv)


def mean_std_text(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return ""
    if len(arr) == 1:
        return f"{arr[0]:.5f}"
    return f"{arr.mean():.5f} ± {arr.std(ddof=1):.5f}"


def flatten_metrics(ablation: str, seed: int, payload: dict[str, Any], path: Path) -> dict[str, Any]:
    provenance = payload.get("provenance", {})
    row: dict[str, Any] = {
        "ablation": ablation,
        "seed": seed,
        "best_epoch": payload.get("best_epoch"),
        "classification_temperature": payload.get("classification_temperature"),
        "result_path": str(path),
        "dataset_table_sha256": provenance.get("dataset_table_sha256"),
        "split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "cache_schema": provenance.get("cache_schema"),
        "global_feature_schema": provenance.get("global_feature_schema"),
        "neighbor_policy": provenance.get("neighbor_policy"),
    }
    for split in ("validation", "test"):
        for key, value in payload.get(split, {}).items():
            row[f"{split}_{key}"] = value
        for key, value in payload.get(f"{split}_calibrated", {}).items():
            row[f"{split}_calibrated_{key}"] = value
    return row


def load_runs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/seed_*/final_metrics.json")):
        ablation = path.parents[1].name
        if ablation not in ABLATION_ORDER or not path.parent.name.startswith("seed_"):
            continue
        try:
            seed = int(path.parent.name.removeprefix("seed_"))
        except ValueError:
            continue
        rows.append(flatten_metrics(ablation, seed, json.loads(path.read_text(encoding="utf-8")), path))
    return rows


def assert_common_provenance(frame: pd.DataFrame) -> None:
    for key in ("dataset_table_sha256", "split_manifest_sha256", "cache_schema", "global_feature_schema", "neighbor_policy"):
        if key not in frame or frame[key].isna().any() or (frame[key].astype(str).str.len() == 0).any():
            raise RuntimeError(f"ablation results contain missing provenance: {key}")
        values = set(frame[key].astype(str))
        if len(values) != 1:
            raise RuntimeError(f"ablation results mix incompatible {key}: {sorted(values)}")


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {"ablation", "result_path", "dataset_table_sha256", "split_manifest_sha256", "cache_schema", "global_feature_schema", "neighbor_policy"}
    numeric = [c for c in frame if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    rows = []
    for ablation, group in frame.groupby("ablation", sort=False):
        row: dict[str, Any] = {"ablation": ablation, "n_runs": len(group)}
        for col in numeric:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(values):
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_order"] = result["ablation"].map(ABLATION_ORDER).fillna(999)
        result = result.sort_values(["_order", "ablation"]).drop(columns="_order")
    return result


def paper_table(frame: pd.DataFrame) -> pd.DataFrame:
    full = frame[frame["ablation"] == "full"]
    full_f1 = pd.to_numeric(full.get("test_macro_f1", pd.Series(dtype=float)), errors="coerce").mean()
    full_mae = pd.to_numeric(full.get("test_NFE_Pseudo_Score_mae", pd.Series(dtype=float)), errors="coerce").mean()
    rows = []
    for ablation, group in frame.groupby("ablation", sort=False):
        row: dict[str, Any] = {"Ablation": DISPLAY_NAMES.get(ablation, ablation), "key": ablation, "Seeds": int(group["seed"].nunique())}
        for metric in PAPER_METRICS:
            values = pd.to_numeric(group.get(metric, pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            row[metric] = mean_std_text(values)
        f1 = pd.to_numeric(group.get("test_macro_f1", pd.Series(dtype=float)), errors="coerce").mean()
        mae = pd.to_numeric(group.get("test_NFE_Pseudo_Score_mae", pd.Series(dtype=float)), errors="coerce").mean()
        row["Δ macro F1 vs full"] = float(f1 - full_f1) if np.isfinite(f1) and np.isfinite(full_f1) else np.nan
        if ablation == "classification_only":
            row["Δ score MAE vs full"] = np.nan
            for metric in ("test_NFE_Pseudo_Score_mae", "test_NFE_Pseudo_Score_rmse", "test_NFE_Pseudo_Score_spearman", "test_NFE_Pseudo_Score_r2"):
                row[metric] = "N/A"
        else:
            row["Δ score MAE vs full"] = float(mae - full_mae) if np.isfinite(mae) and np.isfinite(full_mae) else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_order"] = result["key"].map(ABLATION_ORDER).fillna(999)
        result = result.sort_values(["_order", "key"]).drop(columns=["_order", "key"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.runs_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_runs(root)
    if not rows:
        raise SystemExit(f"no ablation final_metrics.json files found under {root}")
    per_seed = pd.DataFrame(rows)
    assert_common_provenance(per_seed)
    per_seed["_order"] = per_seed["ablation"].map(ABLATION_ORDER).fillna(999)
    per_seed = per_seed.sort_values(["_order", "ablation", "seed"]).drop(columns="_order")
    summary = numeric_summary(per_seed)
    paper = paper_table(per_seed)
    per_seed.to_csv(output / "ablation_per_seed.csv", index=False)
    summary.to_csv(output / "ablation_summary.csv", index=False)
    paper.to_csv(output / "ablation_paper_table.csv", index=False)
    print(paper.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
