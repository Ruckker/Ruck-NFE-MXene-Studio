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
    "no_auxiliary_regression": 5,
    "classification_only": 6,
}

DISPLAY_NAMES = {
    "full": "Full model",
    "no_vector": "− vector/equivariant branch",
    "no_global": "− global slab features",
    "no_masked_pretrain": "− masked-atom objective",
    "no_denoise": "− coordinate denoising",
    "no_auxiliary_regression": "− auxiliary regression (score only)",
    "classification_only": "Classification only",
}

PAPER_METRICS = (
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_macro_roc_auc",
    "test_low_f1",
    "test_medium_f1",
    "test_high_f1",
    "test_low_recall",
    "test_high_recall",
    "test_NFE_Pseudo_Score_mae",
    "test_NFE_Pseudo_Score_rmse",
    "test_calibrated_ece",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate NFE predictor ablation runs.")
    parser.add_argument("--runs-root", default="runs/ablations")
    parser.add_argument("--output-dir", default="training/ablations/results")
    return parser.parse_args(argv)


def mean_std_text(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return ""
    if len(array) == 1:
        return f"{array[0]:.5f}"
    return f"{array.mean():.5f} ± {array.std(ddof=1):.5f}"


def flatten_metrics(
    ablation: str, seed: int, payload: dict[str, Any], path: Path
) -> dict[str, Any]:
    provenance = payload.get("provenance", {})
    row: dict[str, Any] = {
        "ablation": ablation,
        "seed": seed,
        "best_epoch": payload.get("best_epoch"),
        "classification_temperature": payload.get("classification_temperature"),
        "dataset_table_sha256": provenance.get("dataset_table_sha256"),
        "split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "git_commit": provenance.get("git_commit"),
        "result_path": str(path),
    }
    for split in ("validation", "test"):
        for key, value in payload.get(split, {}).items():
            row[f"{split}_{key}"] = value
        for key, value in payload.get(f"{split}_calibrated", {}).items():
            row[f"{split}_calibrated_{key}"] = value
    return row


def load_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/seed_*/final_metrics.json")):
        ablation = path.parents[1].name
        seed_name = path.parent.name
        if not seed_name.startswith("seed_"):
            continue
        try:
            seed = int(seed_name.removeprefix("seed_"))
        except ValueError:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(flatten_metrics(ablation, seed, payload, path))
    return rows


def assert_common_provenance(frame: pd.DataFrame) -> None:
    for column in ("dataset_table_sha256", "split_manifest_sha256"):
        if column not in frame:
            raise RuntimeError(f"ablation results are missing required provenance field {column}")
        values = {str(value) for value in frame[column].dropna().tolist() if str(value)}
        if not values:
            raise RuntimeError(
                f"ablation results contain no {column}; rerun with the audited trainer"
            )
        if len(values) > 1:
            raise RuntimeError(
                f"cannot aggregate ablations with mixed {column}: {sorted(values)}"
            )


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    excluded = {
        "ablation",
        "result_path",
        "dataset_table_sha256",
        "split_manifest_sha256",
        "git_commit",
    }
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for ablation, group in frame.groupby("ablation", sort=False):
        row: dict[str, Any] = {"ablation": ablation, "n_runs": len(group)}
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_order"] = result["ablation"].map(ABLATION_ORDER).fillna(999)
        result = result.sort_values(["_order", "ablation"]).drop(columns="_order")
    return result


def paper_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full = frame[frame["ablation"] == "full"]
    full_f1 = pd.to_numeric(
        full.get("test_macro_f1", pd.Series(dtype=float)), errors="coerce"
    ).mean()
    full_score_mae = pd.to_numeric(
        full.get("test_NFE_Pseudo_Score_mae", pd.Series(dtype=float)), errors="coerce"
    ).mean()

    for ablation, group in frame.groupby("ablation", sort=False):
        row: dict[str, Any] = {
            "Ablation": DISPLAY_NAMES.get(ablation, ablation),
            "Seeds": int(group["seed"].nunique()),
            "key": ablation,
        }
        for metric in PAPER_METRICS:
            if metric not in group:
                row[metric] = ""
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            row[metric] = mean_std_text(values)
        f1 = pd.to_numeric(
            group.get("test_macro_f1", pd.Series(dtype=float)), errors="coerce"
        ).mean()
        score_mae = pd.to_numeric(
            group.get("test_NFE_Pseudo_Score_mae", pd.Series(dtype=float)),
            errors="coerce",
        ).mean()
        row["Δ macro F1 vs full"] = (
            float(f1 - full_f1) if np.isfinite(f1) and np.isfinite(full_f1) else np.nan
        )
        if ablation == "classification_only":
            row["Δ score MAE vs full"] = np.nan
            row["test_NFE_Pseudo_Score_mae"] = "N/A"
            row["test_NFE_Pseudo_Score_rmse"] = "N/A"
        else:
            row["Δ score MAE vs full"] = (
                float(score_mae - full_score_mae)
                if np.isfinite(score_mae) and np.isfinite(full_score_mae)
                else np.nan
            )
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
