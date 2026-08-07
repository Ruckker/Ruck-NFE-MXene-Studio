from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "dummy": "Dummy prior/median",
    "xgboost": "XGBoost (structure-only)",
    "cgcnn": "CGCNN-style (controlled)",
    "schnet": "SchNet-style (controlled)",
    "alignn": "ALIGNN-style (controlled)",
    "m3gnet": "M3GNet-style (controlled)",
    "painn": "Ruck-NFE backbone (matched)",
    "ours_full": "Ruck-NFE Full",
}

MODEL_ORDER = {
    "dummy": 0,
    "xgboost": 1,
    "cgcnn": 2,
    "schnet": 3,
    "alignn": 4,
    "m3gnet": 5,
    "painn": 6,
    "ours_full": 7,
}

TRACK_ORDER = {"architecture": 0, "full-system": 1}

PAPER_METRICS = [
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_macro_roc_auc",
    "test_NFE_Pseudo_Score_mae",
    "test_NFE_Pseudo_Score_rmse",
    "test_low_f1",
    "test_medium_f1",
    "test_high_f1",
    "test_low_recall",
    "test_high_recall",
    "test_ece",
]


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    provenance = result.get("provenance", {})
    row: dict[str, Any] = {
        "track": result.get("track", "architecture"),
        "model": result.get("model"),
        "seed": result.get("seed"),
        "parameter_count": result.get("parameter_count"),
        "training_seconds": result.get("training_seconds"),
        "evaluation_seconds": result.get("evaluation_seconds"),
        "temperature": result.get("temperature"),
        "dataset_table_sha256": provenance.get("dataset_table_sha256"),
        "split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "git_commit": provenance.get("git_commit"),
    }
    for split in ("validation", "test"):
        for key, value in result.get(f"{split}_metrics", {}).items():
            row[f"{split}_{key}"] = value
    return row


def mean_std_text(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return ""
    if len(array) == 1:
        return f"{array[0]:.5f}"
    return f"{array.mean():.5f} ± {array.std(ddof=1):.5f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate audited NFE benchmark tracks.")
    parser.add_argument("--results-root", default="training/baselines/results")
    parser.add_argument("--output-dir", default="training/baselines/results")
    parser.add_argument("--minimum-full-seeds", type=int, default=5)
    return parser.parse_args(argv)


def load_results(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/seed_*/result.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "nfe-baseline-result-2.0":
            continue
        row = flatten_result(payload)
        row["result_path"] = str(path)
        results.append(row)
    return results


def assert_common_provenance(frame: pd.DataFrame) -> None:
    for column in ("dataset_table_sha256", "split_manifest_sha256"):
        values = {str(value) for value in frame[column].dropna().tolist() if str(value)}
        if len(values) > 1:
            raise RuntimeError(
                f"cannot aggregate benchmark results with mixed {column}: {sorted(values)}"
            )


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    excluded = {
        "track",
        "model",
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
    for (track, model), group in frame.groupby(["track", "model"], sort=False):
        row: dict[str, Any] = {"track": track, "model": model, "n_runs": len(group)}
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    subset = frame[frame["track"] == track]
    rows: list[dict[str, Any]] = []
    for model, group in subset.groupby("model", sort=False):
        row: dict[str, Any] = {
            "Track": track,
            "Model": DISPLAY_NAMES.get(model, model),
            "Seeds": int(group["seed"].nunique()),
        }
        for metric in PAPER_METRICS:
            if metric not in group:
                row[metric] = ""
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            row[metric] = mean_std_text(values)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        reverse_names = {value: key for key, value in DISPLAY_NAMES.items()}
        result["_model_key"] = result["Model"].map(reverse_names).fillna(result["Model"])
        result["_order"] = result["_model_key"].map(MODEL_ORDER).fillna(999)
        result = result.sort_values(["_order", "Model"]).drop(
            columns=["_order", "_model_key"]
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_results(root)
    if not rows:
        raise SystemExit(f"no audited result.json files found under {root}")
    per_seed = pd.DataFrame(rows)
    assert_common_provenance(per_seed)
    per_seed["_track_order"] = per_seed["track"].map(TRACK_ORDER).fillna(999)
    per_seed["_model_order"] = per_seed["model"].map(MODEL_ORDER).fillna(999)
    per_seed = per_seed.sort_values(
        ["_track_order", "_model_order", "track", "model", "seed"]
    ).drop(columns=["_track_order", "_model_order"])

    full = per_seed[
        (per_seed["track"] == "full-system") & (per_seed["model"] == "ours_full")
    ]
    if not full.empty and int(full["seed"].nunique()) < int(args.minimum_full_seeds):
        raise RuntimeError(
            "full-system paper summary is incomplete: "
            f"found {full['seed'].nunique()} independent seeds, "
            f"require at least {args.minimum_full_seeds}"
        )

    summary = numeric_summary(per_seed)
    if not summary.empty:
        summary["_track_order"] = summary["track"].map(TRACK_ORDER).fillna(999)
        summary["_model_order"] = summary["model"].map(MODEL_ORDER).fillna(999)
        summary = summary.sort_values(
            ["_track_order", "_model_order", "track", "model"]
        ).drop(columns=["_track_order", "_model_order"])

    architecture = paper_table(per_seed, "architecture")
    full_system = paper_table(per_seed, "full-system")
    combined = pd.concat([architecture, full_system], ignore_index=True)

    per_seed.to_csv(output / "benchmark_per_seed.csv", index=False)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    architecture.to_csv(output / "architecture_paper_table.csv", index=False)
    full_system.to_csv(output / "full_system_paper_table.csv", index=False)
    combined.to_csv(output / "benchmark_paper_table.csv", index=False)
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
