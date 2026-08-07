from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from .common import flatten_result, mean_std_text
except ImportError:
    from common import flatten_result, mean_std_text


MODEL_ORDER = {
    "dummy": 0,
    "xgboost": 1,
    "cgcnn": 2,
    "schnet": 3,
    "alignn": 4,
    "m3gnet": 5,
    "ours": 6,
}

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate NFE baseline benchmark results.")
    parser.add_argument("--results-root", default="training/baselines/results")
    parser.add_argument("--output-dir", default="training/baselines/results")
    return parser.parse_args(argv)


def load_results(root: Path) -> list[dict]:
    results = []
    for path in sorted(root.glob("*/seed_*/result.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "nfe-baseline-result-1.0":
            continue
        row = flatten_result(payload)
        row["result_path"] = str(path)
        results.append(row)
    return results


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"model", "result_path"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for model, group in frame.groupby("model", sort=False):
        row = {"model": model, "n_runs": len(group)}
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        row = {"Model": model}
        for metric in PAPER_METRICS:
            if metric not in group:
                row[metric] = ""
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            row[metric] = mean_std_text(values)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_order"] = result["Model"].map(MODEL_ORDER).fillna(999)
        result = result.sort_values(["_order", "Model"]).drop(columns="_order")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_results(root)
    if not rows:
        raise SystemExit(f"no result.json files found under {root}")
    per_seed = pd.DataFrame(rows)
    per_seed["_order"] = per_seed["model"].map(MODEL_ORDER).fillna(999)
    per_seed = per_seed.sort_values(["_order", "model", "seed"]).drop(columns="_order")
    summary = numeric_summary(per_seed)
    if not summary.empty:
        summary["_order"] = summary["model"].map(MODEL_ORDER).fillna(999)
        summary = summary.sort_values(["_order", "model"]).drop(columns="_order")
    paper = paper_table(per_seed)
    per_seed.to_csv(output / "benchmark_per_seed.csv", index=False)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    paper.to_csv(output / "benchmark_paper_table.csv", index=False)
    print(paper.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
