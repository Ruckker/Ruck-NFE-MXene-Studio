from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics
from nfe_model.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired Split_Group block bootstrap for two audited prediction files."
    )
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default="training/evaluation/results/paired_bootstrap.json")
    return parser.parse_args()


def _classification(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    labels = (
        frame["True_Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_TO_INDEX)
        .fillna(-1)
        .to_numpy(np.int64)
    )
    probs = frame[
        [
            f"{prefix}_Probability_Low",
            f"{prefix}_Probability_Medium",
            f"{prefix}_Probability_High",
        ]
    ].to_numpy(float)
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError(f"{prefix} probabilities must be finite and non-negative")
    sums = probs.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError(f"{prefix} probabilities contain zero-sum rows")
    probs = probs / sums
    return classification_metrics(np.log(np.clip(probs, 1e-12, 1.0)), labels)


def _values(frame: pd.DataFrame) -> dict[str, float]:
    metrics_a = _classification(frame, "A")
    metrics_b = _classification(frame, "B")
    result = {
        "macro_f1": metrics_a.get("macro_f1", np.nan) - metrics_b.get("macro_f1", np.nan),
        "balanced_accuracy": metrics_a.get("balanced_accuracy", np.nan)
        - metrics_b.get("balanced_accuracy", np.nan),
        "macro_average_precision": metrics_a.get("macro_average_precision", np.nan)
        - metrics_b.get("macro_average_precision", np.nan),
        "high_f1": metrics_a.get("high_f1", np.nan) - metrics_b.get("high_f1", np.nan),
    }
    required = {
        "A_Predicted_NFE_Pseudo_Score",
        "B_Predicted_NFE_Pseudo_Score",
        "True_NFE_Pseudo_Score",
    }
    if required <= set(frame.columns):
        truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        prediction_a = pd.to_numeric(
            frame["A_Predicted_NFE_Pseudo_Score"], errors="coerce"
        ).to_numpy(float)
        prediction_b = pd.to_numeric(
            frame["B_Predicted_NFE_Pseudo_Score"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(prediction_a) & np.isfinite(prediction_b)
        if valid.any():
            result["score_mae_improvement"] = float(
                np.mean(np.abs(prediction_b[valid] - truth[valid]))
                - np.mean(np.abs(prediction_a[valid] - truth[valid]))
            )
    return result


def _prepare(path: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame["Structure_Name"].astype(str).duplicated().any():
        raise ValueError(f"{prefix} prediction file has duplicate Structure_Name values")
    keep = [
        "Record_Index",
        "Structure_Name",
        "Split_Group",
        "True_Label",
        "Probability_Low",
        "Probability_Medium",
        "Probability_High",
        "True_NFE_Pseudo_Score",
        "Predicted_NFE_Pseudo_Score",
    ]
    frame = frame[[column for column in keep if column in frame]].copy()
    rename = {
        column: f"{prefix}_{column}"
        for column in frame.columns
        if column not in {"Record_Index", "Structure_Name", "Split_Group"}
    }
    return frame.rename(columns=rename)


def main() -> int:
    args = parse_args()
    if args.iterations < 100:
        raise ValueError("--iterations must be at least 100 for a meaningful bootstrap interval")
    a = _prepare(args.a, "A")
    b = _prepare(args.b, "B")

    if "Record_Index" in a and "Record_Index" in b:
        pair_keys = ["Record_Index"]
    else:
        pair_keys = ["Structure_Name", "Split_Group"]
    joined = a.merge(b, on=pair_keys, how="inner", validate="one_to_one", suffixes=("_a", "_b"))
    if joined.empty:
        raise RuntimeError("prediction files have no paired rows")

    # Recover and verify identity metadata when Record_Index was the primary key.
    if pair_keys == ["Record_Index"]:
        if (joined["Structure_Name_a"].astype(str) != joined["Structure_Name_b"].astype(str)).any():
            raise RuntimeError("paired Record_Index rows disagree on Structure_Name")
        joined["Structure_Name"] = joined["Structure_Name_a"]
        group_a = joined["Split_Group_a"].fillna("").astype(str)
        group_b = joined["Split_Group_b"].fillna("").astype(str)
        if (group_a != group_b).any():
            raise RuntimeError("paired Record_Index rows disagree on Split_Group")
        joined["Split_Group"] = group_a

    if "A_True_Label" in joined and "B_True_Label" in joined:
        left_label = joined["A_True_Label"].astype(str).str.strip().str.lower()
        right_label = joined["B_True_Label"].astype(str).str.strip().str.lower()
        if (left_label != right_label).any():
            raise RuntimeError("paired files disagree on True_Label")
        joined["True_Label"] = left_label
    if "A_True_NFE_Pseudo_Score" in joined and "B_True_NFE_Pseudo_Score" in joined:
        left = pd.to_numeric(joined["A_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(joined["B_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        both = np.isfinite(left) & np.isfinite(right)
        one_missing = np.isfinite(left) ^ np.isfinite(right)
        if one_missing.any() or (both.any() and not np.allclose(left[both], right[both], atol=1e-10, rtol=0.0)):
            raise RuntimeError("paired files disagree on True_NFE_Pseudo_Score")
        joined["True_NFE_Pseudo_Score"] = left

    groups = joined["Split_Group"].fillna("").astype(str).copy()
    missing_group = groups.str.strip() == ""
    groups.loc[missing_group] = "__structure__::" + joined.loc[
        missing_group, "Structure_Name"
    ].astype(str)
    unique_groups = groups.unique()
    if not len(unique_groups):
        raise RuntimeError("paired prediction files contain no bootstrap groups")
    by_group = {group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups}

    rng = np.random.default_rng(args.seed)
    observed = _values(joined)
    samples = {key: [] for key in observed}
    for _ in range(args.iterations):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in selected])
        values = _values(joined.iloc[indices])
        for key in samples:
            samples[key].append(values.get(key, np.nan))

    result = {
        "comparison": f"{args.name_a} - {args.name_b}",
        "paired_rows": len(joined),
        "bootstrap_unit": "Split_Group",
        "bootstrap_groups": len(unique_groups),
        "iterations": args.iterations,
        "metrics": {},
    }
    for key, observed_value in observed.items():
        array = np.asarray(samples[key], dtype=float)
        finite = array[np.isfinite(array)]
        if not len(finite):
            result["metrics"][key] = {
                "observed_improvement": observed_value,
                "ci95_low": None,
                "ci95_high": None,
                "probability_A_better": None,
                "valid_iterations": 0,
            }
            continue
        result["metrics"][key] = {
            "observed_improvement": observed_value,
            "ci95_low": float(np.quantile(finite, 0.025)),
            "ci95_high": float(np.quantile(finite, 0.975)),
            "probability_A_better": float(np.mean(finite > 0) + 0.5 * np.mean(finite == 0)),
            "valid_iterations": int(len(finite)),
        }
    save_json(Path(args.output).resolve(), result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
