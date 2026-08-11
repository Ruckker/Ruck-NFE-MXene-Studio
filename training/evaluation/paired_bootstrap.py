from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics
from nfe_model.utils import save_json


REQUIRED_COLUMNS = (
    "Structure_Name",
    "Split_Group",
    "True_Label",
    "Probability_Low",
    "Probability_Medium",
    "Probability_High",
    "True_NFE_Pseudo_Score",
    "Predicted_NFE_Pseudo_Score",
)


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
    label_text = frame["True_Label"].fillna("").astype(str).str.strip().str.lower()
    invalid = ~label_text.isin(set(LABEL_TO_INDEX))
    if invalid.any():
        raise ValueError(f"paired bootstrap contains invalid True_Label values for {prefix}")
    labels = label_text.map(LABEL_TO_INDEX).to_numpy(np.int64)
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
    metrics = classification_metrics(np.log(np.clip(probs, 1e-12, 1.0)), labels)
    metrics["_all_classes_present"] = float(
        all(metrics.get(f"{name}_support", 0.0) > 0 for name in ("low", "medium", "high"))
    )
    return metrics


def _values(frame: pd.DataFrame) -> dict[str, float]:
    metrics_a = _classification(frame, "A")
    metrics_b = _classification(frame, "B")
    all_classes = bool(metrics_a["_all_classes_present"] and metrics_b["_all_classes_present"])
    result = {
        "macro_f1": (
            metrics_a.get("macro_f1", np.nan) - metrics_b.get("macro_f1", np.nan)
            if all_classes
            else np.nan
        ),
        "balanced_accuracy": (
            metrics_a.get("balanced_accuracy", np.nan)
            - metrics_b.get("balanced_accuracy", np.nan)
            if all_classes
            else np.nan
        ),
        "macro_average_precision": (
            metrics_a.get("macro_average_precision", np.nan)
            - metrics_b.get("macro_average_precision", np.nan)
            if all_classes
            else np.nan
        ),
        "high_f1": metrics_a.get("high_f1", np.nan) - metrics_b.get("high_f1", np.nan),
        "high_enrichment_at_5pct": (
            metrics_a.get("high_enrichment_at_5pct", np.nan)
            - metrics_b.get("high_enrichment_at_5pct", np.nan)
        ),
    }

    truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    prediction_a = pd.to_numeric(
        frame["A_Predicted_NFE_Pseudo_Score"], errors="coerce"
    ).to_numpy(float)
    prediction_b = pd.to_numeric(
        frame["B_Predicted_NFE_Pseudo_Score"], errors="coerce"
    ).to_numpy(float)
    if not (
        np.all(np.isfinite(truth))
        and np.all(np.isfinite(prediction_a))
        and np.all(np.isfinite(prediction_b))
    ):
        raise ValueError("formal paired bootstrap requires finite true/predicted NFE scores on every row")
    result["score_mae_improvement"] = float(
        np.mean(np.abs(prediction_b - truth)) - np.mean(np.abs(prediction_a - truth))
    )
    return result


def _prepare(path: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{prefix} prediction file is missing columns: {sorted(missing)}")
    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (identifiers == "").any() or identifiers.duplicated().any():
        raise ValueError(f"{prefix} prediction file requires unique non-empty Structure_Name values")
    groups = frame["Split_Group"].fillna("").astype(str).str.strip()
    if (groups == "").any():
        raise ValueError(
            f"{prefix} formal prediction file contains blank Split_Group; block bootstrap cannot infer chemistry clusters"
        )

    keep = ["Record_Index", *REQUIRED_COLUMNS]
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
        if set(a["Record_Index"].astype(int)) != set(b["Record_Index"].astype(int)):
            raise RuntimeError("paired files do not contain the same Record_Index set")
        pair_keys = ["Record_Index"]
    else:
        if set(a["Structure_Name"].astype(str)) != set(b["Structure_Name"].astype(str)):
            raise RuntimeError("paired files do not contain the same Structure_Name set")
        pair_keys = ["Structure_Name"]
    if len(a) != len(b):
        raise RuntimeError(f"paired prediction files have different row counts: A={len(a)} B={len(b)}")

    joined = a.merge(b, on=pair_keys, how="inner", validate="one_to_one", suffixes=("_a", "_b"))
    if len(joined) != len(a):
        raise RuntimeError("paired merge lost rows despite identical declared sample sets")

    if pair_keys == ["Record_Index"]:
        if (joined["Structure_Name_a"].astype(str) != joined["Structure_Name_b"].astype(str)).any():
            raise RuntimeError("paired Record_Index rows disagree on Structure_Name")
        joined["Structure_Name"] = joined["Structure_Name_a"]
        group_a = joined["Split_Group_a"].fillna("").astype(str).str.strip()
        group_b = joined["Split_Group_b"].fillna("").astype(str).str.strip()
    else:
        group_a = joined["Split_Group_a"].fillna("").astype(str).str.strip()
        group_b = joined["Split_Group_b"].fillna("").astype(str).str.strip()
    if (group_a != group_b).any() or (group_a == "").any():
        raise RuntimeError("paired rows disagree on or are missing Split_Group")
    joined["Split_Group"] = group_a

    left_label = joined["A_True_Label"].fillna("").astype(str).str.strip().str.lower()
    right_label = joined["B_True_Label"].fillna("").astype(str).str.strip().str.lower()
    if (left_label != right_label).any():
        raise RuntimeError("paired files disagree on True_Label")
    if (~left_label.isin(set(LABEL_TO_INDEX))).any():
        raise ValueError("paired files contain invalid/missing True_Label values")
    joined["True_Label"] = left_label

    left = pd.to_numeric(joined["A_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    right = pd.to_numeric(joined["B_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired files require finite True_NFE_Pseudo_Score values")
    if not np.allclose(left, right, atol=1e-10, rtol=0.0):
        raise RuntimeError("paired files disagree on True_NFE_Pseudo_Score")
    joined["True_NFE_Pseudo_Score"] = left

    unique_groups = joined["Split_Group"].unique()
    if len(unique_groups) < 2:
        raise RuntimeError("paired block bootstrap requires at least two distinct Split_Group clusters")
    groups_array = joined["Split_Group"].to_numpy()
    by_group = {group: np.flatnonzero(groups_array == group) for group in unique_groups}

    rng = np.random.default_rng(args.seed)
    observed = _values(joined)
    if not np.isfinite(observed["macro_f1"]):
        raise RuntimeError("full paired sample must contain all three NFE classes")
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
        valid_fraction = float(len(finite) / len(array)) if len(array) else 0.0
        if not len(finite):
            result["metrics"][key] = {
                "observed_improvement": observed_value,
                "ci95_low": None,
                "ci95_high": None,
                "probability_A_better": None,
                "valid_iterations": 0,
                "valid_iteration_fraction": valid_fraction,
            }
            continue
        result["metrics"][key] = {
            "observed_improvement": observed_value,
            "ci95_low": float(np.quantile(finite, 0.025)),
            "ci95_high": float(np.quantile(finite, 0.975)),
            "probability_A_better": float(
                np.mean(finite > 0) + 0.5 * np.mean(finite == 0)
            ),
            "valid_iterations": int(len(finite)),
            "valid_iteration_fraction": valid_fraction,
        }
    save_json(Path(args.output).resolve(), result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
