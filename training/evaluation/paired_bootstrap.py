from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired Split_Group block bootstrap for two prediction files.")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--name-a", default="A")
    p.add_argument("--name-b", default="B")
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--output", default="training/evaluation/results/paired_bootstrap.json")
    return p.parse_args()


def _classification(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    labels = frame["True_Label"].astype(str).str.lower().map(LABEL_TO_INDEX).fillna(-1).to_numpy(np.int64)
    probs = frame[[f"{prefix}_Probability_Low", f"{prefix}_Probability_Medium", f"{prefix}_Probability_High"]].to_numpy(float)
    return classification_metrics(np.log(np.clip(probs, 1e-12, 1.0)), labels)


def _values(frame: pd.DataFrame) -> dict[str, float]:
    ma = _classification(frame, "A")
    mb = _classification(frame, "B")
    result = {
        "macro_f1": ma.get("macro_f1", np.nan) - mb.get("macro_f1", np.nan),
        "balanced_accuracy": ma.get("balanced_accuracy", np.nan) - mb.get("balanced_accuracy", np.nan),
        "macro_average_precision": ma.get("macro_average_precision", np.nan) - mb.get("macro_average_precision", np.nan),
        "high_f1": ma.get("high_f1", np.nan) - mb.get("high_f1", np.nan),
    }
    if {"A_Predicted_NFE_Pseudo_Score", "B_Predicted_NFE_Pseudo_Score", "True_NFE_Pseudo_Score"} <= set(frame.columns):
        truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        pa = pd.to_numeric(frame["A_Predicted_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        pb = pd.to_numeric(frame["B_Predicted_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(pa) & np.isfinite(pb)
        if valid.any():
            result["score_mae_improvement"] = float(np.mean(np.abs(pb[valid] - truth[valid])) - np.mean(np.abs(pa[valid] - truth[valid])))
    return result


def main() -> int:
    args = parse_args()
    a = pd.read_csv(args.a)
    b = pd.read_csv(args.b)
    keep = ["Structure_Name", "Split_Group", "True_Label", "Probability_Low", "Probability_Medium", "Probability_High", "True_NFE_Pseudo_Score", "Predicted_NFE_Pseudo_Score"]
    a = a[[c for c in keep if c in a]].copy()
    b = b[[c for c in keep if c in b]].copy()
    a = a.rename(columns={c: f"A_{c}" for c in a.columns if c not in {"Structure_Name", "Split_Group"}})
    b = b.rename(columns={c: f"B_{c}" for c in b.columns if c not in {"Structure_Name", "Split_Group"}})
    joined = a.merge(b, on=["Structure_Name", "Split_Group"], how="inner", validate="one_to_one")
    if joined.empty:
        raise RuntimeError("prediction files have no paired rows")
    if "A_True_Label" in joined and "B_True_Label" in joined:
        mismatch = joined["A_True_Label"].astype(str) != joined["B_True_Label"].astype(str)
        if mismatch.any():
            raise RuntimeError("paired files disagree on True_Label")
        joined["True_Label"] = joined["A_True_Label"]
    if "A_True_NFE_Pseudo_Score" in joined and "B_True_NFE_Pseudo_Score" in joined:
        left = pd.to_numeric(joined["A_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(joined["B_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        comparable = np.isfinite(left) & np.isfinite(right)
        if comparable.any() and not np.allclose(left[comparable], right[comparable], atol=1e-10, rtol=0.0):
            raise RuntimeError("paired files disagree on True_NFE_Pseudo_Score")
        joined["True_NFE_Pseudo_Score"] = left
    groups = joined["Split_Group"].fillna("").astype(str)
    if (groups == "").any():
        groups = joined["Structure_Name"].astype(str)
    unique_groups = groups.unique()
    rng = np.random.default_rng(args.seed)
    observed = _values(joined)
    samples = {key: [] for key in observed}
    by_group = {g: np.flatnonzero(groups.to_numpy() == g) for g in unique_groups}
    for _ in range(args.iterations):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[g] for g in selected])
        values = _values(joined.iloc[indices])
        for key in samples:
            samples[key].append(values.get(key, np.nan))
    result = {
        "comparison": f"{args.name_a} - {args.name_b}",
        "paired_rows": len(joined),
        "bootstrap_unit": "Split_Group",
        "iterations": args.iterations,
        "metrics": {},
    }
    for key, observed_value in observed.items():
        arr = np.asarray(samples[key], dtype=float)
        arr = arr[np.isfinite(arr)]
        result["metrics"][key] = {
            "observed_improvement": float(observed_value),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "probability_A_better": float(np.mean(arr > 0)),
        }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
