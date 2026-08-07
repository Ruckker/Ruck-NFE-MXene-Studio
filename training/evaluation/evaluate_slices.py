from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics, regression_metrics


SLICE_COLUMNS = (
    "OOD_Unseen_Metal_Pair",
    "OOD_Unseen_Termination_Pair",
    "OOD_Unseen_X_Element",
    "OOD_Unseen_Element",
    "OOD_Cell_Size",
    "OOD_Any_Chemistry",
    "OOD_Any",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one prediction file over common OOD slices.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--manifest", default="training/evaluation/ood_manifest.csv")
    p.add_argument("--output", default="training/evaluation/results/ood_slice_metrics.json")
    return p.parse_args()


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    true = frame["True_Label"].astype(str).str.lower().map(LABEL_TO_INDEX).fillna(-1).to_numpy(np.int64)
    probs = frame[["Probability_Low", "Probability_Medium", "Probability_High"]].to_numpy(float)
    result = classification_metrics(np.log(np.clip(probs, 1e-12, 1.0)), true)
    if {"True_NFE_Pseudo_Score", "Predicted_NFE_Pseudo_Score"} <= set(frame.columns):
        truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(frame["Predicted_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(pred)
        if valid.any():
            result.update(regression_metrics(pred[:, None], truth[:, None], valid[:, None], ["NFE_Pseudo_Score"]))
    result["support"] = float(len(frame))
    return result


def main() -> int:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    joined = pred.merge(manifest, on="Structure_Name", how="inner", validate="one_to_one")
    if "Suggested_Split" in joined:
        joined = joined[joined["Suggested_Split"].astype(str).str.lower() == "test"]
    results = {"all_test": _metrics(joined)}
    for column in SLICE_COLUMNS:
        if column not in joined:
            continue
        mask = joined[column].astype(str).str.lower().isin({"true", "1", "yes"}) if joined[column].dtype == object else joined[column].astype(bool)
        if mask.any():
            results[column] = _metrics(joined[mask])
        if (~mask).any():
            results[f"not_{column}"] = _metrics(joined[~mask])
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
