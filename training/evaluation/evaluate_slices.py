from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.utils import save_json


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
    parser = argparse.ArgumentParser(description="Evaluate one prediction file over common OOD slices.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", default="training/evaluation/ood_manifest.csv")
    parser.add_argument("--output", default="training/evaluation/results/ood_slice_metrics.json")
    return parser.parse_args()


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    true = (
        frame["True_Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_TO_INDEX)
        .fillna(-1)
        .to_numpy(np.int64)
    )
    probabilities = frame[
        ["Probability_Low", "Probability_Medium", "Probability_High"]
    ].to_numpy(float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("slice probabilities must be finite and non-negative")
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("slice probabilities contain zero-sum rows")
    probabilities = probabilities / totals
    result = classification_metrics(np.log(np.clip(probabilities, 1e-12, 1.0)), true)
    if {"True_NFE_Pseudo_Score", "Predicted_NFE_Pseudo_Score"} <= set(frame.columns):
        truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(
            frame["Predicted_NFE_Pseudo_Score"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(truth) & np.isfinite(prediction)
        if valid.any():
            result.update(
                regression_metrics(
                    prediction[:, None],
                    truth[:, None],
                    valid[:, None],
                    ["NFE_Pseudo_Score"],
                )
            )
    result["support"] = float(len(frame))
    return result


def _boolean_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    invalid = ~text.isin({"true", "false", "1", "0", "yes", "no"})
    if invalid.any():
        examples = sorted(text[invalid].unique())[:5]
        raise ValueError(f"OOD manifest contains unparseable boolean values: {examples}")
    return text.isin({"true", "1", "yes"})


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    for name, frame in (("predictions", predictions), ("manifest", manifest)):
        if frame["Structure_Name"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate Structure_Name values")
    joined = predictions.merge(manifest, on="Structure_Name", how="inner", validate="one_to_one")
    if "Suggested_Split" in joined:
        joined = joined[joined["Suggested_Split"].astype(str).str.lower() == "test"]
    if joined.empty:
        raise RuntimeError("no test predictions matched the OOD manifest")
    if len(joined) != len(predictions):
        missing = set(predictions["Structure_Name"].astype(str)) - set(joined["Structure_Name"].astype(str))
        raise RuntimeError(
            f"OOD manifest did not match every prediction row; missing={len(missing)} examples={sorted(missing)[:5]}"
        )

    results = {"all_test": _metrics(joined)}
    for column in SLICE_COLUMNS:
        if column not in joined:
            raise ValueError(f"OOD manifest is missing required slice column {column}")
        mask = _boolean_mask(joined[column])
        results[column] = _metrics(joined[mask]) if mask.any() else {"support": 0.0}
        results[f"not_{column}"] = _metrics(joined[~mask]) if (~mask).any() else {"support": 0.0}
    save_json(Path(args.output).resolve(), results)
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
