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
    "OOD_Large_Cell_Representation",
    "OOD_Any_Chemistry",
    "OOD_Any",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one formal test prediction file over OOD slices.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", default="training/evaluation/ood_manifest.csv")
    parser.add_argument("--output", default="training/evaluation/results/ood_slice_metrics.json")
    return parser.parse_args()


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    label_text = frame["True_Label"].fillna("").astype(str).str.strip().str.lower()
    invalid_label = ~label_text.isin(set(LABEL_TO_INDEX))
    if invalid_label.any():
        examples = frame.loc[invalid_label, ["Structure_Name", "True_Label"]].head(5).to_dict("records")
        raise ValueError(f"OOD slice contains invalid/missing formal True_Label values: {examples}")
    true = label_text.map(LABEL_TO_INDEX).to_numpy(np.int64)

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

    truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    prediction = pd.to_numeric(
        frame["Predicted_NFE_Pseudo_Score"], errors="coerce"
    ).to_numpy(float)
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError(
            "formal OOD slice requires finite True_NFE_Pseudo_Score and Predicted_NFE_Pseudo_Score on every row"
        )
    result.update(
        regression_metrics(
            prediction[:, None],
            truth[:, None],
            np.ones((len(frame), 1), dtype=bool),
            ["NFE_Pseudo_Score"],
        )
    )
    result["support"] = float(len(frame))
    return result


def _boolean_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.fillna("").astype(str).str.strip().str.lower()
    invalid = ~text.isin({"true", "false", "1", "0", "yes", "no"})
    if invalid.any():
        examples = sorted(text[invalid].unique())[:5]
        raise ValueError(f"OOD manifest contains unparseable boolean values: {examples}")
    return text.isin({"true", "1", "yes"})


def _validate_identifiers(frame: pd.DataFrame, name: str) -> pd.Series:
    if "Structure_Name" not in frame:
        raise ValueError(f"{name} is missing Structure_Name")
    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (identifiers == "").any() or identifiers.duplicated().any():
        raise ValueError(f"{name} requires unique non-empty Structure_Name values")
    return identifiers


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    prediction_required = {
        "Structure_Name",
        "True_Label",
        "Probability_Low",
        "Probability_Medium",
        "Probability_High",
        "True_NFE_Pseudo_Score",
        "Predicted_NFE_Pseudo_Score",
    }
    missing_prediction = prediction_required - set(predictions.columns)
    if missing_prediction:
        raise ValueError(f"prediction file is missing columns: {sorted(missing_prediction)}")
    manifest_required = {"Structure_Name", "Suggested_Split", *SLICE_COLUMNS}
    missing_manifest = manifest_required - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"OOD manifest is missing required columns: {sorted(missing_manifest)}")

    prediction_ids = _validate_identifiers(predictions, "predictions")
    manifest_ids = _validate_identifiers(manifest, "manifest")
    manifest_id_set = set(manifest_ids)
    missing_ids = [identifier for identifier in prediction_ids if identifier not in manifest_id_set]
    if missing_ids:
        raise RuntimeError(
            f"OOD manifest did not match every prediction row; missing={len(missing_ids)} examples={missing_ids[:5]}"
        )

    joined = predictions.merge(manifest, on="Structure_Name", how="left", validate="one_to_one")
    split = joined["Suggested_Split"].fillna("").astype(str).str.strip().str.lower().replace(
        {"val": "validation", "valid": "validation"}
    )
    if not (split == "test").all():
        examples = joined.loc[split != "test", ["Structure_Name", "Suggested_Split"]].head(5).to_dict("records")
        raise RuntimeError(
            "formal OOD evaluator expects a test_predictions.csv file only; "
            f"non-test/missing rows={examples}"
        )
    if joined.empty:
        raise RuntimeError("no test predictions matched the OOD manifest")

    results = {"all_test": _metrics(joined)}
    for column in SLICE_COLUMNS:
        mask = _boolean_mask(joined[column])
        results[column] = _metrics(joined[mask]) if mask.any() else {"support": 0.0}
        results[f"not_{column}"] = _metrics(joined[~mask]) if (~mask).any() else {"support": 0.0}
    save_json(Path(args.output).resolve(), results)
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
