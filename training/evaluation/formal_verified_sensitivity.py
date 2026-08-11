from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.prediction_manifest import load_prediction_manifest
from nfe_model.utils import save_json
from training.evaluation.evaluate_verified_nfe import _parse_bool, _review_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal verified-NFE sensitivity analysis across reviewer-confidence thresholds."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.0,0.6,0.8,0.9",
        help="comma-separated Reviewer_Confidence thresholds; 0.0 should remain the primary all-reviewed analysis",
    )
    parser.add_argument("--require-effective-mass", action="store_true")
    parser.add_argument(
        "--output", default="training/evaluation/results/verified_confidence_sensitivity.json"
    )
    return parser.parse_args()


def _thresholds(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value < 0 or value > 1 for value in values):
        raise ValueError("verified confidence thresholds must be non-empty values in [0,1]")
    if 0.0 not in values:
        raise ValueError("formal verified sensitivity must include threshold 0.0 as the all-reviewed primary analysis")
    return values


def _validate_unique_ids(frame: pd.DataFrame, name: str) -> None:
    if "Structure_Name" not in frame:
        raise ValueError(f"{name} is missing Structure_Name")
    ids = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (ids == "").any() or ids.duplicated().any():
        raise ValueError(f"{name} requires unique non-empty Structure_Name values")


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    label_text = frame["Verified_NFE_Label"].astype(str).str.strip().str.lower()
    labels = label_text.map(LABEL_TO_INDEX).to_numpy(np.int64)
    probabilities = frame[
        ["Probability_Low", "Probability_Medium", "Probability_High"]
    ].to_numpy(float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("verified prediction probabilities must be finite and non-negative")
    row_sum = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("verified prediction probabilities contain zero-sum rows")
    probabilities = probabilities / row_sum
    result = classification_metrics(np.log(np.clip(probabilities, 1e-12, 1.0)), labels)

    if "Verified_NFE_Score" in frame and "Predicted_NFE_Pseudo_Score" in frame:
        truth = pd.to_numeric(frame["Verified_NFE_Score"], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(
            frame["Predicted_NFE_Pseudo_Score"], errors="coerce"
        ).to_numpy(float)
        mask = np.isfinite(truth) & np.isfinite(prediction)
        if mask.any():
            result.update(
                regression_metrics(
                    prediction[:, None], truth[:, None], mask[:, None], ["Verified_NFE_Score"]
                )
            )
            result["verified_score_support"] = float(mask.sum())
    result["support"] = float(len(frame))
    for label, index in LABEL_TO_INDEX.items():
        result[f"{label}_support"] = float(np.sum(labels == index))
    return result


def main() -> int:
    args = parse_args()
    thresholds = _thresholds(args.thresholds)
    load_prediction_manifest(args.predictions, expected_split="test")
    predictions = pd.read_csv(args.predictions)
    verified = pd.read_csv(args.verified)
    _validate_unique_ids(predictions, "predictions")
    _validate_unique_ids(verified, "verified")

    required = {
        "Structure_Name",
        "Verified_NFE_Label",
        "Charge_Localization_Reviewed",
        "Charge_Localization_Confirmed",
        "Parabolic_Dispersion_Reviewed",
        "Parabolic_Dispersion_Confirmed",
        "Reviewer_Confidence",
    }
    missing = required - set(verified.columns)
    if missing:
        raise ValueError(f"verified table is missing columns: {sorted(missing)}")

    charge_complete = _review_complete(
        verified, "Charge_Localization_Reviewed", "Charge_Localization_Confirmed"
    )
    dispersion_complete = _review_complete(
        verified, "Parabolic_Dispersion_Reviewed", "Parabolic_Dispersion_Confirmed"
    )
    complete = charge_complete & dispersion_complete
    if args.require_effective_mass:
        complete &= _review_complete(
            verified, "Effective_Mass_Reviewed", "Effective_Mass_Consistent"
        )

    confidence_raw = verified["Reviewer_Confidence"]
    confidence = pd.to_numeric(confidence_raw, errors="coerce")
    provided = confidence_raw.notna() & (confidence_raw.astype(str).str.strip() != "")
    if (provided & confidence.isna()).any() or ((confidence < 0) | (confidence > 1)).fillna(False).any():
        raise ValueError("Reviewer_Confidence must be numeric in [0,1] when provided")
    if (complete & confidence.isna()).any():
        examples = verified.loc[complete & confidence.isna(), "Structure_Name"].astype(str).head(5).tolist()
        raise ValueError(
            "every review-complete verified row requires Reviewer_Confidence for sensitivity analysis; "
            f"examples={examples}"
        )

    label_text = verified["Verified_NFE_Label"].astype(str).str.strip().str.lower()
    invalid = complete & ~label_text.isin(set(LABEL_TO_INDEX))
    if invalid.any():
        examples = verified.loc[invalid, ["Structure_Name", "Verified_NFE_Label"]].head(5).to_dict("records")
        raise ValueError(f"review-complete rows contain invalid verified labels: {examples}")

    reviewed = verified[complete].copy()
    reviewed["Reviewer_Confidence"] = confidence[complete].astype(float)
    prediction_ids = set(predictions["Structure_Name"].astype(str))
    missing_prediction = [
        value for value in reviewed["Structure_Name"].astype(str) if value not in prediction_ids
    ]
    if missing_prediction:
        raise RuntimeError(
            "formal verified sensitivity requires predictions for every review-complete row; "
            f"missing={len(missing_prediction)} examples={missing_prediction[:5]}"
        )
    joined = reviewed.merge(predictions, on="Structure_Name", how="left", validate="one_to_one")

    result = {
        "primary_definition": "all review-complete cases (Reviewer_Confidence >= 0.0)",
        "thresholds": thresholds,
        "review_complete_support": int(len(joined)),
        "analyses": {},
    }
    for threshold in thresholds:
        subset = joined[joined["Reviewer_Confidence"] >= threshold].copy()
        if subset.empty:
            result["analyses"][str(threshold)] = {"support": 0.0}
        else:
            result["analyses"][str(threshold)] = _metrics(subset)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
