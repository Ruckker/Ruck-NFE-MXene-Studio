from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.utils import save_json


_TRUE = {"1", "true", "yes", "y", "confirmed", "pass", "positive"}
_FALSE = {"0", "false", "no", "n", "rejected", "fail", "negative"}
_BOOL_VALUES = _TRUE | _FALSE


def _parse_bool(series: pd.Series, *, column: str) -> pd.Series:
    raw = series.fillna("").astype(str).str.strip()
    text = raw.str.lower()
    invalid = (raw != "") & ~text.isin(_BOOL_VALUES)
    if invalid.any():
        examples = raw[invalid].unique()[:5].tolist()
        raise ValueError(f"verified boolean column {column} contains unrecognized values: {examples}")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[text.isin(_TRUE)] = True
    result[text.isin(_FALSE)] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions on manually/DFT verified NFE cases without positive-only evidence filtering."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--output-dir", default="training/evaluation/results/verified")
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument(
        "--require-effective-mass",
        action="store_true",
        help="require effective-mass review to be completed; consistency itself may be true or false",
    )
    parser.add_argument(
        "--allow-partial-predictions",
        action="store_true",
        help="exploratory only: allow eligible verified rows to be absent from the prediction file",
    )
    return parser.parse_args()


def _review_complete(frame: pd.DataFrame, reviewed: str, finding: str) -> pd.Series:
    if reviewed not in frame or finding not in frame:
        raise ValueError(f"verified table requires both {reviewed} and {finding}")
    reviewed_value = _parse_bool(frame[reviewed], column=reviewed)
    finding_value = _parse_bool(frame[finding], column=finding)
    return reviewed_value.fillna(False).astype(bool) & finding_value.notna()


def _validate_names(frame: pd.DataFrame, name: str) -> None:
    if "Structure_Name" not in frame:
        raise ValueError(f"{name} table is missing Structure_Name")
    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (identifiers == "").any():
        raise ValueError(f"{name} table contains blank Structure_Name values")
    if identifiers.duplicated().any():
        duplicates = identifiers[identifiers.duplicated(keep=False)].unique()[:5]
        raise ValueError(f"{name} table has duplicate Structure_Name values: {duplicates.tolist()}")


def main() -> int:
    args = parse_args()
    if not 0.0 <= float(args.min_confidence) <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")

    predictions = pd.read_csv(args.predictions)
    verified = pd.read_csv(args.verified)
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
    prediction_required = {
        "Structure_Name",
        "Probability_Low",
        "Probability_Medium",
        "Probability_High",
        "Predicted_NFE_Pseudo_Score",
    }
    missing_prediction = prediction_required - set(predictions.columns)
    if missing_prediction:
        raise ValueError(
            f"prediction table is missing columns: {sorted(missing_prediction)}"
        )
    _validate_names(verified, "verified")
    _validate_names(predictions, "predictions")

    charge_reviewed = _review_complete(
        verified, "Charge_Localization_Reviewed", "Charge_Localization_Confirmed"
    )
    dispersion_reviewed = _review_complete(
        verified, "Parabolic_Dispersion_Reviewed", "Parabolic_Dispersion_Confirmed"
    )
    evidence_complete = charge_reviewed & dispersion_reviewed
    if args.require_effective_mass:
        evidence_complete &= _review_complete(
            verified, "Effective_Mass_Reviewed", "Effective_Mass_Consistent"
        )

    raw_confidence = verified["Reviewer_Confidence"]
    confidence = pd.to_numeric(raw_confidence, errors="coerce")
    nonblank_confidence = raw_confidence.fillna("").astype(str).str.strip() != ""
    invalid_confidence = nonblank_confidence & (
        confidence.isna() | (confidence < 0.0) | (confidence > 1.0)
    )
    if invalid_confidence.any():
        examples = raw_confidence[invalid_confidence].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"Reviewer_Confidence must be numeric in [0,1] when provided; examples={examples}"
        )

    label_text = verified["Verified_NFE_Label"].fillna("").astype(str).str.strip().str.lower()
    high_confidence_complete = evidence_complete & confidence.ge(float(args.min_confidence)).fillna(False)
    invalid_label = high_confidence_complete & ~label_text.isin(set(LABEL_TO_INDEX))
    if invalid_label.any():
        examples = verified.loc[
            invalid_label, ["Structure_Name", "Verified_NFE_Label"]
        ].head(5).to_dict("records")
        raise ValueError(
            "review-complete high-confidence verified rows require a valid low/medium/high label; "
            f"examples={examples}"
        )

    eligible = high_confidence_complete & label_text.isin(set(LABEL_TO_INDEX))
    verified = verified[eligible].copy()
    if verified.empty:
        raise RuntimeError("no reviewed verified structures pass the evidence-completeness/confidence gate")

    prediction_ids = set(predictions["Structure_Name"].astype(str).str.strip())
    eligible_ids = verified["Structure_Name"].astype(str).str.strip()
    missing_ids = [identifier for identifier in eligible_ids if identifier not in prediction_ids]
    if missing_ids and not args.allow_partial_predictions:
        raise RuntimeError(
            "formal verified evaluation refuses selective prediction coverage; "
            f"{len(missing_ids)} eligible verified structures are missing, examples={missing_ids[:5]}. "
            "Use --allow-partial-predictions only for explicitly exploratory analysis."
        )

    joined = verified.merge(predictions, on="Structure_Name", how="inner", validate="one_to_one")
    if joined.empty:
        raise RuntimeError("no verified structures matched the prediction file")

    labels = (
        joined["Verified_NFE_Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_TO_INDEX)
        .to_numpy(dtype=np.int64)
    )
    probabilities = joined[
        ["Probability_Low", "Probability_Medium", "Probability_High"]
    ].to_numpy(float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("prediction probabilities must be finite and non-negative")
    row_sum = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("prediction probabilities contain a zero-sum row")
    probabilities = probabilities / row_sum
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    metrics = classification_metrics(logits, labels)

    if "Verified_NFE_Score" in joined:
        truth_raw = joined["Verified_NFE_Score"]
        truth = pd.to_numeric(truth_raw, errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(
            joined["Predicted_NFE_Pseudo_Score"], errors="coerce"
        ).to_numpy(float)
        provided_truth = truth_raw.fillna("").astype(str).str.strip().to_numpy() != ""
        invalid_truth = provided_truth & ~np.isfinite(truth)
        if np.any(invalid_truth):
            raise ValueError("Verified_NFE_Score contains non-numeric/non-finite provided values")
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Predicted_NFE_Pseudo_Score contains non-finite values")
        mask = np.isfinite(truth)
        if mask.any():
            metrics.update(
                regression_metrics(
                    prediction.reshape(-1, 1),
                    truth.reshape(-1, 1),
                    mask.reshape(-1, 1),
                    ["Verified_NFE_Score"],
                )
            )

    metrics["verified_support"] = float(len(joined))
    metrics["verified_eligible_before_prediction_join"] = float(len(verified))
    metrics["verified_prediction_coverage"] = float(len(joined) / len(verified))
    metrics["evidence_gate_min_confidence"] = float(args.min_confidence)
    metrics["partial_prediction_coverage_allowed"] = bool(args.allow_partial_predictions)
    for label, index in LABEL_TO_INDEX.items():
        metrics[f"verified_{label}_support"] = float(np.sum(labels == index))

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "verified_joined_predictions.csv", index=False)
    save_json(out / "verified_metrics.json", metrics)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
