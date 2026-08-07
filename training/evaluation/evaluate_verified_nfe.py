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


def _parse_bool(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
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
    return parser.parse_args()


def _review_complete(frame: pd.DataFrame, reviewed: str, finding: str) -> pd.Series:
    if reviewed not in frame or finding not in frame:
        raise ValueError(f"verified table requires both {reviewed} and {finding}")
    reviewed_value = _parse_bool(frame[reviewed])
    finding_value = _parse_bool(frame[finding])
    return reviewed_value.fillna(False).astype(bool) & finding_value.notna()


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
    for name, frame in (("verified", verified), ("predictions", predictions)):
        if frame["Structure_Name"].astype(str).duplicated().any():
            duplicates = frame.loc[
                frame["Structure_Name"].astype(str).duplicated(keep=False), "Structure_Name"
            ].astype(str).unique()[:5]
            raise ValueError(f"{name} table has duplicate Structure_Name values: {duplicates.tolist()}")

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

    confidence = pd.to_numeric(verified["Reviewer_Confidence"], errors="coerce").fillna(0.0)
    label_text = verified["Verified_NFE_Label"].astype(str).str.strip().str.lower()
    valid_label = label_text.isin(set(LABEL_TO_INDEX))
    rejected_invalid_labels = int((evidence_complete & ~valid_label).sum())
    eligible = evidence_complete & valid_label & (confidence >= args.min_confidence)
    verified = verified[eligible].copy()
    if verified.empty:
        raise RuntimeError("no reviewed verified structures pass the evidence-completeness/confidence gate")

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

    if "Verified_NFE_Score" in joined and "Predicted_NFE_Pseudo_Score" in joined:
        truth = pd.to_numeric(joined["Verified_NFE_Score"], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(
            joined["Predicted_NFE_Pseudo_Score"], errors="coerce"
        ).to_numpy(float)
        mask = np.isfinite(truth) & np.isfinite(prediction)
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
    metrics["evidence_gate_min_confidence"] = float(args.min_confidence)
    metrics["rejected_invalid_verified_labels"] = float(rejected_invalid_labels)
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
