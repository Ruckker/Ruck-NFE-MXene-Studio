from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.metrics_v2 import classification_metrics, regression_metrics


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "confirmed"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predictions on manually/DFT verified NFE cases.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--verified", required=True)
    p.add_argument("--output-dir", default="training/evaluation/results/verified")
    p.add_argument("--min-confidence", type=float, default=0.8)
    p.add_argument(
        "--require-effective-mass",
        action="store_true",
        help="also require Effective_Mass_Consistent for the evidence gate",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    verified = pd.read_csv(args.verified)
    required = {
        "Structure_Name",
        "Verified_NFE_Label",
        "Charge_Localization_Confirmed",
        "Parabolic_Dispersion_Confirmed",
        "Reviewer_Confidence",
    }
    missing = required - set(verified.columns)
    if missing:
        raise ValueError(f"verified table is missing columns: {sorted(missing)}")
    evidence = _truthy(verified["Charge_Localization_Confirmed"]) & _truthy(
        verified["Parabolic_Dispersion_Confirmed"]
    )
    if args.require_effective_mass:
        if "Effective_Mass_Consistent" not in verified:
            raise ValueError("--require-effective-mass needs Effective_Mass_Consistent")
        evidence &= _truthy(verified["Effective_Mass_Consistent"])
    confidence = pd.to_numeric(verified["Reviewer_Confidence"], errors="coerce").fillna(0.0)
    verified = verified[evidence & (confidence >= args.min_confidence)].copy()
    joined = verified.merge(pred, on="Structure_Name", how="inner", validate="one_to_one")
    if joined.empty:
        raise RuntimeError("no verified structures matched the prediction file")

    label_text = joined["Verified_NFE_Label"].astype(str).str.strip().str.lower()
    labels = label_text.map(LABEL_TO_INDEX).fillna(-1).to_numpy(dtype=np.int64)
    probs = joined[["Probability_Low", "Probability_Medium", "Probability_High"]].to_numpy(float)
    probs = np.clip(probs, 1e-12, 1.0)
    logits = np.log(probs)
    metrics = classification_metrics(logits, labels)

    if "Verified_NFE_Score" in joined and "Predicted_NFE_Pseudo_Score" in joined:
        truth = pd.to_numeric(joined["Verified_NFE_Score"], errors="coerce").to_numpy(float)
        prediction = pd.to_numeric(joined["Predicted_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
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
    metrics["evidence_gate_min_confidence"] = float(args.min_confidence)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "verified_joined_predictions.csv", index=False)
    with (out / "verified_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
