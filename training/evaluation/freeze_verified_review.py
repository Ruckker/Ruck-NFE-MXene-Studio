from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX


FROZEN_SCHEMA = "verified-nfe-frozen-review-1.0"
FORBIDDEN_MODEL_COLUMNS = {
    "Pseudo_Label_Stratum",
    "Label_Index",
    "Predicted_Label",
    "Predicted_NFE_Pseudo_Score",
    "Probability_Low",
    "Probability_Medium",
    "Probability_High",
    "True_Label",
    "True_NFE_Pseudo_Score",
}
TRUE = {"1", "true", "yes", "y", "confirmed", "pass", "positive"}
FALSE = {"0", "false", "no", "n", "rejected", "fail", "negative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze a reviewer-blinded verified-NFE sheet before joining predictions."
    )
    parser.add_argument("--review-sheet", required=True)
    parser.add_argument("--blinding-manifest", required=True)
    parser.add_argument(
        "--reviewer-blinded-to-model-predictions",
        action="store_true",
        help="explicitly record that reviewers did not inspect model predictions before freezing labels",
    )
    parser.add_argument(
        "--output", help="defaults to <review_sheet_stem>.frozen.json"
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool_value(value) -> bool | None:
    text = str(value).strip().lower()
    if text in TRUE:
        return True
    if text in FALSE:
        return False
    return None


def main() -> int:
    args = parse_args()
    sheet_path = Path(args.review_sheet).resolve()
    blinding_path = Path(args.blinding_manifest).resolve()
    if not sheet_path.is_file() or not blinding_path.is_file():
        raise FileNotFoundError("review sheet and blinding manifest must both exist")
    blinding = json.loads(blinding_path.read_text(encoding="utf-8"))
    if blinding.get("schema") != "verified-nfe-blinded-review-sheet-1.0":
        raise ValueError("unsupported or missing blinded review manifest schema")

    frame = pd.read_csv(sheet_path)
    leaked = FORBIDDEN_MODEL_COLUMNS & set(frame.columns)
    if leaked:
        raise RuntimeError(
            f"cannot freeze a review sheet that exposes pseudo/model prediction columns: {sorted(leaked)}"
        )
    required = {
        "Structure_Name",
        "Charge_Localization_Reviewed",
        "Charge_Localization_Confirmed",
        "Parabolic_Dispersion_Reviewed",
        "Parabolic_Dispersion_Confirmed",
        "Verified_NFE_Label",
        "Reviewer_Confidence",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"completed review sheet is missing columns: {sorted(missing)}")
    ids = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (ids == "").any() or ids.duplicated().any():
        raise ValueError("completed review sheet requires unique non-empty Structure_Name values")

    for column in (
        "Charge_Localization_Reviewed",
        "Charge_Localization_Confirmed",
        "Parabolic_Dispersion_Reviewed",
        "Parabolic_Dispersion_Confirmed",
    ):
        parsed = frame[column].map(_bool_value)
        if parsed.isna().any():
            examples = ids[parsed.isna()].head(5).tolist()
            raise ValueError(f"{column} has incomplete/invalid boolean review values: {examples}")
    reviewed = (
        frame["Charge_Localization_Reviewed"].map(_bool_value).astype(bool)
        & frame["Parabolic_Dispersion_Reviewed"].map(_bool_value).astype(bool)
    )
    if not reviewed.all():
        examples = ids[~reviewed].head(5).tolist()
        raise RuntimeError(
            "formal frozen verified set requires charge-localization and parabolic-dispersion review complete "
            f"for every selected row; incomplete={examples}"
        )

    labels = frame["Verified_NFE_Label"].fillna("").astype(str).str.strip().str.lower()
    if (~labels.isin(set(LABEL_TO_INDEX))).any():
        examples = frame.loc[~labels.isin(set(LABEL_TO_INDEX)), ["Structure_Name", "Verified_NFE_Label"]].head(5).to_dict("records")
        raise ValueError(f"frozen review contains invalid verified labels: {examples}")
    confidence = pd.to_numeric(frame["Reviewer_Confidence"], errors="coerce")
    if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("every frozen review row requires Reviewer_Confidence in [0,1]")

    score_definition = None
    if "Verified_NFE_Score" in frame:
        score = pd.to_numeric(frame["Verified_NFE_Score"], errors="coerce")
        provided = frame["Verified_NFE_Score"].notna() & (
            frame["Verified_NFE_Score"].astype(str).str.strip() != ""
        )
        if (provided & score.isna()).any():
            raise ValueError("Verified_NFE_Score contains nonnumeric provided values")
        if provided.any():
            if "Verified_NFE_Score_Definition" not in frame:
                raise ValueError(
                    "Verified_NFE_Score values require a Verified_NFE_Score_Definition column"
                )
            definitions = {
                value
                for value in frame.loc[provided, "Verified_NFE_Score_Definition"]
                .fillna("")
                .astype(str)
                .str.strip()
                if value
            }
            if len(definitions) != 1:
                raise ValueError(
                    "all provided Verified_NFE_Score values must share one explicit non-empty definition"
                )
            score_definition = next(iter(definitions))

    output = (
        Path(args.output).resolve()
        if args.output
        else sheet_path.with_name(f"{sheet_path.stem}.frozen.json")
    )
    payload = {
        "schema": FROZEN_SCHEMA,
        "review_sheet": str(sheet_path),
        "review_sheet_sha256": _sha256(sheet_path),
        "blinding_manifest": str(blinding_path),
        "selection_protocol_sha256": blinding.get("selection_protocol_sha256"),
        "rows": int(len(frame)),
        "verified_class_support": labels.value_counts().sort_index().to_dict(),
        "reviewer_blinded_to_model_predictions": bool(
            args.reviewer_blinded_to_model_predictions
        ),
        "verified_score_definition": score_definition,
        "warning": (
            None
            if args.reviewer_blinded_to_model_predictions
            else "review was not declared prediction-blinded; disclose this limitation and do not call the set independent blinded validation"
        ),
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
