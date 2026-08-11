from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from nfe_model.data_v2 import LABEL_TO_INDEX
from nfe_model.prediction_manifest import prediction_data_identity
from nfe_model.provenance_v2 import canonical_sha256


PAPER_FROZEN_SCHEMA = "verified-nfe-paper-frozen-review-1.2"
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
        description="Freeze the paper verified-NFE review only after proving preregistered membership and prediction blinding."
    )
    parser.add_argument("--review-sheet", required=True)
    parser.add_argument("--selection-queue", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--blinding-manifest", required=True)
    parser.add_argument(
        "--confirm-reviewer-blinded-to-model-predictions",
        action="store_true",
        help="required for paper-ready independent verified validation",
    )
    parser.add_argument(
        "--evidence-root",
        required=True,
        help="root directory for relative evidence paths listed in Evidence_File",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids(frame: pd.DataFrame, name: str) -> list[str]:
    if "Structure_Name" not in frame:
        raise ValueError(f"{name} is missing Structure_Name")
    values = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (values == "").any() or values.duplicated().any():
        raise ValueError(f"{name} requires unique non-empty Structure_Name values")
    return values.tolist()


def _bool(value) -> bool | None:
    text = str(value).strip().lower()
    if text in TRUE:
        return True
    if text in FALSE:
        return False
    return None


def main() -> int:
    args = parse_args()
    if not args.confirm_reviewer_blinded_to_model_predictions:
        raise RuntimeError(
            "paper freeze requires explicit --confirm-reviewer-blinded-to-model-predictions; "
            "use exploratory tools and disclose the limitation for unblinded review"
        )
    review_path = Path(args.review_sheet).resolve()
    queue_path = Path(args.selection_queue).resolve()
    selection_path = Path(args.selection_manifest).resolve()
    blinding_path = Path(args.blinding_manifest).resolve()
    for path in (review_path, queue_path, selection_path, blinding_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    blinding = json.loads(blinding_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "verified-nfe-review-selection-2.0":
        raise ValueError(
            "paper freeze requires the unambiguous v2 review selection manifest from build_verified_review_queue_paper.py"
        )
    if blinding.get("schema") != "verified-nfe-blinded-review-sheet-1.0":
        raise ValueError("paper freeze requires the blinded review-sheet manifest")
    if selection.get("queue_sha256") != _sha256(queue_path):
        raise ValueError("selection queue bytes do not match selection manifest")
    if blinding.get("source_queue_sha256") != _sha256(queue_path):
        raise ValueError("blinding manifest does not point to the supplied selection queue")
    if str(blinding.get("selection_protocol_sha256", "")) != str(
        selection.get("selection_protocol_sha256", "")
    ):
        raise ValueError("selection/blinding manifests disagree on selection protocol")

    selection_provenance = selection.get("provenance")
    if not isinstance(selection_provenance, Mapping):
        raise ValueError("paper selection manifest has no benchmark provenance mapping")
    data_identity = prediction_data_identity(selection_provenance)
    data_identity_sha256 = canonical_sha256(data_identity)
    if str(selection.get("dataset_table_sha256", "")) != str(
        data_identity["dataset_table_sha256"]
    ):
        raise ValueError("selection manifest dataset hash disagrees with its provenance")
    if str(selection.get("split_manifest_sha256", "")) != str(
        data_identity["split_manifest_sha256"]
    ):
        raise ValueError("selection manifest split hash disagrees with its provenance")

    queue = pd.read_csv(queue_path)
    review = pd.read_csv(review_path)
    selected_ids = _ids(queue, "selection queue")
    reviewed_ids = _ids(review, "review sheet")
    if set(selected_ids) != set(reviewed_ids) or len(selected_ids) != len(reviewed_ids):
        missing = sorted(set(selected_ids) - set(reviewed_ids))
        added = sorted(set(reviewed_ids) - set(selected_ids))
        raise RuntimeError(
            "review membership differs from preregistered queue: "
            f"missing={missing[:5]} added={added[:5]}"
        )

    leaked = FORBIDDEN_MODEL_COLUMNS & set(review.columns)
    if leaked:
        raise RuntimeError(
            f"paper review sheet leaks pseudo/model columns: {sorted(leaked)}"
        )
    required = {
        "Charge_Localization_Reviewed",
        "Charge_Localization_Confirmed",
        "Parabolic_Dispersion_Reviewed",
        "Parabolic_Dispersion_Confirmed",
        "Verified_NFE_Label",
        "Reviewer_Confidence",
        "Evidence_File",
    }
    missing_columns = required - set(review.columns)
    if missing_columns:
        raise ValueError(f"paper review sheet is missing columns: {sorted(missing_columns)}")

    evidence_root = Path(args.evidence_root).resolve()
    if not evidence_root.is_dir():
        raise FileNotFoundError(evidence_root)
    evidence_by_structure: dict[str, list[dict[str, str]]] = {}
    for row in review.itertuples(index=False):
        structure_name = str(getattr(row, "Structure_Name")).strip()
        raw = str(getattr(row, "Evidence_File")).strip()
        entries = [value.strip() for value in raw.split(";") if value.strip()]
        if not entries:
            raise RuntimeError(
                f"paper physical NFE review requires evidence files for {structure_name}"
            )
        artifacts: list[dict[str, str]] = []
        for value in entries:
            candidate = (evidence_root / value).resolve()
            try:
                candidate.relative_to(evidence_root)
            except ValueError as exc:
                raise ValueError(
                    f"evidence path escapes --evidence-root for {structure_name}: {value}"
                ) from exc
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"missing physical NFE evidence for {structure_name}: {candidate}"
                )
            artifacts.append(
                {
                    "relative_path": candidate.relative_to(evidence_root).as_posix(),
                    "sha256": _sha256(candidate),
                }
            )
        evidence_by_structure[structure_name] = artifacts

    parsed = {}
    for column in (
        "Charge_Localization_Reviewed",
        "Charge_Localization_Confirmed",
        "Parabolic_Dispersion_Reviewed",
        "Parabolic_Dispersion_Confirmed",
    ):
        parsed[column] = review[column].map(_bool)
        if parsed[column].isna().any():
            examples = review.loc[parsed[column].isna(), "Structure_Name"].astype(str).head(5).tolist()
            raise ValueError(f"{column} has invalid/incomplete boolean values: {examples}")
    if not parsed["Charge_Localization_Reviewed"].astype(bool).all():
        raise RuntimeError("paper verified set requires charge-localization review complete for every selected row")
    if not parsed["Parabolic_Dispersion_Reviewed"].astype(bool).all():
        raise RuntimeError("paper verified set requires parabolic-dispersion review complete for every selected row")

    labels = review["Verified_NFE_Label"].fillna("").astype(str).str.strip().str.lower()
    if (~labels.isin(set(LABEL_TO_INDEX))).any():
        examples = review.loc[
            ~labels.isin(set(LABEL_TO_INDEX)), ["Structure_Name", "Verified_NFE_Label"]
        ].head(5).to_dict("records")
        raise ValueError(f"paper review contains invalid verified labels: {examples}")
    confidence = pd.to_numeric(review["Reviewer_Confidence"], errors="coerce")
    if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("every paper review row requires Reviewer_Confidence in [0,1]")

    score_definition = None
    score_support = 0
    if "Verified_NFE_Score" in review:
        score_raw = review["Verified_NFE_Score"]
        provided = score_raw.notna() & (score_raw.astype(str).str.strip() != "")
        score = pd.to_numeric(score_raw, errors="coerce")
        if (provided & score.isna()).any():
            raise ValueError("Verified_NFE_Score contains nonnumeric provided values")
        score_support = int(provided.sum())
        if score_support:
            if "Verified_NFE_Score_Definition" not in review:
                raise ValueError("provided Verified_NFE_Score values require an explicit definition column")
            definitions = {
                value
                for value in review.loc[provided, "Verified_NFE_Score_Definition"]
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
        else review_path.with_name(f"{review_path.stem}.paper_frozen.json")
    )
    payload = {
        "schema": PAPER_FROZEN_SCHEMA,
        "review_sheet": str(review_path),
        "review_sheet_sha256": _sha256(review_path),
        "selection_queue_sha256": _sha256(queue_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "blinding_manifest_sha256": _sha256(blinding_path),
        "selection_protocol_sha256": selection["selection_protocol_sha256"],
        "selection_mode": selection["mode"],
        "selection_seed": selection["selection_seed"],
        "data_identity": data_identity,
        "data_identity_sha256": data_identity_sha256,
        "rows": int(len(review)),
        "verified_class_support": labels.value_counts().sort_index().to_dict(),
        "verified_score_support": score_support,
        "verified_score_definition": score_definition,
        "reviewer_blinded_to_model_predictions": True,
        "membership_exactly_matches_preregistered_queue": True,
        "physical_evidence_by_structure": evidence_by_structure,
        "physical_evidence_manifest_sha256": canonical_sha256(evidence_by_structure),
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
