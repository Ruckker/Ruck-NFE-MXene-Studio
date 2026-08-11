from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BLINDED_REVIEW_SCHEMA = "verified-nfe-blinded-review-sheet-1.0"
FORBIDDEN_REVIEW_COLUMNS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reviewer-facing sheet with pseudo/model labels removed from a preselected verification queue."
    )
    parser.add_argument("--queue", required=True)
    parser.add_argument(
        "--selection-manifest",
        help="defaults to <queue_stem>.selection.json beside the queue",
    )
    parser.add_argument("--output", default="training/evaluation/results/verified_review_sheet_blinded.csv")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    queue_path = Path(args.queue).resolve()
    if not queue_path.is_file():
        raise FileNotFoundError(queue_path)
    selection_path = (
        Path(args.selection_manifest).resolve()
        if args.selection_manifest
        else queue_path.with_name(f"{queue_path.stem}.selection.json")
    )
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("queue_sha256") != _sha256(queue_path):
        raise ValueError("selected review queue bytes do not match its selection manifest")

    queue = pd.read_csv(queue_path)
    if "Structure_Name" not in queue:
        raise ValueError("review queue is missing Structure_Name")
    ids = queue["Structure_Name"].fillna("").astype(str).str.strip()
    if (ids == "").any() or ids.duplicated().any():
        raise ValueError("review queue requires unique non-empty Structure_Name values")

    reviewer_columns = [
        column for column in queue.columns if column not in FORBIDDEN_REVIEW_COLUMNS
    ]
    sheet = queue[reviewer_columns].copy()
    leaked = FORBIDDEN_REVIEW_COLUMNS & set(sheet.columns)
    if leaked:
        raise RuntimeError(f"blinded review sheet still leaks target/prediction columns: {sorted(leaked)}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output, index=False)
    manifest = {
        "schema": BLINDED_REVIEW_SCHEMA,
        "selection_manifest": str(selection_path),
        "selection_protocol_sha256": selection.get("selection_protocol_sha256"),
        "source_queue_sha256": _sha256(queue_path),
        "blinded_sheet_sha256": _sha256(output),
        "rows": int(len(sheet)),
        "forbidden_columns_removed": sorted(FORBIDDEN_REVIEW_COLUMNS & set(queue.columns)),
        "review_instruction": (
            "Freeze/version the completed blinded sheet before joining pseudo-label strata or model predictions."
        ),
    }
    manifest_path = output.with_name(f"{output.stem}.blinding.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
