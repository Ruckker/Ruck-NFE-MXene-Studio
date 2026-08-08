from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from training.evaluation.freeze_verified_review import main as _legacy_main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify preregistered review membership, then run the strict blinded review freeze."
    )
    parser.add_argument("--review-sheet", required=True)
    parser.add_argument("--selection-queue", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--blinding-manifest", required=True)
    parser.add_argument("--reviewer-blinded-to-model-predictions", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def _ids(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "Structure_Name" not in frame:
        raise ValueError(f"{path} is missing Structure_Name")
    ids = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (ids == "").any() or ids.duplicated().any():
        raise ValueError(f"{path} requires unique non-empty Structure_Name values")
    return ids.tolist()


def main() -> int:
    args = parse_args()
    review_path = Path(args.review_sheet).resolve()
    queue_path = Path(args.selection_queue).resolve()
    selection_path = Path(args.selection_manifest).resolve()
    blinding_path = Path(args.blinding_manifest).resolve()
    for path in (review_path, queue_path, selection_path, blinding_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    blinding = json.loads(blinding_path.read_text(encoding="utf-8"))
    if selection.get("queue_sha256") != _sha256(queue_path):
        raise ValueError("selection queue bytes do not match the preregistered selection manifest")
    if blinding.get("source_queue_sha256") != _sha256(queue_path):
        raise ValueError("blinding manifest does not refer to the supplied preregistered queue")
    if str(blinding.get("selection_protocol_sha256", "")) != str(
        selection.get("selection_protocol_sha256", "")
    ):
        raise ValueError("selection and blinding manifests disagree on selection protocol identity")

    selected_ids = _ids(queue_path)
    reviewed_ids = _ids(review_path)
    if set(selected_ids) != set(reviewed_ids) or len(selected_ids) != len(reviewed_ids):
        missing = sorted(set(selected_ids) - set(reviewed_ids))
        added = sorted(set(reviewed_ids) - set(selected_ids))
        raise RuntimeError(
            "completed verified review membership differs from the preregistered prediction-blind queue: "
            f"missing={missing[:5]} added={added[:5]}"
        )

    # Delegate the field-level blinded/label/confidence/score-definition checks
    # to the base freezer while preserving its CLI contract.
    import sys

    delegated = [
        "freeze_verified_review",
        "--review-sheet",
        str(review_path),
        "--blinding-manifest",
        str(blinding_path),
    ]
    if args.reviewer_blinded_to_model_predictions:
        delegated.append("--reviewer-blinded-to-model-predictions")
    if args.output:
        delegated.extend(["--output", str(Path(args.output).resolve())])
    original_argv = sys.argv
    try:
        sys.argv = delegated
        return _legacy_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
