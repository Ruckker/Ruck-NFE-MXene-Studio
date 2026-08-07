from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
    prediction_data_identity,
)
from nfe_model.provenance_v2 import canonical_sha256
from training.evaluation import formal_verified_sensitivity as evaluator


PAPER_FROZEN_SCHEMA = "verified-nfe-paper-frozen-review-1.1"
PAPER_THRESHOLDS = "0.0,0.6,0.8,0.9"


def _argument_value(flag: str) -> str:
    found: list[str] = []
    index = 0
    while index < len(sys.argv):
        token = sys.argv[index]
        if token == flag:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires a value")
            found.append(sys.argv[index + 1])
            index += 2
            continue
        if token.startswith(flag + "="):
            found.append(token.split("=", 1)[1])
        index += 1
    if len(found) != 1:
        raise ValueError(
            f"paper verified evaluator requires exactly one {flag}; observed={len(found)}"
        )
    return found[0]


def _has_option(flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in sys.argv[1:])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_option_with_value(arguments: list[str], flag: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"{flag} requires a value")
            index += 2
            continue
        if token.startswith(flag + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def main() -> int:
    if _has_option("--thresholds"):
        raise ValueError(
            "paper verified confidence thresholds are preregistered as "
            f"{PAPER_THRESHOLDS}; do not override --thresholds"
        )
    if _has_option("--require-effective-mass"):
        raise ValueError(
            "paper primary verified analysis is defined by completed charge-localization and "
            "parabolic-dispersion review; effective-mass-gated analyses are exploratory sensitivity checks"
        )

    review_path = Path(_argument_value("--verified")).resolve()
    frozen_path = Path(_argument_value("--paper-frozen-manifest")).resolve()
    prediction_path = Path(_argument_value("--predictions")).resolve()
    if not review_path.is_file() or not frozen_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(
            "verified review sheet, paper frozen manifest and test predictions must all exist"
        )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("schema") != PAPER_FROZEN_SCHEMA:
        raise ValueError(
            "paper verified evaluation accepts only the current data-bound "
            "freeze_verified_review_paper.py manifest"
        )
    if frozen.get("review_sheet_sha256") != _sha256(review_path):
        raise RuntimeError("verified review sheet bytes differ from the paper-frozen artifact")
    if frozen.get("reviewer_blinded_to_model_predictions") is not True:
        raise RuntimeError("paper verified review is not declared prediction-blinded")
    if frozen.get("membership_exactly_matches_preregistered_queue") is not True:
        raise RuntimeError("paper verified review does not attest exact preregistered membership")

    frozen_identity = frozen.get("data_identity")
    if not isinstance(frozen_identity, dict):
        raise ValueError("paper-frozen verified manifest has no benchmark data_identity")
    canonical_frozen_identity = prediction_data_identity(frozen_identity)
    if canonical_sha256(canonical_frozen_identity) != frozen.get("data_identity_sha256"):
        raise ValueError("paper-frozen verified data identity hash is inconsistent")

    prediction_manifest = load_prediction_manifest(
        prediction_path, expected_split="test"
    )
    assert_same_prediction_data_identity(frozen, prediction_manifest)

    delegated = _remove_option_with_value(list(sys.argv), "--paper-frozen-manifest")
    delegated.extend(["--thresholds", PAPER_THRESHOLDS])
    original = sys.argv
    try:
        sys.argv = delegated
        return evaluator.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
