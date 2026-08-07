from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from training.evaluation import formal_verified_sensitivity as evaluator


def _argument_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError as exc:
        raise ValueError(f"paper verified evaluator requires {flag}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    return sys.argv[index + 1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    review_path = Path(_argument_value("--verified")).resolve()
    frozen_path = Path(_argument_value("--paper-frozen-manifest")).resolve()
    if not review_path.is_file() or not frozen_path.is_file():
        raise FileNotFoundError("verified review sheet and paper frozen manifest must both exist")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("schema") != "verified-nfe-paper-frozen-review-1.0":
        raise ValueError(
            "paper verified evaluation accepts only freeze_verified_review_paper.py manifests"
        )
    if frozen.get("review_sheet_sha256") != _sha256(review_path):
        raise RuntimeError("verified review sheet bytes differ from the paper-frozen artifact")
    if frozen.get("reviewer_blinded_to_model_predictions") is not True:
        raise RuntimeError("paper verified review is not declared prediction-blinded")
    if frozen.get("membership_exactly_matches_preregistered_queue") is not True:
        raise RuntimeError("paper verified review does not attest exact preregistered membership")

    index = sys.argv.index("--paper-frozen-manifest")
    delegated = sys.argv[:index] + sys.argv[index + 2 :]
    original = sys.argv
    try:
        sys.argv = delegated
        return evaluator.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
