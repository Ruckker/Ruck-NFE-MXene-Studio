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
        raise ValueError(f"strict verified evaluator requires {flag}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    return sys.argv[index + 1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    verified_path = Path(_argument_value("--verified")).resolve()
    frozen_path = Path(_argument_value("--frozen-manifest")).resolve()
    if not verified_path.is_file() or not frozen_path.is_file():
        raise FileNotFoundError("verified review sheet and frozen manifest must both exist")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("schema") != "verified-nfe-frozen-review-1.0":
        raise ValueError("unsupported/missing frozen verified-review schema")
    if frozen.get("review_sheet_sha256") != _sha256(verified_path):
        raise RuntimeError(
            "verified review sheet bytes differ from the frozen pre-prediction review artifact"
        )
    if frozen.get("reviewer_blinded_to_model_predictions") is not True:
        raise RuntimeError(
            "paper-ready independent verified analysis requires reviewer_blinded_to_model_predictions=true; "
            "use the lower-level evaluator only for explicitly disclosed unblinded sensitivity analysis"
        )

    # Remove the strict-wrapper-only option before delegating to the existing
    # sensitivity evaluator, which has its own argument parser.
    index = sys.argv.index("--frozen-manifest")
    delegated = sys.argv[:index] + sys.argv[index + 2 :]
    original = sys.argv
    try:
        sys.argv = delegated
        return evaluator.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
