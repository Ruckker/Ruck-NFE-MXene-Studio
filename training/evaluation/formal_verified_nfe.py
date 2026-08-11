from __future__ import annotations

import sys

from nfe_model.prediction_manifest import load_prediction_manifest
from training.evaluation import evaluate_verified_nfe as evaluator


def _argument_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError as exc:
        raise ValueError(f"formal verified evaluator requires {flag}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    return sys.argv[index + 1]


def main() -> int:
    prediction_path = _argument_value("--predictions")
    load_prediction_manifest(prediction_path, expected_split="test")
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
