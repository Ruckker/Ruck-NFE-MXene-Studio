from __future__ import annotations

import sys

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
)
from training.evaluation import paired_bootstrap as evaluator


def _argument_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError as exc:
        raise ValueError(f"formal paired bootstrap requires {flag}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    return sys.argv[index + 1]


def main() -> int:
    left_path = _argument_value("--a")
    right_path = _argument_value("--b")
    left = load_prediction_manifest(left_path, expected_split="test")
    right = load_prediction_manifest(right_path, expected_split="test")
    assert_same_prediction_data_identity(left, right)
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
