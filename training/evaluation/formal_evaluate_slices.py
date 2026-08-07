from __future__ import annotations

import sys

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
)
from training.evaluation import evaluate_slices as evaluator
from training.evaluation.build_ood_manifest import load_ood_manifest_sidecar


def _argument_value(flag: str, *, default: str | None = None) -> str:
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
    if len(found) > 1:
        raise ValueError(f"formal OOD evaluator accepts {flag} at most once")
    if found:
        return found[0]
    if default is not None:
        return default
    raise ValueError(f"formal OOD evaluator requires {flag}")


def main() -> int:
    prediction_path = _argument_value("--predictions")
    ood_path = _argument_value(
        "--manifest", default="training/evaluation/ood_manifest.csv"
    )
    prediction_manifest = load_prediction_manifest(
        prediction_path, expected_split="test"
    )
    ood_manifest = load_ood_manifest_sidecar(ood_path)
    assert_same_prediction_data_identity(prediction_manifest, ood_manifest)
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
