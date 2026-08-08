from __future__ import annotations

import sys

import pandas as pd

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
)
from nfe_model.provenance_v2 import assert_matching_provenance
from training.baselines.common import load_benchmark_data
from training.evaluation import evaluate_slices as evaluator
from training.evaluation.build_ood_manifest import load_ood_manifest_sidecar
from training.evaluation.sign_predictions_formal import (
    _assert_exact_split_membership,
    _prediction_metrics,
)


FORMAL_TOLERANCE = 5e-6


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
    prediction_path = _argument_value("--predictions")
    ood_path = _argument_value(
        "--manifest", default="training/evaluation/ood_manifest.csv"
    )
    config_path = _argument_value(
        "--config", default="training/configs/nfe_predictor.yaml"
    )
    prediction_manifest = load_prediction_manifest(
        prediction_path, expected_split="test"
    )
    ood_manifest = load_ood_manifest_sidecar(ood_path)
    assert_same_prediction_data_identity(prediction_manifest, ood_manifest)

    data = load_benchmark_data(config_path, rebuild_cache=False)
    assert_matching_provenance(
        prediction_manifest["data_identity"],
        data.provenance,
        require_present=True,
        require_code_match=True,
    )
    frame = pd.read_csv(prediction_path)
    _prediction_metrics(frame)
    _assert_exact_split_membership(frame, data, "test", FORMAL_TOLERANCE)

    delegated = _remove_option_with_value(list(sys.argv), "--config")
    original = sys.argv
    try:
        sys.argv = delegated
        return evaluator.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
