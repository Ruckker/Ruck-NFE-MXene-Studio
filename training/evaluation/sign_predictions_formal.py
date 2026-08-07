from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nfe_model.data_contract import DATA_IMPLEMENTATION_SCHEMA, data_implementation_sha256
from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    STRUCTURE_MANIFEST_SCHEMA,
    TARGET_SCHEMA,
    LABEL_TO_INDEX,
    target_schema_sha256,
)
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.prediction_manifest import write_prediction_manifest
from nfe_model.provenance_v2 import NORMALIZER_SCHEMA


EXPECTED = {
    "structure_manifest_schema": STRUCTURE_MANIFEST_SCHEMA,
    "target_schema": TARGET_SCHEMA,
    "target_schema_sha256": target_schema_sha256(),
    "data_implementation_schema": DATA_IMPLEMENTATION_SCHEMA,
    "data_implementation_sha256": data_implementation_sha256(),
    "normalizer_schema": NORMALIZER_SCHEMA,
    "cache_schema": CACHE_SCHEMA,
    "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
    "neighbor_policy": NEIGHBOR_POLICY,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a prediction CSV against run metrics, then bind it cryptographically to the run."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result", help="result.json/final_metrics.json; auto-detected from prediction directory")
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--metric-tolerance", type=float, default=5e-6)
    return parser.parse_args()


def _result_path(prediction_path: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    existing = [
        path
        for path in (
            prediction_path.with_name("result.json"),
            prediction_path.with_name("final_metrics.json"),
        )
        if path.is_file()
    ]
    if len(existing) != 1:
        raise RuntimeError(
            "formal prediction signer requires exactly one sibling result.json/final_metrics.json"
        )
    return existing[0]


def _split(prediction_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    stem = prediction_path.stem.lower()
    if stem.startswith("validation"):
        return "validation"
    if stem.startswith("test"):
        return "test"
    raise ValueError("cannot infer split from prediction filename; pass --split")


def _seed(payload: Mapping[str, Any], result_path: Path) -> int | None:
    value = payload.get("seed")
    if value is None and isinstance(payload.get("config"), Mapping):
        value = payload["config"].get("seed")
    if value is None and result_path.parent.name.startswith("seed_"):
        try:
            value = int(result_path.parent.name.removeprefix("seed_"))
        except ValueError:
            value = None
    return None if value is None else int(value)


def _track_model(payload: Mapping[str, Any]) -> tuple[str, str]:
    track = str(payload.get("track") or "")
    model = str(payload.get("model") or "")
    ablation = payload.get("ablation_config")
    if isinstance(ablation, Mapping) and ablation.get("name"):
        name = str(ablation["name"])
        return track or "ablation", model or name
    return track or "predictor", model or "ours_full"


def _checkpoint_hash(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("checkpoint_sha256")
    if value:
        return str(value)
    details = payload.get("details")
    if isinstance(details, Mapping) and details.get("checkpoint_sha256"):
        return str(details["checkpoint_sha256"])
    return None


def _validate_provenance(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("run result has no provenance mapping")
    for key, expected in EXPECTED.items():
        if str(provenance.get(key)) != str(expected):
            raise RuntimeError(
                f"cannot formally sign stale/incompatible result {key}: "
                f"observed={provenance.get(key)!r} expected={expected!r}"
            )
    for key in (
        "dataset_table_sha256",
        "structure_manifest_sha256",
        "cache_records_sha256",
        "normalizer_sha256",
        "split_manifest_sha256",
        "git_commit",
    ):
        if not str(provenance.get(key, "")) or str(provenance.get(key)) == "unknown":
            raise RuntimeError(f"run result is missing formal provenance field {key}")
    if provenance.get("git_dirty") is not False:
        raise RuntimeError("cannot formally sign a dirty/unknown-worktree result")
    return provenance


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    required = {
        "Structure_Name",
        "True_Label",
        "Probability_Low",
        "Probability_Medium",
        "Probability_High",
        "True_NFE_Pseudo_Score",
        "Predicted_NFE_Pseudo_Score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction CSV is missing formal columns: {sorted(missing)}")
    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (identifiers == "").any() or identifiers.duplicated().any():
        raise ValueError("prediction CSV requires unique non-empty Structure_Name values")

    label_text = frame["True_Label"].fillna("").astype(str).str.strip().str.lower()
    if (~label_text.isin(set(LABEL_TO_INDEX))).any():
        raise ValueError("prediction CSV contains invalid/missing True_Label values")
    labels = label_text.map(LABEL_TO_INDEX).to_numpy(np.int64)
    probabilities = frame[
        ["Probability_Low", "Probability_Medium", "Probability_High"]
    ].to_numpy(float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("prediction probabilities must be finite and non-negative")
    row_sum = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("prediction probabilities contain zero-sum rows")
    probabilities = probabilities / row_sum
    metrics = classification_metrics(np.log(np.clip(probabilities, 1e-12, 1.0)), labels)

    truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    prediction = pd.to_numeric(
        frame["Predicted_NFE_Pseudo_Score"], errors="coerce"
    ).to_numpy(float)
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction CSV requires finite true/predicted NFE pseudo-scores")
    metrics.update(
        regression_metrics(
            prediction[:, None],
            truth[:, None],
            np.ones((len(frame), 1), dtype=bool),
            ["NFE_Pseudo_Score"],
        )
    )
    return metrics


def _reported_metrics(payload: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    # Baseline result schema.
    direct = payload.get(f"{split}_metrics")
    if isinstance(direct, Mapping):
        return direct
    # Audited full/ablation final metrics: prediction CSVs use calibrated
    # classification probabilities, so prefer calibrated split metrics.
    calibrated = payload.get(f"{split}_calibrated")
    if isinstance(calibrated, Mapping):
        return calibrated
    raw = payload.get(split)
    if isinstance(raw, Mapping):
        return raw
    raise ValueError(f"run result has no reported metrics for split {split}")


def _assert_metrics_match(
    observed: Mapping[str, float], reported: Mapping[str, Any], tolerance: float
) -> None:
    comparisons = (
        "macro_f1",
        "balanced_accuracy",
        "macro_roc_auc",
        "macro_average_precision",
        "NFE_Pseudo_Score_mae",
        "NFE_Pseudo_Score_rmse",
    )
    checked = 0
    for key in comparisons:
        if key not in reported:
            continue
        left = float(observed[key])
        right = float(reported[key])
        if np.isnan(left) and np.isnan(right):
            checked += 1
            continue
        if not np.isfinite(left) or not np.isfinite(right):
            raise RuntimeError(f"non-finite result/prediction metric mismatch for {key}: {left} vs {right}")
        if abs(left - right) > tolerance:
            raise RuntimeError(
                f"prediction CSV does not reproduce sibling run metric {key}: "
                f"csv={left:.10g} result={right:.10g} tolerance={tolerance}"
            )
        checked += 1
    if checked < 2:
        raise RuntimeError(
            "run result exposes too few comparable metrics to cryptographically bind a prediction CSV safely"
        )


def main() -> int:
    args = parse_args()
    if args.metric_tolerance < 0:
        raise ValueError("--metric-tolerance must be non-negative")
    prediction_path = Path(args.predictions).resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    result_path = _result_path(prediction_path, args.result)
    split = _split(prediction_path, args.split)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run result must be a JSON object")
    provenance = _validate_provenance(payload)

    frame = pd.read_csv(prediction_path)
    expected_rows = None
    sizes = payload.get("split_sizes")
    if isinstance(sizes, Mapping) and split in sizes:
        expected_rows = int(sizes[split])
    coverage = provenance.get("primary_target_coverage")
    if expected_rows is None and isinstance(coverage, Mapping) and isinstance(coverage.get(split), Mapping):
        if coverage[split].get("rows") is not None:
            expected_rows = int(coverage[split]["rows"])
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(
            f"prediction row count {len(frame)} does not match formal {split} support {expected_rows}"
        )

    observed_metrics = _prediction_metrics(frame)
    _assert_metrics_match(
        observed_metrics,
        _reported_metrics(payload, split),
        float(args.metric_tolerance),
    )

    track, model = _track_model(payload)
    output = write_prediction_manifest(
        prediction_path,
        split=split,
        provenance=provenance,
        track=track,
        model=model,
        seed=_seed(payload, result_path),
        checkpoint_sha256=_checkpoint_hash(payload),
        training_protocol_sha256=(
            payload.get("training_protocol_sha256")
            or payload.get("benchmark_common_protocol_sha256")
            or payload.get("model_protocol_sha256")
        ),
        model_protocol_sha256=payload.get("model_protocol_sha256"),
        temperature=(
            payload.get("classification_temperature")
            if payload.get("classification_temperature") is not None
            else payload.get("temperature")
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
