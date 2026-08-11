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
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    target_schema_sha256,
)
from nfe_model.metrics_v2 import classification_metrics, regression_metrics
from nfe_model.prediction_manifest import write_prediction_manifest
from nfe_model.provenance_v2 import NORMALIZER_SCHEMA, assert_matching_provenance
from training.baselines.common import load_benchmark_data


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

PAPER_METRIC_BINDINGS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_roc_auc",
    "macro_average_precision",
    "high_average_precision",
    "high_precision_at_5pct",
    "high_recall_at_5pct",
    "high_enrichment_at_5pct",
    "ece",
    "NFE_Pseudo_Score_mae",
    "NFE_Pseudo_Score_rmse",
    "NFE_Pseudo_Score_spearman",
    "NFE_Pseudo_Score_r2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a prediction CSV against the exact formal split and run metrics, then bind it to the run."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result", help="result.json/final_metrics.json; auto-detected from prediction directory")
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
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
        "Record_Index",
        "Structure_Name",
        "Split_Group",
        "True_Label",
        "Predicted_Label",
        "Probability_Low",
        "Probability_Medium",
        "Probability_High",
        "True_NFE_Pseudo_Score",
        "Predicted_NFE_Pseudo_Score",
        "Absolute_Score_Error",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction CSV is missing formal columns: {sorted(missing)}")
    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    groups = frame["Split_Group"].fillna("").astype(str).str.strip()
    if (identifiers == "").any() or identifiers.duplicated().any():
        raise ValueError("prediction CSV requires unique non-empty Structure_Name values")
    if (groups == "").any():
        raise ValueError("prediction CSV requires non-empty Split_Group values")
    record_indices = pd.to_numeric(frame["Record_Index"], errors="coerce")
    if record_indices.isna().any() or record_indices.duplicated().any():
        raise ValueError("prediction CSV requires unique integer Record_Index values")
    if not np.allclose(record_indices.to_numpy(float), np.rint(record_indices.to_numpy(float)), atol=0, rtol=0):
        raise ValueError("prediction CSV Record_Index values must be integers")

    label_text = frame["True_Label"].fillna("").astype(str).str.strip().str.lower()
    if (~label_text.isin(set(LABEL_TO_INDEX))).any():
        raise ValueError("prediction CSV contains invalid/missing True_Label values")
    labels = label_text.map(LABEL_TO_INDEX).to_numpy(np.int64)
    probabilities = frame[
        ["Probability_Low", "Probability_Medium", "Probability_High"]
    ].to_numpy(float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("prediction probabilities must be finite and lie in [0,1]")
    row_sum = probabilities.sum(axis=1)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=5e-6):
        raise ValueError("prediction probability rows must sum to one within 5e-6")
    predicted = np.argmax(probabilities, axis=1)
    predicted_text = frame["Predicted_Label"].fillna("").astype(str).str.strip().str.lower()
    expected_predicted_text = np.asarray([INDEX_TO_LABEL[int(value)] for value in predicted], dtype=object)
    if np.any(predicted_text.to_numpy(object) != expected_predicted_text):
        raise ValueError("Predicted_Label disagrees with the probability argmax")
    metrics = classification_metrics(np.log(np.clip(probabilities, 1e-12, 1.0)), labels)

    truth = pd.to_numeric(frame["True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    prediction = pd.to_numeric(
        frame["Predicted_NFE_Pseudo_Score"], errors="coerce"
    ).to_numpy(float)
    absolute_error = pd.to_numeric(frame["Absolute_Score_Error"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction CSV requires finite true/predicted NFE pseudo-scores")
    if not np.all(np.isfinite(absolute_error)):
        raise ValueError("prediction CSV requires finite Absolute_Score_Error values")
    if not np.allclose(absolute_error, np.abs(prediction - truth), rtol=0.0, atol=5e-6):
        raise ValueError("Absolute_Score_Error disagrees with true/predicted NFE pseudo-scores")
    metrics.update(
        regression_metrics(
            prediction[:, None],
            truth[:, None],
            np.ones((len(frame), 1), dtype=bool),
            ["NFE_Pseudo_Score"],
        )
    )
    return metrics


def _assert_exact_split_membership(frame: pd.DataFrame, data, split: str, tolerance: float) -> None:
    expected_indices = [int(value) for value in data.splits[split]]
    expected_by_index = {index: data.records[index] for index in expected_indices}
    observed_indices = pd.to_numeric(frame["Record_Index"], errors="raise").astype(np.int64).tolist()
    if set(observed_indices) != set(expected_indices) or len(observed_indices) != len(expected_indices):
        missing = sorted(set(expected_indices) - set(observed_indices))[:10]
        extra = sorted(set(observed_indices) - set(expected_indices))[:10]
        raise RuntimeError(
            f"prediction CSV does not contain the exact formal {split} record set: missing={missing} extra={extra}"
        )

    for row in frame.itertuples(index=False):
        record_index = int(getattr(row, "Record_Index"))
        record = expected_by_index[record_index]
        identifier = str(record.get("id", ""))
        group = str(record.get("split_group", ""))
        label = int(record.get("label", -1))
        expected_label = INDEX_TO_LABEL.get(label, "")
        if str(getattr(row, "Structure_Name")).strip() != identifier:
            raise RuntimeError(
                f"prediction Structure_Name mismatch at Record_Index={record_index}: "
                f"csv={getattr(row, 'Structure_Name')!r} cache={identifier!r}"
            )
        if str(getattr(row, "Split_Group")).strip() != group:
            raise RuntimeError(
                f"prediction Split_Group mismatch for {identifier}: "
                f"csv={getattr(row, 'Split_Group')!r} cache={group!r}"
            )
        if str(getattr(row, "True_Label")).strip().lower() != expected_label:
            raise RuntimeError(
                f"prediction True_Label mismatch for {identifier}: "
                f"csv={getattr(row, 'True_Label')!r} cache={expected_label!r}"
            )
        target_mask = record["target_mask"]
        if not bool(target_mask[0]):
            raise RuntimeError(f"formal cache unexpectedly masks primary score for {identifier}")
        expected_score = float(record["targets"][0])
        observed_score = float(getattr(row, "True_NFE_Pseudo_Score"))
        if abs(expected_score - observed_score) > tolerance:
            raise RuntimeError(
                f"prediction true score mismatch for {identifier}: "
                f"csv={observed_score:.10g} cache={expected_score:.10g} tolerance={tolerance}"
            )


def _reported_metrics(payload: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    direct = payload.get(f"{split}_metrics")
    if isinstance(direct, Mapping):
        return direct
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
    for key in PAPER_METRIC_BINDINGS:
        if key not in observed:
            raise RuntimeError(f"prediction CSV metric recomputation is missing required paper metric {key}")
        if key not in reported:
            raise RuntimeError(f"sibling run result is missing required paper metric {key}")
        left = float(observed[key])
        right = float(reported[key])
        if np.isnan(left) and np.isnan(right):
            continue
        if not np.isfinite(left) or not np.isfinite(right):
            raise RuntimeError(f"non-finite result/prediction metric mismatch for {key}: {left} vs {right}")
        if abs(left - right) > tolerance:
            raise RuntimeError(
                f"prediction CSV does not reproduce sibling run metric {key}: "
                f"csv={left:.10g} result={right:.10g} tolerance={tolerance}"
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

    data = load_benchmark_data(args.config, rebuild_cache=False)
    assert_matching_provenance(
        provenance,
        data.provenance,
        require_present=True,
        require_code_match=True,
    )

    frame = pd.read_csv(prediction_path)
    _prediction_metrics(frame)
    _assert_exact_split_membership(frame, data, split, float(args.metric_tolerance))

    expected_rows = len(data.splits[split])
    if len(frame) != expected_rows:
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
