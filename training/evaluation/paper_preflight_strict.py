from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from nfe_model.data_contract import DATA_IMPLEMENTATION_SCHEMA, data_implementation_sha256
from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    STRUCTURE_MANIFEST_SCHEMA,
    TARGET_SCHEMA,
    target_schema_sha256,
)
from nfe_model.prediction_manifest import load_prediction_manifest
from nfe_model.provenance_v2 import NORMALIZER_SCHEMA, git_repository_state
from training.evaluation.sign_predictions_formal import (
    _assert_metrics_match,
    _checkpoint_hash,
    _prediction_metrics,
    _reported_metrics,
    _seed,
    _track_model,
)


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
        description="Strict final paper gate: result, signed CSV, metrics, run identity and data identity must all agree."
    )
    parser.add_argument("results", nargs="+")
    parser.add_argument("--metric-tolerance", type=float, default=5e-6)
    parser.add_argument(
        "--allow-cache-skips",
        action="store_true",
        help="exploratory only; paper-ready default is zero cache skips",
    )
    return parser.parse_args()


def _expected_protocol(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("training_protocol_sha256")
        or payload.get("benchmark_common_protocol_sha256")
        or payload.get("model_protocol_sha256")
        or ""
    )


def _temperature(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("classification_temperature")
    if value is None:
        value = payload.get("temperature")
    return None if value is None else float(value)


def _validate_one(path: Path, tolerance: float, allow_cache_skips: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"result is not a JSON object: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"result has no provenance mapping: {path}")
    for key, expected in EXPECTED.items():
        if str(provenance.get(key)) != str(expected):
            raise RuntimeError(
                f"{path} uses stale {key}: observed={provenance.get(key)!r} expected={expected!r}"
            )
    if provenance.get("git_dirty") is not False:
        raise RuntimeError(f"{path} was produced from a dirty/unknown worktree")
    skipped = int(provenance.get("skipped_cache_records", -1))
    if skipped < 0:
        raise RuntimeError(f"{path} does not record skipped_cache_records")
    if skipped and not allow_cache_skips:
        raise RuntimeError(f"{path} skipped {skipped} cache rows; paper-ready default is zero")

    track, model = _track_model(payload)
    expected_seed = _seed(payload, path)
    expected_checkpoint = _checkpoint_hash(payload) or ""
    expected_protocol = _expected_protocol(payload)
    expected_temperature = _temperature(payload)

    manifest_hash = None
    for split in ("validation", "test"):
        prediction_path = path.with_name(f"{split}_predictions.csv")
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        manifest = load_prediction_manifest(prediction_path, expected_split=split)
        identity = manifest["data_identity"]
        for key in (*EXPECTED,):
            if str(identity.get(key)) != str(EXPECTED[key]):
                raise RuntimeError(f"{prediction_path} manifest uses stale {key}")
        for key in (
            "dataset_table_sha256",
            "structure_manifest_sha256",
            "target_schema_sha256",
            "data_implementation_sha256",
            "cache_records_sha256",
            "normalizer_sha256",
            "split_manifest_sha256",
            "git_commit",
        ):
            if str(identity.get(key, "")) != str(provenance.get(key, "")):
                raise RuntimeError(
                    f"{prediction_path} manifest/result data identity mismatch for {key}"
                )
        if manifest_hash is None:
            manifest_hash = manifest["data_identity_sha256"]
        elif manifest["data_identity_sha256"] != manifest_hash:
            raise RuntimeError("validation/test prediction manifests use different data identities")

        run = manifest["run_identity"]
        if str(run.get("track", "")) != track or str(run.get("model", "")) != model:
            raise RuntimeError(
                f"{prediction_path} run identity model/track differs from current result"
            )
        if run.get("seed") != expected_seed:
            raise RuntimeError(
                f"{prediction_path} run identity seed={run.get('seed')} result seed={expected_seed}"
            )
        if str(run.get("checkpoint_sha256", "")) != expected_checkpoint:
            raise RuntimeError(
                f"{prediction_path} checkpoint hash differs from current result"
            )
        if str(run.get("training_protocol_sha256", "")) != expected_protocol:
            raise RuntimeError(
                f"{prediction_path} training protocol differs from current result"
            )
        if expected_temperature is None:
            if run.get("temperature") is not None:
                raise RuntimeError(f"{prediction_path} records unexpected calibration temperature")
        else:
            if run.get("temperature") is None or abs(float(run["temperature"]) - expected_temperature) > 1e-12:
                raise RuntimeError(
                    f"{prediction_path} calibration temperature differs from current result"
                )

        frame = pd.read_csv(prediction_path)
        observed = _prediction_metrics(frame)
        reported = _reported_metrics(payload, split)
        _assert_metrics_match(observed, reported, tolerance)

    return {
        "result": str(path),
        "git_commit": str(provenance.get("git_commit")),
        "dataset_table_sha256": str(provenance.get("dataset_table_sha256")),
        "cache_records_sha256": str(provenance.get("cache_records_sha256")),
        "normalizer_sha256": str(provenance.get("normalizer_sha256")),
        "split_manifest_sha256": str(provenance.get("split_manifest_sha256")),
        "prediction_data_identity_sha256": str(manifest_hash),
        "track": track,
        "model": model,
        "seed": expected_seed,
        "checkpoint_sha256": expected_checkpoint,
        "skipped_cache_records": skipped,
    }


def main() -> int:
    args = parse_args()
    if args.metric_tolerance < 0:
        raise ValueError("--metric-tolerance must be non-negative")
    runtime = git_repository_state()
    if runtime.get("git_dirty") is not False:
        raise RuntimeError("strict paper preflight requires a clean current Git worktree")
    runtime_commit = str(runtime.get("git_commit", "unknown"))
    if runtime_commit == "unknown":
        raise RuntimeError("strict paper preflight requires a resolvable current Git commit")

    rows = [
        _validate_one(Path(value).resolve(), float(args.metric_tolerance), bool(args.allow_cache_skips))
        for value in args.results
    ]
    if {row["git_commit"] for row in rows} != {runtime_commit}:
        raise RuntimeError(
            f"all final artifacts must come from current commit {runtime_commit}; "
            f"found={sorted({row['git_commit'] for row in rows})}"
        )
    for key in (
        "dataset_table_sha256",
        "cache_records_sha256",
        "normalizer_sha256",
        "split_manifest_sha256",
        "prediction_data_identity_sha256",
    ):
        values = {row[key] for row in rows}
        if len(values) != 1:
            raise RuntimeError(f"strict paper result set mixes {key}: {sorted(values)}")
    print(json.dumps({"paper_ready": True, "results": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
