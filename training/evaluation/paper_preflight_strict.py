from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from nfe_model.checkpoint_contract import assert_checkpoint_internal_contract
from nfe_model.data_contract import DATA_IMPLEMENTATION_SCHEMA, data_implementation_sha256
from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    STRUCTURE_MANIFEST_SCHEMA,
    TARGET_SCHEMA,
    target_schema_sha256,
    torch_load_compat,
)
from nfe_model.prediction_manifest import load_prediction_manifest
from nfe_model.provenance_v2 import (
    NORMALIZER_SCHEMA,
    assert_matching_provenance,
    file_sha256,
    git_repository_state,
)
from training.baselines.common import load_benchmark_data
from training.evaluation.sign_predictions_formal import (
    _assert_exact_split_membership,
    _assert_metrics_match,
    _checkpoint_hash,
    _prediction_metrics,
    _reported_metrics,
    _seed,
    _track_model,
)


PAPER_METRIC_TOLERANCE = 5e-6
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
        description=(
            "Strict final paper gate: result, signed CSV, exact split membership, checkpoint, "
            "metrics, run identity and data identity must all agree."
        )
    )
    parser.add_argument("results", nargs="+")
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--metric-tolerance", type=float, default=PAPER_METRIC_TOLERANCE)
    parser.add_argument(
        "--allow-cache-skips",
        action="store_true",
        help="not allowed by this strict paper gate; retained only to produce an explicit error",
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


def _checkpoint_path(payload: Mapping[str, Any], result_path: Path) -> Path:
    details = payload.get("details")
    if isinstance(details, Mapping) and details.get("checkpoint"):
        return Path(str(details["checkpoint"])).expanduser().resolve()
    return result_path.with_name("best.pt")


def _validate_checkpoint(
    payload: Mapping[str, Any],
    result_path: Path,
    expected_hash: str,
    data,
    *,
    model: str,
) -> None:
    if model in {"dummy", "xgboost"}:
        if expected_hash:
            raise RuntimeError(
                f"classical/parameter-free result {result_path} unexpectedly claims a neural checkpoint hash"
            )
        if model == "xgboost":
            details = payload.get("details")
            state_hash = str(details.get("model_state_sha256", "")) if isinstance(details, Mapping) else ""
            version = str(details.get("xgboost_version", "")) if isinstance(details, Mapping) else ""
            if len(state_hash) != 64 or not version:
                raise RuntimeError(
                    f"paper XGBoost result lacks fitted-state SHA256 or xgboost_version: {result_path}"
                )
        return

    if len(expected_hash) != 64:
        raise RuntimeError(
            f"checkpointed paper model {model} lacks a 64-character checkpoint SHA256: {result_path}"
        )
    checkpoint_path = _checkpoint_path(payload, result_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"paper preflight requires the fitted checkpoint file for {result_path}: {checkpoint_path}"
        )
    observed_hash = file_sha256(checkpoint_path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"checkpoint bytes differ from result identity for {result_path}: "
            f"result={expected_hash} file={observed_hash}"
        )
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint_path}")
    assert_checkpoint_internal_contract(checkpoint)
    assert_matching_provenance(
        checkpoint.get("provenance"),
        data.provenance,
        require_present=True,
        require_code_match=True,
    )


def _validate_one(
    path: Path,
    tolerance: float,
    data,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"result is not a JSON object: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"result has no provenance mapping: {path}")
    assert_matching_provenance(
        provenance,
        data.provenance,
        require_present=True,
        require_code_match=True,
    )
    for key, expected in EXPECTED.items():
        if str(provenance.get(key)) != str(expected):
            raise RuntimeError(
                f"{path} uses stale {key}: observed={provenance.get(key)!r} expected={expected!r}"
            )
    if provenance.get("git_dirty") is not False:
        raise RuntimeError(f"{path} was produced from a dirty/unknown worktree")
    skipped = int(provenance.get("skipped_cache_records", -1))
    if skipped != 0:
        raise RuntimeError(
            f"{path} skipped {skipped} cache rows; strict paper-ready analysis requires exactly zero"
        )

    track, model = _track_model(payload)
    expected_seed = _seed(payload, path)
    expected_checkpoint = _checkpoint_hash(payload) or ""
    expected_protocol = _expected_protocol(payload)
    expected_model_protocol = str(payload.get("model_protocol_sha256") or "")
    expected_temperature = _temperature(payload)
    if not expected_protocol:
        raise RuntimeError(f"{path} has no training/model protocol fingerprint")

    _validate_checkpoint(
        payload,
        path,
        expected_checkpoint,
        data,
        model=model,
    )

    manifest_hash = None
    for split in ("validation", "test"):
        prediction_path = path.with_name(f"{split}_predictions.csv")
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        manifest = load_prediction_manifest(prediction_path, expected_split=split)
        identity = manifest["data_identity"]
        for key in EXPECTED:
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
        if str(run.get("model_protocol_sha256", "")) != expected_model_protocol:
            raise RuntimeError(
                f"{prediction_path} model protocol differs from current result"
            )
        if expected_temperature is None:
            if run.get("temperature") is not None:
                raise RuntimeError(f"{prediction_path} records unexpected calibration temperature")
        elif run.get("temperature") is None or abs(
            float(run["temperature"]) - expected_temperature
        ) > 1e-12:
            raise RuntimeError(
                f"{prediction_path} calibration temperature differs from current result"
            )

        frame = pd.read_csv(prediction_path)
        observed = _prediction_metrics(frame)
        _assert_exact_split_membership(frame, data, split, tolerance)
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
        "training_protocol_sha256": expected_protocol,
        "model_protocol_sha256": expected_model_protocol,
        "skipped_cache_records": skipped,
    }


def main() -> int:
    args = parse_args()
    if args.allow_cache_skips:
        raise ValueError(
            "--allow-cache-skips is forbidden by strict paper preflight; use paper_preflight.py for exploration"
        )
    if abs(float(args.metric_tolerance) - PAPER_METRIC_TOLERANCE) > 1e-15:
        raise ValueError(
            f"strict paper preflight fixes --metric-tolerance={PAPER_METRIC_TOLERANCE}"
        )
    runtime = git_repository_state()
    if runtime.get("git_dirty") is not False:
        raise RuntimeError("strict paper preflight requires a clean current Git worktree")
    runtime_commit = str(runtime.get("git_commit", "unknown"))
    if runtime_commit == "unknown":
        raise RuntimeError("strict paper preflight requires a resolvable current Git commit")

    data = load_benchmark_data(args.config, rebuild_cache=False)
    if data.skipped_cache_records != 0:
        raise RuntimeError(
            f"strict paper preflight requires zero cache skips; observed={data.skipped_cache_records}"
        )

    rows = [
        _validate_one(Path(value).resolve(), PAPER_METRIC_TOLERANCE, data)
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
