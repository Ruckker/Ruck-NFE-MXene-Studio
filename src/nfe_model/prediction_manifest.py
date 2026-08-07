from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .provenance_v2 import canonical_sha256, file_sha256


PREDICTION_MANIFEST_SCHEMA = "nfe-prediction-manifest-1.0"
PREDICTION_DATA_IDENTITY_KEYS = (
    "dataset_table_sha256",
    "structure_manifest_schema",
    "structure_manifest_sha256",
    "target_schema",
    "target_schema_sha256",
    "data_implementation_schema",
    "data_implementation_sha256",
    "cache_records_sha256",
    "normalizer_schema",
    "normalizer_sha256",
    "split_manifest_sha256",
    "cache_schema",
    "global_feature_schema",
    "neighbor_policy",
    "graph_radius_A",
    "max_neighbors",
    "git_commit",
)


def prediction_manifest_path(prediction_path: str | Path) -> Path:
    path = Path(prediction_path)
    return path.with_name(f"{path.stem}.manifest.json")


def prediction_data_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one canonical benchmark-data identity used by formal analyses."""
    result: dict[str, Any] = {}
    for key in PREDICTION_DATA_IDENTITY_KEYS:
        value = provenance.get(key)
        if value is None or str(value) in {"", "unknown"}:
            raise ValueError(f"formal prediction provenance is missing {key}")
        result[key] = value
    if provenance.get("git_dirty") is not False:
        raise ValueError("formal prediction provenance requires a clean training worktree")
    result["git_dirty"] = False
    return result


# Backward-compatible private alias for earlier callers/tests on this branch.
_data_identity = prediction_data_identity


def write_prediction_manifest(
    prediction_path: str | Path,
    *,
    split: str,
    provenance: Mapping[str, Any],
    track: str,
    model: str,
    seed: int | None,
    checkpoint_sha256: str | None,
    training_protocol_sha256: str | None,
    model_protocol_sha256: str | None = None,
    temperature: float | None = None,
) -> Path:
    path = Path(prediction_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"cannot manifest missing prediction file: {path}")
    if split not in {"validation", "test"}:
        raise ValueError(f"formal prediction manifest requires validation/test split, got {split!r}")
    identity = prediction_data_identity(provenance)
    run_identity = {
        "track": str(track),
        "model": str(model),
        "seed": None if seed is None else int(seed),
        "checkpoint_sha256": str(checkpoint_sha256 or ""),
        "training_protocol_sha256": str(training_protocol_sha256 or ""),
        "model_protocol_sha256": str(model_protocol_sha256 or ""),
        "temperature": None if temperature is None else float(temperature),
    }
    manifest = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "split": split,
        "prediction_filename": path.name,
        "prediction_file_sha256": file_sha256(path),
        "data_identity": identity,
        "data_identity_sha256": canonical_sha256(identity),
        "run_identity": run_identity,
        "run_identity_sha256": canonical_sha256(run_identity),
    }
    output = prediction_manifest_path(path)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_prediction_manifest(
    prediction_path: str | Path,
    *,
    expected_split: str | None = None,
) -> dict[str, Any]:
    path = Path(prediction_path).resolve()
    manifest_path = prediction_manifest_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"formal prediction file has no identity manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PREDICTION_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported prediction manifest schema: {manifest.get('schema')!r}"
        )
    if manifest.get("prediction_filename") != path.name:
        raise ValueError("prediction manifest filename does not match the supplied CSV")
    observed_file_hash = file_sha256(path)
    if manifest.get("prediction_file_sha256") != observed_file_hash:
        raise ValueError(
            "prediction CSV bytes do not match its manifest: "
            f"manifest={manifest.get('prediction_file_sha256')} current={observed_file_hash}"
        )
    if expected_split is not None and manifest.get("split") != expected_split:
        raise ValueError(
            f"prediction manifest split={manifest.get('split')!r}, expected {expected_split!r}"
        )
    identity = manifest.get("data_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("prediction manifest has no data_identity mapping")
    canonical_identity = prediction_data_identity(identity)
    if canonical_sha256(canonical_identity) != manifest.get("data_identity_sha256"):
        raise ValueError("prediction manifest data identity hash is inconsistent")
    run_identity = manifest.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise ValueError("prediction manifest has no run_identity mapping")
    if canonical_sha256(dict(run_identity)) != manifest.get("run_identity_sha256"):
        raise ValueError("prediction manifest run identity hash is inconsistent")
    return manifest


def assert_same_prediction_data_identity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    left_hash = str(left.get("data_identity_sha256", ""))
    right_hash = str(right.get("data_identity_sha256", ""))
    if not left_hash or not right_hash or left_hash != right_hash:
        raise RuntimeError(
            "formal prediction files use different benchmark data identities: "
            f"left={left_hash or 'missing'} right={right_hash or 'missing'}"
        )
