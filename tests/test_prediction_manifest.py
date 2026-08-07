from __future__ import annotations

import json

import pytest

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
    write_prediction_manifest,
)


def _provenance(tag: str = "a") -> dict:
    return {
        "dataset_table_sha256": tag * 64,
        "structure_manifest_schema": "source-bytes-v1",
        "structure_manifest_sha256": "b" * 64,
        "target_schema": "regression-target-specs-v1",
        "target_schema_sha256": "c" * 64,
        "data_implementation_schema": "data-source-code-v2",
        "data_implementation_sha256": "d" * 64,
        "cache_records_sha256": "e" * 64,
        "normalizer_schema": "robust-train-normalizers-v1",
        "normalizer_sha256": "f" * 64,
        "split_manifest_sha256": "1" * 64,
        "cache_schema": "nfe-mxene-cache-2.3",
        "global_feature_schema": "intrinsic-slab-v3",
        "neighbor_policy": "radius-shell-complete-v2",
        "graph_radius_A": 6.0,
        "max_neighbors": 36,
        "git_commit": "2" * 40,
        "git_dirty": False,
    }


def test_prediction_manifest_detects_csv_tampering(tmp_path) -> None:
    prediction = tmp_path / "test_predictions.csv"
    prediction.write_text("Structure_Name,value\na,1\n", encoding="utf-8")
    write_prediction_manifest(
        prediction,
        split="test",
        provenance=_provenance(),
        track="architecture",
        model="painn",
        seed=2027,
        checkpoint_sha256="3" * 64,
        training_protocol_sha256="4" * 64,
    )
    manifest = load_prediction_manifest(prediction, expected_split="test")
    assert manifest["run_identity"]["seed"] == 2027

    prediction.write_text("Structure_Name,value\na,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes do not match"):
        load_prediction_manifest(prediction, expected_split="test")


def test_prediction_manifest_detects_manifest_identity_tampering(tmp_path) -> None:
    prediction = tmp_path / "test_predictions.csv"
    prediction.write_text("Structure_Name,value\na,1\n", encoding="utf-8")
    manifest_path = write_prediction_manifest(
        prediction,
        split="test",
        provenance=_provenance(),
        track="full-system",
        model="ours_full",
        seed=2027,
        checkpoint_sha256="3" * 64,
        training_protocol_sha256="4" * 64,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["data_identity"]["dataset_table_sha256"] = "9" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="data identity hash"):
        load_prediction_manifest(prediction, expected_split="test")


def test_paired_prediction_identity_rejects_different_benchmarks(tmp_path) -> None:
    left_path = tmp_path / "left.csv"
    right_path = tmp_path / "right.csv"
    left_path.write_text("x\n1\n", encoding="utf-8")
    right_path.write_text("x\n1\n", encoding="utf-8")
    write_prediction_manifest(
        left_path,
        split="test",
        provenance=_provenance("a"),
        track="architecture",
        model="painn",
        seed=2027,
        checkpoint_sha256="3" * 64,
        training_protocol_sha256="4" * 64,
    )
    write_prediction_manifest(
        right_path,
        split="test",
        provenance=_provenance("z"),
        track="architecture",
        model="cgcnn_controlled",
        seed=2027,
        checkpoint_sha256="5" * 64,
        training_protocol_sha256="4" * 64,
    )
    left = load_prediction_manifest(left_path, expected_split="test")
    right = load_prediction_manifest(right_path, expected_split="test")
    with pytest.raises(RuntimeError, match="different benchmark data identities"):
        assert_same_prediction_data_identity(left, right)
