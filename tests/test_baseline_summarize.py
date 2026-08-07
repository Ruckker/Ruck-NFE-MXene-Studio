from __future__ import annotations

import json

import pandas as pd
import pytest

from nfe_model.data_contract import DATA_IMPLEMENTATION_SCHEMA, data_implementation_sha256
from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    STRUCTURE_MANIFEST_SCHEMA,
    TARGET_SCHEMA,
    target_schema_sha256,
)
from training.baselines.summarize import (
    BASELINE_RESULT_SCHEMA,
    assert_common_provenance,
    assert_independent_full_system,
    assert_seed_coverage,
    assert_training_protocols,
    load_results,
    paper_table,
)


def _formal_provenance() -> dict:
    return {
        "dataset_table_sha256": "d" * 64,
        "structure_manifest_schema": STRUCTURE_MANIFEST_SCHEMA,
        "structure_manifest_sha256": "s" * 64,
        "target_schema": TARGET_SCHEMA,
        "target_schema_sha256": target_schema_sha256(),
        "data_implementation_schema": DATA_IMPLEMENTATION_SCHEMA,
        "data_implementation_sha256": data_implementation_sha256(),
        "cache_records_sha256": "c" * 64,
        "split_manifest_sha256": "p" * 64,
        "cache_schema": CACHE_SCHEMA,
        "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
        "neighbor_policy": NEIGHBOR_POLICY,
        "graph_radius_A": 6.0,
        "max_neighbors": 36,
        "git_commit": "a" * 40,
        "git_dirty": False,
    }


def test_paper_table_reports_mean_and_sample_std() -> None:
    frame = pd.DataFrame(
        [
            {
                "track": "architecture",
                "model": "cgcnn_controlled",
                "seed": 2027,
                "parameter_count": 100,
                "test_macro_f1": 0.60,
                "test_balanced_accuracy": 0.65,
                "test_macro_roc_auc": 0.80,
                "test_NFE_Pseudo_Score_mae": 0.10,
            },
            {
                "track": "architecture",
                "model": "cgcnn_controlled",
                "seed": 2028,
                "parameter_count": 100,
                "test_macro_f1": 0.62,
                "test_balanced_accuracy": 0.67,
                "test_macro_roc_auc": 0.82,
                "test_NFE_Pseudo_Score_mae": 0.09,
            },
        ]
    )
    table = paper_table(frame, "architecture")
    assert len(table) == 1
    assert table.iloc[0]["Track"] == "architecture"
    assert table.iloc[0]["Model"] == "CGCNN-style (controlled)"
    assert table.iloc[0]["Seeds"] == 2
    assert table.iloc[0]["test_macro_f1"].startswith("0.61000 ±")
    assert table.iloc[0]["test_NFE_Pseudo_Score_mae"].startswith("0.09500 ±")


def test_full_system_rejects_same_checkpoint_relabelled_as_multiple_seeds() -> None:
    frame = pd.DataFrame(
        [
            {
                "track": "full-system",
                "model": "ours_full",
                "seed": 2027,
                "checkpoint_seed": 2027,
                "checkpoint_sha256": "same-hash",
                "git_commit": "a" * 40,
                "checkpoint_training_git_commit": "a" * 40,
                "checkpoint_training_git_dirty": False,
            },
            {
                "track": "full-system",
                "model": "ours_full",
                "seed": 2028,
                "checkpoint_seed": 2028,
                "checkpoint_sha256": "same-hash",
                "git_commit": "a" * 40,
                "checkpoint_training_git_commit": "a" * 40,
                "checkpoint_training_git_dirty": False,
            },
        ]
    )
    with pytest.raises(RuntimeError, match="distinct checkpoint"):
        assert_independent_full_system(frame, minimum_seeds=2)


def test_formal_summary_rejects_mixed_git_commits_and_dirty_runs() -> None:
    base = _formal_provenance()
    frame = pd.DataFrame([base, {**base, "git_commit": "b" * 40}])
    with pytest.raises(RuntimeError, match="mixed git_commit"):
        assert_common_provenance(frame)
    dirty = pd.DataFrame([base, {**base, "git_dirty": True}])
    with pytest.raises(RuntimeError, match="dirty"):
        assert_common_provenance(dirty)


def test_formal_summary_rejects_stale_but_internally_consistent_cache_semantics() -> None:
    stale = {**_formal_provenance(), "cache_schema": "nfe-mxene-cache-2.2"}
    frame = pd.DataFrame([stale, stale])
    with pytest.raises(RuntimeError, match="stale cache_schema"):
        assert_common_provenance(frame)


def test_formal_models_must_use_same_seed_set() -> None:
    frame = pd.DataFrame(
        [
            {"track": "architecture", "model": "painn", "seed": 2027},
            {"track": "architecture", "model": "painn", "seed": 2028},
            {"track": "architecture", "model": "cgcnn_controlled", "seed": 2027},
            {"track": "architecture", "model": "cgcnn_controlled", "seed": 2029},
        ]
    )
    with pytest.raises(RuntimeError, match="same seed set"):
        assert_seed_coverage(frame, minimum_model_seeds=2)


def test_neural_models_must_share_common_training_protocol() -> None:
    frame = pd.DataFrame(
        [
            {
                "track": "architecture",
                "model": "painn",
                "seed": 2027,
                "benchmark_common_protocol_sha256": "common-a",
                "model_protocol_sha256": "painn",
                "checkpoint_sha256": "p1",
            },
            {
                "track": "architecture",
                "model": "painn",
                "seed": 2028,
                "benchmark_common_protocol_sha256": "common-a",
                "model_protocol_sha256": "painn",
                "checkpoint_sha256": "p2",
            },
            {
                "track": "official-upstream",
                "model": "schnet_official",
                "seed": 2027,
                "benchmark_common_protocol_sha256": "common-b",
                "model_protocol_sha256": "schnet",
                "checkpoint_sha256": "s1",
            },
            {
                "track": "official-upstream",
                "model": "schnet_official",
                "seed": 2028,
                "benchmark_common_protocol_sha256": "common-b",
                "model_protocol_sha256": "schnet",
                "checkpoint_sha256": "s2",
            },
        ]
    )
    with pytest.raises(RuntimeError, match="training/capacity"):
        assert_training_protocols(frame)


def test_loader_accepts_only_current_formal_result_schema(tmp_path) -> None:
    legacy = tmp_path / "architecture" / "painn" / "seed_2027"
    legacy.mkdir(parents=True)
    (legacy / "result.json").write_text(
        json.dumps({"schema": "nfe-baseline-result-2.1", "track": "architecture"}),
        encoding="utf-8",
    )
    formal = tmp_path / "architecture" / "painn" / "seed_2028"
    formal.mkdir(parents=True)
    (formal / "result.json").write_text(
        json.dumps(
            {
                "schema": BASELINE_RESULT_SCHEMA,
                "track": "architecture",
                "model": "painn",
                "seed": 2028,
            }
        ),
        encoding="utf-8",
    )
    rows = load_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["seed"] == 2028
