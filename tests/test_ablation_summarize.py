from __future__ import annotations

import pandas as pd
import pytest

from training.ablations.summarize import (
    assert_complete_seed_matrix,
    assert_protocol_matrix,
    paper_table,
)


def test_ablation_paper_table_reports_paired_delta_from_full() -> None:
    frame = pd.DataFrame(
        [
            {
                "ablation": "full",
                "seed": 2027,
                "test_macro_f1": 0.74,
                "test_NFE_Pseudo_Score_mae": 0.035,
            },
            {
                "ablation": "full",
                "seed": 2028,
                "test_macro_f1": 0.72,
                "test_NFE_Pseudo_Score_mae": 0.040,
            },
            {
                "ablation": "no_global",
                "seed": 2027,
                "test_macro_f1": 0.70,
                "test_NFE_Pseudo_Score_mae": 0.045,
            },
            {
                "ablation": "no_global",
                "seed": 2028,
                "test_macro_f1": 0.71,
                "test_NFE_Pseudo_Score_mae": 0.050,
            },
            {
                "ablation": "classification_only",
                "seed": 2027,
                "test_macro_f1": 0.68,
            },
            {
                "ablation": "classification_only",
                "seed": 2028,
                "test_macro_f1": 0.67,
            },
        ]
    )
    table = paper_table(frame)
    no_global = table[table["Ablation"] == "− global slab information"].iloc[0]
    classification = table[table["Ablation"] == "Classification only"].iloc[0]
    assert no_global["Δ macro F1 vs full (paired)"].startswith("-0.02500 ±")
    assert no_global["Δ score MAE vs full (paired)"].startswith("0.01000 ±")
    assert classification["test_NFE_Pseudo_Score_mae"] == "N/A"


def test_ablation_matrix_rejects_mismatched_seeds_and_duplicate_checkpoints() -> None:
    mismatched = pd.DataFrame(
        [
            {"ablation": "full", "seed": 2027, "checkpoint_sha256": "f1"},
            {"ablation": "full", "seed": 2028, "checkpoint_sha256": "f2"},
            {"ablation": "no_global", "seed": 2027, "checkpoint_sha256": "g1"},
            {"ablation": "no_global", "seed": 2029, "checkpoint_sha256": "g2"},
        ]
    )
    with pytest.raises(RuntimeError, match="same seed set"):
        assert_complete_seed_matrix(mismatched, minimum_seeds=2)

    duplicated = pd.DataFrame(
        [
            {"ablation": "full", "seed": 2027, "checkpoint_sha256": "same"},
            {"ablation": "full", "seed": 2028, "checkpoint_sha256": "same"},
        ]
    )
    with pytest.raises(RuntimeError, match="distinct checkpoint"):
        assert_complete_seed_matrix(duplicated, minimum_seeds=2)


def _protocol_row(
    ablation: str,
    seed_suffix: str,
    *,
    training: str = "train",
    runtime: str = "runtime",
    model: str = "model",
) -> dict[str, str]:
    return {
        "ablation": ablation,
        "ablation_config_name": ablation,
        "training_protocol_sha256": training,
        "training_runtime_environment_sha256": runtime,
        "model_protocol_sha256": model,
        "experiment_protocol_sha256": f"experiment-{seed_suffix}",
    }


def test_ablation_protocol_matrix_rejects_directory_or_protocol_mismatch() -> None:
    wrong_name = pd.DataFrame(
        [
            {
                **_protocol_row("no_global", "1"),
                "ablation_config_name": "full",
            },
            {
                **_protocol_row("no_global", "2"),
                "ablation_config_name": "full",
            },
        ]
    )
    with pytest.raises(RuntimeError, match="directory/checkpoint"):
        assert_protocol_matrix(wrong_name)

    mixed = pd.DataFrame(
        [
            _protocol_row("no_global", "1", training="p1"),
            _protocol_row("no_global", "2", training="p2"),
        ]
    )
    with pytest.raises(RuntimeError, match="mixes training_protocol_sha256"):
        assert_protocol_matrix(mixed)


def test_ablation_protocol_matrix_rejects_runtime_or_model_protocol_drift() -> None:
    mixed_runtime_within_seed_group = pd.DataFrame(
        [
            _protocol_row("full", "1", runtime="r1"),
            _protocol_row("full", "2", runtime="r2"),
        ]
    )
    with pytest.raises(RuntimeError, match="training_runtime_environment_sha256"):
        assert_protocol_matrix(mixed_runtime_within_seed_group)

    mixed_runtime_across_ablation = pd.DataFrame(
        [
            _protocol_row("full", "1", runtime="r1", model="full-model"),
            _protocol_row("full", "2", runtime="r1", model="full-model"),
            _protocol_row("no_global", "1", runtime="r2", model="ng-model"),
            _protocol_row("no_global", "2", runtime="r2", model="ng-model"),
        ]
    )
    with pytest.raises(RuntimeError, match="causal ablation matrix mixes"):
        assert_protocol_matrix(mixed_runtime_across_ablation)

    mixed_model = pd.DataFrame(
        [
            _protocol_row("full", "1", model="m1"),
            _protocol_row("full", "2", model="m2"),
        ]
    )
    with pytest.raises(RuntimeError, match="model_protocol_sha256"):
        assert_protocol_matrix(mixed_model)


def test_ablation_protocol_matrix_accepts_one_runtime_with_distinct_ablation_models() -> None:
    frame = pd.DataFrame(
        [
            _protocol_row("full", "1", runtime="same", model="full-model"),
            _protocol_row("full", "2", runtime="same", model="full-model"),
            _protocol_row("no_global", "1", runtime="same", model="ng-model"),
            _protocol_row("no_global", "2", runtime="same", model="ng-model"),
        ]
    )
    assert_protocol_matrix(frame)
