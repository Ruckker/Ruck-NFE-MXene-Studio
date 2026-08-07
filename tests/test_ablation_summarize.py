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


def test_ablation_protocol_matrix_rejects_directory_or_protocol_mismatch() -> None:
    wrong_name = pd.DataFrame(
        [
            {
                "ablation": "no_global",
                "ablation_config_name": "full",
                "training_protocol_sha256": "p",
                "experiment_protocol_sha256": "e1",
            },
            {
                "ablation": "no_global",
                "ablation_config_name": "full",
                "training_protocol_sha256": "p",
                "experiment_protocol_sha256": "e2",
            },
        ]
    )
    with pytest.raises(RuntimeError, match="directory/checkpoint"):
        assert_protocol_matrix(wrong_name)

    mixed = pd.DataFrame(
        [
            {
                "ablation": "no_global",
                "ablation_config_name": "no_global",
                "training_protocol_sha256": "p1",
                "experiment_protocol_sha256": "e1",
            },
            {
                "ablation": "no_global",
                "ablation_config_name": "no_global",
                "training_protocol_sha256": "p2",
                "experiment_protocol_sha256": "e2",
            },
        ]
    )
    with pytest.raises(RuntimeError, match="mixes training protocols"):
        assert_protocol_matrix(mixed)