from __future__ import annotations

import pandas as pd
import pytest

from training.baselines.summarize import assert_independent_full_system, paper_table


def test_paper_table_reports_mean_and_sample_std() -> None:
    frame = pd.DataFrame(
        [
            {
                "track": "architecture",
                "model": "cgcnn",
                "seed": 2027,
                "test_macro_f1": 0.60,
                "test_balanced_accuracy": 0.65,
                "test_macro_roc_auc": 0.80,
                "test_NFE_Pseudo_Score_mae": 0.10,
            },
            {
                "track": "architecture",
                "model": "cgcnn",
                "seed": 2028,
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
            },
            {
                "track": "full-system",
                "model": "ours_full",
                "seed": 2028,
                "checkpoint_seed": 2028,
                "checkpoint_sha256": "same-hash",
            },
        ]
    )
    with pytest.raises(RuntimeError, match="identical checkpoint"):
        assert_independent_full_system(frame, minimum_seeds=2)
