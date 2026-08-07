from __future__ import annotations

import pandas as pd

from training.baselines.summarize import paper_table


def test_paper_table_reports_mean_and_sample_std() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "cgcnn",
                "test_macro_f1": 0.60,
                "test_balanced_accuracy": 0.65,
                "test_macro_roc_auc": 0.80,
                "test_NFE_Pseudo_Score_mae": 0.10,
            },
            {
                "model": "cgcnn",
                "test_macro_f1": 0.62,
                "test_balanced_accuracy": 0.67,
                "test_macro_roc_auc": 0.82,
                "test_NFE_Pseudo_Score_mae": 0.09,
            },
        ]
    )
    table = paper_table(frame)
    assert len(table) == 1
    assert table.iloc[0]["Model"] == "cgcnn"
    assert table.iloc[0]["test_macro_f1"].startswith("0.61000 ±")
    assert table.iloc[0]["test_NFE_Pseudo_Score_mae"].startswith("0.09500 ±")
