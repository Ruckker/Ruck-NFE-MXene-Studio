from __future__ import annotations

import pandas as pd

from training.ablations.summarize import paper_table


def test_ablation_paper_table_reports_delta_from_full() -> None:
    frame = pd.DataFrame(
        [
            {
                "ablation": "full",
                "seed": 2027,
                "test_macro_f1": 0.74,
                "test_NFE_Pseudo_Score_mae": 0.035,
                "test_calibrated_ece": 0.014,
            },
            {
                "ablation": "no_global",
                "seed": 2027,
                "test_macro_f1": 0.70,
                "test_NFE_Pseudo_Score_mae": 0.045,
                "test_calibrated_ece": 0.020,
            },
            {
                "ablation": "classification_only",
                "seed": 2027,
                "test_macro_f1": 0.68,
                "test_NFE_Pseudo_Score_mae": 0.50,
                "test_calibrated_ece": 0.025,
            },
        ]
    )
    table = paper_table(frame)
    no_global = table[table["Ablation"] == "− global slab features"].iloc[0]
    classification = table[table["Ablation"] == "Classification only"].iloc[0]
    assert abs(float(no_global["Δ macro F1 vs full"]) + 0.04) < 1e-12
    assert abs(float(no_global["Δ score MAE vs full"]) - 0.01) < 1e-12
    assert classification["test_NFE_Pseudo_Score_mae"] == "N/A"
