from __future__ import annotations

import numpy as np
import pandas as pd

from training.evaluation.formal_multiseed_bootstrap import _observed_seed_mean_delta
from training.evaluation.formal_multiseed_bootstrap_strict import PLANNED_COMPARISONS


def _frame(delta: float) -> pd.DataFrame:
    # Two classes would be insufficient for macro-F1, so this fixture targets
    # the continuous MAE estimand. Model A has absolute error |delta| above B.
    truth = np.asarray([0.2, 0.8], dtype=float)
    pred_b = truth.copy()
    pred_a = truth + float(delta)
    return pd.DataFrame(
        {
            "True_Label_a": ["low", "high"],
            "True_Label_b": ["low", "high"],
            "True_NFE_Pseudo_Score_a": truth,
            "True_NFE_Pseudo_Score_b": truth,
            "Predicted_NFE_Pseudo_Score_a": pred_a,
            "Predicted_NFE_Pseudo_Score_b": pred_b,
        }
    )


def test_observed_point_estimate_is_mean_of_complete_seed_deltas() -> None:
    pairs = [
        (2027, _frame(0.10), "a1", "b1"),
        (2028, _frame(0.20), "a2", "b2"),
        (2029, _frame(0.30), "a3", "b3"),
    ]
    observed, rows = _observed_seed_mean_delta(pairs, "NFE_Pseudo_Score_mae")
    assert observed == np.mean([0.10, 0.20, 0.30])
    assert [row["seed"] for row in rows] == [2027, 2028, 2029]
    assert [row["delta_a_minus_b"] for row in rows] == [0.10, 0.20, 0.30]


def test_preregistered_comparisons_have_one_fixed_direction() -> None:
    assert ("full-system/ours_full", "architecture/painn") in PLANNED_COMPARISONS
    assert ("architecture/painn", "full-system/ours_full") not in PLANNED_COMPARISONS
    assert ("ablation/no_denoise", "ablation/no_vector") in PLANNED_COMPARISONS
    assert ("ablation/no_vector", "ablation/no_denoise") not in PLANNED_COMPARISONS
