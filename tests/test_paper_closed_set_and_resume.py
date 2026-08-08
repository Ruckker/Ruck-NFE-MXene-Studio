from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from nfe_model import train_ablation_safe
from training.ablations.summarize_paper import assert_complete_paper_ablation_set
from training.baselines.summarize_paper import (
    EXPECTED_PAPER_MODELS,
    assert_complete_paper_model_set,
)


def test_cross_run_ablation_resume_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 2027\n", encoding="utf-8")
    args = Namespace(
        config=str(config),
        seed=2027,
        checkpoint_dir=str(tmp_path / "run_a"),
        ablation="full",
        resume=str(tmp_path / "run_b" / "best.pt"),
    )
    with pytest.raises(ValueError, match="same run directory"):
        train_ablation_safe._assert_same_run_resume(args)


def test_same_run_ablation_resume_is_accepted(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 2027\n", encoding="utf-8")
    run_dir = tmp_path / "run_a"
    args = Namespace(
        config=str(config),
        seed=2027,
        checkpoint_dir=str(run_dir),
        ablation="full",
        resume=str(run_dir / "best.pt"),
    )
    train_ablation_safe._assert_same_run_resume(args)


def _complete_benchmark_frame() -> pd.DataFrame:
    rows = []
    for track, models in EXPECTED_PAPER_MODELS.items():
        for model in sorted(models):
            rows.append({"track": track, "model": model})
    return pd.DataFrame(rows)


def test_paper_benchmark_requires_exact_preregistered_model_set() -> None:
    complete = _complete_benchmark_frame()
    assert_complete_paper_model_set(complete)
    incomplete = complete[complete["model"] != "m3gnet_official"].copy()
    with pytest.raises(RuntimeError, match="missing"):
        assert_complete_paper_model_set(incomplete)


def test_paper_benchmark_rejects_unexpected_model() -> None:
    frame = _complete_benchmark_frame()
    frame.loc[len(frame)] = {"track": "architecture", "model": "post_hoc_model"}
    with pytest.raises(RuntimeError, match="unexpected"):
        assert_complete_paper_model_set(frame)


def test_paper_ablation_requires_all_nine_rows() -> None:
    names = [
        "full",
        "no_vector",
        "no_global",
        "no_masked_pretrain",
        "no_denoise",
        "no_self_supervision",
        "no_auxiliary_regression",
        "matched_supervision",
        "classification_only",
    ]
    frame = pd.DataFrame({"ablation": names})
    assert_complete_paper_ablation_set(frame)
    with pytest.raises(RuntimeError, match="missing"):
        assert_complete_paper_ablation_set(frame[frame["ablation"] != "no_global"])
