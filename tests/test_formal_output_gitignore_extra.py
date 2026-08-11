from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "training/evaluation/ood_manifest.csv",
        "training/ablations/results/ablation_paper_table.csv",
        "training/configs/nfe_predictor/best.pt",
        "training/configs/nfe_predictor/final_metrics.json",
        "training/configs/nfe_predictor/test_predictions.csv",
    ],
)
def test_additional_formal_generated_paths_are_gitignored(relative_path: str) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "formal generated artifact is not ignored and can invalidate git_dirty provenance: "
        f"{relative_path}"
    )
