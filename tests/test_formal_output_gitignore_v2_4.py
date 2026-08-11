from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_ignored(relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative_path],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            "formal predictor artifact can dirty Git provenance: " + relative_path
        )


def test_v2_4_development_predictor_output_directory_is_gitignored() -> None:
    _assert_ignored(
        (
            "training/configs/nfe_predictor_v2_4/best.pt",
            "training/configs/nfe_predictor_v2_4/final_metrics.json",
            "training/configs/nfe_predictor_v2_4/test_predictions.csv",
            "training/configs/nfe_predictor_v2_4/test_predictions.manifest.json",
        )
    )


def test_v2_4_paper_ready_predictor_output_directory_is_gitignored() -> None:
    _assert_ignored(
        (
            "training/configs/nfe_predictor_v2_4_paper_ready/best.pt",
            "training/configs/nfe_predictor_v2_4_paper_ready/final_metrics.json",
            "training/configs/nfe_predictor_v2_4_paper_ready/test_predictions.csv",
            "training/configs/nfe_predictor_v2_4_paper_ready/test_predictions.manifest.json",
        )
    )
