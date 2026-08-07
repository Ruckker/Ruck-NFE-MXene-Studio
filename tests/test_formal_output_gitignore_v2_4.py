from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v2_4_predictor_output_directory_is_gitignored() -> None:
    relative_paths = (
        "training/configs/nfe_predictor_v2_4/best.pt",
        "training/configs/nfe_predictor_v2_4/final_metrics.json",
        "training/configs/nfe_predictor_v2_4/test_predictions.csv",
        "training/configs/nfe_predictor_v2_4/test_predictions.manifest.json",
    )
    for relative_path in relative_paths:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative_path],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            "v2.4 formal predictor artifact can dirty Git provenance: " + relative_path
        )
