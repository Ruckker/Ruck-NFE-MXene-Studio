from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "cache/nfe_graphs_v2_3.pt",
        "runs/ablations/full/seed_2027/best.pt",
        "runs/ablations/full/seed_2027/final_metrics.json",
        "runs/ablations/full/seed_2027/test_predictions.csv",
        "runs/ablations/full/seed_2027/test_predictions.manifest.json",
        "training/baselines/results/architecture/painn/seed_2027/best.pt",
        "training/baselines/results/architecture/painn/seed_2027/result.json",
        "training/baselines/results/architecture/painn/seed_2027/test_predictions.csv",
        "training/baselines/results/architecture/painn/seed_2027/test_predictions.manifest.json",
        "training/evaluation/results/paired_bootstrap.json",
        "training/evaluation/results/split_duplicate_audit.json",
        "training/evaluation/results/supercell_consistency.json",
    ],
)
def test_formal_generated_paths_are_gitignored(relative_path: str) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"formal generated artifact is not ignored by Git and can invalidate git_dirty provenance: "
        f"{relative_path}"
    )
