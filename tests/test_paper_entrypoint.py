from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from training import paper


def test_paper_config_matches_explicit_registered_protocol() -> None:
    config = paper._load_paper_config()
    assert config["data"]["max_cache_skip_fraction"] == 0.0
    assert config["training"]["epochs"] == 220
    assert config["training"]["batch_size_per_gpu"] == 96
    assert config["generation"]["minimum_vacuum_A"] == 15.0


def test_paper_config_drift_is_rejected() -> None:
    config = paper._load_paper_config()
    drifted = copy.deepcopy(config)
    drifted["training"]["epochs"] = 5
    with pytest.raises(RuntimeError, match="drifted"):
        paper._validate_paper_config(drifted)


def test_budget_overrides_are_rejected_in_both_cli_forms() -> None:
    with pytest.raises(ValueError, match="immutable"):
        paper._reject_options(
            ["--epochs", "5"], paper.IMMUTABLE_TRAINING_OPTIONS, context="test"
        )
    with pytest.raises(ValueError, match="immutable"):
        paper._reject_options(
            ["--batch-size=48"], paper.IMMUTABLE_TRAINING_OPTIONS, context="test"
        )


def test_baseline_budget_is_injected_from_registered_config() -> None:
    config = paper._load_paper_config()
    arguments = paper._baseline_budget_args("baseline", config)
    pairs = dict(zip(arguments[0::2], arguments[1::2]))
    assert pairs["--epochs"] == "220"
    assert pairs["--batch-size"] == "96"
    assert pairs["--hidden-dim"] == "192"
    assert pairs["--layers"] == "6"
    assert pairs["--dropout"] == "0.12"


def test_paper_training_rejects_ddp_world_size(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(RuntimeError, match="one Python training process"):
        paper._assert_single_process_training("train")


def test_paper_entrypoint_rejects_arbitrary_module_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["training.paper", "nfe_model.train_core"])
    with pytest.raises(ValueError, match="not a paper-ready alias"):
        paper.main()


def test_only_one_canonical_paper_dispatcher_remains() -> None:
    root = Path(__file__).resolve().parents[1] / "training"
    assert (root / "paper.py").is_file()
    assert not (root / "paper_current.py").exists()
    assert not (root / "paper_final.py").exists()
    assert not (root / "paper_v2_4.py").exists()


def test_official_cgcnn_runner_no_longer_claims_fixed_padding() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "training/baselines/official/run.py").read_text(encoding="utf-8")
    assert "fixed-train-max-degree" not in source
    assert "ragged-common-edge-scatter-no-padding" in source
