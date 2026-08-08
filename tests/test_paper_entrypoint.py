from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from training import paper


def test_paper_config_matches_explicit_registered_protocol() -> None:
    config = paper._load_paper_config()
    assert config["seed"] == 2027
    assert paper.EXPECTED_SEEDS == (2027, 2028, 2029, 2030, 2031)
    assert config["data"]["max_cache_skip_fraction"] == 0.0
    assert config["data"]["cache"].endswith("nfe_graphs_v2_4.pt")
    assert config["training"]["epochs"] == 220
    assert config["training"]["batch_size_per_gpu"] == 96
    assert config["inference"]["embedding_bank_size"] == 4096
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
    with pytest.raises(ValueError, match="immutable"):
        paper._reject_options(
            ["--device", "cpu"], paper.IMMUTABLE_TRAINING_OPTIONS, context="test"
        )
    seeds, stripped = paper._consume_registered_seed_subset(
        ["--seeds", "2027,2029", "--model", "painn"]
    )
    assert seeds == [2027, 2029]
    assert stripped == ["--model", "painn"]
    with pytest.raises(ValueError, match="unregistered"):
        paper._consume_registered_seed_subset(["--seeds", "1,2027"])


def test_baseline_budget_is_injected_from_registered_config() -> None:
    config = paper._load_paper_config()
    arguments = paper._baseline_budget_args("baseline", config)
    pairs = dict(zip(arguments[0::2], arguments[1::2]))
    assert pairs["--epochs"] == "220"
    assert pairs["--batch-size"] == "96"
    assert pairs["--hidden-dim"] == "192"
    assert pairs["--layers"] == "6"
    assert pairs["--dropout"] == "0.12"
    assert pairs["--device"] == "cuda"
    assert pairs["--seeds"] == "2027,2028,2029,2030,2031"


def test_paper_baseline_seed_subset_is_injected_without_changing_budget() -> None:
    config = paper._load_paper_config()
    arguments = paper._baseline_budget_args("baseline", config, [2027])
    pairs = dict(zip(arguments[0::2], arguments[1::2]))
    assert pairs["--seeds"] == "2027"
    assert pairs["--epochs"] == "220"


def test_secondary_paper_analysis_knobs_are_fixed() -> None:
    args = paper._fixed_secondary_args("ood-manifest", [])
    assert dict(zip(args[0::2], args[1::2]))["--cell-size-quantile"] == "0.95"
    with pytest.raises(ValueError, match="immutable"):
        paper._fixed_secondary_args(
            "representation-audit", ["--max-score-drift", "0.01"]
        )


def test_paper_training_rejects_ddp_world_size(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(RuntimeError, match="one Python training process"):
        paper._assert_single_process_training("train")


def test_paper_training_rejects_cpu_runtime(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(paper.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        paper._assert_single_process_training("train")


def test_paper_training_accepts_single_cuda_process(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(paper.torch.cuda, "is_available", lambda: True)
    paper._assert_single_process_training("train")


def test_paper_ablation_seed_is_registered() -> None:
    paper._validate_ablation_seed(["--ablation", "full", "--seed", "2027"])
    with pytest.raises(ValueError, match="outside registered set"):
        paper._validate_ablation_seed(["--ablation", "full", "--seed", "1"])


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


def test_paper_aliases_use_existing_strict_components() -> None:
    assert paper.ALIASES["baseline"] == "training.baselines.run_formal"
    assert paper.ALIASES["baseline-summary"] == "training.baselines.summarize"
    assert paper.ALIASES["ablation-summary"] == "training.ablations.summarize"
    assert paper.ALIASES["ood-manifest"] == "training.evaluation.build_ood_manifest"
    assert paper.ALIASES["paper-preflight"] == "training.evaluation.paper_preflight_strict"


def test_paper_ready_checkpoint_directory_is_git_ignored() -> None:
    root = Path(__file__).resolve().parents[1]
    ignore = root / "training/configs/nfe_predictor_v2_4_paper_ready/.gitignore"
    assert ignore.is_file()
    assert "*" in ignore.read_text(encoding="utf-8")


def test_official_cgcnn_runner_no_longer_claims_fixed_padding() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "training/baselines/official/run.py").read_text(encoding="utf-8")
    assert "fixed-train-max-degree" not in source
    assert "ragged-common-edge-scatter-no-padding" in source
