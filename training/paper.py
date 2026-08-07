from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import yaml

from nfe_model.provenance_v2 import git_repository_state
from training import formal_v2_4


PAPER_CONFIG = "training/configs/nfe_predictor_v2_4_paper_ready.yaml"
ALIASES = {
    **formal_v2_4.ALIASES,
    "generator-contract-audit": "training.evaluation.audit_generator_predictor_contract",
}
TRAINING_ALIASES = {"train", "ablation", "baseline", "official"}
CONFIG_ALIASES = {
    "train",
    "ablation",
    "baseline",
    "official",
    "cache-rebuild-audit",
    "cache-sanity-audit",
    "split-duplicate-audit",
    "neighbor-symmetry-audit",
    "verified-queue",
    "generator-contract-audit",
}

# These options change the pre-registered optimization/capacity protocol. They
# are deliberately unavailable through the paper-ready dispatcher. Short smoke
# runs and exploratory overrides must use training.formal_v2_4 instead and must
# not be admitted to paper tables.
IMMUTABLE_TRAINING_OPTIONS = {
    "--config",
    "--epochs",
    "--batch-size",
    "--batch-size-per-gpu",
    "--learning-rate",
    "--min-learning-rate",
    "--warmup-epochs",
    "--weight-decay",
    "--patience",
    "--hidden-dim",
    "--layers",
    "--dropout",
    "--label-smoothing",
    "--no-amp",
    "--amp",
    "--rebuild-cache",
    "--allow-unverified-checkpoint",
}
IMMUTABLE_SUMMARY_OPTIONS = {
    "--minimum-full-seeds",
    "--minimum-model-seeds",
    "--minimum-seeds",
}


def _pytorch_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1").strip() or "1"
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid PyTorch WORLD_SIZE={raw!r}") from exc
    if value <= 0:
        raise ValueError(f"PyTorch WORLD_SIZE must be positive, got {value}")
    return value


def _assert_clean_git() -> None:
    state = git_repository_state()
    commit = str(state.get("git_commit", "unknown"))
    if state.get("git_dirty") is not False:
        raise RuntimeError("paper commands require a clean Git worktree before execution")
    if commit == "unknown" or len(commit) != 40:
        raise RuntimeError("paper commands require a resolvable 40-character Git commit")


def _assert_single_process_training(alias: str) -> None:
    if alias not in TRAINING_ALIASES:
        return
    world_size = _pytorch_world_size()
    if world_size != 1:
        raise RuntimeError(
            "paper optimization protocol fixes one Python training process per independent run; "
            f"observed WORLD_SIZE={world_size}. Run independent seeds/models concurrently across GPUs instead."
        )
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank not in (None, "", "0") or local_rank not in (None, "", "0"):
        raise RuntimeError(
            "paper training must not run as a nonzero torchrun rank; launch one normal Python process per GPU"
        )


def _option_name(token: str) -> str | None:
    if not token.startswith("--"):
        return None
    return token.split("=", 1)[0]


def _reject_options(arguments: Iterable[str], forbidden: set[str], *, context: str) -> None:
    observed = sorted(
        {
            name
            for token in arguments
            if (name := _option_name(token)) is not None and name in forbidden
        }
    )
    if observed:
        raise ValueError(
            f"{context} is immutable for paper-ready runs; forbidden overrides={observed}. "
            "Use python -m training.formal_v2_4 for smoke/exploratory runs instead."
        )


def _load_paper_config() -> dict:
    path = Path(PAPER_CONFIG)
    if not path.is_file():
        raise FileNotFoundError(f"paper-ready configuration not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid paper-ready configuration: {path}")
    return config


def _baseline_budget_args(alias: str, config: dict) -> list[str]:
    if alias not in {"baseline", "official"}:
        return []
    training = config["training"]
    model = config["model"]
    loss = config["loss"]
    values: list[tuple[str, object]] = [
        ("--epochs", training["epochs"]),
        ("--batch-size", training["batch_size_per_gpu"]),
        ("--learning-rate", training["learning_rate"]),
        ("--min-learning-rate", training["min_learning_rate"]),
        ("--warmup-epochs", training["warmup_epochs"]),
        ("--weight-decay", training["weight_decay"]),
        ("--patience", training["early_stopping_patience"]),
        ("--hidden-dim", model["hidden_dim"]),
        ("--layers", model["num_layers"]),
        ("--label-smoothing", loss["label_smoothing"]),
    ]
    if alias == "baseline":
        values.append(("--dropout", model["dropout"]))
    arguments: list[str] = []
    for name, value in values:
        arguments.extend([name, str(value)])
    if bool(training.get("amp", True)) is not True:
        raise RuntimeError("paper-ready architecture/official protocol requires training.amp=true")
    return arguments


def _usage() -> str:
    rows = "\n".join(f"  {alias:26s} -> {module}" for alias, module in ALIASES.items())
    return (
        "Usage: python -m training.paper <alias> [arguments...]\n\n"
        "This is the only paper-ready dispatcher. Arbitrary module passthrough is disabled.\n"
        f"Immutable config: {PAPER_CONFIG}\n"
        "Training runs are one Python process / one GPU; parallelize independent seeds/models across GPUs.\n\n"
        f"{rows}\n"
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    alias = sys.argv[1]
    if alias not in ALIASES:
        raise ValueError(
            f"{alias!r} is not a paper-ready alias. Arbitrary module passthrough is disabled.\n{_usage()}"
        )
    arguments = list(sys.argv[2:])

    _assert_clean_git()
    _assert_single_process_training(alias)
    if alias in TRAINING_ALIASES:
        _reject_options(arguments, IMMUTABLE_TRAINING_OPTIONS, context="paper training budget")
    elif any(_option_name(token) == "--config" for token in arguments):
        raise ValueError(
            f"paper-ready data identity is fixed to {PAPER_CONFIG}; --config overrides are forbidden"
        )
    if alias in {"baseline-summary", "ablation-summary"}:
        _reject_options(arguments, IMMUTABLE_SUMMARY_OPTIONS, context="paper seed-count gate")

    config = _load_paper_config()
    module = ALIASES[alias]
    fixed_arguments = _baseline_budget_args(alias, config)
    if alias in CONFIG_ALIASES:
        fixed_arguments = ["--config", PAPER_CONFIG, *fixed_arguments]

    # formal_v2_4 installs the pair-symmetric data/graph contract before the
    # target module is imported. We pass the explicit audited module here so the
    # lower-level development default config cannot replace PAPER_CONFIG.
    sys.argv = [sys.argv[0], module, *fixed_arguments, *arguments]
    formal_v2_4.DEFAULT_CONFIG = PAPER_CONFIG
    return formal_v2_4.main()


if __name__ == "__main__":
    raise SystemExit(main())
