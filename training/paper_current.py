from __future__ import annotations

import os
import sys

from nfe_model.provenance_v2 import git_repository_state
from training import formal_v2_4


PAPER_CONFIG = "training/configs/nfe_predictor_v2_4_paper.yaml"
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


def _assert_clean_git() -> None:
    state = git_repository_state()
    commit = str(state.get("git_commit", "unknown"))
    if state.get("git_dirty") is not False:
        raise RuntimeError("paper-current commands require a clean Git worktree")
    if commit == "unknown" or len(commit) != 40:
        raise RuntimeError("paper-current commands require a resolvable Git commit")


def _world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1").strip() or "1"
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid WORLD_SIZE={raw!r}") from exc
    if value <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {value}")
    return value


def _assert_training_runtime(alias: str) -> None:
    if alias not in TRAINING_ALIASES:
        return
    if _world_size() != 1:
        raise RuntimeError(
            "paper-current fixes one Python training process/GPU per independent run; "
            "parallelize independent seeds/models across GPUs instead of using DDP inside one run"
        )
    if os.environ.get("RANK") not in (None, "", "0"):
        raise RuntimeError("paper-current training refuses nonzero torchrun RANK")
    if os.environ.get("LOCAL_RANK") not in (None, "", "0"):
        raise RuntimeError("paper-current training refuses nonzero torchrun LOCAL_RANK")


def _usage() -> str:
    rows = "\n".join(f"  {alias:26s} {module}" for alias, module in ALIASES.items())
    return (
        "Usage: python -m training.paper_current <alias> [arguments...]\n\n"
        "Only the audited aliases below are accepted for paper-ready work; arbitrary module passthrough is intentionally disabled.\n"
        f"Default config for data-consuming commands: {PAPER_CONFIG}\n\n{rows}"
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0
    alias = sys.argv[1]
    if alias not in ALIASES:
        raise ValueError(
            f"{alias!r} is not a paper-current alias. Arbitrary module passthrough is disabled.\n{_usage()}"
        )
    _assert_clean_git()
    _assert_training_runtime(alias)

    module = ALIASES[alias]
    arguments = list(sys.argv[2:])
    if alias in CONFIG_ALIASES and "--config" not in arguments:
        arguments = ["--config", PAPER_CONFIG, *arguments]
    # formal_v2_4 performs the crucial pair-symmetric data_v2 patch before it
    # imports the target module. Supplying the explicit target avoids any lower
    # level default-config alias rewrite.
    sys.argv = [sys.argv[0], module, *arguments]
    return formal_v2_4.main()


if __name__ == "__main__":
    raise SystemExit(main())
