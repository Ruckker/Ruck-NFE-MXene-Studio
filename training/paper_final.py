from __future__ import annotations

import os
import sys

from nfe_model.provenance_v2 import git_repository_state
from training import formal_v2_4


PAPER_CONFIG = "training/configs/nfe_predictor_v2_4_paper.yaml"
TRAINING_ALIASES = {"train", "ablation", "baseline", "official"}
EXTRA_ALIASES = {
    "generator-contract-audit": "training.evaluation.audit_generator_predictor_contract",
}
CONFIG_TARGETS = set(formal_v2_4.CONFIG_MODULES) | set(EXTRA_ALIASES.values())


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
        raise RuntimeError("paper-final commands require a clean Git worktree before execution")
    if commit == "unknown" or len(commit) != 40:
        raise RuntimeError("paper-final commands require a resolvable 40-character Git commit")


def _assert_single_process_training(requested: str) -> None:
    if requested not in TRAINING_ALIASES:
        return
    if _pytorch_world_size() != 1:
        raise RuntimeError(
            "paper-final optimization fixes one Python process/GPU per independent run. "
            "Use the available GPUs to run independent seeds/models concurrently rather than DDP."
        )
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank not in (None, "", "0") or local_rank not in (None, "", "0"):
        raise RuntimeError(
            "paper-final training must not be launched as a nonzero torchrun rank"
        )


def _rewrite_arguments() -> None:
    requested = sys.argv[1]
    module = EXTRA_ALIASES.get(requested, formal_v2_4.ALIASES.get(requested, requested))
    arguments = list(sys.argv[2:])
    if module in CONFIG_TARGETS and "--config" not in arguments:
        arguments = ["--config", PAPER_CONFIG, *arguments]
    # Pass an explicit module name so formal_v2_4 cannot replace it with a
    # legacy/default alias or inject the older v2.4 benchmark-only config.
    sys.argv = [sys.argv[0], module, *arguments]


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Final canonical paper-ready dispatcher.\n"
            f"Default configuration: {PAPER_CONFIG}\n"
            "Graph contract: pair-symmetric v2.4; training contract: one process/GPU per independent run.\n\n"
            + formal_v2_4._usage()
            + "\n  generator-contract-audit -> training.evaluation.audit_generator_predictor_contract"
        )
        return 0
    requested = sys.argv[1]
    _assert_clean_git()
    _assert_single_process_training(requested)
    _rewrite_arguments()
    return formal_v2_4.main()


if __name__ == "__main__":
    raise SystemExit(main())
