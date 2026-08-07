from __future__ import annotations

import os
import sys

from nfe_model.provenance_v2 import git_repository_state
from training import formal_v2_4


TRAINING_ALIASES = {"train", "ablation", "baseline", "official"}


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


def _assert_single_process_training(requested: str) -> None:
    if requested not in TRAINING_ALIASES:
        return
    world_size = _pytorch_world_size()
    if world_size != 1:
        raise RuntimeError(
            "paper optimization protocol fixes one Python training process per independent run; "
            f"observed WORLD_SIZE={world_size}. Use the available GPUs to run independent seeds/models concurrently."
        )
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank not in (None, "", "0") or local_rank not in (None, "", "0"):
        raise RuntimeError(
            "paper training must not run as a nonzero torchrun rank; launch one normal Python process per GPU"
        )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Canonical paper-ready NFE benchmark dispatcher.\n"
            "Adds clean-Git and single-process optimization guards to the pair-symmetric v2.4 contract.\n\n"
            + formal_v2_4._usage()
        )
        return 0
    requested = sys.argv[1]
    _assert_clean_git()
    _assert_single_process_training(requested)
    return formal_v2_4.main()


if __name__ == "__main__":
    raise SystemExit(main())
