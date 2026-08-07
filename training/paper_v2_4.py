from __future__ import annotations

import os
import sys

from nfe_model.provenance_v2 import git_repository_state
from training import formal_v2_4


TRAINING_ALIASES = {"train", "ablation", "baseline", "official"}


def _world_size() -> int:
    values = []
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        raw = os.environ.get(key)
        if raw:
            try:
                values.append(int(raw))
            except ValueError as exc:
                raise ValueError(f"invalid {key}={raw!r}") from exc
    return max(values, default=1)


def _assert_clean_git() -> None:
    state = git_repository_state()
    commit = str(state.get("git_commit", "unknown"))
    if state.get("git_dirty") is not False:
        raise RuntimeError(
            "paper v2.4 commands require a clean Git worktree before execution"
        )
    if commit == "unknown" or len(commit) != 40:
        raise RuntimeError("paper v2.4 commands require a resolvable Git commit")


def _assert_single_process_training(requested: str) -> None:
    if requested not in TRAINING_ALIASES:
        return
    world_size = _world_size()
    if world_size != 1:
        raise RuntimeError(
            "paper v2.4 fixes one training process/GPU per independent run so effective batch and "
            f"optimization semantics are identical across full/ablation/baseline models; observed WORLD_SIZE={world_size}. "
            "Use the four GPUs to run four independent seeds/models concurrently, not DDP inside one run."
        )
    # torchrun may set rank variables even if a malformed command reports a
    # default/absent WORLD_SIZE. Reject that ambiguity rather than guessing.
    if os.environ.get("RANK") not in (None, "", "0") or os.environ.get("LOCAL_RANK") not in (None, "", "0"):
        raise RuntimeError(
            "paper v2.4 training must not be launched under multi-process torchrun; "
            "run one normal Python process per GPU"
        )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Paper-ready wrapper around training.formal_v2_4.\n"
            "It additionally requires a clean Git worktree and fixes every training run to one process/GPU.\n\n"
            + formal_v2_4._usage()
        )
        return 0
    requested = sys.argv[1]
    _assert_clean_git()
    _assert_single_process_training(requested)
    return formal_v2_4.main()


if __name__ == "__main__":
    raise SystemExit(main())
