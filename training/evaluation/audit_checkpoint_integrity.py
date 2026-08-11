from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfe_model.checkpoint_contract import assert_checkpoint_internal_contract
from nfe_model.data_v2 import torch_load_compat
from nfe_model.provenance_v2 import file_sha256, git_repository_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify internal normalizer/model metadata and Git identity of formal checkpoints."
    )
    parser.add_argument("checkpoints", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = git_repository_state()
    if runtime.get("git_dirty") is not False:
        raise RuntimeError("formal checkpoint audit requires a clean runtime Git worktree")
    runtime_commit = str(runtime.get("git_commit", "unknown"))
    if runtime_commit == "unknown":
        raise RuntimeError("formal checkpoint audit requires a resolvable runtime Git commit")

    rows = []
    for value in args.checkpoints:
        path = Path(value).resolve()
        checkpoint = torch_load_compat(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint is not a mapping: {path}")
        assert_checkpoint_internal_contract(checkpoint)
        provenance = checkpoint["provenance"]
        training_commit = str(provenance.get("git_commit", "unknown"))
        if provenance.get("git_dirty") is not False:
            raise RuntimeError(f"checkpoint was produced from dirty/unknown worktree: {path}")
        if training_commit != runtime_commit:
            raise RuntimeError(
                f"checkpoint/runtime Git mismatch for {path}: training={training_commit} runtime={runtime_commit}"
            )
        rows.append(
            {
                "checkpoint": str(path),
                "sha256": file_sha256(path),
                "git_commit": training_commit,
                "normalizer_sha256": provenance.get("normalizer_sha256"),
                "format": checkpoint.get("format"),
                "seed": checkpoint.get("config", {}).get("seed"),
            }
        )
    print(json.dumps({"checkpoint_integrity": True, "checkpoints": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
