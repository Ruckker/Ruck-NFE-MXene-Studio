from __future__ import annotations

from pathlib import Path
from typing import Sequence

from nfe_model.checkpoint_contract import assert_checkpoint_internal_contract
from nfe_model.data_v2 import torch_load_compat

from . import run as base


_ORIGINAL_EVALUATE_FULL = base.evaluate_full_checkpoint


def _integrity_checked_full(data, path: Path, seed: int, args, device):
    checkpoint = torch_load_compat(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"full-system checkpoint is not a mapping: {path}")
    if not args.allow_unverified_checkpoint:
        assert_checkpoint_internal_contract(checkpoint)
    return _ORIGINAL_EVALUATE_FULL(data, path, seed, args, device)


def main(argv: Sequence[str] | None = None) -> int:
    original = base.evaluate_full_checkpoint
    try:
        base.evaluate_full_checkpoint = _integrity_checked_full
        return base.main(argv)
    finally:
        base.evaluate_full_checkpoint = original


if __name__ == "__main__":
    raise SystemExit(main())
