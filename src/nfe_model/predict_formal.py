"""Paper/formal prediction entrypoint with checkpoint self-integrity verification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from .checkpoint_contract import assert_checkpoint_internal_contract
from .data_v2 import torch_load_compat
from . import predict_guard as _guard


def _integrity_loader(path: str | Path, device: torch.device):
    checkpoint = torch_load_compat(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    assert_checkpoint_internal_contract(checkpoint)
    return _ORIGINAL_GUARDED_LOADER(path, device)


_ORIGINAL_GUARDED_LOADER = _guard.guarded_load_checkpoint_model


def main(argv: Sequence[str] | None = None) -> int:
    original = _guard.guarded_load_checkpoint_model
    try:
        _guard.guarded_load_checkpoint_model = _integrity_loader
        return _guard.main(argv)
    finally:
        _guard.guarded_load_checkpoint_model = original


if __name__ == "__main__":
    raise SystemExit(main())
