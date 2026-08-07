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
        runtime_hash = str(checkpoint.get("training_runtime_environment_sha256", ""))
        model_protocol = str(checkpoint.get("model_protocol_sha256", ""))
        if len(runtime_hash) != 64:
            raise RuntimeError(
                f"full-system checkpoint lacks a 64-character training runtime identity: {path}"
            )
        if len(model_protocol) != 64:
            raise RuntimeError(
                f"full-system checkpoint lacks a 64-character model protocol identity: {path}"
            )
    else:
        runtime_hash = str(checkpoint.get("training_runtime_environment_sha256", ""))
        model_protocol = str(checkpoint.get("model_protocol_sha256", ""))

    payload = _ORIGINAL_EVALUATE_FULL(data, path, seed, args, device)
    if model_protocol:
        payload["model_protocol_sha256"] = model_protocol
    details = dict(payload.get("details", {}))
    if runtime_hash:
        details["training_runtime_environment_sha256"] = runtime_hash
    if model_protocol:
        details["checkpoint_model_protocol_sha256"] = model_protocol
    payload["details"] = details
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    original = base.evaluate_full_checkpoint
    try:
        base.evaluate_full_checkpoint = _integrity_checked_full
        return base.main(argv)
    finally:
        base.evaluate_full_checkpoint = original


if __name__ == "__main__":
    raise SystemExit(main())
