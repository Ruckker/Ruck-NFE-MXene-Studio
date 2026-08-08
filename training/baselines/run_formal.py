from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from nfe_model.checkpoint_contract import assert_checkpoint_internal_contract
from nfe_model.data_v2 import torch_load_compat
from nfe_model.provenance_v2 import file_sha256

from . import run as base


_ORIGINAL_EVALUATE_FULL = base.evaluate_full_checkpoint
_ORIGINAL_RUN_XGBOOST = base.run_xgboost
_ORIGINAL_SAVE_JSON = base.save_json
_XGBOOST_ARTIFACTS: dict[int, dict[str, bytes | None]] = {}


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


def _run_xgboost_with_artifacts(data, seed: int) -> dict[str, Any]:
    payload = _ORIGINAL_RUN_XGBOOST(data, seed)
    artifacts = payload.pop("model_artifacts", None)
    if not isinstance(artifacts, dict):
        raise RuntimeError("formal XGBoost run did not return persisted booster bytes")
    classifier = artifacts.get("classifier_ubj")
    regressor = artifacts.get("regressor_ubj")
    if not isinstance(classifier, bytes) or not classifier:
        raise RuntimeError("formal XGBoost classifier UBJ bytes are missing")
    fitted = bool(payload.get("details", {}).get("score_regressor_fitted"))
    if fitted and (not isinstance(regressor, bytes) or not regressor):
        raise RuntimeError("fitted formal XGBoost regressor UBJ bytes are missing")
    if not fitted and regressor is not None:
        raise RuntimeError("unfitted formal XGBoost regressor unexpectedly produced artifact bytes")
    _XGBOOST_ARTIFACTS[int(seed)] = {
        "classifier_ubj": classifier,
        "regressor_ubj": regressor,
    }
    return payload


def _write_xgboost_artifacts(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    seed = int(result["seed"])
    artifacts = _XGBOOST_ARTIFACTS.pop(seed, None)
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"missing in-memory XGBoost artifacts for formal seed {seed}")

    classifier_bytes = artifacts["classifier_ubj"]
    regressor_bytes = artifacts["regressor_ubj"]
    assert isinstance(classifier_bytes, bytes)
    classifier_path = path.parent / "xgboost_classifier.ubj"
    regressor_path = path.parent / "xgboost_regressor.ubj"
    classifier_path.unlink(missing_ok=True)
    regressor_path.unlink(missing_ok=True)
    classifier_path.write_bytes(classifier_bytes)
    if isinstance(regressor_bytes, bytes):
        regressor_path.write_bytes(regressor_bytes)

    details = dict(result.get("details", {}))
    fitted = bool(details.get("score_regressor_fitted"))
    state_digest = hashlib.sha256()
    state_digest.update(b"xgboost-classifier\0")
    state_digest.update(classifier_path.read_bytes())
    state_digest.update(b"\0xgboost-regressor\0")
    state_digest.update(
        regressor_path.read_bytes() if fitted else b"UNFITTED"
    )
    recorded_state = str(details.get("model_state_sha256", ""))
    if state_digest.hexdigest() != recorded_state:
        raise RuntimeError(
            "persisted XGBoost artifact bytes do not reproduce model_state_sha256: "
            f"reconstructed={state_digest.hexdigest()} recorded={recorded_state}"
        )

    booster_artifacts: dict[str, Any] = {
        "format": "ubj",
        "classifier": {
            "file": classifier_path.name,
            "sha256": file_sha256(classifier_path),
        },
        "regressor": None,
    }
    if fitted:
        booster_artifacts["regressor"] = {
            "file": regressor_path.name,
            "sha256": file_sha256(regressor_path),
        }
    details["booster_artifacts"] = booster_artifacts
    updated = dict(result)
    updated["details"] = details
    return updated


def _save_json_with_artifacts(path, payload) -> None:
    result_path = Path(path)
    value = payload
    if (
        result_path.name == "result.json"
        and isinstance(payload, dict)
        and payload.get("track") == "architecture"
        and payload.get("model") == "xgboost"
    ):
        value = _write_xgboost_artifacts(result_path, payload)
    _ORIGINAL_SAVE_JSON(result_path, value)


def main(argv: Sequence[str] | None = None) -> int:
    original_eval = base.evaluate_full_checkpoint
    original_xgboost = base.run_xgboost
    original_save_json = base.save_json
    _XGBOOST_ARTIFACTS.clear()
    try:
        base.evaluate_full_checkpoint = _integrity_checked_full
        base.run_xgboost = _run_xgboost_with_artifacts
        base.save_json = _save_json_with_artifacts
        return base.main(argv)
    finally:
        base.evaluate_full_checkpoint = original_eval
        base.run_xgboost = original_xgboost
        base.save_json = original_save_json
        _XGBOOST_ARTIFACTS.clear()


if __name__ == "__main__":
    raise SystemExit(main())
