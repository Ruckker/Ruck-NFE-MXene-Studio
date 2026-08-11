from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from nfe_model.provenance_v2 import file_sha256

from . import paper_preflight_strict_core as _core


_ORIGINAL_VALIDATE_CHECKPOINT = _core._validate_checkpoint

# Preserve the full pre-existing strict preflight API. The wrapper below adds
# only the fitted XGBoost artifact byte contract and delegates everything else
# to the already-audited whole-campaign preflight implementation.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def _artifact_path(result_path: Path, metadata: Mapping[str, Any], label: str) -> Path:
    filename = str(metadata.get("file", "")).strip()
    expected_hash = str(metadata.get("sha256", "")).strip()
    if not filename or len(expected_hash) != 64:
        raise RuntimeError(
            f"paper XGBoost {label} artifact metadata is incomplete: {result_path}"
        )
    root = result_path.parent.resolve()
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"paper XGBoost {label} artifact escapes result directory: {filename}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"paper XGBoost {label} artifact is missing: {path}"
        )
    observed_hash = file_sha256(path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"paper XGBoost {label} artifact SHA256 mismatch for {result_path}: "
            f"result={expected_hash} file={observed_hash}"
        )
    return path


def _validate_xgboost_artifacts(
    payload: Mapping[str, Any], result_path: Path
) -> None:
    details = payload.get("details")
    if not isinstance(details, Mapping):
        raise RuntimeError(f"paper XGBoost result has no details mapping: {result_path}")
    artifacts = details.get("booster_artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("format") != "ubj":
        raise RuntimeError(
            f"paper XGBoost result lacks persisted UBJ booster artifact contract: {result_path}"
        )
    classifier = artifacts.get("classifier")
    if not isinstance(classifier, Mapping):
        raise RuntimeError(f"paper XGBoost classifier artifact is missing: {result_path}")
    classifier_path = _artifact_path(result_path, classifier, "classifier")

    fitted = bool(details.get("score_regressor_fitted"))
    regressor = artifacts.get("regressor")
    regressor_path: Path | None = None
    if fitted:
        if not isinstance(regressor, Mapping):
            raise RuntimeError(
                f"fitted paper XGBoost regressor artifact is missing: {result_path}"
            )
        regressor_path = _artifact_path(result_path, regressor, "regressor")
    elif regressor not in (None, {}):
        raise RuntimeError(
            f"unfitted paper XGBoost result unexpectedly declares a regressor artifact: {result_path}"
        )

    digest = hashlib.sha256()
    digest.update(b"xgboost-classifier\0")
    digest.update(classifier_path.read_bytes())
    digest.update(b"\0xgboost-regressor\0")
    digest.update(
        regressor_path.read_bytes() if regressor_path is not None else b"UNFITTED"
    )
    observed_state = digest.hexdigest()
    expected_state = str(details.get("model_state_sha256", ""))
    if observed_state != expected_state:
        raise RuntimeError(
            "paper XGBoost persisted booster bytes do not reproduce fitted model identity: "
            f"result={expected_state} reconstructed={observed_state} ({result_path})"
        )


def _validate_checkpoint(*args, **kwargs) -> None:
    _ORIGINAL_VALIDATE_CHECKPOINT(*args, **kwargs)
    payload = args[0] if args else kwargs["payload"]
    result_path = args[1] if len(args) > 1 else kwargs["result_path"]
    model = kwargs.get("model")
    if model == "xgboost":
        _validate_xgboost_artifacts(payload, Path(result_path))


def main() -> int:
    original = _core._validate_checkpoint
    try:
        _core._validate_checkpoint = _validate_checkpoint
        return _core.main()
    finally:
        _core._validate_checkpoint = original


if __name__ == "__main__":
    raise SystemExit(main())
