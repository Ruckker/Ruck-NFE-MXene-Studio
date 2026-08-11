from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from .pair_symmetric_graph import install_pair_symmetric_graph_contract

# Install the canonical graph contract before strict_generation/data_v2 imports
# capture any graph implementation constants.
install_pair_symmetric_graph_contract()

from . import strict_generation as _core  # noqa: E402
from .data_v2 import element_features as _element_features  # noqa: E402
from .data_v2 import torch_load_compat as _torch_load_compat  # noqa: E402
from .predict_core import infer_chunk as _infer_chunk  # noqa: E402
from .predict_guard import (  # noqa: E402
    guarded_build_periodic_graph,
    guarded_load_checkpoint_model,
)
from .provenance_v2 import assert_matching_provenance, file_sha256  # noqa: E402


GENERATOR_CHECKPOINT_FORMAT = "nfe-mxene-surface-generator-1.1"
_GENERATOR_PROVENANCE: dict[str, Any] | None = None
_GENERATOR_CHECKPOINT: dict[str, Any] | None = None
_PREDICTOR_CHECKPOINTS: list[tuple[str, dict[str, Any]]] = []
_OUTPUT_DIRECTORY: Path | None = None

# These public hooks intentionally remain assignable. manifold_generation and
# the Windows compatibility layer replace them temporarily; main() propagates
# the active hooks into the legacy implementation while retaining the formal
# graph/checkpoint guard around the whole call.
parse_args = _core.parse_args
choose_templates = _core.choose_templates
sample_structures = _core.sample_structures
create_chgnet_relaxer = _core.create_chgnet_relaxer


def set_progress_callback(callback):
    return _core.set_progress_callback(callback)


def report_progress(message: str, percent: float | None = None) -> None:
    _core.report_progress(message, percent)


def load_generator(
    path: str | Path, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    global _GENERATOR_PROVENANCE, _GENERATOR_CHECKPOINT
    checkpoint = _torch_load_compat(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"generator checkpoint is not a mapping: {path}")
    if checkpoint.get("format") != GENERATOR_CHECKPOINT_FORMAT:
        raise ValueError(
            "formal generation requires provenance-aware generator checkpoint "
            f"{GENERATOR_CHECKPOINT_FORMAT}: {path}"
        )
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"generator checkpoint lacks benchmark provenance: {path}")
    if provenance.get("git_dirty") is not False:
        raise ValueError("formal generation refuses generator checkpoints from dirty/unknown worktrees")
    if len(str(provenance.get("git_commit", ""))) != 40:
        raise ValueError("generator checkpoint lacks a resolvable training Git commit")
    if not str(checkpoint.get("generator_protocol_sha256", "")):
        raise ValueError(f"generator checkpoint lacks generator protocol fingerprint: {path}")
    if checkpoint.get("template_source_split") != "train":
        raise ValueError("formal generator checkpoint must declare template_source_split=train")

    model = _core.SurfaceAwareTemplateFlow(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _GENERATOR_PROVENANCE = provenance
    _GENERATOR_CHECKPOINT = checkpoint
    return model, checkpoint


def _load_predictor(path: str | Path, device: torch.device):
    if _GENERATOR_PROVENANCE is None:
        raise RuntimeError("generator provenance must be loaded before predictor checkpoints")
    model, checkpoint = guarded_load_checkpoint_model(
        path,
        device,
        require_runtime_git_match=not bool(getattr(sys, "_MEIPASS", None)),
    )
    assert_matching_provenance(
        _GENERATOR_PROVENANCE,
        checkpoint.get("provenance"),
        require_present=True,
        require_code_match=True,
    )
    _PREDICTOR_CHECKPOINTS.append((str(Path(path).resolve()), checkpoint))
    return model, checkpoint


def _capture_output_directory(parent: Path) -> Path:
    global _OUTPUT_DIRECTORY
    _OUTPUT_DIRECTORY = _ORIGINAL_UNIQUE_OUTPUT_DIRECTORY(parent)
    return _OUTPUT_DIRECTORY


def _bind_artifact_identity(
    *,
    generator_checkpoint_path: str | Path,
    predictor_checkpoint_paths: Sequence[str | Path],
) -> None:
    if _OUTPUT_DIRECTORY is None:
        return
    path = _OUTPUT_DIRECTORY / "run_info.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"generation run_info is not a JSON object: {path}")
    generator_path = Path(generator_checkpoint_path).resolve()
    generator = _GENERATOR_CHECKPOINT or {}
    payload["graph_contract"] = "pair-symmetric-data-v2-guarded"
    payload["generator_checkpoint"] = {
        "path": str(generator_path),
        "sha256": file_sha256(generator_path),
        "generator_protocol_sha256": str(
            generator.get("generator_protocol_sha256", "")
        ),
        "git_commit": str(generator.get("provenance", {}).get("git_commit", "")),
    }
    if len(_PREDICTOR_CHECKPOINTS) != len(predictor_checkpoint_paths):
        raise RuntimeError(
            "formal generation predictor checkpoint accounting is incomplete: "
            f"loaded={len(_PREDICTOR_CHECKPOINTS)} requested={len(predictor_checkpoint_paths)}"
        )
    payload["predictor_checkpoints"] = [
        {
            "path": resolved_path,
            "sha256": file_sha256(resolved_path),
            "training_protocol_sha256": str(
                checkpoint.get("training_protocol_sha256", "")
            ),
            "model_protocol_sha256": str(
                checkpoint.get("model_protocol_sha256", "")
            ),
            "git_commit": str(
                checkpoint.get("provenance", {}).get("git_commit", "")
            ),
        }
        for resolved_path, checkpoint in _PREDICTOR_CHECKPOINTS
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_ORIGINAL_BUILD_PERIODIC_GRAPH = _core.build_periodic_graph
_ORIGINAL_ELEMENT_FEATURES = _core.element_features
_ORIGINAL_TORCH_LOAD_COMPAT = _core.torch_load_compat
_ORIGINAL_INFER_CHUNK = _core.infer_chunk
_ORIGINAL_LOAD_CHECKPOINT_MODEL = _core.load_checkpoint_model
_ORIGINAL_LOAD_GENERATOR = _core.load_generator
_ORIGINAL_UNIQUE_OUTPUT_DIRECTORY = _core.unique_output_directory
_ORIGINAL_PARSE_ARGS = _core.parse_args
_ORIGINAL_CHOOSE_TEMPLATES = _core.choose_templates
_ORIGINAL_SAMPLE_STRUCTURES = _core.sample_structures
_ORIGINAL_CHGNET_RELAXER = _core.create_chgnet_relaxer


def main(argv: Sequence[str] | None = None) -> int:
    global _GENERATOR_PROVENANCE, _GENERATOR_CHECKPOINT, _PREDICTOR_CHECKPOINTS, _OUTPUT_DIRECTORY
    _GENERATOR_PROVENANCE = None
    _GENERATOR_CHECKPOINT = None
    _PREDICTOR_CHECKPOINTS = []
    _OUTPUT_DIRECTORY = None
    args = parse_args(argv)

    try:
        _core.parse_args = parse_args
        _core.choose_templates = choose_templates
        _core.sample_structures = sample_structures
        _core.create_chgnet_relaxer = create_chgnet_relaxer
        _core.build_periodic_graph = guarded_build_periodic_graph
        _core.element_features = _element_features
        _core.torch_load_compat = _torch_load_compat
        _core.infer_chunk = _infer_chunk
        _core.load_checkpoint_model = _load_predictor
        _core.load_generator = load_generator
        _core.unique_output_directory = _capture_output_directory
        result = _core.main(argv)
        _bind_artifact_identity(
            generator_checkpoint_path=args.generator_checkpoint,
            predictor_checkpoint_paths=args.predictor_checkpoint,
        )
        return result
    finally:
        _core.parse_args = _ORIGINAL_PARSE_ARGS
        _core.choose_templates = _ORIGINAL_CHOOSE_TEMPLATES
        _core.sample_structures = _ORIGINAL_SAMPLE_STRUCTURES
        _core.create_chgnet_relaxer = _ORIGINAL_CHGNET_RELAXER
        _core.build_periodic_graph = _ORIGINAL_BUILD_PERIODIC_GRAPH
        _core.element_features = _ORIGINAL_ELEMENT_FEATURES
        _core.torch_load_compat = _ORIGINAL_TORCH_LOAD_COMPAT
        _core.infer_chunk = _ORIGINAL_INFER_CHUNK
        _core.load_checkpoint_model = _ORIGINAL_LOAD_CHECKPOINT_MODEL
        _core.load_generator = _ORIGINAL_LOAD_GENERATOR
        _core.unique_output_directory = _ORIGINAL_UNIQUE_OUTPUT_DIRECTORY


def __getattr__(name: str):
    return getattr(_core, name)


if __name__ == "__main__":
    raise SystemExit(main())
