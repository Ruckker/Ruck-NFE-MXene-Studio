from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from . import predict_core as _predict
from .checkpoint_contract import assert_checkpoint_internal_contract
from .data_contract import DATA_IMPLEMENTATION_SCHEMA, data_implementation_sha256
from .data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    STRUCTURE_MANIFEST_SCHEMA,
    TARGET_SCHEMA,
    build_periodic_graph,
    target_schema_sha256,
    torch_load_compat,
)
from .formal_data import assert_graph_vacuum_adequacy
from .model import PeriodicNFEModel
from .provenance_v2 import (
    NORMALIZER_SCHEMA,
    git_repository_state,
    runtime_environment,
    training_protocol_sha256,
)


_ORIGINAL_LOADER = _predict.load_checkpoint_model
_ENSEMBLE_GRAPH_CONTRACT: tuple[object, ...] | None = None
_FEATURE_BUILDER_PACKAGES = ("numpy", "pymatgen")


def _checkpoint_training_protocol(checkpoint: dict) -> str:
    value = str(checkpoint.get("training_protocol_sha256", ""))
    if not value and isinstance(checkpoint.get("config"), dict):
        value = training_protocol_sha256(checkpoint["config"])
    if not value:
        raise ValueError("checkpoint is missing a training protocol fingerprint")
    return value


def _assert_feature_builder_environment(provenance: dict) -> tuple[tuple[str, str], ...]:
    """Require graph/feature-builder package versions to match training exactly."""

    training_packages = provenance.get("runtime_environment", {}).get("packages", {})
    if not isinstance(training_packages, dict):
        raise ValueError("checkpoint is missing runtime_environment.packages provenance")
    runtime_packages = runtime_environment().get("packages", {})
    if not isinstance(runtime_packages, dict):
        raise ValueError("runtime environment did not report package versions")

    identity: list[tuple[str, str]] = []
    for package in _FEATURE_BUILDER_PACKAGES:
        training_version = str(training_packages.get(package, "unknown")).strip()
        runtime_version = str(runtime_packages.get(package, "unknown")).strip()
        if not training_version or training_version.lower() == "unknown":
            raise ValueError(
                f"checkpoint requires a resolvable {package} version for audited feature building"
            )
        if not runtime_version or runtime_version.lower() == "unknown":
            raise ValueError(
                f"production inference requires a resolvable {package} version"
            )
        if runtime_version != training_version:
            raise ValueError(
                "feature-builder environment differs from training: "
                f"{package} training={training_version} runtime={runtime_version}"
            )
        identity.append((package, training_version))
    return tuple(identity)


def _load_supported_model(checkpoint: dict, path: str | Path, device: torch.device):
    fmt = checkpoint.get("format")
    if fmt == "nfe-mxene-predictor-1.0":
        return _ORIGINAL_LOADER(path, device)
    if fmt == "nfe-mxene-predictor-ablation-1.0":
        ablation = checkpoint.get("ablation_config", {})
        if ablation.get("name") != "full":
            raise ValueError(
                "production inference accepts only the full ablation checkpoint; "
                f"got ablation={ablation.get('name', 'missing')}"
            )
        model_config = checkpoint.get("base_model_config", checkpoint.get("model_config"))
        if not isinstance(model_config, dict):
            raise ValueError(f"full ablation checkpoint has no base model config: {path}")
        model = PeriodicNFEModel(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, checkpoint
    raise ValueError(f"unsupported checkpoint format for production inference: {fmt}")


def guarded_load_checkpoint_model(path: str | Path, device: torch.device):
    global _ENSEMBLE_GRAPH_CONTRACT
    checkpoint = torch_load_compat(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    assert_checkpoint_internal_contract(checkpoint)
    provenance = checkpoint.get("provenance", {})
    if provenance.get("cache_schema") != CACHE_SCHEMA:
        raise ValueError(
            "legacy/incompatible predictor checkpoint: cache_schema="
            f"{provenance.get('cache_schema', 'missing')}; expected {CACHE_SCHEMA}."
        )
    if provenance.get("global_feature_schema") != GLOBAL_FEATURE_SCHEMA:
        raise ValueError(
            "legacy/incompatible predictor checkpoint: global_feature_schema="
            f"{provenance.get('global_feature_schema', 'missing')}; expected {GLOBAL_FEATURE_SCHEMA}."
        )
    if provenance.get("neighbor_policy") != NEIGHBOR_POLICY:
        raise ValueError(
            "legacy/incompatible predictor checkpoint: neighbor_policy="
            f"{provenance.get('neighbor_policy', 'missing')}; expected {NEIGHBOR_POLICY}."
        )
    if provenance.get("structure_manifest_schema") != STRUCTURE_MANIFEST_SCHEMA:
        raise ValueError(
            "checkpoint is missing the current structure-file manifest contract: "
            f"{provenance.get('structure_manifest_schema', 'missing')} != {STRUCTURE_MANIFEST_SCHEMA}"
        )
    if provenance.get("target_schema") != TARGET_SCHEMA:
        raise ValueError(
            "checkpoint target schema is incompatible with production inference: "
            f"{provenance.get('target_schema', 'missing')} != {TARGET_SCHEMA}"
        )
    current_target_hash = target_schema_sha256()
    if provenance.get("target_schema_sha256") != current_target_hash:
        raise ValueError(
            "checkpoint regression target ordering/transform contract differs from current code: "
            f"{provenance.get('target_schema_sha256', 'missing')} != {current_target_hash}"
        )
    if provenance.get("data_implementation_schema") != DATA_IMPLEMENTATION_SCHEMA:
        raise ValueError(
            "checkpoint data implementation schema is incompatible with current inference: "
            f"{provenance.get('data_implementation_schema', 'missing')} != {DATA_IMPLEMENTATION_SCHEMA}"
        )
    current_data_hash = data_implementation_sha256()
    if provenance.get("data_implementation_sha256") != current_data_hash:
        raise ValueError(
            "checkpoint graph/feature implementation differs from current runtime: "
            f"{provenance.get('data_implementation_sha256', 'missing')} != {current_data_hash}"
        )
    if provenance.get("normalizer_schema") != NORMALIZER_SCHEMA:
        raise ValueError(
            "checkpoint is missing the current train-normalizer contract: "
            f"{provenance.get('normalizer_schema', 'missing')} != {NORMALIZER_SCHEMA}"
        )
    feature_builder_environment = _assert_feature_builder_environment(provenance)

    config = checkpoint.get("config", {}).get("data", {})
    radius = float(config.get("radius", provenance.get("graph_radius_A", -1.0)))
    max_neighbors = int(config.get("max_neighbors", provenance.get("max_neighbors", -1)))
    if radius <= 0 or max_neighbors <= 0:
        raise ValueError("checkpoint is missing a valid graph radius/max_neighbors contract")
    if abs(radius - float(provenance.get("graph_radius_A", radius))) > 1e-12:
        raise ValueError("checkpoint config radius disagrees with checkpoint provenance")
    if max_neighbors != int(provenance.get("max_neighbors", max_neighbors)):
        raise ValueError("checkpoint config max_neighbors disagrees with checkpoint provenance")

    dataset_hash = str(provenance.get("dataset_table_sha256", ""))
    structure_hash = str(provenance.get("structure_manifest_sha256", ""))
    target_hash = str(provenance.get("target_schema_sha256", ""))
    implementation_hash = str(provenance.get("data_implementation_sha256", ""))
    cache_records_hash = str(provenance.get("cache_records_sha256", ""))
    normalizer_hash = str(provenance.get("normalizer_sha256", ""))
    split_hash = str(provenance.get("split_manifest_sha256", ""))
    git_commit = str(provenance.get("git_commit", ""))
    git_dirty = provenance.get("git_dirty")
    protocol_hash = _checkpoint_training_protocol(checkpoint)
    seen_elements = tuple(sorted(int(value) for value in checkpoint.get("seen_elements", [])))
    if not all(
        (
            dataset_hash,
            structure_hash,
            target_hash,
            implementation_hash,
            cache_records_hash,
            normalizer_hash,
            split_hash,
        )
    ):
        raise ValueError(
            "checkpoint is missing dataset/structure/target/implementation/cache/normalizer/split provenance"
        )
    if len(git_commit) != 40 or git_commit == "unknown":
        raise ValueError("checkpoint is missing a resolvable training Git commit")
    if git_dirty is not False:
        raise ValueError(
            "formal ensemble inference refuses checkpoints trained from dirty/unknown worktrees"
        )

    runtime = git_repository_state()
    runtime_commit = str(runtime.get("git_commit", "unknown"))
    if runtime.get("git_dirty") is not False:
        raise ValueError("formal production inference requires a clean runtime Git worktree")
    if runtime_commit != git_commit:
        raise ValueError(
            "formal production inference requires runtime code equal to training code: "
            f"runtime={runtime_commit} checkpoint={git_commit}"
        )
    if not seen_elements:
        raise ValueError("checkpoint is missing seen_elements required for audited OOD inference")

    contract = (
        dataset_hash,
        structure_hash,
        target_hash,
        implementation_hash,
        cache_records_hash,
        normalizer_hash,
        split_hash,
        git_commit,
        protocol_hash,
        seen_elements,
        feature_builder_environment,
        CACHE_SCHEMA,
        GLOBAL_FEATURE_SCHEMA,
        NEIGHBOR_POLICY,
        radius,
        max_neighbors,
    )
    if _ENSEMBLE_GRAPH_CONTRACT is None:
        _ENSEMBLE_GRAPH_CONTRACT = contract
    elif contract != _ENSEMBLE_GRAPH_CONTRACT:
        raise ValueError(
            "ensemble checkpoints use incompatible data/target/code/training/normalizer/graph contracts: "
            f"{contract} != {_ENSEMBLE_GRAPH_CONTRACT}"
        )
    return _load_supported_model(checkpoint, path, device)


def guarded_build_periodic_graph(structure, radius: float, max_neighbors: int, identifier: str = ""):
    graph = build_periodic_graph(structure, radius, max_neighbors, identifier)
    assert_graph_vacuum_adequacy(graph, radius, record_id=identifier or "prediction")
    return graph


def main(argv: Sequence[str] | None = None) -> int:
    global _ENSEMBLE_GRAPH_CONTRACT
    _ENSEMBLE_GRAPH_CONTRACT = None
    original_loader = _predict.load_checkpoint_model
    original_graph = _predict.build_periodic_graph
    try:
        _predict.load_checkpoint_model = guarded_load_checkpoint_model
        _predict.build_periodic_graph = guarded_build_periodic_graph
        return _predict.main(argv)
    finally:
        _predict.load_checkpoint_model = original_loader
        _predict.build_periodic_graph = original_graph
