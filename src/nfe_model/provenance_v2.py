from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .formal_data import (
    assert_formal_primary_target_coverage,
    assert_formal_slab_vacuum,
)

NORMALIZER_SCHEMA = "robust-train-normalizers-v1"


def _run_git(args: list[str], cwd: Path, timeout: int = 8) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def git_repository_state(start: str | Path | None = None) -> dict[str, Any]:
    cwd = Path(start).resolve() if start is not None else Path(__file__).resolve().parents[2]
    commit_text = _run_git(["rev-parse", "HEAD"], cwd)
    commit = commit_text.strip() if commit_text else "unknown"
    if len(commit) != 40:
        commit = "unknown"
    status = _run_git(["status", "--porcelain", "--untracked-files=all"], cwd)
    if status is None:
        return {
            "git_commit": commit,
            "git_dirty": None,
            "git_state_sha256": "unknown",
        }
    dirty = bool(status.strip())
    diff = _run_git(["diff", "--binary", "HEAD", "--"], cwd) or ""
    state_payload = (status + "\n" + diff).encode("utf-8", errors="replace")
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_state_sha256": hashlib.sha256(state_payload).hexdigest(),
    }


def runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "pymatgen", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unknown"
    cuda_available = bool(torch.cuda.is_available())
    gpu_names: list[str] = []
    if cuda_available:
        try:
            gpu_names = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
        except (RuntimeError, AssertionError):
            gpu_names = ["unavailable"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_names": gpu_names,
        "packages": packages,
    }


def git_commit_sha(start: str | Path | None = None) -> str:
    return str(git_repository_state(start)["git_commit"])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_tensor_digest(digest: Any, key: str, value: Any) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"formal tensor field {key!r} is not a tensor")
    tensor = value.detach().cpu().contiguous()
    digest.update(key.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


def tensor_mapping_sha256(mapping: Mapping[str, Any], *, schema: str) -> str:
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8") + b"\0")
    for key in sorted(mapping):
        _update_tensor_digest(digest, str(key), mapping[key])
    return digest.hexdigest()


def cache_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    tensor_keys = (
        "z",
        "atom_features",
        "frac_pos",
        "lattice",
        "edge_index",
        "edge_shift",
        "global_features",
        "targets",
        "target_mask",
    )
    digest = hashlib.sha256()
    digest.update(b"nfe-cache-record-tensors-v1\0")
    for record_index, record in enumerate(records):
        metadata = {
            "record_index": int(record_index),
            "id": str(record.get("id", "")),
            "source_file_sha256": str(record.get("source_file_sha256", "")),
            "split": str(record.get("split", "")),
            "split_group": str(record.get("split_group", "")),
            "label": int(record.get("label", -1)),
            "sample_weight": float(record.get("sample_weight", 1.0)),
            "elements": [int(value) for value in record.get("elements", ())],
        }
        digest.update(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for key in tensor_keys:
            if key not in record:
                raise RuntimeError(
                    f"formal cache record {metadata['id']!r} is missing tensor field {key!r}"
                )
            _update_tensor_digest(digest, key, record[key])
    return digest.hexdigest()


def training_protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}) or {})
    training.pop("checkpoint_dir", None)
    data = dict(config.get("data", {}) or {})
    data_semantics = {
        key: data.get(key)
        for key in ("radius", "max_neighbors", "max_cache_skip_fraction")
        if key in data
    }
    return {
        "data_semantics": data_semantics,
        "model": dict(config.get("model", {}) or {}),
        "training": training,
        "loss": dict(config.get("loss", {}) or {}),
        "inference": dict(config.get("inference", {}) or {}),
        "ablation": dict(config.get("ablation", {}) or {}),
    }


def training_protocol_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(training_protocol_payload(config))


def experiment_protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": config.get("seed"),
        "training_protocol": training_protocol_payload(config),
    }


def experiment_protocol_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(experiment_protocol_payload(config))


def assert_matching_experiment_protocol(
    checkpoint: Mapping[str, Any], current_config: Mapping[str, Any]
) -> None:
    expected = experiment_protocol_sha256(current_config)
    observed = str(checkpoint.get("experiment_protocol_sha256", ""))
    if not observed:
        checkpoint_config = checkpoint.get("config")
        if isinstance(checkpoint_config, Mapping):
            observed = experiment_protocol_sha256(checkpoint_config)
    if not observed or observed != expected:
        raise ValueError(
            "resume experiment protocol mismatch: "
            f"checkpoint={observed or 'missing'} current={expected}. "
            "Start a new run instead of resuming across changed seed/hyperparameters/objectives."
        )


def split_manifest_sha256(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
) -> str:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for record_index in sorted(int(index) for index in splits.get(split, ())):
            record = records[record_index]
            rows.append(
                {
                    "record_index": record_index,
                    "id": str(record.get("id", "")),
                    "split": split,
                    "split_group": str(record.get("split_group", "")),
                }
            )
    encoded = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_provenance(
    *,
    cache: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    primary_coverage = assert_formal_primary_target_coverage(records, splits)
    radius = float(cache.get("radius", float("nan")))
    minimum_normal_vacuum = assert_formal_slab_vacuum(records, radius)
    # Local import avoids coupling the cache builder to provenance while making
    # the actual train-fitted normalization tensors part of the formal identity.
    from .data_v2 import robust_normalizers

    normalizers = robust_normalizers(list(records), list(splits["train"]))
    normalizer_hash = tensor_mapping_sha256(normalizers, schema=NORMALIZER_SCHEMA)
    state = git_repository_state(repository_root)
    return {
        **state,
        "runtime_environment": runtime_environment(),
        "dataset_table_sha256": str(cache.get("table_sha256", "unknown")),
        "structure_manifest_schema": str(cache.get("structure_manifest_schema", "unknown")),
        "structure_manifest_sha256": str(cache.get("structure_manifest_sha256", "unknown")),
        "target_schema": str(cache.get("target_schema", "unknown")),
        "target_schema_sha256": str(cache.get("target_schema_sha256", "unknown")),
        "data_implementation_schema": str(cache.get("data_implementation_schema", "unknown")),
        "data_implementation_sha256": str(cache.get("data_implementation_sha256", "unknown")),
        "cache_records_sha256": cache_records_sha256(records),
        "normalizer_schema": NORMALIZER_SCHEMA,
        "normalizer_sha256": normalizer_hash,
        "split_manifest_sha256": split_manifest_sha256(records, splits),
        "cache_schema": str(cache.get("schema", "unknown")),
        "global_feature_schema": str(cache.get("global_feature_schema", "unknown")),
        "neighbor_policy": str(cache.get("neighbor_policy", "unknown")),
        "graph_radius_A": radius,
        "max_neighbors": int(cache.get("max_neighbors", -1)),
        "minimum_normal_vacuum_A": minimum_normal_vacuum,
        "records": int(len(records)),
        "skipped_cache_records": int(len(cache.get("skipped", []))),
        "primary_target_coverage": primary_coverage,
    }


def assert_matching_provenance(
    checkpoint_provenance: Mapping[str, Any] | None,
    current_provenance: Mapping[str, Any],
    *,
    require_present: bool = True,
    require_code_match: bool = False,
) -> None:
    if not checkpoint_provenance:
        if require_present:
            raise ValueError(
                "checkpoint has no provenance metadata; rerun/re-export it with the audited "
                "trainer or explicitly opt into unverified legacy evaluation"
            )
        return
    required = [
        "dataset_table_sha256",
        "structure_manifest_schema",
        "structure_manifest_sha256",
        "target_schema",
        "target_schema_sha256",
        "data_implementation_schema",
        "data_implementation_sha256",
        "cache_records_sha256",
        "normalizer_schema",
        "normalizer_sha256",
        "split_manifest_sha256",
        "cache_schema",
        "global_feature_schema",
        "neighbor_policy",
        "graph_radius_A",
        "max_neighbors",
    ]
    for key in required:
        expected = str(current_provenance.get(key, ""))
        observed = str(checkpoint_provenance.get(key, ""))
        if not observed or observed != expected:
            raise ValueError(
                f"checkpoint provenance mismatch for {key}: "
                f"checkpoint={observed or 'missing'} current={expected or 'missing'}"
            )

    enforce_code_match = bool(require_code_match or require_present)
    if enforce_code_match:
        current_commit = str(current_provenance.get("git_commit", "unknown"))
        checkpoint_commit = str(checkpoint_provenance.get("git_commit", "unknown"))
        if current_commit == "unknown" or checkpoint_commit == "unknown":
            raise ValueError(
                "formal execution requires resolvable Git commit provenance for runtime and checkpoint"
            )
        if current_commit != checkpoint_commit:
            raise ValueError(
                "formal execution would mix code revisions: "
                f"checkpoint={checkpoint_commit} current={current_commit}"
            )
        if current_provenance.get("git_dirty") is not False:
            raise ValueError("formal execution is blocked from a dirty or unknown Git worktree")
        if checkpoint_provenance.get("git_dirty") is not False:
            raise ValueError("checkpoint was created from a dirty or unknown Git worktree")
