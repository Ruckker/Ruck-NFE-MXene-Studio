from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


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
    """Record software/hardware context without making it part of model semantics."""
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
    """Stable SHA256 for JSON-like scientific configuration payloads."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Training semantics shared across independent random seeds.

    Filesystem locations, dataloader worker plumbing, and the random seed are
    deliberately excluded. Dataset/graph identity is protected separately by
    provenance hashes. Everything that changes model capacity, optimization,
    objectives, calibration, or an explicit ablation is retained.
    """
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
    """Seed-specific protocol used to guard continuation/resume semantics."""
    return {
        "seed": config.get("seed"),
        "training_protocol": training_protocol_payload(config),
    }


def experiment_protocol_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(experiment_protocol_payload(config))


def assert_matching_experiment_protocol(
    checkpoint: Mapping[str, Any], current_config: Mapping[str, Any]
) -> None:
    """Block resume when seed or optimization/model/objective semantics changed."""
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
    state = git_repository_state(repository_root)
    return {
        **state,
        "runtime_environment": runtime_environment(),
        "dataset_table_sha256": str(cache.get("table_sha256", "unknown")),
        "structure_manifest_schema": str(cache.get("structure_manifest_schema", "unknown")),
        "structure_manifest_sha256": str(cache.get("structure_manifest_sha256", "unknown")),
        "split_manifest_sha256": split_manifest_sha256(records, splits),
        "cache_schema": str(cache.get("schema", "unknown")),
        "global_feature_schema": str(cache.get("global_feature_schema", "unknown")),
        "neighbor_policy": str(cache.get("neighbor_policy", "unknown")),
        "graph_radius_A": float(cache.get("radius", float("nan"))),
        "max_neighbors": int(cache.get("max_neighbors", -1)),
        "records": int(len(records)),
        "skipped_cache_records": int(len(cache.get("skipped", []))),
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
    if require_code_match:
        current_commit = str(current_provenance.get("git_commit", "unknown"))
        checkpoint_commit = str(checkpoint_provenance.get("git_commit", "unknown"))
        if current_commit == "unknown" or checkpoint_commit == "unknown":
            raise ValueError("cannot resume a formal run without resolvable Git commit provenance")
        if current_commit != checkpoint_commit:
            raise ValueError(
                "resume would mix training code revisions: "
                f"checkpoint={checkpoint_commit} current={current_commit}"
            )
        if current_provenance.get("git_dirty") is not False:
            raise ValueError("formal resume is blocked from a dirty or unknown Git worktree")
        if checkpoint_provenance.get("git_dirty") is not False:
            raise ValueError("checkpoint was created from a dirty or unknown Git worktree")
