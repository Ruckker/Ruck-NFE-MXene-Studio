from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


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


def git_commit_sha(start: str | Path | None = None) -> str:
    return str(git_repository_state(start)["git_commit"])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest_sha256(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
) -> str:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for record_index in sorted(int(i) for i in splits.get(split, ())):
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
