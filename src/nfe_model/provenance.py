from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


def git_commit_sha(start: str | Path | None = None) -> str:
    """Return the current Git commit when a checkout is available."""
    cwd = Path(start).resolve() if start is not None else Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = completed.stdout.strip()
    return value if len(value) == 40 else "unknown"


def split_manifest_sha256(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
) -> str:
    """Hash the exact record-to-split assignment used by a run."""
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
                    "file_path": str(record.get("file_path", "")),
                }
            )
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_provenance(
    *,
    cache: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "git_commit": git_commit_sha(repository_root),
        "dataset_table_sha256": str(cache.get("table_sha256", "unknown")),
        "split_manifest_sha256": split_manifest_sha256(records, splits),
        "cache_schema": str(cache.get("schema", "unknown")),
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
) -> None:
    """Reject checkpoint/data mismatches before comparative evaluation."""
    if not checkpoint_provenance:
        if require_present:
            raise ValueError(
                "checkpoint has no provenance metadata; rerun/re-export it with the audited trainer "
                "or explicitly opt into unverified legacy evaluation"
            )
        return
    for key in ("dataset_table_sha256", "split_manifest_sha256"):
        expected = str(current_provenance.get(key, ""))
        observed = str(checkpoint_provenance.get(key, ""))
        if not observed or observed != expected:
            raise ValueError(
                f"checkpoint provenance mismatch for {key}: checkpoint={observed or 'missing'} "
                f"current={expected or 'missing'}"
            )
