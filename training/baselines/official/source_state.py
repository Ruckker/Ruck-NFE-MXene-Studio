from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


PINNED_EXTERNAL_COMMITS = {
    "txie-93/cgcnn": "f42ab233c4ee0c416879d6bc2d22a264418413ad",
}


def _git(args: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot audit external Git checkout {cwd}: {exc}") from exc
    return completed.stdout


def clean_external_checkout(path: str | Path, *, name: str) -> dict[str, Any]:
    """Return portable source identity and reject dirty/unpinned checkouts.

    Remote URLs are intentionally excluded: the same pinned upstream commit
    cloned from GitHub or an institutional mirror is the same scientific source
    revision. Named reference backbones must match their preregistered commit.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{name} checkout does not exist: {root}")
    commit = _git(["rev-parse", "HEAD"], root).strip()
    if len(commit) != 40:
        raise RuntimeError(f"{name} checkout has no resolvable 40-character Git commit: {root}")
    expected = PINNED_EXTERNAL_COMMITS.get(name)
    if expected is not None and commit != expected:
        raise RuntimeError(
            f"formal {name} baseline requires pinned upstream commit {expected}; observed {commit}. "
            "Checkout the registered reference revision instead of silently changing the official backbone."
        )
    status = _git(["status", "--porcelain", "--untracked-files=all"], root)
    if status.strip():
        raise RuntimeError(
            f"formal {name} baseline refuses a dirty upstream checkout: {root}"
        )
    return {
        "name": name,
        "git_commit": commit,
        "git_dirty": False,
        "pinned_reference_commit": expected,
    }
