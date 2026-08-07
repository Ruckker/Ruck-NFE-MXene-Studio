from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


DATA_IMPLEMENTATION_SCHEMA = "data-code-dependencies-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def data_implementation_payload() -> dict[str, Any]:
    """Fingerprint code/dependencies that turn structures and CSV rows into cache tensors.

    Hashing the source files is intentionally conservative: even a harmless edit
    may force one cache rebuild, but a semantic graph/feature/target change can
    never silently reuse tensors created by older code.
    """

    package_root = Path(__file__).resolve().parent
    files = {}
    for name in ("data.py", "data_v2.py"):
        path = package_root / name
        files[name] = _file_sha256(path) if path.is_file() else "missing"
    return {
        "schema": DATA_IMPLEMENTATION_SCHEMA,
        "source_sha256": files,
        "dependencies": {
            "numpy": _version("numpy"),
            "pymatgen": _version("pymatgen"),
        },
    }


def data_implementation_sha256() -> str:
    encoded = json.dumps(
        data_implementation_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
