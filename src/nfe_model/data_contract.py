from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DATA_IMPLEMENTATION_SCHEMA = "data-source-code-v2"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_implementation_payload() -> dict[str, Any]:
    """Fingerprint source semantics that create cache tensors.

    Runtime package versions are recorded separately in provenance. They are
    deliberately *not* part of this cache identity: official upstream models run
    in isolated environments and should be able to consume one immutable tensor
    cache without rebuilding it merely because the reader has a different NumPy
    or pymatgen version. If a cache must actually be rebuilt, the resulting
    structure-byte manifest, tensor semantics and source-code hash remain fully
    auditable, while runtime_environment records the builder environment.
    """

    package_root = Path(__file__).resolve().parent
    files = {}
    for name in ("data.py", "data_v2.py", "data_v2_core.py", "data_contract.py"):
        path = package_root / name
        files[name] = _file_sha256(path) if path.is_file() else "missing"
    return {
        "schema": DATA_IMPLEMENTATION_SCHEMA,
        "source_sha256": files,
    }


def data_implementation_sha256() -> str:
    encoded = json.dumps(
        data_implementation_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
