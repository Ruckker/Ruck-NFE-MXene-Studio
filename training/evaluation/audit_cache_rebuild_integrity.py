from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from nfe_model.data_v2 import build_cache, load_or_build_cache, split_indices
from nfe_model.formal_config import validate_formal_config
from nfe_model.provenance_v2 import cache_records_sha256
from nfe_model.utils import load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the formal graph cache from source structures and require exact tensor identity."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument(
        "--output", default="training/evaluation/results/cache_rebuild_integrity.json"
    )
    return parser.parse_args()


def _resolve_config(path: Path) -> dict[str, Any]:
    config = load_config(path)
    base = path.resolve().parent
    for key in ("table", "root", "cache"):
        value = Path(config["data"][key])
        if not value.is_absolute():
            value = base / value
        config["data"][key] = str(value.resolve())
    validate_formal_config(config)
    return config


def main() -> int:
    args = parse_args()
    config = _resolve_config(Path(args.config).resolve())
    data = config["data"]
    existing = load_or_build_cache(
        data["table"],
        data["root"],
        data["cache"],
        radius=float(data["radius"]),
        max_neighbors=int(data["max_neighbors"]),
        rebuild=False,
    )
    split_indices(existing["records"])
    existing_hash = cache_records_sha256(existing["records"])

    with tempfile.TemporaryDirectory(prefix="nfe_cache_rebuild_") as directory:
        temporary = Path(directory) / "fresh.pt"
        rebuilt = build_cache(
            data["table"],
            data["root"],
            temporary,
            radius=float(data["radius"]),
            max_neighbors=int(data["max_neighbors"]),
        )
        split_indices(rebuilt["records"])
        rebuilt_hash = cache_records_sha256(rebuilt["records"])
        metadata_keys = (
            "schema",
            "global_feature_schema",
            "neighbor_policy",
            "structure_manifest_schema",
            "structure_manifest_sha256",
            "target_schema",
            "target_schema_sha256",
            "data_implementation_schema",
            "data_implementation_sha256",
            "table_sha256",
            "radius",
            "max_neighbors",
        )
        metadata_mismatch = {
            key: {"existing": existing.get(key), "rebuilt": rebuilt.get(key)}
            for key in metadata_keys
            if existing.get(key) != rebuilt.get(key)
        }
        skipped_existing = existing.get("skipped", [])
        skipped_rebuilt = rebuilt.get("skipped", [])

    result = {
        "existing_cache": str(Path(data["cache"]).resolve()),
        "existing_records_sha256": existing_hash,
        "fresh_rebuild_records_sha256": rebuilt_hash,
        "records_match": existing_hash == rebuilt_hash,
        "metadata_mismatch": metadata_mismatch,
        "existing_skipped": skipped_existing,
        "fresh_rebuild_skipped": skipped_rebuilt,
        "pass": (
            existing_hash == rebuilt_hash
            and not metadata_mismatch
            and skipped_existing == skipped_rebuilt
        ),
        "interpretation": (
            "pass means the persisted cache is an exact deterministic reconstruction of current source structures/code "
            "in this canonical environment; official isolated environments should consume this audited immutable cache"
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if not result["pass"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
