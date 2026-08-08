from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nfe_model.data_v2 import load_or_build_cache, split_indices
from nfe_model.formal_config import validate_formal_config
from nfe_model.provenance_v2 import file_sha256
from nfe_model.utils import load_config, save_json


INPUT_TENSOR_KEYS = (
    "z",
    "atom_features",
    "frac_pos",
    "lattice",
    "edge_index",
    "edge_shift",
    "global_features",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact and representation-level duplicate leakage across fixed NFE splits."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--output", default="training/evaluation/results/split_duplicate_audit.json")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--near-duplicate-dispositions",
        default="training/evaluation/near_duplicate_dispositions.json",
        help="manual closure keyed by near-duplicate fingerprint",
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


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    header = (
        str(tensor.dtype)
        + ":"
        + json.dumps(list(tensor.shape), separators=(",", ":"))
        + ":"
    ).encode("ascii")
    return header + tensor.numpy().tobytes(order="C")


def exact_model_input_sha256(record: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"nfe-model-input-v1\0")
    for key in INPUT_TENSOR_KEYS:
        value = record.get(key)
        if not torch.is_tensor(value):
            raise RuntimeError(f"record {record.get('id')} is missing tensor {key}")
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def invariant_near_duplicate_signature(record: dict[str, Any]) -> str:
    """Coarse order/basis-insensitive signature for manual near-duplicate review.

    It is intentionally *not* a hard equality proof. The signature combines
    composition, intrinsic globals and a normalized histogram of typed edge
    distances. Collisions require explicit reviewed disposition closure.
    """

    z = record["z"].detach().cpu().numpy().astype(int)
    composition = sorted((int(value), int(np.sum(z == value))) for value in np.unique(z))
    globals_ = np.round(record["global_features"].detach().cpu().numpy().astype(float), 5).tolist()
    source, destination = record["edge_index"].detach().cpu().numpy()
    frac = record["frac_pos"].detach().cpu().double()
    lattice = record["lattice"].detach().cpu().double()
    shift = record["edge_shift"].detach().cpu().double()
    delta = frac[source] + shift - frac[destination]
    cart = torch.einsum("ei,ij->ej", delta, lattice)
    distance = torch.linalg.vector_norm(cart, dim=1).numpy()
    typed = defaultdict(int)
    for s, d, r in zip(source, destination, distance):
        pair = tuple(sorted((int(z[s]), int(z[d]))))
        typed[(pair[0], pair[1], round(float(r), 4))] += 1
    n_atoms = max(len(z), 1)
    edge_profile = [
        (key[0], key[1], key[2], round(count / n_atoms, 6))
        for key, count in sorted(typed.items())
    ]
    payload = {
        "reduced_composition_ratio": [
            (atomic_number, round(count / n_atoms, 8)) for atomic_number, count in composition
        ],
        "intrinsic_globals": globals_,
        "typed_edge_profile_per_atom": edge_profile,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cross_split_collisions(records, key_fn) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    collisions = []
    for fingerprint, items in groups.items():
        splits = {str(item["split"]) for item in items}
        if len(splits) > 1:
            collisions.append(
                {
                    "fingerprint": fingerprint,
                    "splits": sorted(splits),
                    "structures": [str(item["id"]) for item in items],
                }
            )
    return collisions


def _near_duplicate_review(
    candidates: list[dict[str, Any]], path: Path
) -> tuple[bool, dict[str, Any] | None]:
    if not candidates:
        return True, None
    if not path.is_file():
        return False, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("near-duplicate disposition file must be a JSON object")
    rows = payload.get("dispositions")
    if not isinstance(rows, dict):
        raise ValueError(
            "near-duplicate disposition file requires a dispositions mapping"
        )
    allowed = {"distinct_after_review", "acceptable_related_structure"}
    missing: list[str] = []
    invalid: list[str] = []
    for candidate in candidates:
        fingerprint = str(candidate["fingerprint"])
        disposition = rows.get(fingerprint)
        if not isinstance(disposition, dict):
            missing.append(fingerprint)
            continue
        decision = str(disposition.get("decision", "")).strip()
        reviewer = str(disposition.get("reviewer", "")).strip()
        rationale = str(disposition.get("rationale", "")).strip()
        if decision not in allowed or not reviewer or not rationale:
            invalid.append(fingerprint)
    if missing or invalid:
        raise RuntimeError(
            "near-duplicate review is not closed: "
            f"missing={missing[:8]} invalid={invalid[:8]}"
        )
    return True, {
        "path": str(path),
        "sha256": file_sha256(path),
        "allowed_terminal_decisions": sorted(allowed),
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = _resolve_config(config_path)
    data = config["data"]
    cache = load_or_build_cache(
        data["table"],
        data["root"],
        data["cache"],
        radius=float(data["radius"]),
        max_neighbors=int(data["max_neighbors"]),
        rebuild=bool(args.rebuild_cache),
    )
    records = cache["records"]
    split_indices(records)

    source_collisions = _cross_split_collisions(
        records, lambda record: str(record.get("source_file_sha256", ""))
    )
    exact_input_collisions = _cross_split_collisions(records, exact_model_input_sha256)
    near_duplicate_candidates = _cross_split_collisions(
        records, invariant_near_duplicate_signature
    )
    review_closed, review_manifest = _near_duplicate_review(
        near_duplicate_candidates,
        Path(args.near_duplicate_dispositions).resolve(),
    )
    result = {
        "records": len(records),
        "source_byte_cross_split_collisions": source_collisions,
        "exact_model_input_cross_split_collisions": exact_input_collisions,
        "near_duplicate_candidates_for_manual_review": near_duplicate_candidates,
        "near_duplicate_review_closed": review_closed,
        "near_duplicate_disposition_manifest": review_manifest,
        "hard_pass": (
            not source_collisions
            and not exact_input_collisions
            and review_closed
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["hard_pass"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
