from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nfe_model.data_v2 import GLOBAL_FEATURE_DIM, REGRESSION_TARGETS, load_or_build_cache, split_indices
from nfe_model.formal_config import validate_formal_config
from nfe_model.formal_data import assert_formal_primary_target_coverage, assert_formal_slab_vacuum
from nfe_model.utils import load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit every formal cache record for finite tensors, valid periodic edges and target/weight sanity."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--output", default="training/evaluation/results/cache_tensor_sanity.json")
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


def _finite(name: str, tensor: torch.Tensor, identifier: str) -> None:
    if not torch.all(torch.isfinite(tensor)):
        raise RuntimeError(f"{identifier}: {name} contains non-finite values")


def _record(record: dict[str, Any], radius: float) -> dict[str, Any]:
    identifier = str(record.get("id", "<unknown>"))
    required_tensor = (
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
    for key in required_tensor:
        if not torch.is_tensor(record.get(key)):
            raise RuntimeError(f"{identifier}: missing tensor {key}")

    z = record["z"].detach().cpu()
    n_atoms = int(z.numel())
    if n_atoms <= 0 or torch.any(z < 1) or torch.any(z > 118):
        raise RuntimeError(f"{identifier}: atomic numbers must be physical Z=1..118")
    atom_features = record["atom_features"].detach().cpu()
    frac = record["frac_pos"].detach().cpu()
    lattice = record["lattice"].detach().cpu()
    edge_index = record["edge_index"].detach().cpu().long()
    edge_shift = record["edge_shift"].detach().cpu()
    globals_ = record["global_features"].detach().cpu()
    targets = record["targets"].detach().cpu()
    mask = record["target_mask"].detach().cpu().bool()

    if atom_features.ndim != 2 or atom_features.shape[0] != n_atoms:
        raise RuntimeError(f"{identifier}: atom_features shape disagrees with atom count")
    if frac.shape != (n_atoms, 3):
        raise RuntimeError(f"{identifier}: frac_pos must have shape [N,3]")
    if lattice.shape != (3, 3):
        raise RuntimeError(f"{identifier}: lattice must have shape [3,3]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError(f"{identifier}: edge_index must have shape [2,E]")
    n_edges = int(edge_index.shape[1])
    if n_edges <= 0 or edge_shift.shape != (n_edges, 3):
        raise RuntimeError(f"{identifier}: edge_shift/edge count mismatch")
    if globals_.numel() != GLOBAL_FEATURE_DIM:
        raise RuntimeError(
            f"{identifier}: global feature width={globals_.numel()} expected={GLOBAL_FEATURE_DIM}"
        )
    if targets.numel() != len(REGRESSION_TARGETS) or mask.numel() != len(REGRESSION_TARGETS):
        raise RuntimeError(f"{identifier}: regression target/mask width mismatch")

    for name, tensor in (
        ("atom_features", atom_features),
        ("frac_pos", frac),
        ("lattice", lattice),
        ("edge_shift", edge_shift),
        ("global_features", globals_),
    ):
        _finite(name, tensor, identifier)
    if abs(float(torch.linalg.det(lattice))) <= 1e-8:
        raise RuntimeError(f"{identifier}: lattice is singular/near-singular")
    if torch.any(edge_index < 0) or torch.any(edge_index >= n_atoms):
        raise RuntimeError(f"{identifier}: edge_index contains out-of-range atom indices")

    valid_target_values = targets[mask]
    _finite("masked-in regression targets", valid_target_values, identifier)
    if not bool(mask[0]):
        raise RuntimeError(f"{identifier}: primary NFE pseudo-score target is masked")
    label = int(record.get("label", -1))
    if label not in (0, 1, 2):
        raise RuntimeError(f"{identifier}: invalid class label {label}")
    weight = float(record.get("sample_weight", float("nan")))
    if not np.isfinite(weight) or not (0.25 <= weight <= 1.0):
        raise RuntimeError(f"{identifier}: sample_weight={weight} outside audited [0.25,1.0]")

    source, destination = edge_index
    delta_frac = frac[source] + edge_shift - frac[destination]
    delta_cart = torch.einsum("ei,ij->ej", delta_frac.double(), lattice.double())
    distance = torch.linalg.vector_norm(delta_cart, dim=1)
    if torch.any(~torch.isfinite(distance)) or torch.any(distance <= 1e-7):
        raise RuntimeError(f"{identifier}: periodic edge distances contain invalid/zero values")
    if torch.any(distance > float(radius) + 2e-5):
        maximum = float(distance.max())
        raise RuntimeError(
            f"{identifier}: cached edge distance {maximum:.8f} A exceeds graph radius {radius:.8f} A"
        )
    degree = torch.bincount(destination, minlength=n_atoms)
    if torch.any(degree <= 0):
        atoms = torch.nonzero(degree <= 0, as_tuple=False).flatten().tolist()[:5]
        raise RuntimeError(f"{identifier}: atoms without incoming periodic neighbors: {atoms}")

    # Exact directed periodic edges must be unique. Quantize integer image shifts
    # before constructing Python tuples; build_periodic_graph stores integer-valued shifts in float tensors.
    edge_keys = [
        (
            int(source[index]),
            int(destination[index]),
            int(round(float(edge_shift[index, 0]))),
            int(round(float(edge_shift[index, 1]))),
            int(round(float(edge_shift[index, 2]))),
        )
        for index in range(n_edges)
    ]
    if len(edge_keys) != len(set(edge_keys)):
        raise RuntimeError(f"{identifier}: duplicate directed periodic edges detected")

    return {
        "id": identifier,
        "n_atoms": n_atoms,
        "n_edges": n_edges,
        "max_degree": int(degree.max()),
        "max_edge_A": float(distance.max()),
    }


def main() -> int:
    args = parse_args()
    config = _resolve_config(Path(args.config).resolve())
    data = config["data"]
    cache = load_or_build_cache(
        data["table"],
        data["root"],
        data["cache"],
        radius=float(data["radius"]),
        max_neighbors=int(data["max_neighbors"]),
        rebuild=False,
    )
    records = cache["records"]
    splits = split_indices(records)
    assert_formal_primary_target_coverage(records, splits)
    minimum_vacuum = assert_formal_slab_vacuum(records, float(data["radius"]))
    summaries = [_record(record, float(data["radius"])) for record in records]
    result = {
        "pass": True,
        "records": len(records),
        "minimum_normal_vacuum_A": float(minimum_vacuum),
        "max_atoms": max(item["n_atoms"] for item in summaries),
        "max_edges": max(item["n_edges"] for item in summaries),
        "max_realized_degree": max(item["max_degree"] for item in summaries),
        "maximum_edge_distance_A": max(item["max_edge_A"] for item in summaries),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
