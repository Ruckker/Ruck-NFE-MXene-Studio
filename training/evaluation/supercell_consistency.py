from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    REGRESSION_TARGETS,
    collate_graphs,
    inverse_target,
)
from nfe_model.predict_guard import (
    guarded_build_periodic_graph,
    guarded_load_checkpoint_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit representation invariance of a formal full predictor checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("structures", nargs="+")
    parser.add_argument("--radius", type=float)
    parser.add_argument("--max-neighbors", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-probability-drift", type=float, default=1e-3)
    parser.add_argument("--max-score-drift", type=float, default=1e-3)
    parser.add_argument("--output", default="training/evaluation/results/supercell_consistency.json")
    return parser.parse_args()


def _predict(model, checkpoint, structure: Structure, args, device):
    graph = guarded_build_periodic_graph(
        structure, args.radius, args.max_neighbors, identifier="representation-audit"
    )
    normalizers = {key: value.cpu() for key, value in checkpoint["normalizers"].items()}
    graph["global_features"] = torch.clamp(
        (graph["global_features"] - normalizers["global_median"])
        / normalizers["global_scale"],
        -8.0,
        8.0,
    )
    graph["targets"] = torch.zeros(len(REGRESSION_TARGETS))
    graph["target_mask"] = torch.zeros(len(REGRESSION_TARGETS), dtype=torch.bool)
    graph["label"] = -1
    batch = collate_graphs([graph])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        output = model(batch)
    temperature = float(checkpoint.get("classification_temperature", 1.0))
    probability = torch.softmax(output["class_logits"] / temperature, dim=-1)[0].cpu().numpy()
    mean = output["regression_mean"][0].cpu().numpy()
    median = normalizers["target_median"].numpy()
    scale = normalizers["target_scale"].numpy()
    transformed = mean * scale + median
    score = float(inverse_target(transformed[0], REGRESSION_TARGETS[0].transform))
    return probability, score


def _representations(base: Structure) -> dict[str, Structure]:
    result = {"reference": base.copy()}
    for factor in (2, 3):
        structure = base.copy()
        structure.make_supercell([factor, factor, 1])
        result[f"supercell_{factor}x{factor}"] = structure

    result["site_reordered"] = Structure.from_sites(
        [base[index] for index in reversed(range(len(base)))], to_unit_cell=True
    )

    basis = base.copy()
    basis.make_supercell([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    if len(basis) != len(base):
        raise RuntimeError("unimodular basis transform unexpectedly changed atom count")
    result["equivalent_inplane_basis"] = basis

    matrix = np.asarray(base.lattice.matrix, dtype=float).copy()
    matrix[2] *= 1.5
    result["extra_vacuum_same_cartesian_slab"] = Structure(
        Lattice(matrix),
        base.species,
        base.cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )
    return result


def main() -> int:
    args = parse_args()
    if args.max_probability_drift < 0 or args.max_score_drift < 0:
        raise ValueError("consistency drift thresholds must be non-negative")
    device = torch.device(args.device)
    model, checkpoint = guarded_load_checkpoint_model(args.checkpoint, device)
    provenance = checkpoint["provenance"]
    config_data = checkpoint.get("config", {}).get("data", {})
    expected_radius = float(config_data.get("radius", provenance.get("graph_radius_A", -1.0)))
    expected_neighbors = int(
        config_data.get("max_neighbors", provenance.get("max_neighbors", -1))
    )
    if expected_radius <= 0 or expected_neighbors <= 0:
        raise ValueError("checkpoint is missing graph radius/max_neighbors provenance")
    if args.radius is None:
        args.radius = expected_radius
    elif abs(float(args.radius) - expected_radius) > 1e-12:
        raise ValueError(
            f"--radius={args.radius} does not match checkpoint radius={expected_radius}"
        )
    if args.max_neighbors is None:
        args.max_neighbors = expected_neighbors
    elif int(args.max_neighbors) != expected_neighbors:
        raise ValueError(
            f"--max-neighbors={args.max_neighbors} does not match checkpoint={expected_neighbors}"
        )

    rows = []
    overall_pass = True
    for path in args.structures:
        base = Structure.from_file(path)
        variants = _representations(base)
        predictions = {}
        for name, structure in variants.items():
            probability, score = _predict(model, checkpoint, structure, args, device)
            predictions[name] = {
                "probabilities": probability.tolist(),
                "score": score,
                "n_atoms": len(structure),
            }

        base_probability = np.asarray(predictions["reference"]["probabilities"])
        base_score = float(predictions["reference"]["score"])
        comparisons = {}
        for name, values in predictions.items():
            if name == "reference":
                continue
            probability_drift = float(
                np.max(np.abs(np.asarray(values["probabilities"]) - base_probability))
            )
            score_drift = abs(float(values["score"]) - base_score)
            passed = (
                probability_drift <= float(args.max_probability_drift)
                and score_drift <= float(args.max_score_drift)
            )
            overall_pass &= passed
            comparisons[name] = {
                "max_probability_drift": probability_drift,
                "score_drift": score_drift,
                "pass": bool(passed),
            }
        rows.append(
            {
                "structure": str(Path(path).resolve()),
                "predictions": predictions,
                "comparisons": comparisons,
                "pass": bool(all(value["pass"] for value in comparisons.values())),
            }
        )

    result = {
        "cache_schema": CACHE_SCHEMA,
        "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
        "neighbor_policy": NEIGHBOR_POLICY,
        "max_probability_drift": float(args.max_probability_drift),
        "max_score_drift": float(args.max_score_drift),
        "pass": bool(overall_pass),
        "results": rows,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not overall_pass:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
