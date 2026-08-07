from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    REGRESSION_TARGETS,
    build_periodic_graph,
    collate_graphs,
    inverse_target,
    torch_load_compat,
)
from nfe_model.model import PeriodicNFEModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit exact in-plane supercell consistency of a predictor.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("structures", nargs="+")
    p.add_argument("--radius", type=float)
    p.add_argument("--max-neighbors", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="training/evaluation/results/supercell_consistency.json")
    return p.parse_args()


def _load(path: str, device: torch.device):
    checkpoint = torch_load_compat(path, map_location="cpu")
    provenance = checkpoint.get("provenance", {})
    if checkpoint.get("format") == "nfe-mxene-predictor-ablation-1.0":
        if checkpoint.get("ablation_config", {}).get("name") != "full":
            raise ValueError("supercell consistency audit accepts only production/full checkpoints")
    elif checkpoint.get("format") != "nfe-mxene-predictor-1.0":
        raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format')}")
    if provenance.get("cache_schema") != CACHE_SCHEMA:
        raise ValueError(
            "checkpoint cache schema is incompatible with this audit: "
            f"{provenance.get('cache_schema')} != {CACHE_SCHEMA}"
        )
    if provenance.get("global_feature_schema") != GLOBAL_FEATURE_SCHEMA:
        raise ValueError(
            "checkpoint was not trained with the current supercell-invariant global feature schema: "
            f"{provenance.get('global_feature_schema')} != {GLOBAL_FEATURE_SCHEMA}"
        )
    if provenance.get("neighbor_policy") != NEIGHBOR_POLICY:
        raise ValueError(
            "checkpoint neighbor policy is incompatible with this audit: "
            f"{provenance.get('neighbor_policy')} != {NEIGHBOR_POLICY}"
        )
    config = checkpoint.get("base_model_config", checkpoint.get("model_config"))
    model = PeriodicNFEModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _predict(model, checkpoint, structure: Structure, args, device):
    graph = build_periodic_graph(structure, args.radius, args.max_neighbors)
    normalizers = {k: v.cpu() for k, v in checkpoint["normalizers"].items()}
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
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
    temperature = float(checkpoint.get("classification_temperature", 1.0))
    prob = torch.softmax(out["class_logits"] / temperature, dim=-1)[0].cpu().numpy()
    mean = out["regression_mean"][0].cpu().numpy()
    median = normalizers["target_median"].numpy()
    scale = normalizers["target_scale"].numpy()
    transformed = mean * scale + median
    score = float(inverse_target(transformed[0], REGRESSION_TARGETS[0].transform))
    return prob, score


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = _load(args.checkpoint, device)
    provenance = checkpoint["provenance"]
    config_data = checkpoint.get("config", {}).get("data", {})
    expected_radius = float(config_data.get("radius", provenance.get("graph_radius_A", -1.0)))
    expected_neighbors = int(config_data.get("max_neighbors", provenance.get("max_neighbors", -1)))
    if expected_radius <= 0 or expected_neighbors <= 0:
        raise ValueError("checkpoint is missing graph radius/max_neighbors provenance")
    if args.radius is None:
        args.radius = expected_radius
    elif abs(float(args.radius) - expected_radius) > 1e-12:
        raise ValueError(f"--radius={args.radius} does not match checkpoint radius={expected_radius}")
    if args.max_neighbors is None:
        args.max_neighbors = expected_neighbors
    elif int(args.max_neighbors) != expected_neighbors:
        raise ValueError(
            f"--max-neighbors={args.max_neighbors} does not match checkpoint={expected_neighbors}"
        )
    rows = []
    for path in args.structures:
        base = Structure.from_file(path)
        predictions = {}
        for factor in (1, 2, 3):
            structure = base.copy()
            if factor > 1:
                structure.make_supercell([factor, factor, 1])
            prob, score = _predict(model, checkpoint, structure, args, device)
            predictions[f"{factor}x{factor}"] = {"probabilities": prob.tolist(), "score": score}
        base_prob = np.asarray(predictions["1x1"]["probabilities"])
        base_score = float(predictions["1x1"]["score"])
        max_prob_drift = max(
            float(np.max(np.abs(np.asarray(v["probabilities"]) - base_prob)))
            for k, v in predictions.items()
            if k != "1x1"
        )
        max_score_drift = max(
            abs(float(v["score"]) - base_score)
            for k, v in predictions.items()
            if k != "1x1"
        )
        rows.append(
            {
                "structure": str(Path(path).resolve()),
                "predictions": predictions,
                "max_probability_drift": max_prob_drift,
                "max_score_drift": max_score_drift,
            }
        )
    result = {
        "cache_schema": CACHE_SCHEMA,
        "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
        "neighbor_policy": NEIGHBOR_POLICY,
        "results": rows,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
