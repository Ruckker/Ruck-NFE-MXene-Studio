# ==============================================================================
# Internal predictor implementation. Public CLI: python -m nfe_model.predict
# ==============================================================================

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from .data import (
    INDEX_TO_LABEL,
    REGRESSION_TARGETS,
    build_periodic_graph,
    collate_graphs,
    inverse_target,
    move_batch,
    torch_load_compat,
)
from .model import PeriodicNFEModel, enable_mc_dropout


# This loader intentionally remains small; the public predict guard wraps it
# with v2.1 data/code/training provenance validation.
def load_checkpoint_model(
    path: str | Path, device: torch.device
) -> tuple[PeriodicNFEModel, dict[str, Any]]:
    checkpoint = torch_load_compat(path, map_location="cpu")
    if checkpoint.get("format") != "nfe-mxene-predictor-1.0":
        raise ValueError(f"unsupported checkpoint format: {path}")
    model = PeriodicNFEModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def prediction_batch(
    graphs: list[dict[str, Any]],
    normalizers: dict[str, torch.Tensor],
) -> dict[str, Any]:
    items = []
    for graph in graphs:
        item = dict(graph)
        item["global_features"] = torch.clamp(
            (graph["global_features"] - normalizers["global_median"])
            / normalizers["global_scale"],
            -8.0,
            8.0,
        )
        item["targets"] = torch.zeros(len(REGRESSION_TARGETS))
        item["target_mask"] = torch.zeros(len(REGRESSION_TARGETS), dtype=torch.bool)
        item["label"] = -1
        item["file_path"] = graph.get("file_path", "")
        items.append(item)
    return collate_graphs(items)


def physical_regression(
    normalized: np.ndarray, checkpoint: dict[str, Any]
) -> np.ndarray:
    normalizers = checkpoint["normalizers"]
    median = normalizers["target_median"].cpu().numpy()
    scale = normalizers["target_scale"].cpu().numpy()
    transformed = normalized * scale + median
    result = np.zeros_like(transformed)
    for index, spec in enumerate(REGRESSION_TARGETS):
        result[:, index] = inverse_target(transformed[:, index], spec.transform)
    return result


def _score_interval_metadata(checkpoint: dict[str, Any]) -> tuple[float, str, bool, float]:
    config = checkpoint.get("config", {})
    confidence = float(config.get("inference", {}).get("confidence_level", 0.90))
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"invalid inference confidence_level={confidence}; expected 0 < p < 1")
    if "empirical_validation_score_radius" in checkpoint:
        radius = float(checkpoint["empirical_validation_score_radius"])
        method = str(
            checkpoint.get(
                "score_interval_method",
                "validation-residual-plus-mc-normal-heuristic",
            )
        )
        guarantee = bool(checkpoint.get("score_interval_coverage_guarantee", False))
    elif "conformal_score_radius" in checkpoint:
        # Compatibility only for an audited checkpoint produced before the
        # terminology correction. The validation split was used for model
        # selection, so this is not granted conformal coverage.
        radius = float(checkpoint["conformal_score_radius"])
        method = "legacy-validation-residual-plus-mc-normal-heuristic"
        guarantee = False
    else:
        radius = 0.25
        method = "fallback-plus-mc-normal-heuristic"
        guarantee = False
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(f"invalid score interval radius in checkpoint: {radius}")
    return radius, method, guarantee, confidence


@torch.no_grad()
def infer_chunk(
    graphs: list[dict[str, Any]],
    models: list[tuple[PeriodicNFEModel, dict[str, Any]]],
    device: torch.device,
    mc_samples: int,
) -> list[dict[str, Any]]:
    all_probabilities: list[np.ndarray] = []
    all_regression: list[np.ndarray] = []
    score_aleatoric: list[np.ndarray] = []
    ood_records: list[list[tuple[float, float, str]]] = [[] for _ in graphs]
    interval_radii: list[float] = []
    interval_methods: list[str] = []
    interval_guarantees: list[bool] = []
    interval_confidences: list[float] = []

    for model, checkpoint in models:
        normalizers = {
            key: value.cpu() for key, value in checkpoint["normalizers"].items()
        }
        batch = prediction_batch(graphs, normalizers)
        batch = move_batch(batch, device)
        model.eval()
        deterministic = model(batch)
        embedding = deterministic["embedding"].float().cpu()

        if all(
            key in checkpoint
            for key in ("embedding_mean", "embedding_std", "embedding_bank")
        ):
            normalized_embedding = (
                embedding - checkpoint["embedding_mean"].cpu()
            ) / checkpoint["embedding_std"].cpu()
            z_rms = torch.sqrt(torch.mean(normalized_embedding**2, dim=1))
            nearest = torch.cdist(
                normalized_embedding, checkpoint["embedding_bank"].cpu()
            ).min(dim=1).values
            q95 = float(checkpoint.get("embedding_nearest_q95", math.inf))
            q99 = float(checkpoint.get("embedding_nearest_q99", math.inf))
            z99 = float(checkpoint.get("embedding_z_rms_q99", math.inf))
            seen = set(int(value) for value in checkpoint.get("seen_elements", []))
            for index, graph in enumerate(graphs):
                unseen = bool(set(graph["elements"]) - seen)
                if unseen or float(nearest[index]) > q99 or float(z_rms[index]) > z99:
                    risk = "high"
                elif float(nearest[index]) > q95:
                    risk = "medium"
                else:
                    risk = "low"
                ood_records[index].append(
                    (float(z_rms[index]), float(nearest[index]), risk)
                )

        enable_mc_dropout(model)
        temperature = float(checkpoint.get("classification_temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"invalid classification temperature: {temperature}")
        for _ in range(mc_samples):
            output = model(batch)
            probability = (
                torch.softmax(output["class_logits"] / temperature, dim=-1)
                .float()
                .cpu()
                .numpy()
            )
            regression = physical_regression(
                output["regression_mean"].float().cpu().numpy(), checkpoint
            )
            if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(regression)):
                raise ValueError("predictor produced non-finite probability/regression values")
            all_probabilities.append(probability)
            all_regression.append(regression)
            score_scale = float(checkpoint["normalizers"]["target_scale"][0])
            score_sigma = (
                torch.exp(0.5 * output["regression_log_variance"][:, 0])
                .float()
                .cpu()
                .numpy()
                * score_scale
            )
            if not np.all(np.isfinite(score_sigma)):
                raise ValueError("predictor produced non-finite aleatoric score uncertainty")
            score_aleatoric.append(score_sigma)
        model.eval()
        radius, method, guarantee, confidence = _score_interval_metadata(checkpoint)
        interval_radii.append(radius)
        interval_methods.append(method)
        interval_guarantees.append(guarantee)
        interval_confidences.append(confidence)

    probabilities = np.stack(all_probabilities, axis=0)
    regressions = np.stack(all_regression, axis=0)
    aleatoric = np.stack(score_aleatoric, axis=0)
    probability_mean = probabilities.mean(axis=0)
    regression_mean = regressions.mean(axis=0)
    regression_std = regressions.std(axis=0)
    score_total_std = np.sqrt(
        regression_std[:, 0] ** 2 + np.mean(aleatoric**2, axis=0)
    )

    empirical_radius = max(interval_radii) if interval_radii else 0.25
    method_set = set(interval_methods)
    interval_method = next(iter(method_set)) if len(method_set) == 1 else "mixed-heuristic"
    coverage_guarantee = bool(interval_guarantees) and all(interval_guarantees)
    confidence_set = {round(value, 12) for value in interval_confidences}
    if len(confidence_set) != 1:
        raise ValueError(f"ensemble checkpoints use different interval confidence levels: {sorted(confidence_set)}")
    nominal_confidence = next(iter(confidence_set)) if confidence_set else 0.90
    normal_multiplier = NormalDist().inv_cdf(0.5 + nominal_confidence / 2.0)

    rows: list[dict[str, Any]] = []
    for index, graph in enumerate(graphs):
        probs = probability_mean[index]
        predicted_class = int(np.argmax(probs))
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        score = float(np.clip(regression_mean[index, 0], 0.0, 1.0))
        radius = max(
            empirical_radius,
            float(normal_multiplier) * float(score_total_std[index]),
        )
        risks = [record[2] for record in ood_records[index]]
        risk = (
            "high"
            if "high" in risks
            else ("medium" if "medium" in risks else ("low" if risks else "unknown"))
        )
        row: dict[str, Any] = {
            "Structure_Name": graph["id"],
            "File_Path": graph.get("file_path", ""),
            "Predicted_NFE_Label": INDEX_TO_LABEL[predicted_class],
            "Probability_Low": float(probs[0]),
            "Probability_Medium": float(probs[1]),
            "Probability_High": float(probs[2]),
            "Predictive_Entropy": entropy,
            "Predicted_NFE_Score": score,
            "NFE_Score_Std": float(score_total_std[index]),
            "NFE_Score_Lower": max(0.0, score - radius),
            "NFE_Score_Upper": min(1.0, score + radius),
            "NFE_Score_Interval_Method": interval_method,
            "NFE_Score_Interval_Nominal_Level": float(nominal_confidence),
            "NFE_Score_Interval_Coverage_Guarantee": coverage_guarantee,
            "OOD_Risk": risk,
            "OOD_Embedding_Z_RMS": (
                float(np.mean([record[0] for record in ood_records[index]]))
                if ood_records[index]
                else math.nan
            ),
            "OOD_Nearest_Embedding_Distance": (
                float(np.mean([record[1] for record in ood_records[index]]))
                if ood_records[index]
                else math.nan
            ),
            "Unseen_Elements": "|".join(
                str(atomic_number)
                for atomic_number in sorted(
                    set(graph["elements"])
                    - set(
                        int(value)
                        for _, checkpoint in models
                        for value in checkpoint.get("seen_elements", [])
                    )
                )
            ),
        }
        for target_index, spec in enumerate(REGRESSION_TARGETS[1:], start=1):
            row[f"Predicted_{spec.name}"] = float(regression_mean[index, target_index])
            row[f"Std_{spec.name}"] = float(regression_std[index, target_index])
        row["Recommended_Low_NFE"] = bool(
            row["Predicted_NFE_Label"] == "low"
            and row["Probability_Low"] >= 0.65
            and score <= 0.40
            and risk != "high"
        )
        row["Recommended_Medium_NFE"] = bool(
            row["Predicted_NFE_Label"] == "medium"
            and row["Probability_Medium"] >= 0.65
            and 0.35 <= score <= 0.75
            and risk != "high"
        )
        row["Recommended_High_NFE"] = bool(
            row["Predicted_NFE_Label"] == "high"
            and row["Probability_High"] >= 0.65
            and score >= 0.70
            and risk != "high"
        )
        row["Class_Probability_Ranking"] = "|".join(
            f"{INDEX_TO_LABEL[class_index]}:{float(probs[class_index]):.6f}"
            for class_index in np.argsort(probs)[::-1]
        )
        rows.append(row)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NFE behavior from POSCAR/CIF structures."
    )
    parser.add_argument("structures", nargs="+")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", default="nfe_predictions.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mc-samples", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    models = [load_checkpoint_model(path, device) for path in args.checkpoint]
    first_config = models[0][1]["config"]
    radius = float(first_config["data"]["radius"])
    max_neighbors = int(first_config["data"]["max_neighbors"])
    mc_samples = (
        int(args.mc_samples)
        if args.mc_samples is not None
        else int(first_config["inference"]["mc_samples"])
    )
    if mc_samples <= 0:
        raise ValueError("--mc-samples/config inference.mc_samples must be positive")
    graphs: list[dict[str, Any]] = []
    for path_text in args.structures:
        path = Path(path_text).resolve()
        structure = Structure.from_file(path)
        graph = build_periodic_graph(
            structure,
            radius,
            max_neighbors,
            identifier=path.stem,
        )
        graph["file_path"] = str(path)
        graphs.append(graph)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(graphs), args.batch_size):
        rows.extend(
            infer_chunk(
                graphs[start : start + args.batch_size],
                models,
                device,
                mc_samples,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved predictions to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "nfe_model.predict_core is internal; use `python -m nfe_model.predict` "
        "or training/entrypoints/predict.py so provenance guards cannot be bypassed"
    )
