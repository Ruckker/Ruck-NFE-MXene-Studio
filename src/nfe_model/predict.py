# ==============================================================================
# 中文概述：对 CIF/POSCAR 执行三分类、连续 NFE 强度、多物性和 OOD 推理。
# English overview: Infer tri-class, continuous NFE strength, auxiliary properties, and OOD status from CIF/POSCAR.
#
# 中文输入：预测器检查点与一个或多个晶体结构。
# English inputs: A predictor checkpoint and one or more crystal structures.
# 中文输出：概率、类别、NFE 分数、MC-dropout 不确定性、OOD 与物性预测。
# English outputs: Probabilities, class, NFE score, MC-dropout uncertainty, OOD, and property predictions.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: load_checkpoint_model, prediction_batch, physical_regression, infer_chunk, parse_args, main
#
# Author: Ruck
# Generated: 2026-07-29 20:36:56 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import math
from pathlib import Path
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


# 中文：顶层接口 `load_checkpoint_model`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `load_checkpoint_model`; review type hints and callers before extending it.
def load_checkpoint_model(
    path: str | Path, device: torch.device
) -> tuple[PeriodicNFEModel, dict[str, Any]]:
    # Optimizer state may be present for resumable training and is not needed on GPU.
    checkpoint = torch_load_compat(path, map_location="cpu")
    if checkpoint.get("format") != "nfe-mxene-predictor-1.0":
        raise ValueError(f"unsupported checkpoint format: {path}")
    model = PeriodicNFEModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


# 中文：顶层接口 `prediction_batch`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `prediction_batch`; review type hints and callers before extending it.
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


# 中文：顶层接口 `physical_regression`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `physical_regression`; review type hints and callers before extending it.
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


# 中文：顶层接口 `infer_chunk`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `infer_chunk`; review type hints and callers before extending it.
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
    conformal_radii: list[float] = []

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
            seen = set(int(x) for x in checkpoint.get("seen_elements", []))
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
        for _ in range(max(1, mc_samples)):
            output = model(batch)
            all_probabilities.append(
                torch.softmax(
                    output["class_logits"] / temperature, dim=-1
                )
                .float()
                .cpu()
                .numpy()
            )
            all_regression.append(
                physical_regression(
                    output["regression_mean"].float().cpu().numpy(), checkpoint
                )
            )
            score_scale = float(checkpoint["normalizers"]["target_scale"][0])
            score_sigma = (
                torch.exp(0.5 * output["regression_log_variance"][:, 0])
                .float()
                .cpu()
                .numpy()
                * score_scale
            )
            score_aleatoric.append(score_sigma)
        model.eval()
        conformal_radii.append(float(checkpoint.get("conformal_score_radius", 0.25)))

    probabilities = np.stack(all_probabilities, axis=0)
    regressions = np.stack(all_regression, axis=0)
    aleatoric = np.stack(score_aleatoric, axis=0)
    probability_mean = probabilities.mean(axis=0)
    regression_mean = regressions.mean(axis=0)
    regression_std = regressions.std(axis=0)
    score_total_std = np.sqrt(
        regression_std[:, 0] ** 2 + np.mean(aleatoric**2, axis=0)
    )
    conformal = max(conformal_radii) if conformal_radii else 0.25

    rows: list[dict[str, Any]] = []
    for index, graph in enumerate(graphs):
        probs = probability_mean[index]
        predicted_class = int(np.argmax(probs))
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        score = float(np.clip(regression_mean[index, 0], 0.0, 1.0))
        radius = max(conformal, 1.645 * float(score_total_std[index]))
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
                str(z)
                for z in sorted(
                    set(graph["elements"])
                    - set(
                        int(x)
                        for _, checkpoint in models
                        for x in checkpoint.get("seen_elements", [])
                    )
                )
            ),
        }
        for target_index, spec in enumerate(REGRESSION_TARGETS[1:], start=1):
            row[f"Predicted_{spec.name}"] = float(
                regression_mean[index, target_index]
            )
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


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
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


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    raise SystemExit(main())
