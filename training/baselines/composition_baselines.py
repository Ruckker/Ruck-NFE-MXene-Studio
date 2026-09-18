# ==============================================================================
# 中文概述：在固定的 group-aware 划分上训练组成基线（规则、逻辑回归、MLP），给等变 GNN 提供对照。
# English overview: Train composition-only baselines (rule, logistic regression, MLP) on the fixed
#                   group-aware split as the reference point for the equivariant GNN.
#
# 中文输入：nfe_dataset.csv；特征只用结构名中的金属、核心、端基、堆垛，可选加入四个几何标量。
# English inputs: nfe_dataset.csv; features are the metals, core, terminations and stackings parsed from
#                 the structure name, optionally plus four relaxed-geometry scalars.
# 中文输出：多 seed 均值 ± 标准差的 JSON 与 Markdown 表。
# English outputs: JSON and a Markdown table with mean ± std over seeds.
#
# 关键约束 / Key invariants:
# - 与 GNN 使用同一 Suggested_Split，同一 metrics 实现；不用 test 集选择任何东西。
#   Same Suggested_Split and the same metrics implementation as the GNN; nothing is selected on test.
# - 主要接口 / Main APIs: build_features, train_classifier, train_regressor, main
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from nfe_model.data import LABEL_TO_INDEX
from nfe_model.metrics import classification_metrics, r_squared, spearman_rho
from nfe_model.utils import save_json

REPORT_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_roc_auc",
    "low_f1",
    "high_f1",
    "high_vs_rest_f1",
    "high_vs_rest_roc_auc",
    "high_vs_rest_average_precision",
    "ece",
)
CATEGORICAL = (
    "Metal_Top",
    "Metal_Bottom",
    "X_Element",
    "Termination_Top",
    "Termination_Bottom",
    "Stacking_Top",
    "Stacking_Bottom",
)
GEOMETRY = ("Lattice_a_A", "slab_thickness_A", "Min_Interatomic_Distance_A", "N_Atoms")


# 中文：顶层接口 `build_features`；有序 one-hot + 无序“含某金属/端基”指示 + 可选几何标量。
# English: Top-level function `build_features`; ordered one-hot plus unordered indicators plus optional geometry.
def build_features(frame: pd.DataFrame, with_geometry: bool) -> tuple[np.ndarray, list[str]]:
    table = pd.get_dummies(frame[list(CATEGORICAL)].astype(str))
    for metal in sorted(set(frame["Metal_Top"]) | set(frame["Metal_Bottom"])):
        table[f"metal_any_{metal}"] = (
            (frame["Metal_Top"] == metal) | (frame["Metal_Bottom"] == metal)
        ).astype(int)
    for termination in sorted(set(frame["Termination_Top"]) | set(frame["Termination_Bottom"])):
        table[f"term_any_{termination}"] = (
            (frame["Termination_Top"] == termination)
            | (frame["Termination_Bottom"] == termination)
        ).astype(int)
    features = table.to_numpy(dtype=np.float32)
    names = list(table.columns)
    if with_geometry:
        geometry = frame[list(GEOMETRY)].to_numpy(dtype=np.float32)
        train = frame["Suggested_Split"].to_numpy() == "train"
        geometry = (geometry - geometry[train].mean(axis=0)) / (geometry[train].std(axis=0) + 1e-6)
        features = np.concatenate([features, geometry], axis=1)
        names.extend(GEOMETRY)
    return features, names


def _network(dimension: int, hidden: int, outputs: int) -> nn.Module:
    if hidden <= 0:
        return nn.Linear(dimension, outputs)
    return nn.Sequential(
        nn.Linear(dimension, hidden),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outputs),
    )


# 中文：顶层接口 `train_classifier`；按验证集 macro F1 选 epoch，返回 test 指标。
# English: Top-level function `train_classifier`; epoch selected on validation macro F1, returns test metrics.
def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    *,
    hidden: int,
    seed: int,
    epochs: int,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(seed)
    train, validation, test = split == "train", split == "validation", split == "test"
    counts = np.bincount(labels[train], minlength=3).astype(np.float32) + 1.0
    weights = np.sqrt(counts.sum() / counts)
    class_weight = torch.tensor(weights / weights.mean(), device=device)
    network = _network(features.shape[1], hidden, 3).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-3, weight_decay=1e-4)
    x_train = torch.tensor(features[train], device=device)
    y_train = torch.tensor(labels[train], device=device)
    x_validation = torch.tensor(features[validation], device=device)
    x_test = torch.tensor(features[test], device=device)
    best, best_state = -1.0, None
    for epoch in range(epochs):
        network.train()
        permutation = torch.randperm(len(x_train), device=device)
        for start in range(0, len(x_train), 256):
            index = permutation[start : start + 256]
            loss = F.cross_entropy(
                network(x_train[index]), y_train[index], weight=class_weight, label_smoothing=0.04
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 4:
            network.eval()
            with torch.no_grad():
                score = classification_metrics(
                    network(x_validation).cpu().numpy(), labels[validation]
                )["macro_f1"]
            if score > best:
                best, best_state = score, {k: v.clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        metrics = classification_metrics(network(x_test).cpu().numpy(), labels[test])
    metrics["validation_macro_f1"] = float(best)
    return metrics


# 中文：顶层接口 `train_regressor`；回归 NFE_Pseudo_Score，按验证 MAE 选 epoch。
# English: Top-level function `train_regressor`; regress NFE_Pseudo_Score with validation-MAE epoch selection.
def train_regressor(
    features: np.ndarray,
    scores: np.ndarray,
    split: np.ndarray,
    *,
    hidden: int,
    seed: int,
    epochs: int,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(seed)
    train, validation, test = split == "train", split == "validation", split == "test"
    network = _network(features.shape[1], hidden, 1).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-3, weight_decay=1e-4)
    x_train = torch.tensor(features[train], device=device)
    y_train = torch.tensor(scores[train], dtype=torch.float32, device=device)
    x_validation = torch.tensor(features[validation], device=device)
    x_test = torch.tensor(features[test], device=device)
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        network.train()
        permutation = torch.randperm(len(x_train), device=device)
        for start in range(0, len(x_train), 256):
            index = permutation[start : start + 256]
            loss = F.smooth_l1_loss(network(x_train[index]).squeeze(-1), y_train[index], beta=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 4:
            network.eval()
            with torch.no_grad():
                mae = float(np.mean(np.abs(network(x_validation).squeeze(-1).cpu().numpy() - scores[validation])))
            if mae < best:
                best, best_state = mae, {k: v.clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        prediction = np.clip(network(x_test).squeeze(-1).cpu().numpy(), 0.0, 1.0)
    error = prediction - scores[test]
    return {
        "NFE_Pseudo_Score_mae": float(np.mean(np.abs(error))),
        "NFE_Pseudo_Score_rmse": float(np.sqrt(np.mean(error**2))),
        "NFE_Pseudo_Score_spearman": spearman_rho(prediction, scores[test]),
        "NFE_Pseudo_Score_r2": r_squared(prediction, scores[test]),
        "validation_mae": float(best),
    }


def _summarize(runs: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in runs[0]:
        values = [float(run[key]) for run in runs if key in run]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composition-only baselines for the NFE predictor.")
    parser.add_argument("--table", default="data/full/nfe_dataset.csv")
    parser.add_argument("--output", default="models/metadata/baseline_metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    table_path = Path(args.table)
    frame = pd.read_csv(table_path)
    labels = frame["NFE_Pseudo_Label"].map(LABEL_TO_INDEX).to_numpy()
    scores = frame["NFE_Pseudo_Score"].to_numpy(dtype=float)
    split = frame["Suggested_Split"].to_numpy()
    composition, composition_names = build_features(frame, with_geometry=False)
    with_geometry, geometry_names = build_features(frame, with_geometry=True)
    test = split == "test"
    started = time.time()

    results: dict[str, Any] = {}
    has_oh = (frame["Termination_Top"] == "OH") | (frame["Termination_Bottom"] == "OH")
    rule = np.where(has_oh.to_numpy()[test], 2, 1)
    results["rule_has_OH_is_high"] = {
        "description": "high if any termination is OH, otherwise medium; no training",
        "runs": [classification_metrics(np.eye(3)[rule] * 8.0, labels[test])],
    }
    variants = {
        "logistic_regression_composition": (composition, 0, "one-hot composition, linear softmax"),
        "mlp128_composition": (composition, 128, "one-hot composition, 2-layer MLP"),
        "mlp128_composition_geometry": (with_geometry, 128, "one-hot composition + a, slab thickness, min distance, N atoms"),
    }
    for name, (features, hidden, description) in variants.items():
        runs = [
            train_classifier(features, labels, split, hidden=hidden, seed=seed, epochs=args.epochs, device=device)
            for seed in args.seeds
        ]
        results[name] = {"description": description, "runs": runs}
        means = {key: round(item["mean"], 4) for key, item in _summarize(runs).items() if key in REPORT_KEYS}
        print(name, means, flush=True)
    regression_runs = [
        train_regressor(with_geometry, scores, split, hidden=128, seed=seed, epochs=args.epochs, device=device)
        for seed in args.seeds
    ]
    results["mlp128_score_regression_composition_geometry"] = {
        "description": "NFE_Pseudo_Score regression from composition + geometry scalars",
        "runs": regression_runs,
    }
    for name, payload in results.items():
        payload["summary"] = _summarize(payload["runs"])
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "table": str(table_path),
        "table_sha256": digest,
        "split_counts": {key: int(np.sum(split == key)) for key in ("train", "validation", "test")},
        "seeds": list(args.seeds),
        "feature_counts": {"composition": len(composition_names), "composition_geometry": len(geometry_names)},
        "device": str(device),
        "elapsed_seconds": round(time.time() - started, 1),
        "results": results,
    }
    save_json(args.output, report)
    lines = ["| baseline | macro F1 | high-vs-rest F1 | high-vs-rest AUC | low F1 | score MAE |", "|---|---:|---:|---:|---:|---:|"]
    for name, payload in results.items():
        summary = payload["summary"]

        def cell(key: str) -> str:
            if key not in summary:
                return "-"
            return f"{summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}"

        lines.append(
            f"| {name} | {cell('macro_f1')} | {cell('high_vs_rest_f1')} | {cell('high_vs_rest_roc_auc')} | {cell('low_f1')} | {cell('NFE_Pseudo_Score_mae')} |"
        )
    print("\n".join(lines))
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
