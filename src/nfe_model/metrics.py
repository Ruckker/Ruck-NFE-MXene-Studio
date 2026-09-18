# ==============================================================================
# 中文概述：计算分类、回归、校准和检查点选择指标。
# English overview: Compute classification, regression, calibration, and checkpoint-selection metrics.
#
# 中文输入：真实标签、预测概率、预测值和有效掩码。
# English inputs: Ground-truth labels, probabilities, predictions, and validity masks.
# 中文输出：逐类别 precision/recall/F1/AUC、macro 指标、ECE 与回归误差。
# English outputs: Per-class precision/recall/F1/AUC, macro metrics, ECE, and regression errors.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: binary_roc_auc, expected_calibration_error, classification_metrics, regression_metrics, selection_score
#
# Author: Ruck
# Generated: 2026-07-29 20:36:56 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import math

import numpy as np


CLASS_NAMES = ("low", "medium", "high")


# 中文：顶层接口 `binary_roc_auc`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `binary_roc_auc`; review type hints and callers before extending it.
def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""
    labels = labels.astype(bool)
    positives = int(np.sum(labels))
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(np.sum(ranks[labels]))
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


# 中文：顶层接口 `expected_calibration_error`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `expected_calibration_error`; review type hints and callers before extending it.
def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == labels).astype(float)
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return ece


# 中文：顶层接口 `average_precision`；按分数降序累计的精度均值（PR 曲线下面积的阶梯估计）。
# English: Top-level function `average_precision`; step estimate of the area under the precision-recall curve.
def average_precision(scores: np.ndarray, positives: np.ndarray) -> float:
    positives = positives.astype(bool)
    if not np.any(positives):
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    hits = positives[order].astype(float)
    cumulative = np.cumsum(hits)
    ranks = np.arange(1, len(hits) + 1)
    return float(np.sum(hits * cumulative / ranks) / positives.sum())


# 中文：顶层接口 `enrichment_at`；按分数排序取前若干比例，正例密度相对总体基率的倍数。
# English: Top-level function `enrichment_at`; positive density in the top fraction divided by the base rate.
def enrichment_at(scores: np.ndarray, positives: np.ndarray, fraction: float = 0.05) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives).astype(bool)
    total = len(scores)
    base = float(positives.mean()) if total else 0.0
    count = max(1, int(round(total * float(fraction))))
    if not total or base <= 0.0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")[:count]
    return float(positives[order].mean() / base)


# 中文：顶层接口 `spearman_rho`；秩相关系数（平均秩处理并列）。
# English: Top-level function `spearman_rho`; rank correlation with average ranks for ties.
def spearman_rho(first: np.ndarray, second: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(1, len(values) + 1, dtype=np.float64)
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.zeros(len(unique), dtype=np.float64)
        np.add.at(sums, inverse, result)
        return (sums / counts)[inverse]

    first_ranks, second_ranks = ranks(first), ranks(second)
    first_ranks -= first_ranks.mean()
    second_ranks -= second_ranks.mean()
    denominator = float(np.sqrt(np.sum(first_ranks**2) * np.sum(second_ranks**2)))
    return float(np.sum(first_ranks * second_ranks) / denominator) if denominator > 0 else float("nan")


# 中文：顶层接口 `r_squared`；决定系数。/ English: Top-level function `r_squared`; coefficient of determination.
def r_squared(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    total = float(np.sum((target - target.mean()) ** 2))
    return float(1.0 - np.sum((target - prediction) ** 2) / total) if total > 0 else float("nan")


# 中文：顶层接口 `binary_metrics`；high 对非 high 的二分类指标，阈值作用在 P(high) 上。
# English: Top-level function `binary_metrics`; high-vs-rest metrics with a threshold on P(high).
def binary_metrics(
    scores: np.ndarray,
    positives: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "high_vs_rest",
) -> dict[str, float]:
    positives = positives.astype(bool)
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & positives))
    false_positive = int(np.sum(predicted & ~positives))
    false_negative = int(np.sum(~predicted & positives))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(2 * true_positive / denominator) if denominator else 0.0,
        f"{prefix}_roc_auc": float(binary_roc_auc(positives.astype(int), scores)),
        f"{prefix}_average_precision": average_precision(scores, positives),
        f"{prefix}_support": float(positives.sum()),
    }


# 中文：顶层接口 `best_binary_threshold`；在验证集上选使 high-vs-rest F1 最大的 P(high) 阈值。
# English: Top-level function `best_binary_threshold`; pick the P(high) threshold maximizing validation F1.
def best_binary_threshold(
    scores: np.ndarray, positives: np.ndarray, grid: np.ndarray | None = None
) -> float:
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    positives = positives.astype(bool)
    if not np.any(positives) or np.all(positives):
        return 0.5
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in grid:
        f1 = binary_metrics(scores, positives, float(threshold))["high_vs_rest_f1"]
        if f1 > best_f1 + 1e-12 or (
            abs(f1 - best_f1) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold


# 中文：顶层接口 `classification_metrics`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `classification_metrics`; review type hints and callers before extending it.
def classification_metrics(
    logits: np.ndarray, labels: np.ndarray, high_threshold: float = 0.5
) -> dict[str, float]:
    valid = labels >= 0
    logits = logits[valid]
    labels = labels[valid]
    if not len(labels):
        return {}
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = probabilities.argmax(axis=1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, predicted in zip(labels, prediction):
        confusion[int(actual), int(predicted)] += 1

    precisions = []
    recalls = []
    f1_values = []
    result: dict[str, float] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(
            np.sum((prediction == class_index) & (labels == class_index))
        )
        false_positive = int(
            np.sum((prediction == class_index) & (labels != class_index))
        )
        false_negative = int(
            np.sum((prediction != class_index) & (labels == class_index))
        )
        support = true_positive + false_negative
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        auc = binary_roc_auc(
            (labels == class_index).astype(int),
            probabilities[:, class_index],
        )
        precisions.append(precision)
        if support:
            recalls.append(recall)
        f1_values.append(f1)
        result[f"{class_name}_precision"] = float(precision)
        result[f"{class_name}_recall"] = float(recall)
        result[f"{class_name}_f1"] = float(f1)
        result[f"{class_name}_roc_auc"] = float(auc)
        result[f"{class_name}_support"] = float(support)
    result.update(
        {
            "accuracy": float(np.mean(labels == prediction)),
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
            "macro_precision": float(np.mean(precisions)),
            "macro_f1": float(np.mean(f1_values)),
            "macro_roc_auc": float(
                np.mean([result[f"{name}_roc_auc"] for name in CLASS_NAMES])
            ),
            "ece": float(expected_calibration_error(probabilities, labels)),
            "macro_average_precision": float(
                np.mean([average_precision(probabilities[:, index], labels == index) for index in range(len(CLASS_NAMES))])
            ),
            "high_enrichment_at_5pct": enrichment_at(probabilities[:, 2], labels == 2, 0.05),
        }
    )
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            result[
                f"confusion_true_{actual_name}_pred_{predicted_name}"
            ] = float(confusion[actual_index, predicted_index])
    # The scientifically meaningful decision is "high" versus everything else:
    # low and medium are both non-NFE candidate bands that differ mainly in
    # energy position, so their boundary is a threshold artefact of the
    # pseudo-score (see docs/SCIENTIFIC_OVERVIEW.md).
    result.update(
        binary_metrics(probabilities[:, 2], labels == 2, float(high_threshold))
    )
    return result


# 中文：顶层接口 `regression_metrics`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `regression_metrics`; review type hints and callers before extending it.
def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    names: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, name in enumerate(names):
        valid = mask[:, index].astype(bool)
        if not np.any(valid):
            continue
        error = prediction[valid, index] - target[valid, index]
        result[f"{name}_mae"] = float(np.mean(np.abs(error)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[f"{name}_spearman"] = spearman_rho(prediction[valid, index], target[valid, index])
        result[f"{name}_r2"] = r_squared(prediction[valid, index], target[valid, index])
    return result


# 中文：顶层接口 `selection_score`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `selection_score`; review type hints and callers before extending it.
def selection_score(
    metrics: dict[str, float], primary_task: str = "three_class"
) -> float:
    """Checkpoint-selection score.

    ``primary_task="three_class"`` reproduces the 1.0 behaviour (macro F1 and
    macro AUC).  ``"high_vs_rest"`` weights the binary high-vs-rest F1 and AUC
    instead, which is the recommended target for NFE screening.
    """
    if primary_task == "high_vs_rest":
        task_f1 = metrics.get("high_vs_rest_f1", metrics.get("high_f1", 0.0))
        auc = metrics.get("high_vs_rest_roc_auc", metrics.get("high_roc_auc", 0.5))
    else:
        task_f1 = metrics.get("macro_f1", 0.0)
        auc = metrics.get(
            "macro_roc_auc",
            metrics.get("high_roc_auc", 0.5),
        )
    score_mae = metrics.get("NFE_Pseudo_Score_mae", 1.0)
    regression_quality = math.exp(-score_mae / 0.15)
    calibration_penalty = max(0.0, 1.0 - metrics.get("ece", 1.0))
    return (
        0.40 * task_f1
        + 0.25 * auc
        + 0.25 * regression_quality
        + 0.10 * calibration_penalty
    )
