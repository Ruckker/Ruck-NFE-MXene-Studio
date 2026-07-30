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


# 中文：顶层接口 `classification_metrics`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `classification_metrics`; review type hints and callers before extending it.
def classification_metrics(
    logits: np.ndarray, labels: np.ndarray
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
        }
    )
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            result[
                f"confusion_true_{actual_name}_pred_{predicted_name}"
            ] = float(confusion[actual_index, predicted_index])
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
    return result


# 中文：顶层接口 `selection_score`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `selection_score`; review type hints and callers before extending it.
def selection_score(metrics: dict[str, float]) -> float:
    macro_f1 = metrics.get("macro_f1", 0.0)
    auc = metrics.get(
        "macro_roc_auc",
        metrics.get("high_roc_auc", 0.5),
    )
    score_mae = metrics.get("NFE_Pseudo_Score_mae", 1.0)
    regression_quality = math.exp(-score_mae / 0.15)
    calibration_penalty = max(0.0, 1.0 - metrics.get("ece", 1.0))
    return (
        0.40 * macro_f1
        + 0.25 * auc
        + 0.25 * regression_quality
        + 0.10 * calibration_penalty
    )
