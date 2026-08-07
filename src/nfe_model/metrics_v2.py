from __future__ import annotations

import math

import numpy as np

CLASS_NAMES = ("low", "medium", "high")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(np.sum(labels))
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    rank_sum = float(np.sum(ranks[labels]))
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-safe average precision for one-vs-rest class probabilities."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    tp = 0
    fp = 0
    previous_recall = 0.0
    ap = 0.0
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        group = labels[start:stop]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(ap)


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


def _ranking_at_fraction(
    truth: np.ndarray, scores: np.ndarray, fraction: float
) -> tuple[float, float, float]:
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n = len(truth)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = max(1, int(math.ceil(float(fraction) * n)))
    order = np.argsort(-scores, kind="mergesort")[:k]
    selected_positive = int(truth[order].sum())
    total_positive = int(truth.sum())
    precision = selected_positive / k
    recall = selected_positive / total_positive if total_positive else float("nan")
    prevalence = total_positive / n
    enrichment = precision / prevalence if prevalence > 0 else float("nan")
    return float(precision), float(recall), float(enrichment)


def _nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def classification_metrics(
    logits: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    labels = np.asarray(labels)
    valid = labels >= 0
    logits = np.asarray(logits)[valid]
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

    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    aps: list[float] = []
    aucs: list[float] = []
    result: dict[str, float] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(np.sum((prediction == class_index) & (labels == class_index)))
        false_positive = int(np.sum((prediction == class_index) & (labels != class_index)))
        false_negative = int(np.sum((prediction != class_index) & (labels == class_index)))
        support = true_positive + false_negative
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        one_vs_rest = labels == class_index
        auc = binary_roc_auc(one_vs_rest, probabilities[:, class_index])
        ap = binary_average_precision(one_vs_rest, probabilities[:, class_index])
        precisions.append(precision)
        if support:
            recalls.append(recall)
        f1_values.append(f1)
        aps.append(ap)
        aucs.append(auc)
        result[f"{class_name}_precision"] = float(precision)
        result[f"{class_name}_recall"] = float(recall)
        result[f"{class_name}_f1"] = float(f1)
        result[f"{class_name}_roc_auc"] = float(auc)
        result[f"{class_name}_average_precision"] = float(ap)
        result[f"{class_name}_support"] = float(support)

    result.update(
        {
            "accuracy": float(np.mean(labels == prediction)),
            "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
            "macro_precision": float(np.mean(precisions)),
            "macro_f1": float(np.mean(f1_values)),
            "macro_roc_auc": _nanmean(aucs),
            "macro_average_precision": _nanmean(aps),
            "ece": float(expected_calibration_error(probabilities, labels)),
        }
    )
    high_truth = labels == 2
    for fraction, suffix in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")):
        precision, recall, enrichment = _ranking_at_fraction(
            high_truth, probabilities[:, 2], fraction
        )
        result[f"high_precision_at_{suffix}"] = precision
        result[f"high_recall_at_{suffix}"] = recall
        result[f"high_enrichment_at_{suffix}"] = enrichment

    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            result[f"confusion_true_{actual_name}_pred_{predicted_name}"] = float(
                confusion[actual_index, predicted_index]
            )
    return result


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / denom) if denom > 0 else float("nan")


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
        pred = np.asarray(prediction[valid, index], dtype=np.float64)
        truth = np.asarray(target[valid, index], dtype=np.float64)
        error = pred - truth
        result[f"{name}_mae"] = float(np.mean(np.abs(error)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[f"{name}_spearman"] = spearman_correlation(pred, truth)
        centered = truth - truth.mean()
        denominator = float(np.sum(centered**2))
        result[f"{name}_r2"] = (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan")
        )
    return result


def _finite_or(value: float | None, fallback: float) -> float:
    if value is None:
        return fallback
    value = float(value)
    return value if math.isfinite(value) else fallback


def selection_score(metrics: dict[str, float]) -> float:
    """Historical checkpoint-selection score retained for comparability."""
    macro_f1 = _finite_or(metrics.get("macro_f1"), 0.0)
    auc = _finite_or(metrics.get("macro_roc_auc", metrics.get("high_roc_auc")), 0.5)
    score_mae = _finite_or(metrics.get("NFE_Pseudo_Score_mae"), 1.0)
    regression_quality = math.exp(-score_mae / 0.15)
    ece = _finite_or(metrics.get("ece"), 1.0)
    calibration_quality = max(0.0, 1.0 - ece)
    return 0.40 * macro_f1 + 0.25 * auc + 0.25 * regression_quality + 0.10 * calibration_quality
