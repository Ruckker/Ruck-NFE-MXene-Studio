from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

try:
    from .common import (
        BenchmarkData,
        calibrate_metrics,
        class_weight_array,
        score_arrays,
    )
except ImportError:
    from common import (
        BenchmarkData,
        calibrate_metrics,
        class_weight_array,
        score_arrays,
    )


def structural_feature_vector(record: dict[str, Any]) -> np.ndarray:
    """Build a leakage-safe structure-only feature vector.

    Only cached atomic numbers, elemental descriptors, and geometric global
    invariants are used. No electronic-structure-derived CSV field is read.
    """
    z = record["z"].detach().cpu().numpy().astype(int)
    descriptors = record["atom_features"].detach().cpu().numpy().astype(np.float64)
    global_features = record["global_features"].detach().cpu().numpy().astype(np.float64)

    composition = np.bincount(z, minlength=119)[1:119].astype(np.float64)
    composition /= max(float(composition.sum()), 1.0)

    if descriptors.size:
        descriptor_stats = np.concatenate(
            [
                descriptors.mean(axis=0),
                descriptors.std(axis=0),
                descriptors.min(axis=0),
                descriptors.max(axis=0),
            ]
        )
    else:
        descriptor_stats = np.zeros(56, dtype=np.float64)

    simple_stats = np.asarray(
        [
            np.log1p(len(z)),
            float(np.count_nonzero(composition)),
            float(np.max(composition)) if composition.size else 0.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([composition, descriptor_stats, global_features, simple_stats])


def feature_matrix(records: Sequence[dict[str, Any]], indices: Sequence[int]) -> np.ndarray:
    return np.vstack([structural_feature_vector(records[i]) for i in indices]).astype(np.float32)


def _metrics_payload(
    data: BenchmarkData,
    *,
    validation_logits: np.ndarray,
    test_logits: np.ndarray,
    validation_score_prediction: np.ndarray,
    test_score_prediction: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, float]]:
    val_labels, val_score, val_mask, _ = score_arrays(data.records, data.splits["validation"])
    test_labels, test_score, test_mask, _ = score_arrays(data.records, data.splits["test"])
    return calibrate_metrics(
        validation_logits,
        val_labels,
        test_logits,
        validation_score_prediction,
        val_score,
        val_mask,
        test_labels,
        test_score_prediction,
        test_score,
        test_mask,
    )


def run_dummy(data: BenchmarkData, seed: int) -> dict[str, Any]:
    del seed
    start = time.time()
    train_labels, train_score, train_mask, _ = score_arrays(data.records, data.splits["train"])
    valid_train_labels = train_labels[train_labels >= 0]
    counts = np.bincount(valid_train_labels, minlength=3).astype(np.float64) + 1e-6
    prior = counts / counts.sum()
    train_median = float(np.median(train_score[train_mask])) if np.any(train_mask) else 0.5

    n_val = len(data.splits["validation"])
    n_test = len(data.splits["test"])
    validation_logits = np.tile(np.log(prior)[None, :], (n_val, 1))
    test_logits = np.tile(np.log(prior)[None, :], (n_test, 1))
    validation_score_prediction = np.full(n_val, train_median, dtype=np.float64)
    test_score_prediction = np.full(n_test, train_median, dtype=np.float64)
    temperature, validation_metrics, test_metrics = _metrics_payload(
        data,
        validation_logits=validation_logits,
        test_logits=test_logits,
        validation_score_prediction=validation_score_prediction,
        test_score_prediction=test_score_prediction,
    )
    return {
        "parameter_count": 0,
        "training_seconds": time.time() - start,
        "temperature": temperature,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "details": {"class_prior": prior.tolist(), "score_median": train_median},
    }


def run_xgboost(data: BenchmarkData, seed: int) -> dict[str, Any]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "xgboost baseline requires the optional dependency: "
            "python -m pip install -r training/baselines/requirements-classical.txt"
        ) from exc

    start = time.time()
    train_idx = data.splits["train"]
    val_idx = data.splits["validation"]
    test_idx = data.splits["test"]
    x_train = feature_matrix(data.records, train_idx)
    x_val = feature_matrix(data.records, val_idx)
    x_test = feature_matrix(data.records, test_idx)
    y_train, score_train, score_mask_train, base_weights = score_arrays(data.records, train_idx)

    valid_class = y_train >= 0
    class_weights = class_weight_array(data)
    classification_weights = base_weights[valid_class] * class_weights[y_train[valid_class]]
    classifier = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=2.0,
        reg_alpha=1e-4,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=int(seed),
        n_jobs=-1,
        tree_method="hist",
    )
    classifier.fit(
        x_train[valid_class],
        y_train[valid_class],
        sample_weight=classification_weights,
    )

    regressor = XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=2.0,
        reg_alpha=1e-4,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=int(seed),
        n_jobs=-1,
        tree_method="hist",
    )
    if np.any(score_mask_train):
        regressor.fit(
            x_train[score_mask_train],
            score_train[score_mask_train],
            sample_weight=base_weights[score_mask_train],
        )
        validation_score_prediction = regressor.predict(x_val)
        test_score_prediction = regressor.predict(x_test)
    else:
        fallback = 0.5
        validation_score_prediction = np.full(len(x_val), fallback)
        test_score_prediction = np.full(len(x_test), fallback)

    val_probability = np.asarray(classifier.predict_proba(x_val), dtype=np.float64)
    test_probability = np.asarray(classifier.predict_proba(x_test), dtype=np.float64)
    validation_logits = np.log(np.clip(val_probability, 1e-8, 1.0))
    test_logits = np.log(np.clip(test_probability, 1e-8, 1.0))
    temperature, validation_metrics, test_metrics = _metrics_payload(
        data,
        validation_logits=validation_logits,
        test_logits=test_logits,
        validation_score_prediction=validation_score_prediction,
        test_score_prediction=test_score_prediction,
    )
    parameter_count = int(
        classifier.get_booster().num_boosted_rounds() + regressor.get_booster().num_boosted_rounds()
    )
    return {
        "parameter_count": parameter_count,
        "training_seconds": time.time() - start,
        "temperature": temperature,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "details": {
            "feature_dim": int(x_train.shape[1]),
            "classifier_trees": int(classifier.get_booster().num_boosted_rounds()),
            "regressor_trees": int(regressor.get_booster().num_boosted_rounds()),
            "parameter_count_note": "tree rounds, not neural-network scalar parameters",
        },
    }
