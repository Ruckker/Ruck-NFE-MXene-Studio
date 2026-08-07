from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import numpy as np

from nfe_model.provenance_v2 import canonical_sha256

try:
    from .common import BenchmarkData, calibrate_metrics, class_weight_array, score_arrays
except ImportError:
    from common import BenchmarkData, calibrate_metrics, class_weight_array, score_arrays


def structural_feature_vector(record: dict[str, Any]) -> np.ndarray:
    """Leakage-safe and in-plane-supercell-intensive structure-only features."""
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
    n_atoms = max(len(z), 1)
    mean_degree = float(record["edge_index"].shape[1]) / n_atoms
    simple_stats = np.asarray(
        [
            mean_degree,
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
):
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


def _prediction_payload(logits: np.ndarray, score_prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "logits": np.asarray(logits, dtype=np.float64),
        "score_prediction": np.asarray(score_prediction, dtype=np.float64),
    }


def run_dummy(data: BenchmarkData, seed: int) -> dict[str, Any]:
    del seed
    start = time.time()
    train_labels, train_score, train_mask, _ = score_arrays(data.records, data.splits["train"])
    counts = np.bincount(train_labels[train_labels >= 0], minlength=3).astype(np.float64) + 1e-6
    prior = counts / counts.sum()
    median = float(np.median(train_score[train_mask])) if np.any(train_mask) else 0.5
    n_val, n_test = len(data.splits["validation"]), len(data.splits["test"])
    val_logits = np.tile(np.log(prior)[None, :], (n_val, 1))
    test_logits = np.tile(np.log(prior)[None, :], (n_test, 1))
    val_score = np.full(n_val, median)
    test_score = np.full(n_test, median)
    temperature, validation_metrics, test_metrics = _metrics_payload(
        data,
        validation_logits=val_logits,
        test_logits=test_logits,
        validation_score_prediction=val_score,
        test_score_prediction=test_score,
    )
    protocol = {
        "backend": "deterministic_dummy",
        "classification": "Laplace-smoothed train class prior with epsilon=1e-6",
        "regression": "median train NFE_Pseudo_Score",
        "train_only_statistics": True,
    }
    return {
        "parameter_count": 0,
        "training_seconds": time.time() - start,
        "temperature": temperature,
        "model_protocol_sha256": canonical_sha256(protocol),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_predictions": _prediction_payload(val_logits, val_score),
        "test_predictions": _prediction_payload(test_logits, test_score),
        "details": {
            "class_prior": prior.tolist(),
            "score_median": median,
            "model_protocol": protocol,
            "artifact_policy": "deterministic parameter-free baseline; no fitted model artifact",
        },
    }


def run_xgboost(data: BenchmarkData, seed: int) -> dict[str, Any]:
    try:
        import xgboost
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise RuntimeError("install training/baselines/requirements-classical.txt") from exc
    start = time.time()
    train_idx, val_idx, test_idx = (
        data.splits["train"],
        data.splits["validation"],
        data.splits["test"],
    )
    x_train = feature_matrix(data.records, train_idx)
    x_val = feature_matrix(data.records, val_idx)
    x_test = feature_matrix(data.records, test_idx)
    y_train, score_train, score_mask, base_weights = score_arrays(data.records, train_idx)
    valid_class = y_train >= 0
    cw = class_weight_array(data)

    classifier_params = {
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.035,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 2.0,
        "reg_alpha": 1e-4,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": int(seed),
        "n_jobs": -1,
        "tree_method": "hist",
    }
    regressor_params = {
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.035,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 2.0,
        "reg_alpha": 1e-4,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": int(seed),
        "n_jobs": -1,
        "tree_method": "hist",
    }
    classifier = XGBClassifier(**classifier_params)
    classifier.fit(
        x_train[valid_class],
        y_train[valid_class],
        sample_weight=base_weights[valid_class] * cw[y_train[valid_class]],
    )
    regressor = XGBRegressor(**regressor_params)
    fitted = bool(np.any(score_mask))
    if fitted:
        regressor.fit(
            x_train[score_mask],
            score_train[score_mask],
            sample_weight=base_weights[score_mask],
        )
        val_score = regressor.predict(x_val)
        test_score = regressor.predict(x_test)
    else:
        val_score = np.full(len(x_val), 0.5)
        test_score = np.full(len(x_test), 0.5)
    val_prob = np.asarray(classifier.predict_proba(x_val), dtype=np.float64)
    test_prob = np.asarray(classifier.predict_proba(x_test), dtype=np.float64)
    val_logits = np.log(np.clip(val_prob, 1e-8, 1.0))
    test_logits = np.log(np.clip(test_prob, 1e-8, 1.0))
    temperature, validation_metrics, test_metrics = _metrics_payload(
        data,
        validation_logits=val_logits,
        test_logits=test_logits,
        validation_score_prediction=val_score,
        test_score_prediction=test_score,
    )
    classifier_rounds = int(classifier.get_booster().num_boosted_rounds())
    regressor_rounds = int(regressor.get_booster().num_boosted_rounds()) if fitted else 0
    classifier_raw = bytes(classifier.get_booster().save_raw())
    regressor_raw = bytes(regressor.get_booster().save_raw()) if fitted else b"UNFITTED"
    state_digest = hashlib.sha256()
    state_digest.update(b"xgboost-classifier\0")
    state_digest.update(classifier_raw)
    state_digest.update(b"\0xgboost-regressor\0")
    state_digest.update(regressor_raw)

    # random_state is seed-specific and therefore belongs in the experiment
    # identity, not the across-seed model protocol. Remove it before hashing the
    # common model/training semantics used to compare the five independent seeds.
    protocol_classifier = dict(classifier_params)
    protocol_regressor = dict(regressor_params)
    protocol_classifier.pop("random_state", None)
    protocol_regressor.pop("random_state", None)
    protocol = {
        "backend": "xgboost",
        "xgboost_version": str(xgboost.__version__),
        "feature_schema": "composition_fraction_118 + elemental_descriptor_stats_56 + intrinsic_global_11 + simple_stats_3",
        "classifier": protocol_classifier,
        "regressor": protocol_regressor,
        "score_regressor_fitted": fitted,
    }
    return {
        # Tree boosting has no scalar parameter count comparable to a neural
        # network's trainable tensors. Keep the paper-table parameter cell empty
        # and report tree complexity explicitly below instead.
        "parameter_count": None,
        "training_seconds": time.time() - start,
        "temperature": temperature,
        "model_protocol_sha256": canonical_sha256(protocol),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_predictions": _prediction_payload(val_logits, val_score),
        "test_predictions": _prediction_payload(test_logits, test_score),
        "details": {
            "feature_dim": int(x_train.shape[1]),
            "classifier_trees": classifier_rounds,
            "regressor_trees": regressor_rounds,
            "complexity_measure": "boosted-tree rounds; not comparable to neural scalar parameters",
            "supercell_intensive_features": True,
            "xgboost_version": str(xgboost.__version__),
            "model_protocol": protocol,
            "model_state_sha256": state_digest.hexdigest(),
            "artifact_policy": (
                "fitted booster state is content-hashed; paper predictions/results bind this identity, "
                "while deterministic reconstruction also requires the recorded XGBoost version and seed"
            ),
        },
    }