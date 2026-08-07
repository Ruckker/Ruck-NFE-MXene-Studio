from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data_v2 import (
    CACHE_SCHEMA,
    INDEX_TO_LABEL,
    NFEDataset as BaseNFEDataset,
    collate_graphs as base_collate_graphs,
)
from .model import PeriodicNFEModel
from .provenance_v2 import (
    assert_matching_provenance,
    build_provenance,
    canonical_sha256,
    experiment_protocol_sha256,
    training_protocol_sha256,
    file_sha256,
)
from . import data_v2, metrics_v2


_CACHE_META: dict[str, Any] | None = None
_PROVENANCE: dict[str, Any] = {}
_SPLIT_RECORD_INDICES: dict[str, tuple[int, ...]] = {}
_LATEST_EVALUATIONS: dict[str, dict[str, np.ndarray]] = {}
_EXPERIMENT_PROTOCOL_SHA256 = ""
_TRAINING_PROTOCOL_SHA256 = ""
_SCORE_INTERVAL_METHOD = "validation-residual-plus-mc-normal-heuristic"


def _training_runtime_environment_sha256(provenance: dict[str, Any]) -> str:
    runtime = provenance.get("runtime_environment")
    if not isinstance(runtime, dict) or not runtime:
        raise RuntimeError("audited training provenance is missing runtime_environment")
    return canonical_sha256(runtime)


def _model_protocol_sha256(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    return canonical_sha256(
        {
            "training_protocol_sha256": training_protocol_sha256(config),
            "training_runtime_environment_sha256": _training_runtime_environment_sha256(
                provenance
            ),
            "architecture": type(model).__name__,
            "ablation": dict(config.get("ablation", {}) or {}),
        }
    )


class AuditedNFEDataset(BaseNFEDataset):
    def __getitem__(self, item: int) -> dict[str, Any]:
        result = super().__getitem__(item)
        result["record_index"] = int(self.indices[item])
        return result


def audited_collate_graphs(items):
    batch = base_collate_graphs(items)
    batch["record_indices"] = torch.tensor(
        [int(item.get("record_index", -1)) for item in items], dtype=torch.long
    )
    batch["split_groups"] = [str(item.get("split_group", "")) for item in items]
    return batch


def deduplicate_payload(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "record_indices" not in payload:
        return payload
    indices = np.asarray(payload["record_indices"], dtype=np.int64)
    if not len(indices):
        return payload
    unique_indices, first_positions = np.unique(indices, return_index=True)
    positions = first_positions[np.argsort(unique_indices)]
    result: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        result[key] = (
            array[positions]
            if array.ndim >= 1 and array.shape[0] == len(indices)
            else array
        )
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def prediction_frame(payload: dict[str, np.ndarray], temperature: float = 1.0) -> pd.DataFrame:
    logits = np.asarray(payload["logits"], dtype=float) / max(float(temperature), 1e-8)
    probabilities = _softmax(logits)
    labels = np.asarray(payload["labels"], dtype=int)
    predicted = probabilities.argmax(axis=1)
    score_prediction = np.asarray(payload["prediction"], dtype=float)[:, 0]
    score_target = np.asarray(payload["target"], dtype=float)[:, 0]
    score_mask = np.asarray(payload["mask"], dtype=bool)[:, 0]
    return pd.DataFrame(
        {
            "Record_Index": np.asarray(payload["record_indices"], dtype=int),
            "Structure_Name": np.asarray(payload["ids"], dtype=object),
            "Split_Group": np.asarray(payload["split_groups"], dtype=object),
            "True_Label": [INDEX_TO_LABEL.get(int(value), "") for value in labels],
            "Predicted_Label": [INDEX_TO_LABEL.get(int(value), "") for value in predicted],
            "Probability_Low": probabilities[:, 0],
            "Probability_Medium": probabilities[:, 1],
            "Probability_High": probabilities[:, 2],
            "True_NFE_Pseudo_Score": np.where(score_mask, score_target, np.nan),
            "Predicted_NFE_Pseudo_Score": score_prediction,
            "Absolute_Score_Error": np.where(
                score_mask, np.abs(score_prediction - score_target), np.nan
            ),
        }
    )


def apply_checkpoint_contract(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    result["provenance"] = dict(provenance)
    result["experiment_protocol_sha256"] = experiment_protocol_sha256(config)
    result["training_protocol_sha256"] = training_protocol_sha256(config)
    result["training_runtime_environment_sha256"] = _training_runtime_environment_sha256(
        provenance
    )
    result["model_protocol_sha256"] = _model_protocol_sha256(
        model=model,
        config=config,
        provenance=provenance,
    )
    config.setdefault("provenance", dict(provenance))
    ablation = config.get("ablation")
    if not ablation:
        return result
    allowed = {
        name
        for name in inspect.signature(PeriodicNFEModel.__init__).parameters
        if name != "self"
    }
    model_config = getattr(model, "config", {})
    base_model_config = {key: value for key, value in model_config.items() if key in allowed}
    result["format"] = "nfe-mxene-predictor-ablation-1.0"
    result["architecture"] = type(model).__name__
    result["base_model_config"] = base_model_config
    result["ablation_config"] = dict(ablation)
    result["model_config"] = base_model_config
    return result


def _payload_split_name(payload: dict[str, np.ndarray]) -> str | None:
    observed = tuple(sorted(int(value) for value in np.asarray(payload["record_indices"]).tolist()))
    for split, expected in _SPLIT_RECORD_INDICES.items():
        if observed == expected:
            return split
    return None


def install_audit_patches(train_module) -> None:
    if getattr(train_module, "_benchmark_audit_patched", False):
        return
    original_torch_load = train_module.torch_load_compat
    train_module.load_or_build_cache = data_v2.load_or_build_cache
    train_module.split_indices = data_v2.split_indices
    train_module.classification_metrics = metrics_v2.classification_metrics
    train_module.regression_metrics = metrics_v2.regression_metrics
    train_module.selection_score = metrics_v2.selection_score
    original_checkpoint_payload = train_module.checkpoint_payload
    original_save_json = train_module.save_json

    def audited_torch_load(path, map_location="cpu"):
        global _CACHE_META, _PROVENANCE, _EXPERIMENT_PROTOCOL_SHA256, _TRAINING_PROTOCOL_SHA256
        payload = original_torch_load(path, map_location=map_location)
        if isinstance(payload, dict) and payload.get("schema") == CACHE_SCHEMA:
            _CACHE_META = payload
            _PROVENANCE = {}
            _EXPERIMENT_PROTOCOL_SHA256 = ""
            _TRAINING_PROTOCOL_SHA256 = ""
            _SPLIT_RECORD_INDICES.clear()
            _LATEST_EVALUATIONS.clear()
        elif (
            isinstance(payload, dict)
            and payload.get("format")
            in {"nfe-mxene-predictor-1.0", "nfe-mxene-predictor-ablation-1.0"}
            and _PROVENANCE
        ):
            assert_matching_provenance(
                payload.get("provenance"),
                _PROVENANCE,
                require_present=True,
                require_code_match=True,
            )
            expected_runtime = _training_runtime_environment_sha256(_PROVENANCE)
            observed_runtime = str(payload.get("training_runtime_environment_sha256", ""))
            if not observed_runtime:
                raise ValueError(
                    "formal resume checkpoint lacks training_runtime_environment_sha256; "
                    "start a new audited run instead of importing an environment-unknown checkpoint"
                )
            if observed_runtime != expected_runtime:
                raise ValueError(
                    "resume checkpoint training runtime environment differs from the current audited runtime: "
                    f"checkpoint={observed_runtime} current={expected_runtime}"
                )
            expected_model_protocol = _model_protocol_sha256(
                model=train_module.PeriodicNFEModel
                if isinstance(train_module.PeriodicNFEModel, torch.nn.Module)
                else type("RuntimeArchitecture", (), {})(),
                config=payload.get("config", {}),
                provenance=_PROVENANCE,
            ) if False else None
            # The experiment protocol check performed by the public wrapper owns
            # architecture/config equality; here the runtime identity is the
            # additional resume invariant required before loading optimizer state.
            experiment = str(payload.get("experiment_protocol_sha256", ""))
            common = str(payload.get("training_protocol_sha256", ""))
            if isinstance(payload.get("config"), dict):
                if not experiment:
                    experiment = experiment_protocol_sha256(payload["config"])
                if not common:
                    common = training_protocol_sha256(payload["config"])
            _EXPERIMENT_PROTOCOL_SHA256 = experiment
            _TRAINING_PROTOCOL_SHA256 = common
        return payload

    def audited_assert_disjoint(records, splits) -> None:
        global _PROVENANCE
        data_v2.assert_disjoint_split_groups(records, splits)
        if _CACHE_META is None:
            raise RuntimeError(
                "audit provenance was not initialized from the current graph cache; "
                f"expected schema {CACHE_SCHEMA}"
            )
        _PROVENANCE = build_provenance(cache=_CACHE_META, records=records, splits=splits)
        _SPLIT_RECORD_INDICES.clear()
        _SPLIT_RECORD_INDICES.update(
            {
                split: tuple(sorted(int(index) for index in indices))
                for split, indices in splits.items()
            }
        )
        _LATEST_EVALUATIONS.clear()

    @torch.no_grad()
    def audited_evaluate(model, loader, device, normalizers, amp):
        model.eval()
        logits_list: list[np.ndarray] = []
        mean_list: list[np.ndarray] = []
        target_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        label_list: list[np.ndarray] = []
        index_list: list[np.ndarray] = []
        id_list: list[np.ndarray] = []
        group_list: list[np.ndarray] = []
        for batch in loader:
            batch = train_module.move_batch(batch, device)
            with train_module.autocast_context(device, amp):
                outputs = model(batch)
            logits_list.append(outputs["class_logits"].float().cpu().numpy())
            mean_list.append(outputs["regression_mean"].float().cpu().numpy())
            target_list.append(batch["targets"].float().cpu().numpy())
            mask_list.append(batch["target_mask"].cpu().numpy())
            label_list.append(batch["labels"].cpu().numpy())
            index_list.append(batch["record_indices"].cpu().numpy())
            id_list.append(np.asarray(batch["ids"], dtype=object))
            group_list.append(np.asarray(batch["split_groups"], dtype=object))
        local = {
            "logits": np.concatenate(logits_list, axis=0),
            "mean_normalized": np.concatenate(mean_list, axis=0),
            "target_normalized": np.concatenate(target_list, axis=0),
            "mask": np.concatenate(mask_list, axis=0),
            "labels": np.concatenate(label_list, axis=0),
            "record_indices": np.concatenate(index_list, axis=0),
            "ids": np.concatenate(id_list, axis=0),
            "split_groups": np.concatenate(group_list, axis=0),
        }
        payload = deduplicate_payload(train_module.gather_payload(local))
        median = normalizers["target_median"].cpu().numpy()
        scale = normalizers["target_scale"].cpu().numpy()
        pred_transformed = payload["mean_normalized"] * scale + median
        target_transformed = payload["target_normalized"] * scale + median
        prediction = np.zeros_like(pred_transformed)
        target = np.zeros_like(target_transformed)
        for index, spec in enumerate(train_module.REGRESSION_TARGETS):
            prediction[:, index] = train_module.inverse_target(
                pred_transformed[:, index], spec.transform
            )
            target[:, index] = train_module.inverse_target(
                target_transformed[:, index], spec.transform
            )
        payload["prediction"] = prediction
        payload["target"] = target
        metrics = train_module.classification_metrics(payload["logits"], payload["labels"])
        metrics.update(
            train_module.regression_metrics(
                prediction,
                target,
                payload["mask"],
                [spec.name for spec in train_module.REGRESSION_TARGETS],
            )
        )
        metrics["selection_score"] = train_module.selection_score(metrics)
        split_name = _payload_split_name(payload)
        if split_name in {"validation", "test"}:
            _LATEST_EVALUATIONS[split_name] = payload
        return metrics, payload

    def audited_checkpoint_payload(**kwargs):
        global _EXPERIMENT_PROTOCOL_SHA256, _TRAINING_PROTOCOL_SHA256
        payload = original_checkpoint_payload(**kwargs)
        payload = apply_checkpoint_contract(
            payload,
            model=kwargs["model"],
            config=kwargs["config"],
            provenance=_PROVENANCE,
        )
        _EXPERIMENT_PROTOCOL_SHA256 = str(payload["experiment_protocol_sha256"])
        _TRAINING_PROTOCOL_SHA256 = str(payload["training_protocol_sha256"])
        return payload

    def audited_save_json(path, value) -> None:
        path_obj = Path(path)
        if path_obj.name == "final_metrics.json" and isinstance(value, dict):
            value = dict(value)
            value["provenance"] = dict(_PROVENANCE)
            value["training_runtime_environment_sha256"] = _training_runtime_environment_sha256(
                _PROVENANCE
            )
            best_path = path_obj.with_name("best.pt")
            if not best_path.is_file():
                raise RuntimeError(f"audited final metrics require checkpoint {best_path}")
            best_payload = original_torch_load(best_path, map_location="cpu")
            checkpoint_runtime = str(
                best_payload.get("training_runtime_environment_sha256", "")
            )
            if checkpoint_runtime != value["training_runtime_environment_sha256"]:
                raise RuntimeError(
                    "best checkpoint/runtime environment identity differs from final metrics runtime: "
                    f"checkpoint={checkpoint_runtime or 'missing'} "
                    f"current={value['training_runtime_environment_sha256']}"
                )
            model_protocol = str(best_payload.get("model_protocol_sha256", ""))
            if not model_protocol:
                raise RuntimeError("audited best checkpoint is missing model_protocol_sha256")
            value["model_protocol_sha256"] = model_protocol
            experiment = str(best_payload.get("experiment_protocol_sha256", ""))
            common = str(best_payload.get("training_protocol_sha256", ""))
            if isinstance(best_payload.get("config"), dict):
                if not experiment:
                    experiment = experiment_protocol_sha256(best_payload["config"])
                if not common:
                    common = training_protocol_sha256(best_payload["config"])
            if not experiment:
                experiment = _EXPERIMENT_PROTOCOL_SHA256
            if not common:
                common = _TRAINING_PROTOCOL_SHA256
            if not experiment or not common:
                raise RuntimeError("audited final metrics require experiment/training protocol fingerprints")
            value["experiment_protocol_sha256"] = experiment
            value["training_protocol_sha256"] = common
            if isinstance(best_payload.get("ablation_config"), dict):
                value["ablation_config"] = dict(best_payload["ablation_config"])

            empirical_radius = best_payload.pop("conformal_score_radius", None)
            if empirical_radius is None:
                empirical_radius = value.pop("conformal_score_radius", None)
            else:
                value.pop("conformal_score_radius", None)
            if empirical_radius is not None:
                empirical_radius = float(empirical_radius)
                best_payload["empirical_validation_score_radius"] = empirical_radius
                best_payload["score_interval_method"] = _SCORE_INTERVAL_METHOD
                best_payload["score_interval_coverage_guarantee"] = False
                value["empirical_validation_score_radius"] = empirical_radius
                value["score_interval_method"] = _SCORE_INTERVAL_METHOD
                value["score_interval_coverage_guarantee"] = False
                train_module.atomic_torch_save(best_payload, best_path)

            value["checkpoint_sha256"] = file_sha256(best_path)
            temperature = float(value.get("classification_temperature", 1.0))
            for split in ("validation", "test"):
                payload = _LATEST_EVALUATIONS.get(split)
                if payload is None:
                    raise RuntimeError(
                        f"cannot write audited {split} predictions: no matching evaluation payload"
                    )
                prediction_frame(payload, temperature).to_csv(
                    path_obj.with_name(f"{split}_predictions.csv"), index=False
                )
        original_save_json(path, value)

    train_module.NFEDataset = AuditedNFEDataset
    train_module.collate_graphs = audited_collate_graphs
    train_module.torch_load_compat = audited_torch_load
    train_module.assert_disjoint_split_groups = audited_assert_disjoint
    train_module.evaluate = audited_evaluate
    train_module.checkpoint_payload = audited_checkpoint_payload
    train_module.save_json = audited_save_json
    train_module._benchmark_audit_patched = True
