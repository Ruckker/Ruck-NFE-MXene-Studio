from __future__ import annotations

import inspect
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import INDEX_TO_LABEL, NFEDataset as BaseNFEDataset, collate_graphs as base_collate_graphs
from .model import PeriodicNFEModel
from .provenance import build_provenance


_CACHE_META: dict[str, Any] | None = None
_PROVENANCE: dict[str, Any] = {}
_RECENT_EVALUATIONS: deque[dict[str, np.ndarray]] = deque(maxlen=2)


class AuditedNFEDataset(BaseNFEDataset):
    """NFEDataset that exposes the stable cache-record index for audit-safe evaluation."""

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
    """Remove DistributedSampler padding duplicates using stable record indices."""
    if "record_indices" not in payload:
        return payload
    indices = np.asarray(payload["record_indices"], dtype=np.int64)
    if not len(indices):
        return payload
    unique_indices, first_positions = np.unique(indices, return_index=True)
    order = np.argsort(unique_indices)
    positions = first_positions[order]
    result: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == len(indices):
            result[key] = array[positions]
        else:
            result[key] = array
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
            "True_Label": [INDEX_TO_LABEL.get(int(x), "") for x in labels],
            "Predicted_Label": [INDEX_TO_LABEL.get(int(x), "") for x in predicted],
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


def install_audit_patches(train_module) -> None:
    """Install numerical-audit patches on the existing trainer without duplicating it."""
    if getattr(train_module, "_benchmark_audit_patched", False):
        return

    original_torch_load = train_module.torch_load_compat
    original_assert_disjoint = train_module.assert_disjoint_split_groups
    original_checkpoint_payload = train_module.checkpoint_payload
    original_save_json = train_module.save_json

    def audited_torch_load(path, map_location="cpu"):
        global _CACHE_META
        payload = original_torch_load(path, map_location=map_location)
        if isinstance(payload, dict) and payload.get("schema") == "nfe-mxene-cache-1.0":
            _CACHE_META = payload
        return payload

    def audited_assert_disjoint(records, splits) -> None:
        global _PROVENANCE
        original_assert_disjoint(records, splits)
        if _CACHE_META is not None:
            _PROVENANCE = build_provenance(
                cache=_CACHE_META,
                records=records,
                splits=splits,
            )

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
        _RECENT_EVALUATIONS.append(payload)
        return metrics, payload

    def audited_checkpoint_payload(**kwargs):
        payload = original_checkpoint_payload(**kwargs)
        config = kwargs["config"]
        model = kwargs["model"]
        payload["provenance"] = dict(_PROVENANCE)
        config.setdefault("provenance", dict(_PROVENANCE))
        if config.get("ablation"):
            allowed = {
                name
                for name in inspect.signature(PeriodicNFEModel.__init__).parameters
                if name != "self"
            }
            base_model_config = {
                key: value for key, value in model.config.items() if key in allowed
            }
            payload["format"] = "nfe-mxene-predictor-ablation-1.0"
            payload["architecture"] = type(model).__name__
            payload["base_model_config"] = base_model_config
            payload["ablation_config"] = dict(config["ablation"])
            payload["model_config"] = base_model_config
        return payload

    def audited_save_json(path, value) -> None:
        path_obj = Path(path)
        if path_obj.name == "final_metrics.json" and isinstance(value, dict):
            value = dict(value)
            value["provenance"] = dict(_PROVENANCE)
            if len(_RECENT_EVALUATIONS) >= 2:
                # Final trainer order is test then validation.
                test_payload, validation_payload = list(_RECENT_EVALUATIONS)[-2:]
                temperature = float(value.get("classification_temperature", 1.0))
                prediction_frame(test_payload, temperature).to_csv(
                    path_obj.with_name("test_predictions.csv"), index=False
                )
                prediction_frame(validation_payload, temperature).to_csv(
                    path_obj.with_name("validation_predictions.csv"), index=False
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
