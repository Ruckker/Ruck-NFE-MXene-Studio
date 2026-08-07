from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from nfe_model.data_v2 import (
    INDEX_TO_LABEL,
    NFEDataset,
    assert_disjoint_split_groups,
    class_weights,
    collate_graphs,
    inverse_target,
    load_or_build_cache,
    robust_normalizers,
    split_indices,
)
from nfe_model.metrics_v2 import classification_metrics, regression_metrics, selection_score
from nfe_model.provenance_v2 import build_provenance
from nfe_model.utils import save_json as _strict_save_json


@dataclass
class BenchmarkData:
    records: list[dict[str, Any]]
    splits: dict[str, list[int]]
    normalizers: dict[str, torch.Tensor]
    config: dict[str, Any]
    table_path: Path
    root: Path
    cache_path: Path
    skipped_cache_records: int
    provenance: dict[str, Any]


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_benchmark_data(
    config_path: str | Path = "training/configs/nfe_predictor.yaml",
    *,
    rebuild_cache: bool = False,
) -> BenchmarkData:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "data" not in config:
        raise ValueError(f"invalid predictor config: {config_path}")
    data_config = config["data"]
    base = config_path.parent
    table_path = _resolve_path(base, data_config["table"])
    root = _resolve_path(base, data_config["root"])
    cache_path = _resolve_path(base, data_config["cache"])
    cache = load_or_build_cache(
        table_path,
        root,
        cache_path,
        radius=float(data_config["radius"]),
        max_neighbors=int(data_config["max_neighbors"]),
        rebuild=bool(rebuild_cache),
    )
    records = list(cache["records"])
    skipped = list(cache.get("skipped", []))
    skip_fraction = len(skipped) / max(1, len(records) + len(skipped))
    maximum = float(data_config.get("max_cache_skip_fraction", 0.01))
    if skip_fraction > maximum:
        examples = "; ".join(
            f"{item.get('id', '?')}: {item.get('error', '?')}" for item in skipped[:5]
        )
        raise RuntimeError(
            f"benchmark graph cache skipped {skip_fraction:.2%} of structures, exceeding "
            f"the configured {maximum:.2%} limit; examples: {examples}"
        )

    splits = split_indices(records)
    assert_disjoint_split_groups(records, splits)
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        sizes = {key: len(value) for key, value in splits.items()}
        raise RuntimeError(
            "benchmark requires non-empty train/validation/test splits; "
            f"got {sizes}"
        )
    normalizers = robust_normalizers(records, splits["train"])
    provenance = build_provenance(cache=cache, records=records, splits=splits)
    return BenchmarkData(
        records=records,
        splits=splits,
        normalizers=normalizers,
        config=config,
        table_path=table_path,
        root=root,
        cache_path=cache_path,
        skipped_cache_records=len(skipped),
        provenance=provenance,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_loader(
    data: BenchmarkData,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalizers: dict[str, torch.Tensor] | None = None,
) -> DataLoader:
    dataset = NFEDataset(
        data.records,
        data.splits[split],
        normalizers or data.normalizers,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_graphs,
        drop_last=False,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    if int(np.sum(valid)) < 3:
        return 1.0
    x = torch.tensor(logits[valid], dtype=torch.float32)
    y = torch.tensor(labels[valid], dtype=torch.long)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=75, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(0.05, 20.0))


def metrics_from_arrays(
    logits: np.ndarray,
    labels: np.ndarray,
    score_prediction: np.ndarray,
    score_target: np.ndarray,
    score_mask: np.ndarray,
) -> dict[str, float]:
    metrics = classification_metrics(logits, labels)
    metrics.update(
        regression_metrics(
            np.asarray(score_prediction, dtype=float).reshape(-1, 1),
            np.asarray(score_target, dtype=float).reshape(-1, 1),
            np.asarray(score_mask, dtype=bool).reshape(-1, 1),
            ["NFE_Pseudo_Score"],
        )
    )
    metrics["selection_score"] = selection_score(metrics)
    return metrics


def calibrate_metrics(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    test_logits: np.ndarray,
    validation_score_prediction: np.ndarray,
    validation_score_target: np.ndarray,
    validation_score_mask: np.ndarray,
    test_labels: np.ndarray,
    test_score_prediction: np.ndarray,
    test_score_target: np.ndarray,
    test_score_mask: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, float]]:
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_metrics = metrics_from_arrays(
        validation_logits / temperature,
        validation_labels,
        validation_score_prediction,
        validation_score_target,
        validation_score_mask,
    )
    test_metrics = metrics_from_arrays(
        test_logits / temperature,
        test_labels,
        test_score_prediction,
        test_score_target,
        test_score_mask,
    )
    return temperature, validation_metrics, test_metrics


def score_arrays(
    records: Sequence[dict[str, Any]], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([int(records[i]["label"]) for i in indices], dtype=np.int64)
    score = np.asarray([float(records[i]["targets"][0]) for i in indices], dtype=np.float64)
    mask = np.asarray([bool(records[i]["target_mask"][0]) for i in indices], dtype=bool)
    weights = np.asarray(
        [float(records[i].get("sample_weight", 1.0)) for i in indices], dtype=np.float64
    )
    return labels, score, mask, weights


def class_weight_array(data: BenchmarkData) -> np.ndarray:
    return class_weights(data.records, data.splits["train"]).cpu().numpy()


def manifest_frame(data: BenchmarkData) -> pd.DataFrame:
    split_lookup = {
        index: split for split, indices in data.splits.items() for index in indices
    }
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(data.records):
        label_index = int(record.get("label", -1))
        rows.append(
            {
                "Record_Index": index,
                "Structure_Name": record.get("id", ""),
                "Split_Group": record.get("split_group", ""),
                "Suggested_Split": split_lookup.get(index, "train"),
                "File_Path": record.get("file_path", ""),
                "Source_File_SHA256": record.get("source_file_sha256", ""),
                "NFE_Pseudo_Label": INDEX_TO_LABEL.get(label_index, ""),
                "NFE_Pseudo_Score": (
                    float(record["targets"][0])
                    if bool(record["target_mask"][0])
                    else np.nan
                ),
                "Data_Quality_Weight": float(record.get("sample_weight", 1.0)),
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(
    data: BenchmarkData,
    split: str,
    *,
    logits: np.ndarray,
    score_prediction: np.ndarray,
    temperature: float,
) -> pd.DataFrame:
    indices = data.splits[split]
    labels, score_target, score_mask, _ = score_arrays(data.records, indices)
    scaled = np.asarray(logits, dtype=float) / max(float(temperature), 1e-8)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    score_prediction = np.asarray(score_prediction, dtype=float)
    if len(score_prediction) != len(indices) or len(probabilities) != len(indices):
        raise ValueError(
            f"prediction length mismatch for {split}: rows={len(indices)} "
            f"probabilities={len(probabilities)} scores={len(score_prediction)}"
        )
    return pd.DataFrame(
        {
            "Record_Index": indices,
            "Structure_Name": [str(data.records[i].get("id", "")) for i in indices],
            "Split_Group": [str(data.records[i].get("split_group", "")) for i in indices],
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


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    _strict_save_json(path, payload)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    provenance = result.get("provenance", {})
    row: dict[str, Any] = {
        "track": result.get("track", "architecture"),
        "model": result.get("model"),
        "seed": result.get("seed"),
        "parameter_count": result.get("parameter_count"),
        "training_seconds": result.get("training_seconds"),
        "temperature": result.get("temperature"),
        "dataset_table_sha256": provenance.get("dataset_table_sha256"),
        "structure_manifest_sha256": provenance.get("structure_manifest_sha256"),
        "split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
    }
    for split in ("validation", "test"):
        metrics = result.get(f"{split}_metrics", {})
        for key, value in metrics.items():
            row[f"{split}_{key}"] = value
    return row


def mean_std_text(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return ""
    if len(array) == 1:
        return f"{array[0]:.5f}"
    return f"{array.mean():.5f} ± {array.std(ddof=1):.5f}"


def inverse_score_from_normalized(
    normalized: np.ndarray,
    normalizers: dict[str, torch.Tensor],
) -> np.ndarray:
    median = float(normalizers["target_median"][0])
    scale = float(normalizers["target_scale"][0])
    transformed = np.asarray(normalized, dtype=float) * scale + median
    return np.asarray(inverse_target(transformed, "identity"), dtype=float)
