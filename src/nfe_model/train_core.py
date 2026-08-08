# ==============================================================================
# 中文概述：以 DDP/AMP 训练、验证并校准 NFE 预测器。
# English overview: Train, validate, and calibrate the NFE predictor with DDP and AMP.
#
# 中文输入：YAML 配置、图缓存、数据划分与 1–4 张 GPU。
# English inputs: YAML config, graph cache, dataset splits, and one to four GPUs.
# 中文输出：最佳检查点、训练历史、最终指标、温度与 OOD 嵌入库。
# English outputs: Best checkpoint, history, final metrics, temperature, and OOD embedding bank.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: resolve_config_paths, make_loader, autocast_context, make_grad_scaler, corrupt_structure, compute_loss, gather_payload, evaluate, collect_embedding_bank, checkpoint_payload, fit_temperature, parse_args, main
#
# Author: Ruck
# Generated: 2026-07-29 19:06:31 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import (
    INDEX_TO_LABEL,
    REGRESSION_TARGETS,
    NFEDataset,
    assert_disjoint_split_groups,
    class_weights,
    collate_graphs,
    inverse_target,
    load_or_build_cache,
    move_batch,
    robust_normalizers,
    split_indices,
    torch_load_compat,
)
from .metrics import (
    classification_metrics,
    regression_metrics,
    selection_score,
)
from .model import PeriodicNFEModel, heteroscedastic_loss
from .utils import (
    EarlyStopping,
    atomic_torch_save,
    barrier,
    cleanup_distributed,
    cosine_schedule,
    init_distributed,
    is_main_process,
    load_config,
    reduce_sum,
    save_json,
    seed_everything,
)


# 中文：顶层接口 `resolve_config_paths`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `resolve_config_paths`; review type hints and callers before extending it.
def resolve_config_paths(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    base = config_path.resolve().parent
    data = config["data"]
    training = config["training"]
    for key in ("table", "root", "cache"):
        path = Path(data[key])
        if not path.is_absolute():
            path = base / path
        data[key] = str(path.resolve())
    checkpoint_dir = Path(training["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = base / checkpoint_dir
    training["checkpoint_dir"] = str(checkpoint_dir.resolve())
    return config


# 中文：顶层接口 `make_loader`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `make_loader`; review type hints and callers before extending it.
def make_loader(
    dataset: NFEDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_graphs,
        drop_last=False,
    )
    return loader, sampler


# 中文：顶层接口 `autocast_context`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `autocast_context`; review type hints and callers before extending it.
def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


# 中文：顶层接口 `make_grad_scaler`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `make_grad_scaler`; review type hints and callers before extending it.
def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# 中文：顶层接口 `corrupt_structure`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `corrupt_structure`; review type hints and callers before extending it.
def corrupt_structure(
    batch: dict[str, Any],
    *,
    mask_probability: float,
    noise_min: float,
    noise_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_original = batch["z"]
    mask = torch.rand(z_original.shape[0], device=z_original.device) < mask_probability
    if not torch.any(mask):
        mask[torch.randint(0, z_original.shape[0], (1,), device=z_original.device)] = True
    z_corrupted = z_original.clone()
    z_corrupted[mask] = 0

    n_graphs = batch["lattice"].shape[0]
    log_sigma = torch.empty(n_graphs, device=z_original.device).uniform_(
        math.log(noise_min), math.log(noise_max)
    )
    sigma = torch.exp(log_sigma)
    noise_cart = torch.randn_like(batch["frac_pos"]) * sigma[batch["batch"]].unsqueeze(-1)
    inverse_lattice = torch.linalg.inv(batch["lattice"])
    noise_fractional = torch.einsum(
        "ni,nij->nj", noise_cart, inverse_lattice[batch["batch"]]
    )
    # Do not wrap the noisy coordinates: edge image shifts were built for the
    # original topology, and unwrapped positions keep the denoising target
    # continuous when an atom crosses a periodic cell boundary.
    noisy_fractional = batch["frac_pos"] + noise_fractional
    denoise_target = -noise_cart
    return z_corrupted, noisy_fractional, mask, denoise_target


# 中文：顶层接口 `compute_loss`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `compute_loss`; review type hints and callers before extending it.
def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    class_weights_tensor: torch.Tensor,
    target_weights: torch.Tensor,
    loss_config: dict[str, float],
    masked_nodes: torch.Tensor,
    denoise_target: torch.Tensor,
    pretraining: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    def disabled_loss_anchor(tensor: torch.Tensor) -> torch.Tensor:
        # Keep disabled heads in the autograd graph for DDP without allowing an
        # unused FP16 head overflow (0 * inf) to poison the active objective.
        return torch.nan_to_num(tensor.reshape(-1)[0]) * 0.0

    valid_labels = batch["labels"] >= 0
    if torch.any(valid_labels):
        class_loss = F.cross_entropy(
            outputs["class_logits"][valid_labels],
            batch["labels"][valid_labels],
            weight=class_weights_tensor,
            label_smoothing=float(loss_config["label_smoothing"]),
            reduction="none",
        )
        supervised_weights = batch["sample_weights"][valid_labels]
        class_loss = torch.sum(class_loss * supervised_weights) / supervised_weights.sum().clamp_min(
            1e-6
        )
    else:
        class_loss = outputs["class_logits"].sum() * 0.0

    if bool(torch.any(target_weights > 0).item()):
        regression_loss = heteroscedastic_loss(
            outputs["regression_mean"],
            outputs["regression_log_variance"],
            batch["targets"],
            batch["target_mask"],
            target_weights,
            batch["sample_weights"],
        )
    else:
        regression_loss = disabled_loss_anchor(outputs["regression_mean"])
        regression_loss = regression_loss + disabled_loss_anchor(
            outputs["regression_log_variance"]
        )
    masked_weight = float(loss_config["masked_atom_weight"])
    if masked_weight > 0.0 and torch.any(masked_nodes):
        masked_loss = F.cross_entropy(
            outputs["masked_atom_logits"][masked_nodes],
            batch["z"][masked_nodes],
        )
    else:
        masked_loss = disabled_loss_anchor(outputs["masked_atom_logits"])
    denoise_weight = float(loss_config["denoise_weight"])
    if denoise_weight > 0.0:
        denoise_loss = F.smooth_l1_loss(
            outputs["denoise_vector"], denoise_target, beta=0.05
        )
    else:
        denoise_loss = disabled_loss_anchor(outputs["denoise_vector"])

    supervised_factor = 0.25 if pretraining else 1.0
    ssl_factor = 1.0 if pretraining else 0.20
    total = supervised_factor * (
        float(loss_config["class_weight"]) * class_loss + regression_loss
    ) + ssl_factor * (
        masked_weight * masked_loss + denoise_weight * denoise_loss
    )
    values = {
        "loss": float(total.detach()),
        "class_loss": float(class_loss.detach()),
        "regression_loss": float(regression_loss.detach()),
        "masked_loss": float(masked_loss.detach()),
        "denoise_loss": float(denoise_loss.detach()),
    }
    return total, values


# 中文：顶层接口 `gather_payload`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `gather_payload`; review type hints and callers before extending it.
def gather_payload(local: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not dist.is_initialized():
        return local
    gathered: list[dict[str, np.ndarray] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    keys = local.keys()
    return {
        key: np.concatenate([item[key] for item in gathered if item is not None], axis=0)
        for key in keys
    }


# 中文：顶层接口 `evaluate`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `evaluate`; review type hints and callers before extending it.
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizers: dict[str, torch.Tensor],
    amp: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    logits_list, mean_list, target_list, mask_list, label_list = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, amp):
            outputs = model(batch)
        logits_list.append(outputs["class_logits"].float().cpu().numpy())
        mean_list.append(outputs["regression_mean"].float().cpu().numpy())
        target_list.append(batch["targets"].float().cpu().numpy())
        mask_list.append(batch["target_mask"].cpu().numpy())
        label_list.append(batch["labels"].cpu().numpy())
    local = {
        "logits": np.concatenate(logits_list, axis=0),
        "mean_normalized": np.concatenate(mean_list, axis=0),
        "target_normalized": np.concatenate(target_list, axis=0),
        "mask": np.concatenate(mask_list, axis=0),
        "labels": np.concatenate(label_list, axis=0),
    }
    payload = gather_payload(local)
    median = normalizers["target_median"].cpu().numpy()
    scale = normalizers["target_scale"].cpu().numpy()
    pred_transformed = payload["mean_normalized"] * scale + median
    target_transformed = payload["target_normalized"] * scale + median
    prediction = np.zeros_like(pred_transformed)
    target = np.zeros_like(target_transformed)
    for index, spec in enumerate(REGRESSION_TARGETS):
        prediction[:, index] = inverse_target(
            pred_transformed[:, index], spec.transform
        )
        target[:, index] = inverse_target(target_transformed[:, index], spec.transform)
    payload["prediction"] = prediction
    payload["target"] = target
    metrics = classification_metrics(payload["logits"], payload["labels"])
    metrics.update(
        regression_metrics(
            prediction,
            target,
            payload["mask"],
            [spec.name for spec in REGRESSION_TARGETS],
        )
    )
    metrics["selection_score"] = selection_score(metrics)
    return metrics, payload


# 中文：顶层接口 `collect_embedding_bank`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `collect_embedding_bank`; review type hints and callers before extending it.
@torch.no_grad()
def collect_embedding_bank(
    model: PeriodicNFEModel,
    dataset: NFEDataset,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    bank_size: int,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_graphs,
    )
    model.eval()
    embeddings = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        embeddings.append(output["embedding"].float().cpu())
    all_embeddings = torch.cat(embeddings, dim=0)
    mean = all_embeddings.mean(dim=0)
    std = all_embeddings.std(dim=0).clamp_min(1e-5)
    normalized = (all_embeddings - mean) / std
    if normalized.shape[0] > bank_size:
        indices = torch.linspace(
            0, normalized.shape[0] - 1, bank_size
        ).round().long()
        normalized = normalized[indices]
    z_rms = torch.sqrt(torch.mean(normalized**2, dim=1))
    nearest_parts = []
    chunk_size = 256
    for start in range(0, normalized.shape[0], chunk_size):
        stop = min(start + chunk_size, normalized.shape[0])
        distance = torch.cdist(normalized[start:stop], normalized)
        row = torch.arange(stop - start)
        col = torch.arange(start, stop)
        distance[row, col] = torch.inf
        nearest_parts.append(distance.min(dim=1).values)
    nearest = torch.cat(nearest_parts)
    return {
        "embedding_mean": mean,
        "embedding_std": std,
        "embedding_bank": normalized,
        "embedding_z_rms_q99": torch.quantile(z_rms, 0.99),
        "embedding_nearest_q95": torch.quantile(nearest, 0.95),
        "embedding_nearest_q99": torch.quantile(nearest, 0.99),
    }


def _capture_local_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _gather_rng_state_by_rank() -> list[dict[str, Any]] | None:
    state = _capture_local_rng_state()
    if dist.is_available() and dist.is_initialized():
        gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, state)
        if dist.get_rank() != 0:
            return None
        if any(item is None for item in gathered):
            raise RuntimeError("failed to gather complete per-rank RNG state")
        return [item for item in gathered if item is not None]
    return [state]


def _restore_rng_state(checkpoint: dict[str, Any], rank: int) -> None:
    states = checkpoint.get("rng_state_by_rank")
    if not isinstance(states, list) or rank >= len(states):
        raise RuntimeError("formal resume requires last.pt with complete per-rank RNG state")
    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("cuda", [])
    if torch.cuda.is_available():
        if not cuda_states:
            raise RuntimeError("CUDA formal resume checkpoint lacks CUDA RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


# 中文：顶层接口 `checkpoint_payload`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `checkpoint_payload`; review type hints and callers before extending it.
def checkpoint_payload(
    *,
    model: PeriodicNFEModel,
    config: dict[str, Any],
    normalizers: dict[str, torch.Tensor],
    epoch: int,
    metrics: dict[str, float],
    seen_elements: Sequence[int],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any | None = None,
    early_stopping: EarlyStopping | None = None,
    checkpoint_purpose: str = "best_model_selection",
    rng_state_by_rank: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "format": "nfe-mxene-predictor-1.0",
        "model_state": model.state_dict(),
        "model_config": model.config,
        "config": config,
        "normalizers": {key: value.cpu() for key, value in normalizers.items()},
        "target_specs": [spec.__dict__ for spec in REGRESSION_TARGETS],
        "label_map": INDEX_TO_LABEL,
        "epoch": epoch,
        "metrics": metrics,
        "seen_elements": list(sorted(set(int(x) for x in seen_elements))),
        "checkpoint_purpose": checkpoint_purpose,
    }
    if rng_state_by_rank is not None:
        payload["rng_state_by_rank"] = rng_state_by_rank
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if early_stopping is not None:
        payload["early_stopping"] = {
            "best": early_stopping.best,
            "bad_epochs": early_stopping.bad_epochs,
        }
    return payload


# 中文：顶层接口 `fit_temperature`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `fit_temperature`; review type hints and callers before extending it.
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    if int(np.sum(valid)) < 3:
        return 1.0
    logits_tensor = torch.tensor(logits[valid], dtype=torch.float32)
    labels_tensor = torch.tensor(labels[valid], dtype=torch.long)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=75,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0))


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the periodic equivariant NFE model.")
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = resolve_config_paths(load_config(config_path), config_path)
    rank, world_size, local_rank, device = init_distributed()
    seed_everything(int(config["seed"]), rank)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_config = config["data"]
    train_config = config["training"]
    loss_config = config["loss"]
    checkpoint_dir = Path(train_config["checkpoint_dir"])
    if is_main_process():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        load_or_build_cache(
            data_config["table"],
            data_config["root"],
            data_config["cache"],
            radius=float(data_config["radius"]),
            max_neighbors=int(data_config["max_neighbors"]),
            rebuild=bool(args.rebuild_cache or data_config.get("rebuild_cache", False)),
        )
    barrier()
    cache = torch_load_compat(data_config["cache"])
    records = cache["records"]
    skip_fraction = len(cache.get("skipped", [])) / max(
        1, len(records) + len(cache.get("skipped", []))
    )
    if skip_fraction > float(data_config.get("max_cache_skip_fraction", 0.01)):
        examples = "; ".join(
            f"{item['id']}: {item['error']}"
            for item in cache.get("skipped", [])[:5]
        )
        raise RuntimeError(
            f"graph cache skipped {skip_fraction:.2%} of structures, exceeding "
            f"the configured limit; examples: {examples}"
        )
    splits = split_indices(records)
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise RuntimeError(
            f"empty split detected: { {key: len(value) for key, value in splits.items()} }"
        )
    assert_disjoint_split_groups(records, splits)
    normalizers = robust_normalizers(records, splits["train"])
    train_dataset = NFEDataset(records, splits["train"], normalizers)
    validation_dataset = NFEDataset(records, splits["validation"], normalizers)
    test_dataset = NFEDataset(records, splits["test"], normalizers)
    distributed = world_size > 1
    loader_kwargs = {
        "batch_size": int(train_config["batch_size_per_gpu"]),
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": bool(data_config["pin_memory"]),
        "distributed": distributed,
    }
    train_loader, train_sampler = make_loader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    validation_loader, _ = make_loader(
        validation_dataset, shuffle=False, **loader_kwargs
    )
    test_loader, _ = make_loader(test_dataset, shuffle=False, **loader_kwargs)

    model = PeriodicNFEModel(
        **config["model"], num_regression_targets=len(REGRESSION_TARGETS)
    ).to(device)
    raw_model = model
    if args.resume:
        resume = torch_load_compat(args.resume, map_location="cpu")
        if resume.get("checkpoint_purpose") != "last_resume":
            raise ValueError(
                "formal --resume accepts only last.pt "
                "(checkpoint_purpose=last_resume), not best.pt"
            )
        raw_model.load_state_dict(resume["model_state"])
    train_module: torch.nn.Module = raw_model
    if bool(train_config.get("compile", False)) and hasattr(torch, "compile"):
        train_module = torch.compile(raw_model, dynamic=True)
    if distributed:
        train_module = DDP(
            train_module,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    accumulation = int(train_config["grad_accum_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = int(train_config["epochs"]) * updates_per_epoch
    warmup_steps = int(train_config["warmup_epochs"]) * updates_per_epoch
    scheduler = cosine_schedule(
        optimizer,
        total_steps,
        warmup_steps,
        float(train_config["min_learning_rate"])
        / float(train_config["learning_rate"]),
    )
    amp = bool(train_config["amp"] and device.type == "cuda")
    scaler = make_grad_scaler(amp)
    class_weights_tensor = class_weights(records, splits["train"]).to(device)
    target_weights = torch.tensor(
        [
            float(loss_config["score_weight"])
            if index == 0
            else (1.0 if spec.main else float(loss_config["auxiliary_weight"]))
            for index, spec in enumerate(REGRESSION_TARGETS)
        ],
        device=device,
    )
    early_stopping = EarlyStopping(
        int(train_config["early_stopping_patience"]), mode="max"
    )
    if args.resume:
        if "optimizer_state" in resume:
            optimizer.load_state_dict(resume["optimizer_state"])
        if "scheduler_state" in resume:
            scheduler.load_state_dict(resume["scheduler_state"])
        if "scaler_state" in resume:
            scaler.load_state_dict(resume["scaler_state"])
        if "early_stopping" in resume:
            early_stopping.best = float(resume["early_stopping"]["best"])
            early_stopping.bad_epochs = int(
                resume["early_stopping"]["bad_epochs"]
            )
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    log_path = checkpoint_dir / "history.jsonl"
    seen_elements = sorted(
        {
            atomic_number
            for index in splits["train"]
            for atomic_number in records[index]["elements"]
        }
    )
    if is_main_process():
        print(
            json.dumps(
                {
                    "device": str(device),
                    "world_size": world_size,
                    "parameters": raw_model.parameter_count(),
                    "records": len(records),
                    "skipped_cache_records": len(cache.get("skipped", [])),
                    "splits": {key: len(value) for key, value in splits.items()},
                    "class_weights": class_weights_tensor.cpu().tolist(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    start_epoch = 0
    if args.resume:
        start_epoch = int(resume.get("epoch", -1)) + 1
        _restore_rng_state(resume, rank)
    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(start_epoch, int(train_config["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_module.train()
        running = torch.zeros(6, device=device)
        pretraining = epoch < int(train_config["pretrain_epochs"])
        epoch_start = time.time()
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            z_corrupted, noisy_pos, masked_nodes, denoise_target = corrupt_structure(
                batch,
                mask_probability=float(loss_config["mask_probability"]),
                noise_min=float(loss_config["coordinate_noise_min_A"]),
                noise_max=float(loss_config["coordinate_noise_max_A"]),
            )
            should_step = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            sync_context = (
                train_module.no_sync()
                if distributed and not should_step
                else contextlib.nullcontext()
            )
            with sync_context:
                with autocast_context(device, amp):
                    outputs = train_module(
                        batch,
                        z_override=z_corrupted,
                        frac_pos_override=noisy_pos,
                    )
                    loss, components = compute_loss(
                        outputs,
                        batch,
                        class_weights_tensor=class_weights_tensor,
                        target_weights=target_weights,
                        loss_config=loss_config,
                        masked_nodes=masked_nodes,
                        denoise_target=denoise_target,
                        pretraining=pretraining,
                    )
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), float(train_config["grad_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running[:5] += torch.tensor(
                [
                    components["loss"],
                    components["class_loss"],
                    components["regression_loss"],
                    components["masked_loss"],
                    components["denoise_loss"],
                ],
                device=device,
            )
            running[5] += 1
            if (
                is_main_process()
                and (step + 1) % int(train_config["log_every"]) == 0
            ):
                print(
                    f"epoch={epoch:03d} step={step + 1:04d}/{len(train_loader)} "
                    f"loss={components['loss']:.5f} lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )

        running = reduce_sum(running)
        train_averages = (running[:5] / running[5].clamp_min(1)).cpu().tolist()
        validation_metrics, _ = evaluate(
            train_module, validation_loader, device, normalizers, amp
        )
        improved = False
        if is_main_process():
            improved, stop_training = early_stopping.update(
                validation_metrics["selection_score"]
            )
        if distributed:
            control = torch.tensor(
                [int(improved), int(stop_training)], device=device, dtype=torch.int32
            )
            dist.broadcast(control, src=0)
            improved = bool(control[0].item())
            stop_training = bool(control[1].item())
        rng_state_by_rank = _gather_rng_state_by_rank()
        if is_main_process():
            record = {
                "epoch": epoch,
                "phase": "pretrain" if pretraining else "finetune",
                "seconds": time.time() - epoch_start,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_loss": train_averages[0],
                "train_class_loss": train_averages[1],
                "train_regression_loss": train_averages[2],
                "train_masked_loss": train_averages[3],
                "train_denoise_loss": train_averages[4],
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if improved:
                atomic_torch_save(
                    checkpoint_payload(
                        model=raw_model,
                        config=config,
                        normalizers=normalizers,
                        epoch=epoch,
                        metrics=validation_metrics,
                        seen_elements=seen_elements,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        early_stopping=early_stopping,
                        checkpoint_purpose="best_model_selection",
                        rng_state_by_rank=rng_state_by_rank,
                    ),
                    best_path,
                )
            atomic_torch_save(
                checkpoint_payload(
                    model=raw_model,
                    config=config,
                    normalizers=normalizers,
                    epoch=epoch,
                    metrics=validation_metrics,
                    seen_elements=seen_elements,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    early_stopping=early_stopping,
                    checkpoint_purpose="last_resume",
                    rng_state_by_rank=rng_state_by_rank,
                ),
                last_path,
            )
        if stop_training:
            break

    barrier()
    # Keep optimizer/checkpoint tensors on CPU while copying only model weights
    # to the GPU; this avoids a transient memory spike at final evaluation.
    best = torch_load_compat(best_path, map_location="cpu")
    raw_model.load_state_dict(best["model_state"])
    test_metrics, test_payload = evaluate(
        train_module, test_loader, device, normalizers, amp
    )
    validation_metrics, validation_payload = evaluate(
        train_module, validation_loader, device, normalizers, amp
    )
    if is_main_process():
        embedding_stats = collect_embedding_bank(
            raw_model,
            train_dataset,
            device,
            batch_size=int(train_config["batch_size_per_gpu"]),
            num_workers=int(data_config["num_workers"]),
            bank_size=int(config["inference"]["embedding_bank_size"]),
        )
        score_valid = validation_payload["mask"][:, 0].astype(bool)
        score_residual = np.abs(
            validation_payload["prediction"][score_valid, 0]
            - validation_payload["target"][score_valid, 0]
        )
        conformal_quantile = (
            float(
                np.quantile(
                    score_residual,
                    float(config["inference"]["confidence_level"]),
                    method="higher",
                )
            )
            if len(score_residual)
            else 0.25
        )
        temperature = fit_temperature(
            validation_payload["logits"], validation_payload["labels"]
        )
        validation_calibrated = classification_metrics(
            validation_payload["logits"] / temperature,
            validation_payload["labels"],
        )
        test_calibrated = classification_metrics(
            test_payload["logits"] / temperature,
            test_payload["labels"],
        )
        best.update(embedding_stats)
        best["conformal_score_radius"] = conformal_quantile
        best["classification_temperature"] = temperature
        best["validation_metrics"] = validation_metrics
        best["test_metrics"] = test_metrics
        best["validation_calibrated_metrics"] = validation_calibrated
        best["test_calibrated_metrics"] = test_calibrated
        atomic_torch_save(best, best_path)
        save_json(
            checkpoint_dir / "final_metrics.json",
            {
                "best_epoch": best["epoch"],
                "validation": validation_metrics,
                "test": test_metrics,
                "classification_temperature": temperature,
                "validation_calibrated": validation_calibrated,
                "test_calibrated": test_calibrated,
                "conformal_score_radius": conformal_quantile,
            },
        )
        print(
            json.dumps(
                {
                    "training_complete": True,
                    "best_checkpoint": str(best_path),
                    "resume_checkpoint": str(last_path),
                    "test_metrics": test_metrics,
                    "test_calibrated_metrics": test_calibrated,
                    "classification_temperature": temperature,
                    "conformal_score_radius": conformal_quantile,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
