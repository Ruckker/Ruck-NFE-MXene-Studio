# ==============================================================================
# 中文概述：配置、随机种子、分布式通信、学习率、早停与原子写盘工具。
# English overview: Utilities for config, seeds, distributed communication, schedules, early stopping, and atomic saves.
# ==============================================================================

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def json_safe(value: Any) -> Any:
    """Convert scientific scalar containers to strict-JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


def seed_everything(seed: int, rank: int = 0) -> None:
    final_seed = seed + 1009 * rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def distributed_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank, world_size, local_rank = distributed_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    value = value.clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> tuple[bool, bool]:
        numeric = float(value)
        if not math.isfinite(numeric):
            self.bad_epochs += 1
            return False, self.bad_epochs >= self.patience
        improved = numeric > self.best if self.mode == "max" else numeric < self.best
        if improved:
            self.best = numeric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, path)
