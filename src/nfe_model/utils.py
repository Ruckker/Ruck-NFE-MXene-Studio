# ==============================================================================
# 中文概述：配置、随机种子、分布式通信、学习率、早停与原子写盘工具。
# English overview: Utilities for config, seeds, distributed communication, schedules, early stopping, and atomic saves.
#
# 中文输入：训练配置、进程环境变量和 PyTorch 张量。
# English inputs: Training config, process environment variables, and PyTorch tensors.
# 中文输出：可复现运行状态、DDP 设备信息和安全检查点。
# English outputs: Reproducible run state, DDP device information, and safe checkpoints.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: load_config, save_json, seed_everything, distributed_info, init_distributed, cleanup_distributed, is_main_process, barrier, reduce_sum, cosine_schedule, EarlyStopping, atomic_torch_save
#
# Author: Ruck
# Generated: 2026-07-29 19:25:36 Asia/Shanghai
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


# 中文：顶层接口 `load_config`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `load_config`; review type hints and callers before extending it.
def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# 中文：顶层接口 `save_json`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `save_json`; review type hints and callers before extending it.
def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


# 中文：顶层接口 `seed_everything`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `seed_everything`; review type hints and callers before extending it.
def seed_everything(seed: int, rank: int = 0) -> None:
    final_seed = seed + 1009 * rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


# 中文：顶层接口 `distributed_info`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `distributed_info`; review type hints and callers before extending it.
def distributed_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


# 中文：顶层接口 `init_distributed`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `init_distributed`; review type hints and callers before extending it.
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


# 中文：顶层接口 `cleanup_distributed`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `cleanup_distributed`; review type hints and callers before extending it.
def cleanup_distributed() -> None:
    if dist.is_initialized():
        barrier()
        dist.destroy_process_group()


# 中文：顶层接口 `is_main_process`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `is_main_process`; review type hints and callers before extending it.
def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# 中文：顶层接口 `barrier`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `barrier`; review type hints and callers before extending it.
def barrier() -> None:
    if dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


# 中文：顶层接口 `reduce_sum`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `reduce_sum`; review type hints and callers before extending it.
def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    value = value.clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


# 中文：顶层接口 `cosine_schedule`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `cosine_schedule`; review type hints and callers before extending it.
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


# 中文：顶层类 `EarlyStopping`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `EarlyStopping`; review type hints and callers before extending it.
class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


# 中文：顶层接口 `atomic_torch_save`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `atomic_torch_save`; review type hints and callers before extending it.
def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, path)
