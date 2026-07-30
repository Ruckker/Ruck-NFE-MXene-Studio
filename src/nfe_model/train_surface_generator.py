# ==============================================================================
# 中文概述：训练 surface generator 表面模板流，并显式优化端点、层序、锚点、OH 和配对损失。
# English overview: Train the surface-template flow with endpoint, layer, anchor, OH, and pair losses.
#
# 中文输入：surface generator 配置、表面模板记录、几何先验与 1–4 张 GPU。
# English inputs: surface generator config, surface-template records, geometry priors, and one to four GPUs.
# 中文输出：surface generator 最佳检查点、历史和分解后的物理验证指标。
# English outputs: surface generator best checkpoint, history, and decomposed physical validation metrics.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: make_loader, xy_minimum_delta, flow_corruption, repulsion_loss, endpoint_and_topology_losses, compute_flow_loss, condition_dropout, evaluate, parse_args, main
#
# Author: Ruck
# Generated: 2026-07-29 22:30:47 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import (
    assert_disjoint_split_groups,
    load_or_build_cache,
    move_batch,
    split_indices,
    torch_load_compat,
)
from .generator_data import (
    center_slab_fractional_tensor,
    composition_catalog,
    novelty_reference,
    params_to_lattice,
)
from .surface_generator_data import (
    GROUP_ATOMIC_TERMINATION,
    GROUP_OH_HYDROGEN,
    GROUP_OH_OXYGEN,
    GROUP_SURFACE_HYDROGEN,
    SurfaceTemplateDataset,
    collate_surface_templates,
    lattice_normalizers,
    prepare_surface_generator_records,
    surface_template_catalog,
)
from .surface_generator import (
    SurfaceAwareTemplateFlow,
    surface_coordinate_length_scale,
)
from .train import autocast_context, make_grad_scaler, resolve_config_paths
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

METRIC_KEYS = (
    "loss",
    "coordinate_loss",
    "lattice_loss",
    "repulsion_loss",
    "endpoint_loss",
    "endpoint_rmse_A",
    "core_mae_A",
    "surface_mae_A",
    "pair_loss",
    "layer_loss",
    "anchor_loss",
    "oh_loss",
)


# 中文：顶层接口 `make_loader`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `make_loader`; review type hints and callers before extending it.
def make_loader(
    dataset: SurfaceTemplateDataset,
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
        collate_fn=collate_surface_templates,
        drop_last=False,
    )
    return loader, sampler


# 中文：顶层接口 `xy_minimum_delta`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `xy_minimum_delta`; review type hints and callers before extending it.
def xy_minimum_delta(delta: torch.Tensor) -> torch.Tensor:
    result = delta.clone()
    result[..., :2] = torch.remainder(result[..., :2] + 0.5, 1.0) - 0.5
    return result


# 中文：顶层接口 `flow_corruption`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `flow_corruption`; review type hints and callers before extending it.
def flow_corruption(
    batch: dict[str, torch.Tensor],
    normalizers: dict[str, torch.Tensor],
    loss_config: dict[str, float],
) -> dict[str, torch.Tensor]:
    n_graphs = len(batch["target_lattice"])
    device = batch["target_frac"].device
    time = torch.rand(n_graphs, device=device).clamp(1e-4, 1.0 - 1e-4)
    target_physical_params = (
        batch["target_lattice"] * normalizers["lattice_scale"]
        + normalizers["lattice_median"]
    )
    target_physical_lattice = params_to_lattice(target_physical_params)
    template_physical_params = (
        batch["template_lattice"] * normalizers["lattice_scale"]
        + normalizers["lattice_median"]
    )
    template_physical_lattice = params_to_lattice(template_physical_params)

    core_noise = float(loss_config.get("core_template_noise_A", 0.18))
    surface_noise = float(loss_config.get("surface_template_noise_A", 0.35))
    hydrogen_noise = float(loss_config.get("hydrogen_template_noise_A", 0.16))
    node_sigma = torch.full(
        (len(batch["z"]),), core_noise, device=device, dtype=torch.float32
    )
    node_sigma = torch.where(
        batch["surface_side"] != 0,
        torch.full_like(node_sigma, surface_noise),
        node_sigma,
    )
    node_sigma = torch.where(
        batch["group_type"] == GROUP_OH_HYDROGEN,
        torch.full_like(node_sigma, hydrogen_noise),
        node_sigma,
    )
    cartesian_noise = torch.randn_like(batch["template_frac"]) * node_sigma.unsqueeze(-1)
    fractional_noise = torch.einsum(
        "ni,nij->nj",
        cartesian_noise,
        torch.linalg.inv(template_physical_lattice)[batch["batch"]],
    )
    base_frac = batch["template_frac"] + fractional_noise
    base_frac = base_frac.clone()
    base_frac[:, :2] = torch.remainder(base_frac[:, :2], 1.0)
    base_frac[:, 2] = base_frac[:, 2].clamp(0.05, 0.95)
    base_frac = center_slab_fractional_tensor(base_frac, batch["batch"])

    delta_frac = xy_minimum_delta(batch["target_frac"] - base_frac)
    current_frac = torch.remainder(
        base_frac + time[batch["batch"]].unsqueeze(-1) * delta_frac,
        1.0,
    )
    current_frac = center_slab_fractional_tensor(current_frac, batch["batch"])

    lattice_mask = torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        device=device,
        dtype=batch["target_lattice"].dtype,
    )
    lattice_noise = float(loss_config.get("lattice_template_noise", 0.10))
    base_lattice = (
        batch["template_lattice"]
        + torch.randn_like(batch["template_lattice"])
        * lattice_noise
        * lattice_mask
    ).clamp(-8.0, 8.0)
    lattice_velocity = (
        batch["target_lattice"] - base_lattice
    ) * lattice_mask
    current_lattice_state = (
        base_lattice
        + time.unsqueeze(-1) * lattice_velocity
    ).clamp(-8.0, 8.0)
    physical_params = (
        current_lattice_state * normalizers["lattice_scale"]
        + normalizers["lattice_median"]
    )
    lattice = params_to_lattice(physical_params)
    coordinate_velocity_cart = torch.einsum(
        "ni,nij->nj",
        delta_frac,
        lattice[batch["batch"]],
    )
    coordinate_scale = surface_coordinate_length_scale(
        lattice, batch["batch"]
    )
    return {
        "time": time,
        "frac_pos": current_frac,
        "lattice_state": current_lattice_state,
        "lattice": lattice,
        "coordinate_target": coordinate_velocity_cart
        / coordinate_scale[batch["batch"]].unsqueeze(-1),
        "coordinate_scale": coordinate_scale,
        "lattice_target": lattice_velocity,
        "lattice_mask": lattice_mask,
        "target_frac": batch["target_frac"],
        "target_physical_lattice": target_physical_lattice,
    }


# 中文：顶层接口 `repulsion_loss`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `repulsion_loss`; review type hints and callers before extending it.
def repulsion_loss(
    frac_pos: torch.Tensor,
    lattice: torch.Tensor,
    batch_index: torch.Tensor,
    atom_features: torch.Tensor,
    sample_weights: torch.Tensor,
    minimum_factor: float,
) -> torch.Tensor:
    penalties = []
    weights = []
    shifts = torch.tensor(
        [
            [i, j, 0.0]
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
        ],
        device=frac_pos.device,
        dtype=frac_pos.dtype,
    )
    for graph_index in range(lattice.shape[0]):
        nodes = torch.where(batch_index == graph_index)[0]
        if len(nodes) <= 1:
            continue
        local_frac = frac_pos[nodes]
        upper = torch.triu_indices(
            len(nodes), len(nodes), offset=1, device=frac_pos.device
        )
        raw_delta = local_frac[upper[0]] - local_frac[upper[1]]
        candidates = raw_delta.unsqueeze(1) + shifts.unsqueeze(0)
        cartesian_candidates = torch.einsum(
            "pki,ij->pkj", candidates, lattice[graph_index]
        )
        distance = torch.linalg.vector_norm(
            cartesian_candidates, dim=-1
        ).min(dim=1).values.clamp_min(1e-6)
        # Descriptor index 5 stores atomic radius / 3.
        radii = atom_features[nodes, 5] * 3.0
        minimum = float(minimum_factor) * (
            radii[upper[0]] + radii[upper[1]]
        ).clamp_min(1.2)
        penalties.append(torch.relu(minimum - distance).square().mean())
        weights.append(sample_weights[graph_index])
    if not penalties:
        return frac_pos.sum() * 0.0
    values = torch.stack(penalties)
    graph_weights = torch.stack(weights)
    return torch.sum(values * graph_weights) / graph_weights.sum().clamp_min(1e-6)


# 中文：顶层接口 `endpoint_and_topology_losses`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `endpoint_and_topology_losses`; review type hints and callers before extending it.
def endpoint_and_topology_losses(
    predicted_frac: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target_lattice: torch.Tensor,
) -> dict[str, torch.Tensor]:
    graph_weights = batch["sample_weights"]
    node_weights = batch["role_weight"] * graph_weights[batch["batch"]]
    endpoint_delta = xy_minimum_delta(predicted_frac - batch["target_frac"])
    endpoint_cartesian = torch.einsum(
        "ni,nij->nj", endpoint_delta, target_lattice[batch["batch"]]
    )
    endpoint_distance = torch.linalg.vector_norm(endpoint_cartesian, dim=-1)
    endpoint_loss = torch.sum(
        F.smooth_l1_loss(
            endpoint_distance,
            torch.zeros_like(endpoint_distance),
            beta=0.25,
            reduction="none",
        )
        * node_weights
    ) / node_weights.sum().clamp_min(1e-6)
    endpoint_rmse = torch.sqrt(
        torch.sum(endpoint_distance.square() * node_weights)
        / node_weights.sum().clamp_min(1e-6)
    )
    core_mask = batch["surface_side"] == 0
    surface_mask = ~core_mask
    core_mae = (
        endpoint_distance[core_mask].mean()
        if torch.any(core_mask)
        else endpoint_distance.sum() * 0.0
    )
    surface_mae = (
        endpoint_distance[surface_mask].mean()
        if torch.any(surface_mask)
        else endpoint_distance.sum() * 0.0
    )

    pair_losses, layer_losses, anchor_errors, oh_errors = [], [], [], []
    pair_weights = []
    for graph_index in range(target_lattice.shape[0]):
        nodes = torch.where(batch["batch"] == graph_index)[0]
        if len(nodes) <= 1:
            continue
        upper = torch.triu_indices(
            len(nodes), len(nodes), offset=1, device=predicted_frac.device
        )
        left, right = nodes[upper[0]], nodes[upper[1]]
        target_delta = xy_minimum_delta(
            batch["target_frac"][left] - batch["target_frac"][right]
        )
        predicted_delta = xy_minimum_delta(
            predicted_frac[left] - predicted_frac[right]
        )
        target_cart = target_delta @ target_lattice[graph_index]
        predicted_cart = predicted_delta @ target_lattice[graph_index]
        target_distance = torch.linalg.vector_norm(target_cart, dim=-1)
        predicted_distance = torch.linalg.vector_norm(predicted_cart, dim=-1)
        local_mask = target_distance <= 4.25
        if torch.any(local_mask):
            pair_losses.append(
                F.smooth_l1_loss(
                    predicted_distance[local_mask],
                    target_distance[local_mask],
                    beta=0.15,
                )
            )
            pair_weights.append(graph_weights[graph_index])
        target_z = target_cart[:, 2]
        predicted_z = predicted_cart[:, 2]
        ordered = target_z.abs() >= 0.25
        if torch.any(ordered):
            sign = torch.sign(target_z[ordered])
            violation = torch.relu(0.15 - sign * predicted_z[ordered])
            layer_losses.append(violation.square().mean())

        leader_mask = (
            (batch["group_type"][nodes] == GROUP_ATOMIC_TERMINATION)
            | (batch["group_type"][nodes] == GROUP_OH_OXYGEN)
            | (batch["group_type"][nodes] == GROUP_SURFACE_HYDROGEN)
        )
        leaders = nodes[leader_mask]
        leaders = leaders[batch["anchor_index"][leaders] >= 0]
        if len(leaders):
            anchors = batch["anchor_index"][leaders]
            target_anchor_delta = xy_minimum_delta(
                batch["target_frac"][leaders]
                - batch["target_frac"][anchors]
            )
            predicted_anchor_delta = xy_minimum_delta(
                predicted_frac[leaders] - predicted_frac[anchors]
            )
            target_anchor = torch.linalg.vector_norm(
                target_anchor_delta @ target_lattice[graph_index], dim=-1
            )
            predicted_anchor = torch.linalg.vector_norm(
                predicted_anchor_delta @ target_lattice[graph_index], dim=-1
            )
            anchor_errors.append(
                F.smooth_l1_loss(
                    predicted_anchor, target_anchor, beta=0.10
                )
            )

        for side in (-1, 1):
            oxygen = nodes[
                (batch["surface_side"][nodes] == side)
                & (batch["group_type"][nodes] == GROUP_OH_OXYGEN)
            ]
            hydrogen = nodes[
                (batch["surface_side"][nodes] == side)
                & (batch["group_type"][nodes] == GROUP_OH_HYDROGEN)
            ]
            if len(oxygen) == 1 and len(hydrogen) == 1:
                target_oh_delta = xy_minimum_delta(
                    batch["target_frac"][hydrogen]
                    - batch["target_frac"][oxygen]
                )
                predicted_oh_delta = xy_minimum_delta(
                    predicted_frac[hydrogen] - predicted_frac[oxygen]
                )
                target_oh = torch.linalg.vector_norm(
                    target_oh_delta @ target_lattice[graph_index], dim=-1
                )
                predicted_oh = torch.linalg.vector_norm(
                    predicted_oh_delta @ target_lattice[graph_index], dim=-1
                )
                oh_errors.append(
                    F.smooth_l1_loss(predicted_oh, target_oh, beta=0.04)
                )

    zero = endpoint_loss * 0.0
    if pair_losses:
        pair_values = torch.stack(pair_losses)
        pair_weight_values = torch.stack(pair_weights)
        pair_loss = torch.sum(pair_values * pair_weight_values) / pair_weight_values.sum().clamp_min(1e-6)
    else:
        pair_loss = zero
    layer_loss = torch.stack(layer_losses).mean() if layer_losses else zero
    anchor_loss = torch.stack(anchor_errors).mean() if anchor_errors else zero
    oh_loss = torch.stack(oh_errors).mean() if oh_errors else zero
    return {
        "endpoint_loss": endpoint_loss,
        "endpoint_rmse_A": endpoint_rmse,
        "core_mae_A": core_mae,
        "surface_mae_A": surface_mae,
        "pair_loss": pair_loss,
        "layer_loss": layer_loss,
        "anchor_loss": anchor_loss,
        "oh_loss": oh_loss,
    }


# 中文：顶层接口 `compute_flow_loss`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `compute_flow_loss`; review type hints and callers before extending it.
def compute_flow_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    flow: dict[str, torch.Tensor],
    normalizers: dict[str, torch.Tensor],
    loss_config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    graph_weights = batch["sample_weights"]
    node_weights = graph_weights[batch["batch"]]
    coordinate_error = (
        outputs["coordinate_velocity_cart"] - flow["coordinate_target"]
    ).square().sum(dim=-1)
    node_weights = node_weights * batch["role_weight"]
    coordinate_loss = torch.sum(
        coordinate_error * node_weights
    ) / node_weights.sum().clamp_min(1e-6)
    lattice_error = (
        (outputs["lattice_velocity"] - flow["lattice_target"])
        * flow["lattice_mask"]
    ).square().mean(dim=-1)
    lattice_loss = torch.sum(lattice_error * graph_weights) / graph_weights.sum().clamp_min(
        1e-6
    )

    inverse_lattice = torch.linalg.inv(flow["lattice"])
    predicted_fractional_velocity = torch.einsum(
        "ni,nij->nj",
        outputs["coordinate_velocity_cart"]
        * flow["coordinate_scale"][batch["batch"]].unsqueeze(-1),
        inverse_lattice[batch["batch"]],
    )
    remaining = 1.0 - flow["time"]
    predicted_final_frac = torch.remainder(
        flow["frac_pos"]
        + remaining[batch["batch"]].unsqueeze(-1)
        * predicted_fractional_velocity,
        1.0,
    )
    predicted_final_state = (
        flow["lattice_state"]
        + remaining.unsqueeze(-1)
        * outputs["lattice_velocity"]
        * flow["lattice_mask"]
    ).clamp(-8.0, 8.0)
    predicted_final_params = (
        predicted_final_state * normalizers["lattice_scale"]
        + normalizers["lattice_median"]
    )
    predicted_final_lattice = params_to_lattice(predicted_final_params)
    predicted_final_frac = center_slab_fractional_tensor(
        predicted_final_frac, batch["batch"]
    )
    geometry_loss = repulsion_loss(
        predicted_final_frac,
        predicted_final_lattice,
        batch["batch"],
        batch["atom_features"],
        graph_weights,
        float(loss_config.get("minimum_distance_factor", 0.72)),
    )
    topology = endpoint_and_topology_losses(
        predicted_final_frac,
        batch,
        flow["target_physical_lattice"],
    )
    total = (
        float(loss_config["coordinate_weight"]) * coordinate_loss
        + float(loss_config["lattice_weight"]) * lattice_loss
        + float(loss_config["repulsion_weight"]) * geometry_loss
        + float(loss_config.get("endpoint_weight", 1.0))
        * topology["endpoint_loss"]
        + float(loss_config.get("pair_distance_weight", 0.8))
        * topology["pair_loss"]
        + float(loss_config.get("layer_order_weight", 0.5))
        * topology["layer_loss"]
        + float(loss_config.get("surface_anchor_weight", 0.8))
        * topology["anchor_loss"]
        + float(loss_config.get("oh_geometry_weight", 1.2))
        * topology["oh_loss"]
    )
    return total, {
        "loss": float(total.detach()),
        "coordinate_loss": float(coordinate_loss.detach()),
        "lattice_loss": float(lattice_loss.detach()),
        "repulsion_loss": float(geometry_loss.detach()),
        **{
            key: float(value.detach())
            for key, value in topology.items()
        },
    }


# 中文：顶层接口 `condition_dropout`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `condition_dropout`; review type hints and callers before extending it.
def condition_dropout(
    labels: torch.Tensor,
    scores: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    drop = torch.rand(len(labels), device=labels.device) < probability
    return (
        torch.where(drop, torch.full_like(labels, -1), labels),
        torch.where(drop, torch.zeros_like(scores), scores),
    )


# 中文：顶层接口 `evaluate`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `evaluate`; review type hints and callers before extending it.
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizers: dict[str, torch.Tensor],
    config: dict[str, Any],
    amp: bool,
) -> dict[str, float]:
    model.eval()
    running = torch.zeros(len(METRIC_KEYS) + 1, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        flow = flow_corruption(
            batch, normalizers, config["generator_loss"]
        )
        with autocast_context(device, amp):
            outputs = model(
                batch,
                flow["frac_pos"],
                flow["lattice_state"],
                flow["lattice"],
                flow["time"],
                batch["labels"],
                batch["scores"],
            )
            _, components = compute_flow_loss(
                outputs,
                batch,
                flow,
                normalizers,
                config["generator_loss"],
            )
        running[: len(METRIC_KEYS)] += torch.tensor(
            [components[key] for key in METRIC_KEYS],
            device=device,
        )
        running[-1] += 1
    running = reduce_sum(running)
    values = running[: len(METRIC_KEYS)] / running[-1].clamp_min(1)
    return {
        key: float(values[index])
        for index, key in enumerate(METRIC_KEYS)
    }


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train conditional periodic crystal flow matching."
    )
    parser.add_argument("--config", default="training/configs/surface_generator.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = resolve_config_paths(load_config(config_path), config_path)
    generator_training = config["generator_training"]
    checkpoint_dir = Path(generator_training["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = config_path.parent / checkpoint_dir
    checkpoint_dir = checkpoint_dir.resolve()

    rank, world_size, local_rank, device = init_distributed()
    seed_everything(int(config["seed"]) + 7919, rank)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_config = config["data"]
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
    base_records = cache["records"]
    splits = split_indices(base_records)
    if not all(splits.values()):
        raise RuntimeError(
            f"empty split detected: { {key: len(value) for key, value in splits.items()} }"
        )
    assert_disjoint_split_groups(base_records, splits)
    records = prepare_surface_generator_records(base_records)
    normalizers_cpu = lattice_normalizers(records, splits["train"])
    normalizers = {
        key: value.to(device)
        for key, value in normalizers_cpu.items()
    }
    train_dataset = SurfaceTemplateDataset(
        records, splits["train"], normalizers_cpu
    )
    validation_dataset = SurfaceTemplateDataset(
        records, splits["validation"], normalizers_cpu
    )
    test_dataset = SurfaceTemplateDataset(
        records, splits["test"], normalizers_cpu
    )
    distributed = world_size > 1
    loader_kwargs = {
        "batch_size": int(generator_training["batch_size_per_gpu"]),
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
    test_loader, _ = make_loader(
        test_dataset, shuffle=False, **loader_kwargs
    )

    model = SurfaceAwareTemplateFlow(**config["generator_model"]).to(device)
    raw_model = model
    resume = None
    if args.resume:
        resume = torch_load_compat(args.resume, map_location=device)
        raw_model.load_state_dict(resume["model_state"])
    train_module: torch.nn.Module = raw_model
    if distributed:
        train_module = DDP(
            raw_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(generator_training["learning_rate"]),
        weight_decay=float(generator_training["weight_decay"]),
    )
    accumulation = int(generator_training["grad_accum_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = int(generator_training["epochs"]) * updates_per_epoch
    warmup_steps = int(generator_training["warmup_epochs"]) * updates_per_epoch
    scheduler = cosine_schedule(
        optimizer,
        total_steps,
        warmup_steps,
        float(generator_training["min_learning_rate"])
        / float(generator_training["learning_rate"]),
    )
    amp = bool(generator_training["amp"] and device.type == "cuda")
    scaler = make_grad_scaler(amp)
    early_stopping = EarlyStopping(
        int(generator_training["early_stopping_patience"]), mode="min"
    )
    if resume is not None:
        if "optimizer_state" in resume:
            optimizer.load_state_dict(resume["optimizer_state"])
        if "scheduler_state" in resume:
            scheduler.load_state_dict(resume["scheduler_state"])
        if "scaler_state" in resume:
            scaler.load_state_dict(resume["scaler_state"])
    best_path = checkpoint_dir / "best_generator.pt"
    history_path = checkpoint_dir / "generator_history.jsonl"
    catalog = (
        composition_catalog(records, splits["train"])
        if is_main_process()
        else None
    )
    template_catalog = (
        surface_template_catalog(records, splits["train"])
        if is_main_process()
        else None
    )
    training_reference = (
        novelty_reference(records, splits["train"])
        if is_main_process()
        else None
    )
    start_epoch = int(resume.get("epoch", -1)) + 1 if resume else 0
    if is_main_process():
        print(
            json.dumps(
                {
                    "task": "surface_aware_template_flow",
                    "device": str(device),
                    "world_size": world_size,
                    "parameters": raw_model.parameter_count(),
                    "splits": {key: len(value) for key, value in splits.items()},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(start_epoch, int(generator_training["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_module.train()
        running = torch.zeros(len(METRIC_KEYS) + 1, device=device)
        started = time.time()
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            flow = flow_corruption(
                batch, normalizers, config["generator_loss"]
            )
            labels, scores = condition_dropout(
                batch["labels"],
                batch["scores"],
                float(config["generator_loss"]["condition_dropout"]),
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
                        flow["frac_pos"],
                        flow["lattice_state"],
                        flow["lattice"],
                        flow["time"],
                        labels,
                        scores,
                    )
                    loss, components = compute_flow_loss(
                        outputs,
                        batch,
                        flow,
                        normalizers,
                        config["generator_loss"],
                    )
                scaler.scale(loss / accumulation).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    float(generator_training["grad_clip"]),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running[: len(METRIC_KEYS)] += torch.tensor(
                [components[key] for key in METRIC_KEYS],
                device=device,
            )
            running[-1] += 1
        running = reduce_sum(running)
        train_values = (
            running[: len(METRIC_KEYS)] / running[-1].clamp_min(1)
        )
        validation = evaluate(
            train_module,
            validation_loader,
            device,
            normalizers,
            config,
            amp,
        )
        improved = False
        if is_main_process():
            improved, stop_training = early_stopping.update(validation["loss"])
            record = {
                "epoch": epoch,
                "seconds": time.time() - started,
                "learning_rate": scheduler.get_last_lr()[0],
                **{
                    f"train_{key}": float(train_values[index])
                    for index, key in enumerate(METRIC_KEYS)
                },
                **{f"val_{key}": value for key, value in validation.items()},
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if improved:
                atomic_torch_save(
                    {
                        "format": "nfe-mxene-surface-generator-1.0",
                        "model_state": raw_model.state_dict(),
                        "model_config": raw_model.config,
                        "config": config,
                        "normalizers": {
                            key: value.cpu() for key, value in normalizers.items()
                        },
                        "composition_catalog": catalog,
                        "surface_template_catalog": template_catalog,
                        "novelty_reference": training_reference,
                        "epoch": epoch,
                        "validation": validation,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                    },
                    best_path,
                )
        if distributed:
            control = torch.tensor(
                [int(improved), int(stop_training)],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(control, src=0)
            stop_training = bool(control[1].item())
        if stop_training:
            break

    barrier()
    best = torch_load_compat(best_path, map_location="cpu")
    raw_model.load_state_dict(best["model_state"])
    test_metrics = evaluate(
        train_module, test_loader, device, normalizers, config, amp
    )
    if is_main_process():
        best["test"] = test_metrics
        atomic_torch_save(best, best_path)
        save_json(
            checkpoint_dir / "generator_final_metrics.json",
            {
                "best_epoch": best["epoch"],
                "validation": best["validation"],
                "test": test_metrics,
            },
        )
        print(
            json.dumps(
                {
                    "training_complete": True,
                    "generator_checkpoint": str(best_path),
                    "test": test_metrics,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
