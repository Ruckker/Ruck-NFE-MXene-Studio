from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from . import train_surface_generator as core
from .data import assert_disjoint_split_groups
from .data_v2 import load_or_build_cache, move_batch, split_indices, torch_load_compat
from .generator_data import composition_catalog, novelty_reference
from .provenance_v2 import (
    assert_matching_provenance,
    build_provenance,
    canonical_sha256,
    file_sha256,
)
from .surface_generator import SurfaceAwareTemplateFlow
from .surface_generator_data import (
    SurfaceTemplateDataset,
    collate_surface_templates,
    lattice_normalizers,
    prepare_surface_generator_records,
    surface_template_catalog,
)
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


GENERATOR_CHECKPOINT_FORMAT = "nfe-mxene-surface-generator-1.1"
GENERATOR_PROTOCOL_SCHEMA = "surface-generator-training-protocol-1.0"


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without DistributedSampler padding duplicates."""

    def __init__(self, dataset: SurfaceTemplateDataset) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "ExactDistributedEvalSampler requires initialized distributed runtime"
            )
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


def make_loader(
    dataset: SurfaceTemplateDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
) -> tuple[DataLoader, Sampler[int] | None]:
    if distributed and shuffle:
        sampler: Sampler[int] | None = DistributedSampler(
            dataset, shuffle=True, drop_last=False
        )
    elif distributed:
        sampler = ExactDistributedEvalSampler(dataset)
    else:
        sampler = None
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
    running = torch.zeros(len(core.METRIC_KEYS) + 1, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        flow = core.flow_corruption(batch, normalizers, config["generator_loss"])
        with core.autocast_context(device, amp):
            outputs = model(
                batch,
                flow["frac_pos"],
                flow["lattice_state"],
                flow["lattice"],
                flow["time"],
                batch["labels"],
                batch["scores"],
            )
            _, components = core.compute_flow_loss(
                outputs,
                batch,
                flow,
                normalizers,
                config["generator_loss"],
            )
        support = float(len(batch["labels"]))
        running[: len(core.METRIC_KEYS)] += (
            torch.tensor(
                [components[key] for key in core.METRIC_KEYS],
                device=device,
            )
            * support
        )
        running[-1] += support
    running = reduce_sum(running)
    values = running[: len(core.METRIC_KEYS)] / running[-1].clamp_min(1)
    return {
        key: float(values[index])
        for index, key in enumerate(core.METRIC_KEYS)
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the formal v2 surface-aware conditional generator."
    )
    parser.add_argument("--config", default="training/configs/surface_generator.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def _generator_protocol(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": GENERATOR_PROTOCOL_SCHEMA,
        "generator_model": config["generator_model"],
        "generator_training": {
            key: value
            for key, value in config["generator_training"].items()
            if key != "checkpoint_dir"
        },
        "generator_loss": config["generator_loss"],
        "template_source_split": "train",
        "cache_schema": provenance["cache_schema"],
        "neighbor_policy": provenance["neighbor_policy"],
        "data_implementation_sha256": provenance["data_implementation_sha256"],
    }


def _checkpoint_payload(
    *,
    raw_model: SurfaceAwareTemplateFlow,
    config: dict[str, Any],
    normalizers: dict[str, torch.Tensor],
    catalog: Any,
    template_catalog: Any,
    training_reference: Any,
    epoch: int,
    validation: dict[str, float],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    provenance: dict[str, Any],
    generator_protocol: dict[str, Any],
    generator_protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "format": GENERATOR_CHECKPOINT_FORMAT,
        "model_state": raw_model.state_dict(),
        "model_config": raw_model.config,
        "config": config,
        "seed": int(config["seed"]),
        "parameter_count": raw_model.parameter_count(),
        "provenance": provenance,
        "generator_protocol": generator_protocol,
        "generator_protocol_sha256": generator_protocol_sha256,
        "template_source_split": "train",
        "normalizers": {key: value.cpu() for key, value in normalizers.items()},
        "composition_catalog": catalog,
        "surface_template_catalog": template_catalog,
        "novelty_reference": training_reference,
        "epoch": epoch,
        "validation": validation,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = core.resolve_config_paths(load_config(config_path), config_path)
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
    if cache.get("skipped"):
        raise RuntimeError(
            "formal surface-generator training requires zero skipped v2 cache records; "
            f"observed={len(cache.get('skipped', []))}"
        )
    base_records = cache["records"]
    splits = split_indices(base_records)
    if not all(splits.values()):
        raise RuntimeError(
            f"empty split detected: { {key: len(value) for key, value in splits.items()} }"
        )
    assert_disjoint_split_groups(base_records, splits)
    provenance = build_provenance(cache=cache, records=base_records, splits=splits)
    if provenance.get("git_dirty") is not False:
        raise RuntimeError("formal surface-generator training requires a clean Git worktree")
    if len(str(provenance.get("git_commit", ""))) != 40:
        raise RuntimeError("formal surface-generator training requires a resolvable Git commit")
    generator_protocol = _generator_protocol(config, provenance)
    generator_protocol_sha256 = canonical_sha256(generator_protocol)

    records = prepare_surface_generator_records(base_records)
    normalizers_cpu = lattice_normalizers(records, splits["train"])
    normalizers = {key: value.to(device) for key, value in normalizers_cpu.items()}
    train_dataset = SurfaceTemplateDataset(
        records,
        splits["train"],
        normalizers_cpu,
        template_indices=splits["train"],
        deterministic_templates=False,
    )
    validation_dataset = SurfaceTemplateDataset(
        records,
        splits["validation"],
        normalizers_cpu,
        template_indices=splits["train"],
        deterministic_templates=True,
    )
    test_dataset = SurfaceTemplateDataset(
        records,
        splits["test"],
        normalizers_cpu,
        template_indices=splits["train"],
        deterministic_templates=True,
    )

    distributed = world_size > 1
    loader_kwargs = {
        "batch_size": int(generator_training["batch_size_per_gpu"]),
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": bool(data_config["pin_memory"]),
        "distributed": distributed,
    }
    train_loader, train_sampler = make_loader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader, _ = make_loader(
        validation_dataset, shuffle=False, **loader_kwargs
    )
    test_loader, _ = make_loader(test_dataset, shuffle=False, **loader_kwargs)

    model = SurfaceAwareTemplateFlow(**config["generator_model"]).to(device)
    raw_model = model
    resume = None
    if args.resume:
        resume = torch_load_compat(args.resume, map_location="cpu")
        if resume.get("format") != GENERATOR_CHECKPOINT_FORMAT:
            raise ValueError(
                "formal generator resume requires provenance-aware checkpoint format 1.1"
            )
        assert_matching_provenance(
            resume.get("provenance"),
            provenance,
            require_present=True,
            require_code_match=True,
        )
        if str(resume.get("generator_protocol_sha256", "")) != generator_protocol_sha256:
            raise ValueError("generator resume protocol differs from current formal protocol")
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
    scaler = core.make_grad_scaler(amp)
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
    catalog = composition_catalog(records, splits["train"]) if is_main_process() else None
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
                    "task": "formal_surface_aware_template_flow",
                    "device": str(device),
                    "world_size": world_size,
                    "parameters": raw_model.parameter_count(),
                    "splits": {key: len(value) for key, value in splits.items()},
                    "generator_protocol_sha256": generator_protocol_sha256,
                    "git_commit": provenance["git_commit"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(start_epoch, int(generator_training["epochs"])):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        train_module.train()
        running = torch.zeros(len(core.METRIC_KEYS) + 1, device=device)
        started = time.time()
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            flow = core.flow_corruption(batch, normalizers, config["generator_loss"])
            labels, scores = core.condition_dropout(
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
                with core.autocast_context(device, amp):
                    outputs = train_module(
                        batch,
                        flow["frac_pos"],
                        flow["lattice_state"],
                        flow["lattice"],
                        flow["time"],
                        labels,
                        scores,
                    )
                    loss, components = core.compute_flow_loss(
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
                    raw_model.parameters(), float(generator_training["grad_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            support = float(len(batch["labels"]))
            running[: len(core.METRIC_KEYS)] += (
                torch.tensor(
                    [components[key] for key in core.METRIC_KEYS], device=device
                )
                * support
            )
            running[-1] += support
        running = reduce_sum(running)
        train_values = running[: len(core.METRIC_KEYS)] / running[-1].clamp_min(1)
        validation = evaluate(
            train_module, validation_loader, device, normalizers, config, amp
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
                    for index, key in enumerate(core.METRIC_KEYS)
                },
                **{f"val_{key}": value for key, value in validation.items()},
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if improved:
                atomic_torch_save(
                    _checkpoint_payload(
                        raw_model=raw_model,
                        config=config,
                        normalizers=normalizers,
                        catalog=catalog,
                        template_catalog=template_catalog,
                        training_reference=training_reference,
                        epoch=epoch,
                        validation=validation,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        provenance=provenance,
                        generator_protocol=generator_protocol,
                        generator_protocol_sha256=generator_protocol_sha256,
                    ),
                    best_path,
                )
        if distributed:
            control = torch.tensor(
                [int(improved), int(stop_training)], device=device, dtype=torch.int32
            )
            dist.broadcast(control, src=0)
            stop_training = bool(control[1].item())
        if stop_training:
            break

    barrier()
    best = torch_load_compat(best_path, map_location="cpu")
    raw_model.load_state_dict(best["model_state"])
    test_metrics = evaluate(train_module, test_loader, device, normalizers, config, amp)
    if is_main_process():
        best["test"] = test_metrics
        atomic_torch_save(best, best_path)
        checkpoint_sha256 = file_sha256(best_path)
        save_json(
            checkpoint_dir / "generator_final_metrics.json",
            {
                "best_epoch": best["epoch"],
                "validation": best["validation"],
                "test": test_metrics,
                "checkpoint_sha256": checkpoint_sha256,
                "generator_protocol_sha256": generator_protocol_sha256,
                "provenance": provenance,
            },
        )
        print(
            json.dumps(
                {
                    "training_complete": True,
                    "generator_checkpoint": str(best_path),
                    "generator_checkpoint_sha256": checkpoint_sha256,
                    "generator_protocol_sha256": generator_protocol_sha256,
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
