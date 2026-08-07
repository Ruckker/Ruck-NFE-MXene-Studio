from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from nfe_model.data import torch_load_compat
from nfe_model.model import PeriodicNFEModel

try:
    from .classical import run_dummy, run_xgboost
    from .common import (
        BenchmarkData,
        calibrate_metrics,
        class_weight_array,
        inverse_score_from_normalized,
        load_benchmark_data,
        make_loader,
        metrics_from_arrays,
        move_batch,
        resolve_device,
        save_json,
        seed_everything,
    )
    from .models import build_model
except ImportError:
    from classical import run_dummy, run_xgboost
    from common import (
        BenchmarkData,
        calibrate_metrics,
        class_weight_array,
        inverse_score_from_normalized,
        load_benchmark_data,
        make_loader,
        metrics_from_arrays,
        move_batch,
        resolve_device,
        save_json,
        seed_everything,
    )
    from models import build_model


CONTROLLED_GRAPH_MODELS = ("cgcnn", "schnet", "alignn", "m3gnet")
ALL_MODELS = ("dummy", "xgboost", *CONTROLLED_GRAPH_MODELS, "ours")


def parse_seeds(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("seed list contains duplicates")
    return values


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate_controlled_graph(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    normalizers: dict[str, torch.Tensor],
    *,
    amp: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    logits, score_prediction, score_target, score_mask, labels = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, amp):
            output = model(batch)
        logits.append(output["class_logits"].float().cpu().numpy())
        score_prediction.append(output["score"].float().cpu().numpy())
        score_target.append(batch["targets"][:, 0].float().cpu().numpy())
        score_mask.append(batch["target_mask"][:, 0].cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    logits_array = np.concatenate(logits)
    prediction_normalized = np.concatenate(score_prediction)
    target_normalized = np.concatenate(score_target)
    mask_array = np.concatenate(score_mask).astype(bool)
    label_array = np.concatenate(labels)
    prediction = inverse_score_from_normalized(prediction_normalized, normalizers)
    target = inverse_score_from_normalized(target_normalized, normalizers)
    metrics = metrics_from_arrays(logits_array, label_array, prediction, target, mask_array)
    return metrics, {
        "logits": logits_array,
        "labels": label_array,
        "score_prediction": prediction,
        "score_target": target,
        "score_mask": mask_array,
    }


def run_controlled_graph(
    name: str,
    data: BenchmarkData,
    seed: int,
    *,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_workers: int,
    amp: bool,
) -> dict[str, Any]:
    seed_everything(seed)
    start = time.time()
    cutoff = float(data.config["data"].get("radius", 6.0))
    model = build_model(
        name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff,
        dropout=dropout,
    ).to(device)
    train_loader = make_loader(
        data, "train", batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_loader = make_loader(
        data, "validation", batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = make_loader(
        data, "test", batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=learning_rate * 0.02
    )
    scaler = make_scaler(amp and device.type == "cuda")
    class_weights = torch.tensor(
        class_weight_array(data), dtype=torch.float32, device=device
    )

    best_score = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp and device.type == "cuda"):
                output = model(batch)
                valid_labels = batch["labels"] >= 0
                if torch.any(valid_labels):
                    class_loss_raw = F.cross_entropy(
                        output["class_logits"][valid_labels],
                        batch["labels"][valid_labels],
                        weight=class_weights,
                        reduction="none",
                    )
                    class_sample_weight = batch["sample_weights"][valid_labels]
                    class_loss = torch.sum(
                        class_loss_raw * class_sample_weight
                    ) / class_sample_weight.sum().clamp_min(1e-6)
                else:
                    class_loss = output["class_logits"].sum() * 0.0
                valid_score = batch["target_mask"][:, 0]
                if torch.any(valid_score):
                    regression_raw = F.smooth_l1_loss(
                        output["score"][valid_score],
                        batch["targets"][valid_score, 0],
                        beta=0.5,
                        reduction="none",
                    )
                    regression_weight = batch["sample_weights"][valid_score]
                    regression_loss = torch.sum(
                        regression_raw * regression_weight
                    ) / regression_weight.sum().clamp_min(1e-6)
                else:
                    regression_loss = output["score"].sum() * 0.0
                loss = class_loss + 1.5 * regression_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            total_batches += 1
        scheduler.step()

        validation_metrics, _ = evaluate_controlled_graph(
            model,
            validation_loader,
            device,
            data.normalizers,
            amp=amp and device.type == "cuda",
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_batches, 1),
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        current = float(validation_metrics.get("selection_score", -float("inf")))
        if current > best_score + 1e-8:
            best_score = current
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "format": "nfe-controlled-baseline-1.0",
                    "model_name": name,
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "model_config": {
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "cutoff": cutoff,
                        "dropout": dropout,
                    },
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    checkpoint = torch_load_compat(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    _, validation_payload = evaluate_controlled_graph(
        model,
        validation_loader,
        device,
        data.normalizers,
        amp=amp and device.type == "cuda",
    )
    _, test_payload = evaluate_controlled_graph(
        model,
        test_loader,
        device,
        data.normalizers,
        amp=amp and device.type == "cuda",
    )
    temperature, validation_metrics, test_metrics = calibrate_metrics(
        validation_payload["logits"],
        validation_payload["labels"],
        test_payload["logits"],
        validation_payload["score_prediction"],
        validation_payload["score_target"],
        validation_payload["score_mask"],
        test_payload["labels"],
        test_payload["score_prediction"],
        test_payload["score_target"],
        test_payload["score_mask"],
    )
    return {
        "parameter_count": int(model.parameter_count()),
        "training_seconds": time.time() - start,
        "temperature": temperature,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "details": {
            "best_epoch": int(best_epoch),
            "controlled_reimplementation": True,
            "checkpoint": str(best_path),
        },
    }


@torch.no_grad()
def evaluate_ours(
    model: PeriodicNFEModel,
    loader,
    device: torch.device,
    normalizers: dict[str, torch.Tensor],
    *,
    amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    logits, prediction_norm, target_norm, score_mask, labels = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, amp and device.type == "cuda"):
            output = model(batch)
        logits.append(output["class_logits"].float().cpu().numpy())
        prediction_norm.append(output["regression_mean"][:, 0].float().cpu().numpy())
        target_norm.append(batch["targets"][:, 0].float().cpu().numpy())
        score_mask.append(batch["target_mask"][:, 0].cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return {
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
        "score_prediction": inverse_score_from_normalized(
            np.concatenate(prediction_norm), normalizers
        ),
        "score_target": inverse_score_from_normalized(
            np.concatenate(target_norm), normalizers
        ),
        "score_mask": np.concatenate(score_mask).astype(bool),
    }


def run_ours(
    data: BenchmarkData,
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> dict[str, Any]:
    start = time.time()
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError(f"not an NFE predictor checkpoint: {checkpoint_path}")
    normalizers = checkpoint.get("normalizers", data.normalizers)
    normalizers = {key: value.cpu() for key, value in normalizers.items()}
    model = PeriodicNFEModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    validation_loader = make_loader(
        data,
        "validation",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        normalizers=normalizers,
    )
    test_loader = make_loader(
        data,
        "test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        normalizers=normalizers,
    )
    validation_payload = evaluate_ours(
        model, validation_loader, device, normalizers, amp=amp
    )
    test_payload = evaluate_ours(model, test_loader, device, normalizers, amp=amp)
    temperature, validation_metrics, test_metrics = calibrate_metrics(
        validation_payload["logits"],
        validation_payload["labels"],
        test_payload["logits"],
        validation_payload["score_prediction"],
        validation_payload["score_target"],
        validation_payload["score_mask"],
        test_payload["labels"],
        test_payload["score_prediction"],
        test_payload["score_target"],
        test_payload["score_mask"],
    )
    return {
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": 0.0,
        "evaluation_seconds": time.time() - start,
        "temperature": temperature,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "details": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "evaluation_only": True,
        },
    }


def build_result(
    name: str,
    seed: int,
    data: BenchmarkData,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "nfe-baseline-result-1.0",
        "model": name,
        "seed": int(seed),
        "parameter_count": payload.get("parameter_count"),
        "training_seconds": payload.get("training_seconds", 0.0),
        "evaluation_seconds": payload.get("evaluation_seconds"),
        "temperature": payload.get("temperature", 1.0),
        "split_sizes": {key: len(value) for key, value in data.splits.items()},
        "skipped_cache_records": data.skipped_cache_records,
        "validation_metrics": payload["validation_metrics"],
        "test_metrics": payload["test_metrics"],
        "details": payload.get("details", {}),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe NFE predictor baselines.")
    parser.add_argument("--model", choices=(*ALL_MODELS, "all"), default="all")
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument(
        "--seeds", type=parse_seeds, default=parse_seeds("2027,2028,2029")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="training/baselines/results")
    parser.add_argument("--ours-checkpoint")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_benchmark_data(args.config, rebuild_cache=args.rebuild_cache)
    device = resolve_device(args.device)
    num_workers = (
        int(data.config["data"].get("num_workers", 0))
        if args.num_workers < 0
        else int(args.num_workers)
    )
    selected = list(ALL_MODELS) if args.model == "all" else [args.model]
    if "ours" in selected and not args.ours_checkpoint:
        if args.model == "ours":
            raise SystemExit("--ours-checkpoint is required for --model ours")
        selected.remove("ours")
        print("Skipping ours: no --ours-checkpoint supplied", flush=True)

    output_root = Path(args.output_root).resolve()
    for name in selected:
        seeds = args.seeds if name != "ours" else [args.seeds[0]]
        for seed in seeds:
            run_dir = output_root / name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running model={name} seed={seed}", flush=True)
            if name == "dummy":
                payload = run_dummy(data, seed)
            elif name == "xgboost":
                seed_everything(seed)
                payload = run_xgboost(data, seed)
            elif name in CONTROLLED_GRAPH_MODELS:
                payload = run_controlled_graph(
                    name,
                    data,
                    seed,
                    device=device,
                    output_dir=run_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.layers,
                    dropout=args.dropout,
                    num_workers=num_workers,
                    amp=not args.no_amp,
                )
            elif name == "ours":
                payload = run_ours(
                    data,
                    Path(args.ours_checkpoint),
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=num_workers,
                    amp=not args.no_amp,
                )
            else:
                raise AssertionError(name)
            result = build_result(name, seed, data, payload)
            save_json(run_dir / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
