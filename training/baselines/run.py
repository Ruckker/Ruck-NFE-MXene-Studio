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
from nfe_model.provenance import assert_matching_provenance, file_sha256
from nfe_model.utils import cosine_schedule

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
        prediction_frame,
        resolve_device,
        save_json,
        seed_everything,
    )
    from .matched_painn import MatchedPaiNNBaseline
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
        prediction_frame,
        resolve_device,
        save_json,
        seed_everything,
    )
    from matched_painn import MatchedPaiNNBaseline
    from models import build_model


CONTROLLED_GRAPH_MODELS = ("cgcnn", "schnet", "alignn", "m3gnet")
ARCHITECTURE_MODELS = ("dummy", "xgboost", *CONTROLLED_GRAPH_MODELS, "painn")
FULL_SYSTEM_MODELS = ("ours_full",)
ALL_MODELS = (*ARCHITECTURE_MODELS, *FULL_SYSTEM_MODELS)


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


def build_architecture_model(
    name: str,
    *,
    hidden_dim: int,
    num_layers: int,
    cutoff: float,
    dropout: float,
):
    if name == "painn":
        return MatchedPaiNNBaseline(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cutoff=cutoff,
            dropout=dropout,
        )
    return build_model(
        name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cutoff=cutoff,
        dropout=dropout,
    )


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
    metrics = metrics_from_arrays(
        logits_array,
        label_array,
        prediction,
        target,
        mask_array,
    )
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
    min_learning_rate: float,
    warmup_epochs: int,
    weight_decay: float,
    patience: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_workers: int,
    amp: bool,
    label_smoothing: float,
) -> dict[str, Any]:
    seed_everything(seed)
    start = time.time()
    cutoff = float(data.config["data"].get("radius", 6.0))
    model = build_architecture_model(
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
    updates_per_epoch = max(len(train_loader), 1)
    scheduler = cosine_schedule(
        optimizer,
        max(int(epochs) * updates_per_epoch, 1),
        max(int(warmup_epochs) * updates_per_epoch, 0),
        float(min_learning_rate) / float(learning_rate),
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
                        label_smoothing=float(label_smoothing),
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
            scheduler.step()
            total_loss += float(loss.detach())
            total_batches += 1

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
                    "format": "nfe-controlled-baseline-2.0",
                    "track": "architecture",
                    "model_name": name,
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "provenance": data.provenance,
                    "model_config": {
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "cutoff": cutoff,
                        "dropout": dropout,
                    },
                    "training_protocol": {
                        "supervision": "NFE class + NFE pseudo-score only",
                        "auxiliary_regression": False,
                        "masked_atom": False,
                        "coordinate_denoising": False,
                        "epochs": int(epochs),
                        "warmup_epochs": int(warmup_epochs),
                        "label_smoothing": float(label_smoothing),
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
        "validation_predictions": validation_payload,
        "test_predictions": test_payload,
        "details": {
            "best_epoch": int(best_epoch),
            "controlled_reimplementation": name in CONTROLLED_GRAPH_MODELS,
            "matched_supervision": True,
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


def run_ours_full(
    data: BenchmarkData,
    checkpoint_path: Path,
    *,
    expected_seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    allow_unverified_checkpoint: bool,
) -> dict[str, Any]:
    start = time.time()
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format == "nfe-mxene-predictor-ablation-1.0":
        ablation = checkpoint.get("ablation_config", {})
        if ablation.get("name") != "full":
            raise ValueError(
                f"full-system track requires the full ablation checkpoint, got {ablation.get('name')}"
            )
        if checkpoint.get("architecture") not in {None, "PeriodicNFEModel"}:
            raise ValueError(
                "full-system checkpoint must use PeriodicNFEModel, got "
                f"{checkpoint.get('architecture')}"
            )
        model_config = checkpoint.get("base_model_config", checkpoint.get("model_config"))
    elif checkpoint_format == "nfe-mxene-predictor-1.0":
        model_config = checkpoint.get("model_config")
    else:
        raise ValueError(f"unsupported predictor checkpoint format: {checkpoint_format}")
    if not isinstance(model_config, dict) or "model_state" not in checkpoint:
        raise ValueError(f"not an NFE predictor checkpoint: {checkpoint_path}")

    checkpoint_seed = checkpoint.get("config", {}).get("seed")
    if checkpoint_seed is None:
        if not allow_unverified_checkpoint:
            raise ValueError(
                f"checkpoint has no config.seed and cannot verify requested seed {expected_seed}: "
                f"{checkpoint_path}"
            )
    elif int(checkpoint_seed) != int(expected_seed):
        raise ValueError(
            f"checkpoint seed mismatch: path/request={expected_seed}, checkpoint={checkpoint_seed}"
        )

    assert_matching_provenance(
        checkpoint.get("provenance"),
        data.provenance,
        require_present=not allow_unverified_checkpoint,
    )
    checkpoint_hash = file_sha256(checkpoint_path)
    normalizers = checkpoint.get("normalizers", data.normalizers)
    normalizers = {key: value.cpu() for key, value in normalizers.items()}
    model = PeriodicNFEModel(**model_config)
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
        "validation_predictions": validation_payload,
        "test_predictions": test_payload,
        "details": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_seed": checkpoint_seed,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "evaluation_only": True,
            "independent_training_seed_checkpoint": True,
            "checkpoint_format": checkpoint_format,
        },
    }


def build_result(
    track: str,
    name: str,
    seed: int,
    data: BenchmarkData,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "nfe-baseline-result-2.0",
        "track": track,
        "model": name,
        "seed": int(seed),
        "parameter_count": payload.get("parameter_count"),
        "training_seconds": payload.get("training_seconds", 0.0),
        "evaluation_seconds": payload.get("evaluation_seconds"),
        "temperature": payload.get("temperature", 1.0),
        "split_sizes": {key: len(value) for key, value in data.splits.items()},
        "skipped_cache_records": data.skipped_cache_records,
        "provenance": data.provenance,
        "validation_metrics": payload["validation_metrics"],
        "test_metrics": payload["test_metrics"],
        "details": payload.get("details", {}),
    }


def save_prediction_outputs(
    run_dir: Path,
    data: BenchmarkData,
    payload: dict[str, Any],
) -> None:
    temperature = float(payload.get("temperature", 1.0))
    for split in ("validation", "test"):
        values = payload.get(f"{split}_predictions")
        if values is None:
            continue
        frame = prediction_frame(
            data,
            split,
            logits=values["logits"],
            score_prediction=values["score_prediction"],
            temperature=temperature,
        )
        frame.to_csv(run_dir / f"{split}_predictions.csv", index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audited NFE predictor benchmark tracks.")
    parser.add_argument(
        "--track", choices=("architecture", "full-system"), default="architecture"
    )
    parser.add_argument("--model", choices=(*ALL_MODELS, "all"), default="all")
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument(
        "--seeds", type=parse_seeds, default=parse_seeds("2027,2028,2029,2030,2031")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="training/baselines/results")
    parser.add_argument("--ours-root", default="runs/ablations/full")
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
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

    if args.track == "architecture":
        if args.model == "ours_full":
            raise SystemExit("ours_full belongs to --track full-system")
        selected = list(ARCHITECTURE_MODELS) if args.model == "all" else [args.model]
    else:
        if args.model not in {"all", "ours_full"}:
            raise SystemExit("--track full-system currently accepts only --model ours_full/all")
        selected = ["ours_full"]

    output_root = Path(args.output_root).resolve()
    for name in selected:
        seeds = [args.seeds[0]] if name == "dummy" else args.seeds
        for seed in seeds:
            run_dir = output_root / args.track / name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running track={args.track} model={name} seed={seed}", flush=True)
            if name == "dummy":
                payload = run_dummy(data, seed)
            elif name == "xgboost":
                seed_everything(seed)
                payload = run_xgboost(data, seed)
            elif name in (*CONTROLLED_GRAPH_MODELS, "painn"):
                payload = run_controlled_graph(
                    name,
                    data,
                    seed,
                    device=device,
                    output_dir=run_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    min_learning_rate=args.min_learning_rate,
                    warmup_epochs=args.warmup_epochs,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.layers,
                    dropout=args.dropout,
                    num_workers=num_workers,
                    amp=not args.no_amp,
                    label_smoothing=args.label_smoothing,
                )
            elif name == "ours_full":
                checkpoint_path = (
                    Path(args.ours_root).resolve() / f"seed_{seed}" / "best.pt"
                )
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"missing independently trained full-system checkpoint: {checkpoint_path}. "
                        "Run the full ablation for every requested seed first."
                    )
                payload = run_ours_full(
                    data,
                    checkpoint_path,
                    expected_seed=seed,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=num_workers,
                    amp=not args.no_amp,
                    allow_unverified_checkpoint=args.allow_unverified_checkpoint,
                )
            else:
                raise AssertionError(name)
            result = build_result(args.track, name, seed, data, payload)
            save_prediction_outputs(run_dir, data, payload)
            save_json(run_dir / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
