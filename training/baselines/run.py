from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from nfe_model.data_v2 import torch_load_compat
from nfe_model.model import PeriodicNFEModel
from nfe_model.provenance_v2 import assert_matching_provenance, file_sha256
from nfe_model.utils import cosine_schedule

try:
    from .classical import run_dummy, run_xgboost
    from .common import (
        BenchmarkData,
        class_weight_array,
        fit_temperature,
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
    from .controlled_aliases import CONTROLLED_MODEL_KEYS, build_controlled_model
    from .matched_painn import MatchedPaiNNBaseline
    from .protocol import (
        common_neural_training_protocol,
        common_neural_training_protocol_sha256,
        neural_model_protocol_sha256,
    )
except ImportError:
    from classical import run_dummy, run_xgboost
    from common import (
        BenchmarkData,
        class_weight_array,
        fit_temperature,
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
    from controlled_aliases import CONTROLLED_MODEL_KEYS, build_controlled_model
    from matched_painn import MatchedPaiNNBaseline
    from protocol import (
        common_neural_training_protocol,
        common_neural_training_protocol_sha256,
        neural_model_protocol_sha256,
    )

ARCHITECTURE_MODELS = ("dummy", "xgboost", *CONTROLLED_MODEL_KEYS, "painn")
ALL_MODELS = (*ARCHITECTURE_MODELS, "ours_full")


def parse_seeds(text: str) -> list[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("provide one or more unique integer seeds")
    return seeds


def _amp(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _model(name: str, hidden: int, layers: int, cutoff: float, dropout: float):
    if name == "painn":
        return MatchedPaiNNBaseline(
            hidden_dim=hidden, num_layers=layers, cutoff=cutoff, dropout=dropout
        )
    return build_controlled_model(
        name, hidden_dim=hidden, num_layers=layers, cutoff=cutoff, dropout=dropout
    )


@torch.no_grad()
def _evaluate(model, loader, device, normalizers, amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    parts = {key: [] for key in ("logits", "pred", "truth", "mask", "labels")}
    for batch in loader:
        batch = move_batch(batch, device)
        with _amp(device, amp):
            out = model(batch)
        parts["logits"].append(out["class_logits"].float().cpu().numpy())
        parts["pred"].append(out["score"].float().cpu().numpy())
        parts["truth"].append(batch["targets"][:, 0].float().cpu().numpy())
        parts["mask"].append(batch["target_mask"][:, 0].cpu().numpy())
        parts["labels"].append(batch["labels"].cpu().numpy())
    return {
        "logits": np.concatenate(parts["logits"]),
        "labels": np.concatenate(parts["labels"]),
        "score_prediction": inverse_score_from_normalized(
            np.concatenate(parts["pred"]), normalizers
        ),
        "score_target": inverse_score_from_normalized(
            np.concatenate(parts["truth"]), normalizers
        ),
        "score_mask": np.concatenate(parts["mask"]).astype(bool),
    }


def _metrics(payload: dict[str, np.ndarray], temperature: float = 1.0):
    return metrics_from_arrays(
        payload["logits"] / temperature,
        payload["labels"],
        payload["score_prediction"],
        payload["score_target"],
        payload["score_mask"],
    )


def train_architecture(name: str, data: BenchmarkData, seed: int, args, device, run_dir: Path):
    seed_everything(seed)
    cutoff = float(data.config["data"].get("radius", 6.0))
    early_supervised_epochs = int(data.config.get("training", {}).get("pretrain_epochs", 0))
    early_supervised_factor = 0.25
    common_protocol = common_neural_training_protocol(args, data)
    common_protocol_hash = common_neural_training_protocol_sha256(args, data)
    model_protocol_hash = neural_model_protocol_sha256(name, args, data)
    model = _model(name, args.hidden_dim, args.layers, cutoff, args.dropout).to(device)
    workers = (
        int(data.config["data"].get("num_workers", 0))
        if args.num_workers < 0
        else args.num_workers
    )
    train = make_loader(
        data, "train", batch_size=args.batch_size, shuffle=True, num_workers=workers
    )
    val = make_loader(
        data, "validation", batch_size=args.batch_size, shuffle=False, num_workers=workers
    )
    test = make_loader(
        data, "test", batch_size=args.batch_size, shuffle=False, num_workers=workers
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = cosine_schedule(
        optimizer,
        max(args.epochs * max(len(train), 1), 1),
        max(args.warmup_epochs * max(len(train), 1), 0),
        args.min_learning_rate / args.learning_rate,
    )
    scaler = _scaler((not args.no_amp) and device.type == "cuda")
    class_weights = torch.tensor(class_weight_array(data), dtype=torch.float32, device=device)
    history = run_dir / "history.jsonl"
    history.unlink(missing_ok=True)
    best_path = run_dir / "best.pt"
    best_score, best_epoch, bad_epochs = -float("inf"), -1, 0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        batches = 0
        supervised_factor = early_supervised_factor if epoch < early_supervised_epochs else 1.0
        for batch in train:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device, not args.no_amp):
                out = model(batch)
                valid_label = batch["labels"] >= 0
                if torch.any(valid_label):
                    raw = F.cross_entropy(
                        out["class_logits"][valid_label],
                        batch["labels"][valid_label],
                        weight=class_weights,
                        label_smoothing=args.label_smoothing,
                        reduction="none",
                    )
                    sw = batch["sample_weights"][valid_label]
                    class_loss = torch.sum(raw * sw) / sw.sum().clamp_min(1e-6)
                else:
                    class_loss = out["class_logits"].sum() * 0.0
                valid_score = batch["target_mask"][:, 0]
                if torch.any(valid_score):
                    raw = F.smooth_l1_loss(
                        out["score"][valid_score],
                        batch["targets"][valid_score, 0],
                        beta=0.5,
                        reduction="none",
                    )
                    sw = batch["sample_weights"][valid_score]
                    score_loss = torch.sum(raw * sw) / sw.sum().clamp_min(1e-6)
                else:
                    score_loss = out["score"].sum() * 0.0
                loss = supervised_factor * (class_loss + 1.5 * score_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            batches += 1

        val_payload = _evaluate(model, val, device, data.normalizers, not args.no_amp)
        val_metrics = _metrics(val_payload)
        record = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "supervised_factor": supervised_factor,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
        score = float(val_metrics["selection_score"])
        if score > best_score + 1e-8:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save(
                {
                    "format": "nfe-controlled-baseline-3.3",
                    "track": "architecture",
                    "model_name": name,
                    "model_state": model.state_dict(),
                    "normalizers": {key: value.cpu() for key, value in data.normalizers.items()},
                    "seed": seed,
                    "epoch": epoch,
                    "provenance": data.provenance,
                    "benchmark_common_protocol_sha256": common_protocol_hash,
                    "model_protocol_sha256": model_protocol_hash,
                    "model_config": {
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.layers,
                        "cutoff": cutoff,
                        "dropout": args.dropout,
                    },
                    "training_protocol": common_protocol,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch_load_compat(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    val_payload = _evaluate(model, val, device, data.normalizers, not args.no_amp)
    test_payload = _evaluate(model, test, device, data.normalizers, not args.no_amp)
    temperature = fit_temperature(val_payload["logits"], val_payload["labels"])
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": time.time() - start,
        "temperature": temperature,
        "benchmark_common_protocol_sha256": common_protocol_hash,
        "model_protocol_sha256": model_protocol_hash,
        "validation_metrics": _metrics(val_payload, temperature),
        "test_metrics": _metrics(test_payload, temperature),
        "validation_predictions": val_payload,
        "test_predictions": test_payload,
        "details": {
            "best_epoch": best_epoch,
            "controlled_reimplementation": name in CONTROLLED_MODEL_KEYS,
            "matched_supervision": True,
            "matched_supervised_schedule": True,
            "checkpoint": str(best_path),
            "checkpoint_sha256": file_sha256(best_path),
            "checkpoint_training_git_commit": data.provenance.get("git_commit"),
            "checkpoint_training_git_dirty": data.provenance.get("git_dirty"),
        },
    }


@torch.no_grad()
def evaluate_full(model, loader, device, normalizers, amp: bool):
    model.eval()
    logits, pred, truth, mask, labels = [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with _amp(device, amp):
            out = model(batch)
        logits.append(out["class_logits"].float().cpu().numpy())
        pred.append(out["regression_mean"][:, 0].float().cpu().numpy())
        truth.append(batch["targets"][:, 0].float().cpu().numpy())
        mask.append(batch["target_mask"][:, 0].cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return {
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
        "score_prediction": inverse_score_from_normalized(np.concatenate(pred), normalizers),
        "score_target": inverse_score_from_normalized(np.concatenate(truth), normalizers),
        "score_mask": np.concatenate(mask).astype(bool),
    }


def evaluate_full_checkpoint(data, path: Path, seed: int, args, device):
    start = time.time()
    checkpoint = torch_load_compat(path, map_location="cpu")
    fmt = checkpoint.get("format")
    if fmt == "nfe-mxene-predictor-ablation-1.0":
        if checkpoint.get("ablation_config", {}).get("name") != "full":
            raise ValueError("full-system track requires the full ablation checkpoint")
        model_config = checkpoint.get("base_model_config", checkpoint.get("model_config"))
    elif fmt == "nfe-mxene-predictor-1.0":
        model_config = checkpoint.get("model_config")
    else:
        raise ValueError(f"unsupported predictor checkpoint format: {fmt}")
    if not isinstance(model_config, dict):
        raise ValueError(f"checkpoint has no valid model configuration: {path}")
    checkpoint_seed = checkpoint.get("config", {}).get("seed")
    if checkpoint_seed is None and not args.allow_unverified_checkpoint:
        raise ValueError(f"checkpoint has no config.seed: {path}")
    if checkpoint_seed is not None and int(checkpoint_seed) != seed:
        raise ValueError(
            f"checkpoint seed mismatch: request={seed}, checkpoint={checkpoint_seed}"
        )
    assert_matching_provenance(
        checkpoint.get("provenance"),
        data.provenance,
        require_present=not args.allow_unverified_checkpoint,
    )
    normalizer_source = checkpoint.get("normalizers")
    if normalizer_source is None:
        if not args.allow_unverified_checkpoint:
            raise ValueError(f"checkpoint has no train-fitted normalizers: {path}")
        normalizer_source = data.normalizers
    normalizers = {key: value.cpu() for key, value in normalizer_source.items()}
    model = PeriodicNFEModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    workers = (
        int(data.config["data"].get("num_workers", 0))
        if args.num_workers < 0
        else args.num_workers
    )
    val = make_loader(
        data,
        "validation",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        normalizers=normalizers,
    )
    test = make_loader(
        data,
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        normalizers=normalizers,
    )
    val_payload = evaluate_full(model, val, device, normalizers, not args.no_amp)
    test_payload = evaluate_full(model, test, device, normalizers, not args.no_amp)
    temperature = fit_temperature(val_payload["logits"], val_payload["labels"])
    checkpoint_provenance = checkpoint.get("provenance", {})
    training_protocol_hash = checkpoint.get("training_protocol_sha256")
    if not training_protocol_hash and not args.allow_unverified_checkpoint:
        raise ValueError(f"checkpoint has no training_protocol_sha256: {path}")
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": 0.0,
        "evaluation_seconds": time.time() - start,
        "temperature": temperature,
        "model_protocol_sha256": training_protocol_hash,
        "validation_metrics": _metrics(val_payload, temperature),
        "test_metrics": _metrics(test_payload, temperature),
        "validation_predictions": val_payload,
        "test_predictions": test_payload,
        "details": {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_seed": checkpoint_seed,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_format": fmt,
            "checkpoint_training_git_commit": checkpoint_provenance.get("git_commit"),
            "checkpoint_training_git_dirty": checkpoint_provenance.get("git_dirty"),
            "evaluation_only": True,
        },
    }


def _result(track, name, seed, data, payload):
    return {
        "schema": "nfe-baseline-result-2.1",
        "track": track,
        "model": name,
        "seed": int(seed),
        "parameter_count": payload.get("parameter_count"),
        "training_seconds": payload.get("training_seconds", 0.0),
        "evaluation_seconds": payload.get("evaluation_seconds"),
        "temperature": payload.get("temperature", 1.0),
        "benchmark_common_protocol_sha256": payload.get("benchmark_common_protocol_sha256"),
        "model_protocol_sha256": payload.get("model_protocol_sha256"),
        "split_sizes": {key: len(value) for key, value in data.splits.items()},
        "skipped_cache_records": data.skipped_cache_records,
        "provenance": data.provenance,
        "validation_metrics": payload["validation_metrics"],
        "test_metrics": payload["test_metrics"],
        "details": payload.get("details", {}),
    }


def _save_predictions(run_dir, data, payload):
    for split in ("validation", "test"):
        values = payload.get(f"{split}_predictions")
        if values is not None:
            prediction_frame(
                data,
                split,
                logits=values["logits"],
                score_prediction=values["score_prediction"],
                temperature=float(payload.get("temperature", 1.0)),
            ).to_csv(run_dir / f"{split}_predictions.csv", index=False)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Run audited NFE benchmark tracks.")
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
    if args.track == "architecture":
        if args.model == "ours_full":
            raise SystemExit("ours_full belongs to --track full-system")
        selected = list(ARCHITECTURE_MODELS) if args.model == "all" else [args.model]
        if any(name not in ARCHITECTURE_MODELS for name in selected):
            raise SystemExit(f"invalid architecture-track selection: {selected}")
    else:
        if args.model not in {"all", "ours_full"}:
            raise SystemExit("full-system accepts only ours_full/all")
        selected = ["ours_full"]

    root = Path(args.output_root).resolve()
    for name in selected:
        seeds = [args.seeds[0]] if name == "dummy" else args.seeds
        for seed in seeds:
            run_dir = root / args.track / name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running track={args.track} model={name} seed={seed}", flush=True)
            if name == "dummy":
                payload = run_dummy(data, seed)
            elif name == "xgboost":
                seed_everything(seed)
                payload = run_xgboost(data, seed)
            elif name in (*CONTROLLED_MODEL_KEYS, "painn"):
                payload = train_architecture(name, data, seed, args, device, run_dir)
            else:
                path = Path(args.ours_root).resolve() / f"seed_{seed}" / "best.pt"
                if not path.is_file():
                    raise FileNotFoundError(f"missing independently trained checkpoint: {path}")
                payload = evaluate_full_checkpoint(data, path, seed, args, device)
            result = _result(args.track, name, seed, data, payload)
            _save_predictions(run_dir, data, payload)
            save_json(run_dir / "result.json", result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
