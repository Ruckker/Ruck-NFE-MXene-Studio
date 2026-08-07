from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import time
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.core import Element

from nfe_model.metrics_v2 import classification_metrics, regression_metrics, selection_score
from nfe_model import data_v2, metrics_v2, provenance_v2
from nfe_model.utils import cosine_schedule
import training.baselines.common as _common
_common.load_or_build_cache = data_v2.load_or_build_cache
_common.classification_metrics = metrics_v2.classification_metrics
_common.regression_metrics = metrics_v2.regression_metrics
_common.selection_score = metrics_v2.selection_score
_common.build_provenance = provenance_v2.build_provenance
from training.baselines.common import (
    class_weight_array,
    inverse_score_from_normalized,
    load_benchmark_data,
    make_loader,
    prediction_frame,
    resolve_device,
    save_json,
    seed_everything,
)
from training.baselines.official.backends import build_official_backend


OFFICIAL_MODELS = (
    "cgcnn_official",
    "schnet_official",
    "alignn_official",
    "m3gnet_official",
)


def parse_seeds(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("provide one or more unique integer seeds")
    return values


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    if int(valid.sum()) < 3:
        return 1.0
    x = torch.tensor(logits[valid], dtype=torch.float32)
    y = torch.tensor(labels[valid], dtype=torch.long)
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=75, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.05, 20.0)
        loss = F.cross_entropy(x / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t.detach()).clamp(0.05, 20.0))


def _metrics(payload: dict[str, np.ndarray], temperature: float = 1.0) -> dict[str, float]:
    result = classification_metrics(payload["logits"] / temperature, payload["labels"])
    result.update(
        regression_metrics(
            payload["score_prediction"].reshape(-1, 1),
            payload["score_target"].reshape(-1, 1),
            payload["score_mask"].reshape(-1, 1),
            ["NFE_Pseudo_Score"],
        )
    )
    result["selection_score"] = selection_score(result)
    return result


@torch.no_grad()
def evaluate(model, loader, device, normalizers, amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    logits, pred, truth, mask, labels = [], [], [], [], []
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with _autocast(device, amp):
            out = model(batch)
        logits.append(out["class_logits"].float().cpu().numpy())
        pred.append(out["score"].float().cpu().numpy())
        truth.append(batch["targets"][:, 0].float().cpu().numpy())
        mask.append(batch["target_mask"][:, 0].cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    pred_norm = np.concatenate(pred)
    truth_norm = np.concatenate(truth)
    return {
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
        "score_prediction": inverse_score_from_normalized(pred_norm, normalizers),
        "score_target": inverse_score_from_normalized(truth_norm, normalizers),
        "score_mask": np.concatenate(mask).astype(bool),
    }


def _package_versions(name: str) -> dict[str, str]:
    packages = {
        "cgcnn_official": [],
        "schnet_official": ["schnetpack"],
        "alignn_official": ["alignn", "dgl"],
        "m3gnet_official": ["matgl", "torch-geometric"],
    }[name]
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unknown"
    return result


def train_one(args, name: str, seed: int) -> None:
    seed_everything(seed)
    data = load_benchmark_data(args.config, rebuild_cache=args.rebuild_cache)
    device = resolve_device(args.device)
    workers = int(data.config["data"].get("num_workers", 0)) if args.num_workers < 0 else args.num_workers
    train_loader = make_loader(data, "train", batch_size=args.batch_size, shuffle=True, num_workers=workers)
    val_loader = make_loader(data, "validation", batch_size=args.batch_size, shuffle=False, num_workers=workers)
    test_loader = make_loader(data, "test", batch_size=args.batch_size, shuffle=False, num_workers=workers)

    atomic_numbers = sorted(
        {int(z) for i in data.splits["train"] for z in data.records[i]["z"].tolist()}
    )
    element_types = [Element.from_Z(z).symbol for z in atomic_numbers]
    model = build_official_backend(
        name,
        element_types=element_types,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        cutoff=float(data.config["data"]["radius"]),
        max_neighbors=int(data.config["data"]["max_neighbors"]),
        cgcnn_repo=args.cgcnn_repo,
        cgcnn_atom_init=args.cgcnn_atom_init,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(args.epochs * len(train_loader), 1)
    scheduler = cosine_schedule(
        optimizer,
        total_steps,
        max(args.warmup_epochs * len(train_loader), 0),
        args.min_learning_rate / args.learning_rate,
    )
    scaler = _scaler((not args.no_amp) and device.type == "cuda")
    class_weights = torch.tensor(class_weight_array(data), device=device, dtype=torch.float32)
    out_dir = Path(args.output_root).resolve() / "official-upstream" / name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history_path = out_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    best_score = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, (not args.no_amp) and device.type == "cuda"):
                out = model(batch)
                valid_label = batch["labels"] >= 0
                class_raw = F.cross_entropy(
                    out["class_logits"][valid_label],
                    batch["labels"][valid_label],
                    weight=class_weights,
                    label_smoothing=args.label_smoothing,
                    reduction="none",
                )
                sw = batch["sample_weights"][valid_label]
                class_loss = torch.sum(class_raw * sw) / sw.sum().clamp_min(1e-6)
                valid_score = batch["target_mask"][:, 0]
                if torch.any(valid_score):
                    score_raw = F.smooth_l1_loss(
                        out["score"][valid_score],
                        batch["targets"][valid_score, 0],
                        beta=0.5,
                        reduction="none",
                    )
                    score_sw = batch["sample_weights"][valid_score]
                    score_loss = torch.sum(score_raw * score_sw) / score_sw.sum().clamp_min(1e-6)
                else:
                    score_loss = out["score"].sum() * 0.0
                loss = class_loss + 1.5 * score_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            batches += 1

        val_payload = evaluate(model, val_loader, device, data.normalizers, amp=(not args.no_amp))
        val_metrics = _metrics(val_payload, 1.0)
        record = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        current = float(val_metrics["selection_score"])
        if current > best_score + 1e-8:
            best_score, best_epoch, bad_epochs = current, epoch, 0
            torch.save(
                {
                    "format": "nfe-official-upstream-baseline-1.0",
                    "track": "official-upstream",
                    "model_name": name,
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "provenance": data.provenance,
                    "training_protocol": "class + NFE score; common optimizer/split/metric budget",
                    "package_versions": _package_versions(name),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    val_payload = evaluate(model, val_loader, device, data.normalizers, amp=(not args.no_amp))
    test_payload = evaluate(model, test_loader, device, data.normalizers, amp=(not args.no_amp))
    temperature = _temperature(val_payload["logits"], val_payload["labels"])
    val_metrics = _metrics(val_payload, temperature)
    test_metrics = _metrics(test_payload, temperature)
    result = {
        "schema": "nfe-baseline-result-2.0",
        "track": "official-upstream",
        "model": name,
        "seed": seed,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": time.time() - start,
        "temperature": temperature,
        "split_sizes": {k: len(v) for k, v in data.splits.items()},
        "provenance": data.provenance,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "details": {
            "best_epoch": best_epoch,
            "official_upstream_backbone": True,
            "project_nfe_head_or_adapter": True,
            "package_versions": _package_versions(name),
        },
    }
    save_json(out_dir / "result.json", result)
    for split, payload in (("validation", val_payload), ("test", test_payload)):
        prediction_frame(
            data,
            split,
            logits=payload["logits"],
            score_prediction=payload["score_prediction"],
            temperature=temperature,
        ).to_csv(out_dir / f"{split}_predictions.csv", index=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official-upstream backbone baselines on the fixed NFE benchmark."
    )
    parser.add_argument("--model", choices=(*OFFICIAL_MODELS, "all"), default="all")
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("2027,2028,2029,2030,2031"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="training/baselines/results")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cgcnn-repo")
    parser.add_argument("--cgcnn-atom-init")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = OFFICIAL_MODELS if args.model == "all" else (args.model,)
    for name in selected:
        for seed in args.seeds:
            train_one(args, name, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
