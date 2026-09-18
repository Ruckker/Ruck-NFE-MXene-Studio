# ==============================================================================
# 中文概述：用检查点自身的建图设置，在数据表的 test（或 validation）划分上重算一个或多个 NFE 预测器检查点的
#           指标，并与检查点内存储的指标并列，供发布复核。前向为确定性（model.eval()，无 MC dropout）。
# English overview: Re-evaluate one or more NFE predictor checkpoints on the test (or validation) split of a table
#           with each checkpoint's own graph settings, next to the metrics stored in the checkpoint. The forward
#           pass is deterministic (model.eval(), no MC dropout).
#
# 中文输入：数据表、结构目录、若干 名称=检查点路径。
# English inputs: Dataset table, structure directory, one or more name=checkpoint pairs.
# 中文输出：每个检查点的温度校准指标、high-vs-rest 指标、混淆矩阵、分数 MAE 与检查点 SHA256（JSON）；
#           可选地按基准套件的列格式导出每个检查点的 test 预测 CSV。
# English outputs: Per-checkpoint calibrated metrics, high-vs-rest metrics, confusion matrix, score MAE and checkpoint
#           SHA256 (JSON); optionally the test predictions of each checkpoint in the benchmark suite column layout.
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nfe_model.data import LABEL_TO_INDEX, move_batch, structure_to_graph  # noqa: E402
from nfe_model.metrics import classification_metrics, r_squared, spearman_rho  # noqa: E402
from nfe_model.predict import load_checkpoint_model, physical_regression, prediction_batch  # noqa: E402

LABELS = ("low", "medium", "high")
KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_roc_auc",
    "ece",
    "low_f1",
    "medium_f1",
    "high_f1",
    "high_vs_rest_f1",
    "high_vs_rest_precision",
    "high_vs_rest_recall",
    "high_vs_rest_roc_auc",
    "high_vs_rest_average_precision",
    "high_vs_rest_threshold",
    "macro_average_precision",
    "high_enrichment_at_5pct",
)


# 中文：流式计算 SHA256。/ English: Compute SHA256 as a stream.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


# 中文：解析 名称=路径 参数。/ English: Parse name=path arguments.
def parse_pairs(values: Sequence[str]) -> list[tuple[str, Path]]:
    pairs = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--checkpoint expects name=path, got {value!r}")
        name, path = value.split("=", 1)
        pairs.append((name.strip(), Path(path.strip())))
    return pairs


# 中文：评估单个检查点。/ English: Evaluate one checkpoint.
def evaluate_checkpoint(
    path: Path,
    frame: pd.DataFrame,
    structure_dir: Path,
    device: torch.device,
    batch_size: int,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint_model(str(path), device)
    data_config = checkpoint["config"]["data"]
    canonicalize = bool(data_config.get("canonicalize", False))
    complete_shells = bool(data_config.get("complete_shells", False))
    graphs = [
        structure_to_graph(
            Structure.from_file(str(structure_dir / Path(row.File_Path).name)),
            float(data_config["radius"]),
            int(data_config["max_neighbors"]),
            identifier=row.Structure_Name,
            canonicalize=canonicalize,
            complete_shells=complete_shells,
        )
        for row in frame.itertuples()
    ]
    normalizers = {key: value.cpu() for key, value in checkpoint["normalizers"].items()}
    logits, means = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = move_batch(prediction_batch(graphs[start : start + batch_size], normalizers), device)
            output = model(batch)
            logits.append(output["class_logits"].float().cpu().numpy())
            means.append(output["regression_mean"].float().cpu().numpy())
    logits_array = np.concatenate(logits)
    means_array = np.concatenate(means)
    labels = frame["NFE_Pseudo_Label"].map(LABEL_TO_INDEX).to_numpy()
    temperature = float(checkpoint.get("classification_temperature", 1.0))
    threshold = float(checkpoint.get("high_probability_threshold", 0.5))
    metrics = classification_metrics(logits_array / temperature, labels, high_threshold=threshold)
    predicted_score = np.clip(physical_regression(means_array, checkpoint)[:, 0], 0.0, 1.0)
    reference_score = frame["NFE_Pseudo_Score"].to_numpy(dtype=float)
    metrics["NFE_Pseudo_Score_mae"] = float(np.mean(np.abs(predicted_score - reference_score)))
    metrics["NFE_Pseudo_Score_spearman"] = spearman_rho(predicted_score, reference_score)
    metrics["NFE_Pseudo_Score_r2"] = r_squared(predicted_score, reference_score)
    if predictions_path is not None:
        probabilities = np.exp(logits_array / temperature - (logits_array / temperature).max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "Record_Index": frame.index.to_numpy(),
            "Structure_Name": frame["Structure_Name"].to_numpy(),
            "Split_Group": frame["Split_Group"].to_numpy(),
            "True_Label": frame["NFE_Pseudo_Label"].to_numpy(),
            "Predicted_Label": [LABELS[index] for index in probabilities.argmax(axis=1)],
            "Probability_Low": probabilities[:, 0],
            "Probability_Medium": probabilities[:, 1],
            "Probability_High": probabilities[:, 2],
            "True_NFE_Pseudo_Score": reference_score,
            "Predicted_NFE_Pseudo_Score": predicted_score,
            "Absolute_Score_Error": np.abs(predicted_score - reference_score),
        }).to_csv(predictions_path, index=False)
    stored = checkpoint.get("test_calibrated_metrics", {})
    return {
        "checkpoint": path.as_posix(),
        "checkpoint_sha256": sha256(path),
        "epoch": checkpoint.get("epoch"),
        "canonicalize": canonicalize,
        "complete_shells": complete_shells,
        "global_features": checkpoint["model_config"].get("global_features"),
        "temperature": temperature,
        "high_threshold": threshold,
        "local_test": {key: round(float(metrics[key]), 4) for key in KEYS + ("NFE_Pseudo_Score_mae", "NFE_Pseudo_Score_spearman", "NFE_Pseudo_Score_r2") if key in metrics},
        "stored_test_calibrated": {key: round(float(stored[key]), 4) for key in KEYS if key in stored},
        "confusion": [
            [int(metrics[f"confusion_true_{true}_pred_{pred}"]) for pred in LABELS] for true in LABELS
        ],
    }


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/full_v1_1/nfe_dataset.csv")
    parser.add_argument("--root", default="data/full_v1_1", help="directory whose data/ holds the structure copies")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="models/metadata/predictor_v1_1_evaluation.json")
    parser.add_argument(
        "--predictions-dir",
        help="write <name>_test_predictions.csv per checkpoint, in the column layout of the benchmark suite",
    )
    args = parser.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    table = (ROOT / args.table) if not Path(args.table).is_absolute() else Path(args.table)
    structure_dir = ((ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)) / "data"
    frame = pd.read_csv(table)
    frame = frame[frame["Suggested_Split"] == args.split].reset_index(drop=True)
    started = time.time()
    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/evaluate_predictor_checkpoints.py",
        "table": args.table,
        "table_sha256": sha256(table),
        "split": args.split,
        "rows": int(len(frame)),
        "device": str(device),
        "forward": "deterministic (model.eval(), no MC dropout); logits divided by the stored temperature",
    }
    predictions_dir = (ROOT / args.predictions_dir) if args.predictions_dir else None
    for name, path in parse_pairs(args.checkpoint):
        checkpoint_path = path if path.is_absolute() else ROOT / path
        predictions_path = (predictions_dir / f"{name}_test_predictions.csv") if predictions_dir else None
        entry = evaluate_checkpoint(checkpoint_path, frame, structure_dir, device, args.batch_size, predictions_path)
        entry["checkpoint"] = path.as_posix()
        report[name] = entry
        print(name, json.dumps(entry["local_test"]), flush=True)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
