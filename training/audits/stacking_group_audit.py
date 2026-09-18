# ==============================================================================
# 中文概述：把测试集按“固定组成后标签是否唯一”分成两组，分别统计准确率。组成可定的那组衡量
#           化学先验，靠堆垛区分的那组衡量几何辨别力，后者是本项目误差的主要来源。
# English overview: Split the test set by whether a fixed composition already fixes the label, then score each
#           group separately. The composition-decided group measures the chemical prior; the stacking-decided
#           group measures geometric discrimination, and that is where this project's errors concentrate.
#
# 中文输入：数据表与若干预测 CSV（列格式同基准套件），可按模型分组给出多 seed 均值。
# English inputs: The dataset table and prediction CSV files in the benchmark suite column layout, optionally
#           grouped per model so several seeds are averaged.
# 中文输出：每个模型在两组上的准确率、误差占比与逐档 F1（JSON）。
# English outputs: Per-model accuracy on both groups, the share of errors they carry and per-class F1 (JSON).
#
# 关键约束 / Key invariants:
# - 分组只用组成字段（两侧金属、X、两侧端基），不使用堆垛，也不使用任何预测值。
#   The grouping uses only the composition fields (both metals, X, both terminations); neither the stacking nor
#   any prediction enters it.
# - 每个模型的 seed 之间按概率平均，单 seed 结果也一并给出。
#   Seeds of one model are averaged in probability space, and the single-seed result is reported next to it.
# - 主要接口 / Main APIs: group_mask, evaluate, main
#
# Author: Ruck
# Generated: 2026-09-16
# ==============================================================================

from __future__ import annotations

import argparse
import json
import time
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("low", "medium", "high")
PROBABILITY_COLUMNS = ["Probability_Low", "Probability_Medium", "Probability_High"]
COMPOSITION_FIELDS = ("Metal_Top", "Metal_Bottom", "X_Element", "Termination_Top", "Termination_Bottom")


# 中文：按组成分组，标出标签不唯一的那部分。/ English: Group by composition and mark the rows whose label is not unique.
def group_mask(table: pd.DataFrame, names: Sequence[str], truth: np.ndarray) -> np.ndarray:
    ordered = table.set_index("Structure_Name").loc[list(names)]
    key = ordered[list(COMPOSITION_FIELDS)].astype(str).agg("|".join, axis=1).to_numpy()
    unique = pd.DataFrame({"key": key, "truth": truth}).groupby("key")["truth"].nunique()
    return np.isin(key, unique[unique > 1].index.to_numpy())


# 中文：单个模型的评估。/ English: Evaluate one model.
def evaluate(paths: Sequence[Path], truth: np.ndarray, mixed: np.ndarray) -> dict[str, Any]:
    probabilities = [pd.read_csv(path).sort_values("Structure_Name")[PROBABILITY_COLUMNS].to_numpy(dtype=float)
                     for path in paths]
    ensemble = np.mean(probabilities, axis=0).argmax(axis=1)
    singles = [p.argmax(axis=1) for p in probabilities]

    def scores(prediction: np.ndarray) -> dict[str, float]:
        per_class = []
        for index in range(3):
            tp = float(((prediction == index) & (truth == index)).sum())
            fp = float(((prediction == index) & (truth != index)).sum())
            fn = float(((prediction != index) & (truth == index)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            per_class.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        wrong = prediction != truth
        return {
            "macro_f1": float(np.mean(per_class)),
            **{f"{name}_f1": float(value) for name, value in zip(CLASSES, per_class)},
            "accuracy": float((~wrong).mean()),
            "accuracy_stacking_decided": float((~wrong[mixed]).mean()) if mixed.any() else float("nan"),
            "accuracy_composition_decided": float((~wrong[~mixed]).mean()) if (~mixed).any() else float("nan"),
            "errors": int(wrong.sum()),
            "error_share_stacking_decided": float(wrong[mixed].sum() / wrong.sum()) if wrong.sum() else 0.0,
        }

    single = [scores(prediction) for prediction in singles]
    return {
        "seeds": len(paths),
        "single_mean": {key: float(np.mean([entry[key] for entry in single])) for key in single[0]},
        "single_std": {key: float(np.std([entry[key] for entry in single], ddof=1)) if len(single) > 1 else 0.0
                       for key in single[0]},
        "ensemble": scores(ensemble),
    }


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/full_v1_1/nfe_dataset.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", action="append", required=True,
                        help="name=glob, repeatable; the glob selects one prediction CSV per seed")
    parser.add_argument("--output", default="models/metadata/stacking_group_audit.json")
    args = parser.parse_args(argv)

    table = pd.read_csv(ROOT / args.table if not Path(args.table).is_absolute() else args.table)
    table = table[table["Suggested_Split"] == args.split]

    models: dict[str, list[Path]] = {}
    for entry in args.model:
        if "=" not in entry:
            raise SystemExit(f"--model expects name=glob, got {entry!r}")
        name, pattern = entry.split("=", 1)
        paths = sorted(Path(path) for path in glob(pattern.strip()))
        if not paths:
            raise SystemExit(f"no prediction files match {pattern!r}")
        models[name.strip()] = paths

    reference = pd.read_csv(next(iter(models.values()))[0]).sort_values("Structure_Name")
    names = reference["Structure_Name"].to_numpy()
    truth = reference["True_Label"].map({name: index for index, name in enumerate(CLASSES)}).to_numpy()
    mixed = group_mask(table, names, truth)

    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/stacking_group_audit.py",
        "table": args.table,
        "split": args.split,
        "rows": int(len(truth)),
        "stacking_decided_rows": int(mixed.sum()),
        "composition_decided_rows": int((~mixed).sum()),
        "models": {},
    }
    for name, paths in models.items():
        frame = pd.read_csv(paths[0]).sort_values("Structure_Name")
        if not np.array_equal(frame["Structure_Name"].to_numpy(), names):
            raise SystemExit(f"{name} does not cover the same structures")
        report["models"][name] = evaluate(paths, truth, mixed)
        single, ensemble = report["models"][name]["single_mean"], report["models"][name]["ensemble"]
        print(f"{name:28s} seeds={report['models'][name]['seeds']} "
              f"macro_f1={single['macro_f1']:.4f} stacking={single['accuracy_stacking_decided']:.4f} "
              f"composition={single['accuracy_composition_decided']:.4f} "
              f"| ensemble macro_f1={ensemble['macro_f1']:.4f} stacking={ensemble['accuracy_stacking_decided']:.4f}",
              flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
