# ==============================================================================
# 中文概述：把组成基线、官方上游骨干与本模型的 nfe-v1.1 测试集结果汇成一张 Markdown 表，
#           供 README、科学概述与模型卡直接引用，避免手工誊抄数字。
# English overview: Collect the nfe-v1.1 test results of the composition baselines, the official-upstream backbones
#           and this work into one Markdown table for the README, the scientific overview and the model card, so
#           the numbers are never retyped by hand.
#
# 中文输入：baseline_metrics_v1_1.json、official_baselines_v1_1.json、predictor_v1_1_evaluation.json。
# English inputs: baseline_metrics_v1_1.json, official_baselines_v1_1.json, predictor_v1_1_evaluation.json.
# 中文输出：Markdown 表（stdout 与文件）以及同样数字的 JSON。
# English outputs: A Markdown table (stdout and file) plus the same numbers as JSON.
#
# 关键约束 / Key invariants:
# - 三个来源必须指向同一张表；否则拒绝出表。
#   The three sources must point at one dataset table, or the table is refused.
# - 只读取已经算好的数字。/ Only numbers that were already computed are read.
# - 主要接口 / Main APIs: main
#
# Author: Ruck
# Generated: 2026-09-16
# ==============================================================================

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"high_average_precision": "high_vs_rest_average_precision"}
COLUMNS = [
    ("macro F1", "macro_f1", "{:.3f}"),
    ("macro AP", "macro_average_precision", "{:.3f}"),
    ("macro ROC-AUC", "macro_roc_auc", "{:.3f}"),
    ("high AP", "high_average_precision", "{:.3f}"),
    ("high F1", "high_f1", "{:.3f}"),
    ("富集 @5%", "high_enrichment_at_5pct", "{:.2f}"),
    ("ECE", "ece", "{:.4f}"),
    ("分数 MAE", "NFE_Pseudo_Score_mae", "{:.4f}"),
]
BASELINE_ROWS = [
    ("rule_has_OH_is_high", "规则：含 OH 即 high", 1),
    ("logistic_regression_composition", "逻辑回归（组成）", None),
    ("mlp128_composition", "MLP-128（组成）", None),
    ("mlp128_composition_geometry", "MLP-128（组成 + 几何）", None),
]
OFFICIAL_ROWS = [
    ("cgcnn_official", "CGCNN（官方骨干）"),
    ("schnet_official", "SchNet（SchNetPack）"),
    ("alignn_official", "ALIGNN（官方骨干）"),
    ("m3gnet_official", "M3GNet（MatGL）"),
]


# 中文：把均值与标准差写成单元格。/ English: Render a mean and standard deviation as a cell.
def cell(mean: float | None, deviation: float | None, fmt: str) -> str:
    if mean is None:
        return "—"
    if not deviation:
        return fmt.format(mean)
    return f"{fmt.format(mean)} ± {fmt.format(deviation)}"


def summarize(values: Sequence[float]) -> tuple[float, float]:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers), (statistics.stdev(numbers) if len(numbers) > 1 else 0.0)


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", default="models/metadata/baseline_metrics_v1_1.json")
    parser.add_argument("--official", default="models/metadata/official_baselines_v1_1.json")
    parser.add_argument("--evaluation", default="models/metadata/predictor_v1_1_evaluation.json")
    parser.add_argument("--seeds", nargs="+",
                        default=["v1_1_seed2027", "v1_1_seed2028", "v1_1_seed2029",
                                 "v1_1_seed2030", "v1_1_seed2031"])
    parser.add_argument("--output", default="models/metadata/benchmark_table_v1_1.md")
    parser.add_argument("--json-output", default="models/metadata/benchmark_table_v1_1.json")
    args = parser.parse_args(argv)

    baselines = json.loads((ROOT / args.baselines).read_text(encoding="utf-8"))
    official = json.loads((ROOT / args.official).read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / args.evaluation).read_text(encoding="utf-8"))
    tables = {official["table_sha256"].upper(), evaluation["table_sha256"].upper(),
              str(baselines.get("table_sha256", "")).upper()}
    if len(tables) != 1:
        raise SystemExit(f"sources disagree on the dataset table: {sorted(tables)}")

    rows: list[dict[str, Any]] = []
    for key, label, repeats in BASELINE_ROWS:
        summary = baselines["results"].get(key, {}).get("summary", {})
        values = {}
        for _name, metric, _fmt in COLUMNS:
            entry = summary.get(ALIASES.get(metric, metric))
            values[metric] = (float(entry["mean"]), float(entry["std"])) if entry else None
        rows.append({"model": label, "seeds": repeats or len(baselines.get("seeds", [])), "values": values})
    for key, label in OFFICIAL_ROWS:
        model = official["models"].get(key)
        if model is None:
            continue
        values = {}
        for _name, metric, _fmt in COLUMNS:
            entry = model["summary"].get(metric)
            values[metric] = (float(entry["mean"]), float(entry["std"])) if entry else None
        rows.append({"model": label, "seeds": len(model["seeds"]), "values": values})
    values = {}
    for _name, metric, _fmt in COLUMNS:
        local = ALIASES.get(metric, metric)
        collected = [evaluation[name]["local_test"][local] for name in args.seeds
                     if name in evaluation and local in evaluation[name]["local_test"]]
        values[metric] = summarize(collected) if collected else None
    rows.append({"model": "**本模型（NFE 预测器 1.1.0）**", "seeds": len(args.seeds), "values": values})

    header = "| 模型 | seed | " + " | ".join(name for name, _m, _f in COLUMNS) + " |"
    divider = "|---|---:|" + "---:|" * len(COLUMNS)
    lines = [header, divider]
    for row in rows:
        cells = [cell(*(row["values"][metric] or (None, None)), fmt) for _n, metric, fmt in COLUMNS]
        lines.append(f"| {row['model']} | {row['seeds']} | " + " | ".join(cells) + " |")
    markdown = "\n".join(lines) + "\n"
    (ROOT / args.output).write_text(markdown, encoding="utf-8")
    (ROOT / args.json_output).write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/write_benchmark_table.py",
        "table_sha256": next(iter(tables)),
        "columns": [{"name": name, "metric": metric} for name, metric, _fmt in COLUMNS],
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
