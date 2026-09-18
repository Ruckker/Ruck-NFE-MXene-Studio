# ==============================================================================
# 中文概述：汇总多 seed 训练结果，输出均值 ± 标准差；单个 seed 的指标不应作为最终报告。
# English overview: Aggregate multi-seed runs into mean ± std; a single seed is never the final report.
#
# 中文输入：形如 <root>/<variant>/seed_*/final_metrics.json 或 <root>/seed_*/final_metrics.json 的目录树。
# English inputs: Directory trees shaped like <root>/<variant>/seed_*/final_metrics.json or <root>/seed_*/final_metrics.json.
# 中文输出：JSON 汇总与 Markdown 表。
# English outputs: A JSON summary and a Markdown table.
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

DEFAULT_KEYS = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "macro_roc_auc",
    "low_f1",
    "high_f1",
    "high_vs_rest_f1",
    "high_vs_rest_roc_auc",
    "NFE_Pseudo_Score_mae",
    "ece",
)


def collect(root: Path, block: str) -> dict[str, dict[str, list[float]]]:
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.glob("**/seed_*/final_metrics.json")):
        variant = path.parent.parent.name if path.parent.parent != root else root.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get(block) or payload.get("test") or {}
        fallback = payload.get("test", {})
        for key in set(metrics) | set(fallback):
            value = metrics.get(key, fallback.get(key))
            if isinstance(value, (int, float)):
                table[variant][key].append(float(value))
    return table


def summarize(table: dict[str, dict[str, list[float]]], keys: Sequence[str]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for variant, metrics in sorted(table.items()):
        summary[variant] = {}
        for key in keys:
            values = metrics.get(key, [])
            if values:
                summary[variant][key] = {
                    "mean": statistics.mean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
    return summary


def markdown(summary: dict[str, dict[str, dict[str, float]]], keys: Sequence[str]) -> str:
    header = "| variant | n | " + " | ".join(keys) + " |"
    rule = "|---|---:|" + "|".join("---:" for _ in keys) + "|"
    lines = [header, rule]
    for variant, metrics in summary.items():
        n = max((item["n"] for item in metrics.values()), default=0)
        cells = [
            f"{metrics[key]['mean']:.4f} ± {metrics[key]['std']:.4f}" if key in metrics else "-"
            for key in keys
        ]
        lines.append(f"| {variant} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed final_metrics.json files.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--block", default="test_calibrated", help="metrics block to read (falls back to 'test')")
    parser.add_argument("--keys", nargs="+", default=list(DEFAULT_KEYS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    table = collect(args.root.resolve(), args.block)
    if not table:
        raise SystemExit(f"no seed_*/final_metrics.json under {args.root}")
    summary = summarize(table, args.keys)
    text = markdown(summary, args.keys)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "root": str(args.root.resolve()),
                    "block": args.block,
                    "summary": summary,
                    "markdown": text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
