# ==============================================================================
# 中文概述：把官方上游骨干基准套件（CGCNN、SchNet、ALIGNN、M3GNet）在 nfe-v1.1 表上的逐 seed 结果
#           收进本仓库的元数据，并核对它们共享同一张表、同一份结构清单与同一个划分。
# English overview: Collect the per-seed results of the official-upstream backbone suite (CGCNN, SchNet, ALIGNN,
#           M3GNet) on the nfe-v1.1 table into this repository's metadata, checking that every run shares one
#           table, one structure manifest and one split.
#
# 中文输入：基准套件的结果目录，形如 <root>/official-upstream/<model>/seed_<seed>/result.json。
# English inputs: The benchmark result directory, laid out as <root>/official-upstream/<model>/seed_<seed>/result.json.
# 中文输出：逐 seed 指标、均值 ± 标准差、参数量与来源指纹（JSON）。
# English outputs: Per-seed metrics, mean +/- standard deviation, parameter counts and source fingerprints (JSON).
#
# 关键约束 / Key invariants:
# - 只搬运结果文件里已经算好的数字，不重新评估，也不重新训练。
#   Only numbers already present in the result files are copied; nothing is re-evaluated or retrained.
# - 表哈希、结构清单哈希和划分大小必须在所有 run 之间一致，否则报错退出。
#   The table hash, structure manifest hash and split sizes must agree across every run, or the import fails.
# - 这些骨干网络由上游实现提供，任务头、划分与优化器是本项目代码；命名保留“official backbone”。
#   The backbones come from the upstream projects while the task head, split and optimizer are project code, so
#   the names keep the "official backbone" qualifier.
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
LABELS = {
    "cgcnn_official": "CGCNN (official backbone)",
    "schnet_official": "SchNet (SchNetPack)",
    "alignn_official": "ALIGNN (official backbone)",
    "m3gnet_official": "M3GNet (MatGL)",
}
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_average_precision",
    "macro_roc_auc",
    "ece",
    "low_f1",
    "medium_f1",
    "high_f1",
    "high_average_precision",
    "high_enrichment_at_5pct",
    "NFE_Pseudo_Score_mae",
    "NFE_Pseudo_Score_rmse",
    "NFE_Pseudo_Score_spearman",
    "NFE_Pseudo_Score_r2",
)


# 中文：均值与样本标准差。/ English: Mean and sample standard deviation.
def summarize(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "count": len(numbers),
    }


# 中文：读入一个模型的全部 seed。/ English: Read every seed of one model.
def read_model(directory: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload["provenance"]
        environment = provenance["runtime_environment"]
        runs.append({
            "seed": int(payload["seed"]),
            "parameter_count": payload.get("parameter_count"),
            "training_seconds": payload.get("training_seconds"),
            "best_epoch": payload.get("details", {}).get("best_epoch"),
            "temperature": payload.get("temperature"),
            "table_sha256": provenance["dataset_table_sha256"].upper(),
            "structure_manifest_sha256": provenance["structure_manifest_sha256"].upper(),
            "cache_records_sha256": provenance["cache_records_sha256"].upper(),
            "split_sizes": payload["split_sizes"],
            "git_commit": provenance["git_commit"][:12],
            "git_dirty": provenance["git_dirty"],
            "python": environment["python"],
            "torch": environment["torch"],
            "packages": environment.get("packages", {}),
            "metrics": {key: float(payload["test_metrics"][key])
                        for key in METRICS if key in payload["test_metrics"]},
        })
    if not runs:
        raise SystemExit(f"no result.json under {directory}")
    return {
        "label": LABELS.get(directory.name, directory.name),
        "seeds": [run["seed"] for run in runs],
        "parameter_count": sorted({run["parameter_count"] for run in runs}),
        "runs": runs,
        "summary": {key: summarize([run["metrics"][key] for run in runs if key in run["metrics"]])
                    for key in METRICS
                    if any(key in run["metrics"] for run in runs)},
    }


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True,
                        help="benchmark results directory holding official-upstream/<model>/seed_<seed>/result.json")
    parser.add_argument("--track", default="official-upstream")
    parser.add_argument("--output", default="models/metadata/official_baselines_v1_1.json")
    parser.add_argument("--expect-table-sha256", default=None,
                        help="fail unless every run used this dataset table")
    args = parser.parse_args(argv)

    track = Path(args.results_root) / args.track
    if not track.is_dir():
        raise SystemExit(f"{track} is not a directory")
    models = {directory.name: read_model(directory)
              for directory in sorted(track.iterdir()) if directory.is_dir()}
    if not models:
        raise SystemExit(f"no model directories under {track}")

    identities = {name: {
        "table": {run["table_sha256"] for run in model["runs"]},
        "manifest": {run["structure_manifest_sha256"] for run in model["runs"]},
        "cache": {run["cache_records_sha256"] for run in model["runs"]},
        "split": {json.dumps(run["split_sizes"], sort_keys=True) for run in model["runs"]},
        "dirty": {run["git_dirty"] for run in model["runs"]},
    } for name, model in models.items()}
    for field in ("table", "manifest", "cache", "split"):
        values = set().union(*(identity[field] for identity in identities.values()))
        if len(values) != 1:
            raise SystemExit(f"runs disagree on {field}: {sorted(values)}")
    if any(True in identity["dirty"] or None in identity["dirty"] for identity in identities.values()):
        raise SystemExit("at least one run came from a dirty or unknown Git worktree")
    table_sha256 = next(iter(identities[next(iter(identities))]["table"]))
    if args.expect_table_sha256 and table_sha256 != args.expect_table_sha256.upper():
        raise SystemExit(f"runs used table {table_sha256}, expected {args.expect_table_sha256.upper()}")

    first = next(iter(models.values()))["runs"][0]
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/import_official_baselines.py",
        "source_root": Path(args.results_root).as_posix(),
        "track": args.track,
        "table_sha256": table_sha256,
        "structure_manifest_sha256": first["structure_manifest_sha256"],
        "cache_records_sha256": first["cache_records_sha256"],
        "split_sizes": first["split_sizes"],
        "note": (
            "Upstream message-passing backbones with this project's split, task head, optimizer and calibration; "
            "not untouched upstream training pipelines. Every run is pure supervised training on the NFE class "
            "and pseudo-score, without the self-supervised schedule of this work."
        ),
        "models": models,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, model in models.items():
        summary = model["summary"]
        print(f"{name:18s} seeds={model['seeds']} macro_f1="
              f"{summary['macro_f1']['mean']:.4f}±{summary['macro_f1']['std']:.4f}", flush=True)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
