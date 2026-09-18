# ==============================================================================
# 中文概述：用 Windows 程序的严格生成流水线（流 + 流形投影 + 拓扑门控 + NFE 复评 + CHGNet 预弛豫）比较多个生成器
#           检查点：同一预测器、同一骨架、同一随机种子，统计成功组数、导出候选、目标概率、CHGNet 受力与淘汰原因。
# English overview: Compare generator checkpoints with the Windows strict generation pipeline (flow + manifold
#           projection + topology gates + NFE re-scoring + CHGNet pre-relaxation): same predictor, skeletons and
#           seeds; report successful cases, exported candidates, target probability, CHGNet forces and rejections.
#
# 中文输入：预测器、表面几何摘要、若干 名称=生成器路径（生成器须位于 <root>/models/，摘要位于 <root>/resources/）。
# English inputs: Predictor, surface profile, and name=generator pairs (generator under <root>/models/ with the
#                 profile at <root>/resources/, as in the Windows application layout).
# 中文输出：逐组记录与汇总（JSON）。需要 CHGNet（Windows 构建环境）。
# English outputs: Per-case records and a summary (JSON). Requires CHGNet (the Windows build environment).
#
# Author: Ruck
# Generated: 2026-09-16
# ==============================================================================

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.windows.nfe_mxene_studio.backend import ModelPaths, NFEEngine  # noqa: E402

DEFAULT_SKELETONS = ("Ti-C-Nb", "Sc-C-Ta", "Zr-N-Ti", "Mo-C-V")
TARGETS = ("low", "medium", "high")


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--surface-profile", required=True)
    parser.add_argument("--generator", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--skeleton", action="append", help="bottom-core-top, repeatable (default: 4 skeletons)")
    parser.add_argument("--number", type=int, default=2)
    parser.add_argument("--oversample", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--relax-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--work-dir", default="runs/generation_benchmark")
    parser.add_argument("--output", default="models/metadata/generator_strict_generation_benchmark.json")
    args = parser.parse_args(argv)

    skeletons = [tuple(item.split("-")) for item in (args.skeleton or DEFAULT_SKELETONS)]
    generators = [tuple(item.split("=", 1)) for item in args.generator]
    work_dir = ROOT / args.work_dir
    records: list[dict[str, Any]] = []
    for name, path in generators:
        engine = NFEEngine(ModelPaths(Path(args.predictor), Path(path), Path(args.surface_profile)))
        for bottom, core, top in skeletons:
            for target in TARGETS:
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                started = time.time()
                record: dict[str, Any] = {"generator": name, "skeleton": f"{bottom}-{core}-{top}", "target": target}
                attempts: list[dict[str, Any]] = []
                try:
                    payload = engine.generate_skeleton(
                        bottom_metal=bottom, core_element=core, top_metal=top, target=target, number=args.number,
                        output_parent=work_dir / name, oversample=args.oversample, mc_samples=args.mc_samples,
                        sampling_steps=args.steps, relax_steps=args.relax_steps,
                    )
                    rows = payload["rows"]
                    attempts = payload.get("attempts", [])
                    record.update(
                        status="ok",
                        accepted=len(rows),
                        attempts=len(attempts),
                        target_probability=[round(float(row.get("Target_Probability", float("nan"))), 3) for row in rows],
                        chgnet_fmax_eV_A=[round(float(row.get("CHGNet_Max_Force_eV_A", float("nan"))), 4) for row in rows],
                        formulas=[row.get("Formula") for row in rows],
                    )
                except Exception as exc:  # noqa: BLE001 - a failed case is a result, not a crash
                    record.update(status="failed", accepted=0, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                reasons: Counter[str] = Counter()
                for attempt in attempts:
                    for reason, count in (attempt.get("rejection_reasons") or {}).items():
                        reasons[reason] += int(count)
                record["rejections"] = dict(reasons.most_common(6))
                record["seconds"] = round(time.time() - started, 1)
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {}
    for name, _path in generators:
        subset = [record for record in records if record["generator"] == name]
        probabilities = [value for record in subset for value in record.get("target_probability", [])]
        forces = [value for record in subset for value in record.get("chgnet_fmax_eV_A", [])]
        summary[name] = {
            "cases": len(subset),
            "succeeded": sum(1 for record in subset if record["status"] == "ok"),
            "candidates": sum(record["accepted"] for record in subset),
            "succeeded_by_target": {target: sum(1 for r in subset if r["target"] == target and r["status"] == "ok") for target in TARGETS},
            "mean_target_probability": round(float(np.mean(probabilities)), 3) if probabilities else None,
            "mean_chgnet_fmax_eV_A": round(float(np.mean(forces)), 4) if forces else None,
            "seconds": round(sum(record["seconds"] for record in subset), 1),
        }
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/benchmark_strict_generation.py",
        "predictor": args.predictor,
        "generators": dict(generators),
        "settings": {key: getattr(args, key) for key in ("number", "oversample", "mc_samples", "steps", "relax_steps", "seed")},
        "skeletons": ["-".join(item) for item in skeletons],
        "summary": summary,
        "records": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
