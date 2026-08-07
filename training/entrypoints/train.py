# ==============================================================================
# 中文概述：提供 NFE MXene 项目中的单一、可复用源码职责。
# English overview: Provide one reusable source-code responsibility in the NFE MXene project.
#
# 中文输入：请结合类型标注、命令行帮助和调用方查看输入。
# English inputs: Read type hints, CLI help, and callers for the expected inputs.
# 中文输出：返回值或生成文件由公开接口和命令行参数定义。
# English outputs: Return values or generated files are defined by public APIs and CLI arguments.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: parse_launcher_args, worker_arguments, main
#
# Author: Ruck
# Generated: 2026-07-29 19:06:31 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parse_launcher_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch RUCK-DP training on one to four GPUs.",
        add_help=True,
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        choices=range(1, 5),
        metavar="{1,2,3,4}",
        help="number of local GPUs/processes",
    )
    parser.add_argument(
        "--task",
        choices=("predictor", "generator", "all"),
        default="predictor",
        help="model stage to train",
    )
    parser.add_argument(
        "--devices",
        help="optional visible GPU IDs, for example 0,2; count must match --gpus",
    )
    parser.add_argument(
        "--config",
        help="optional explicit YAML; defaults to the NFE predictor or surface-generator configuration",
    )
    parser.add_argument("--resume")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv), []


def worker_arguments(args: argparse.Namespace, task: str) -> list[str]:
    default_config = (
        "training/configs/nfe_predictor.yaml"
        if task == "predictor"
        else "training/configs/surface_generator.yaml"
    )
    result = ["--config", str(Path(args.config or default_config).resolve())]
    if args.resume:
        result.extend(["--resume", str(Path(args.resume).resolve())])
    if args.rebuild_cache:
        result.append("--rebuild-cache")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = parse_launcher_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    env = os.environ.copy()

    if args.devices:
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        if len(devices) != args.gpus:
            raise SystemExit(
                f"--devices contains {len(devices)} IDs but --gpus={args.gpus}"
            )
        if len(set(devices)) != len(devices):
            raise SystemExit("--devices contains duplicate GPU IDs")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build and run "
            "`python check_environment.py` before training."
        )
    if torch.cuda.device_count() < args.gpus:
        raise SystemExit(
            f"requested {args.gpus} GPUs, but only {torch.cuda.device_count()} "
            "CUDA devices are visible"
        )

    if args.task == "all" and args.resume:
        raise SystemExit("--resume cannot be combined with --task all")

    # Predictor work must enter through nfe_model.train: that public module owns
    # v2.2 formal-config validation, audited cache/provenance and resume guards.
    # train_audited remains only as a backward-compatible alias.
    modules = {
        "predictor": "nfe_model.train",
        "generator": "nfe_model.train_surface_generator",
    }
    if int(env.get("WORLD_SIZE", "1")) > 1:
        if args.task == "all":
            raise SystemExit("--task all must be launched directly, not inside torchrun")
        if args.task == "predictor":
            from nfe_model.train import main as worker_main
        else:
            from nfe_model.train_surface_generator import main as worker_main
        return worker_main(worker_arguments(args, args.task))

    tasks = ("predictor", "generator") if args.task == "all" else (args.task,)
    for task in tasks:
        task_arguments = worker_arguments(args, task)
        if (
            args.task == "all"
            and task == "generator"
            and "--rebuild-cache" in task_arguments
        ):
            task_arguments.remove("--rebuild-cache")
        if args.gpus == 1:
            if task == "predictor":
                from nfe_model.train import main as worker_main
            else:
                from nfe_model.train_surface_generator import main as worker_main
            return_code = worker_main(task_arguments)
        else:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.gpus}",
                "-m",
                modules[task],
                *task_arguments,
            ]
            print(f"Launching {task}:", " ".join(command), flush=True)
            return_code = subprocess.run(
                command, cwd=project_root, env=env, check=False
            ).returncode
        if return_code:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
