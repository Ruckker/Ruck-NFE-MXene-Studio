# ==============================================================================
# 中文概述：无需 GUI 的 Windows 后端冒烟测试。
# English overview: Headless smoke test for the Windows backend.
#
# 中文输入：应用资源目录和样例结构。
# English inputs: Application resource directory and sample structures.
# 中文输出：模型加载、预测和小规模生成的机器可读结果。
# English outputs: Machine-readable model-loading, prediction, and small-generation results.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: parse_args, main
#
# Author: Ruck
# Generated: 2026-07-30 07:33:19 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 中文：兼容可重建源码包与 GitHub 分层源码两种包名。
# English: Support both the rebuildable source package and layered GitHub package.
try:
    from windows_app.backend import NFEEngine
except ModuleNotFoundError:
    from .backend import NFEEngine


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFE MXene Windows backend smoke test")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=[
            "examples/structures/sample_low_ScTaCSeBr.cif",
            "examples/structures/sample_medium_TiNbCSeCl.cif",
            "examples/structures/sample_high_ZrTiHSNO.cif",
        ],
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--bottom-metal", default="Sc")
    parser.add_argument("--core", choices=("C", "N"), default="C")
    parser.add_argument("--top-metal", default="Ta")
    parser.add_argument("--target", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--output", default="runs/windows_app_smoke_generation")
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--relax-steps", type=int, default=100)
    return parser.parse_args()


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main() -> int:
    args = parse_args()
    engine = NFEEngine()
    rows = engine.predict_files(
        [Path(value) for value in args.samples],
        mc_samples=5,
        progress=print,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    if args.generate:
        generated = engine.generate_skeleton(
            bottom_metal=args.bottom_metal,
            core_element=args.core,
            top_metal=args.top_metal,
            target=args.target,
            number=1,
            output_parent=Path(args.output),
            oversample=args.oversample,
            mc_samples=5,
            sampling_steps=args.sampling_steps,
            relax_steps=args.relax_steps,
            progress=print,
        )
        print(json.dumps(generated, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
