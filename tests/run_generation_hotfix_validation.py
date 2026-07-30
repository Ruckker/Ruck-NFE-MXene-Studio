"""Windows strict-generation hotfix integration check.

中文：使用正式模型执行一次真实的严格生成，验证 GUI 后端的两轮重试、
CHGNet 候选池扩展以及失败诊断。此脚本只创建新的输出目录，不删除或覆盖文件。
English: Run a real strict-generation request with the release models to verify
two-attempt retrying, CHGNet pool expansion, and actionable failure diagnostics.
The script only creates new outputs and never deletes or overwrites files.

Author: Ruck
Generated: 2026-07-30 22:02:00 Asia/Shanghai
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.windows.nfe_mxene_studio.backend import ModelPaths, NFEEngine


def parse_args() -> argparse.Namespace:
    """Parse paths separately so the check also works outside the build tree."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--number", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    """Run the same Nb-C-Nb/low request that exposed the desktop freeze."""
    args = parse_args()
    root = args.model_root.expanduser().resolve()
    paths = ModelPaths(
        predictor=root / "models" / "nfe_predictor.pt",
        generator=root / "models" / "mxene_generator.pt",
        surface_profile=root / "resources" / "surface_geometry_summary.json",
    )
    engine = NFEEngine(paths, device=args.device)

    def progress(message: str, percent: float | None = None) -> None:
        """Flush every update so a supervising process can follow the run."""
        marker = "-" if percent is None else f"{percent:.2f}"
        print(f"[{marker}] {message}", flush=True)

    try:
        result = engine.generate_skeleton(
            bottom_metal="Nb",
            core_element="C",
            top_metal="Nb",
            target="low",
            number=args.number,
            output_parent=args.output.expanduser().resolve(),
            oversample=48,
            sampling_steps=80,
            mc_samples=20,
            relax_steps=250,
            progress=progress,
        )
    except RuntimeError as exc:
        # A strict 0-candidate outcome is allowed for this regression check.
        # The important contract is that the exception is prompt and diagnostic.
        print(f"STRICT_RESULT: {type(exc).__name__}: {exc}", flush=True)
        return 2

    print(f"STRICT_RESULT: success: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
