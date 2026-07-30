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
# - 主要接口 / Main APIs: main
#
# Author: Ruck
# Generated: 2026-07-29 19:06:31 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import json
import platform
import sys
from importlib import import_module, metadata


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    errors: list[str] = []
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pymatgen": "pymatgen.core",
        "PyYAML": "yaml",
    }
    for distribution, module in packages.items():
        try:
            import_module(module)
            report[distribution] = metadata.version(distribution)
        except Exception as exc:
            report[distribution] = "missing/broken"
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")
    try:
        import_module("chgnet")
        report["chgnet_optional"] = metadata.version("chgnet")
    except Exception:
        report["chgnet_optional"] = "not installed"
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "memory_GiB": round(
                            torch.cuda.get_device_properties(index).total_memory
                            / 1024**3,
                            2,
                        ),
                    }
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
        python_version = tuple(int(value) for value in sys.version_info[:2])
        if python_version < (3, 10):
            errors.append(
                "Python 3.10+ is recommended for current CUDA PyTorch releases"
            )
        torch_version = tuple(
            int(part)
            for part in torch.__version__.split("+", 1)[0].split(".")[:2]
        )
        if torch_version < (2, 6):
            errors.append(
                "PyTorch 2.6+ is recommended on the training server; "
                "the local compatibility smoke test may still run on older builds"
            )
        if not torch.cuda.is_available():
            errors.append("PyTorch cannot see a CUDA GPU")
        if torch.cuda.device_count() < 1:
            errors.append("no CUDA device is visible")
    except Exception as exc:
        report["torch"] = "missing/broken"
        errors.append(f"torch: {type(exc).__name__}: {exc}")
    report["status"] = "ok" if not errors else "attention"
    report["messages"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
