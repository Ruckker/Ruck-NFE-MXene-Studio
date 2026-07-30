# ==============================================================================
# 中文概述：仅打包最终 NFE 预测器、表面约束生成器训练骨干、manifold generator 生成层及全部最终测试。
# English overview: Package only the final NFE predictor, surface generator training backbone,
# manifold generation layer, and all final tests.
#
# Author: Ruck
# Generated: 2026-07-30 10:36:00 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path


INCLUDED_ROOTS = (
    "src",
    "training",
    "data_tools",
    "tests",
    "environment",
)
INCLUDED_FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "CITATION.cff",
    "docs/SCIENTIFIC_OVERVIEW.md",
    "docs/TECH_STACK.md",
    "docs/ARCHITECTURE.md",
    "models/MODEL_CARD.md",
)


# 中文：计算归档哈希。/ English: Compute the archive hash.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


# 中文：只收集最终源码，明确排除缓存、解压环境与非源码数据。
# English: Collect final source only, excluding caches, installed assets, and data.
def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in INCLUDED_ROOTS:
        base = root / name
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if (
                "__pycache__" in relative.parts
                or path.suffix == ".pyc"
                or "server" in relative.parts
                or "full" in relative.parts
            ):
                continue
            files.append(path)
    files.extend(root / name for name in INCLUDED_FILES)
    return sorted(set(files))


# 中文：写入 final-only ZIP 和清单。/ English: Write the final-only ZIP and manifest.
def package(root: Path, output: Path, manifest_path: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    files = source_files(root)
    prefix = Path("NFE-MXene-Studio-Final-Source")
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        archive.comment = b"NFE MXene Studio Final Training Source | Author: Ruck"
        for path in files:
            archive.write(path, (prefix / path.relative_to(root)).as_posix())
    result = {
        "project": "NFE MXene Studio",
        "scope": "final training and inference source only",
        "author": "Ruck",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_count": len(files),
        "archive_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "included_components": {
            "predictor": "NFE predictor",
            "generator_training": "surface-template generator",
            "generator_inference": "manifold generator",
            "windows": "Windows application 1.0",
        },
        "removed_legacy_categories": [
            "baseline generator implementation",
            "development-only generation entry points",
            "obsolete default configuration",
            "intermediate test copies",
        ],
        "final_test_files": [
            "tests/test_smoke.py",
            "tests/test_manifold_generation.py",
            "tests/test_windows_preview.py",
            "tests/test_generation_progress.py",
        ],
    }
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


# 中文：命令行入口。/ English: Command-line entry point.
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package(
                args.root.resolve(),
                args.output.resolve(),
                args.manifest.resolve(),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
