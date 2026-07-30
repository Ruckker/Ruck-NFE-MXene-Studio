"""
中文：把完整 GitHub 目录连同服务器/Windows 发行资产打包为一个通用 ZIP64。
English: Package the complete GitHub tree and its server/Windows release assets
as one portable ZIP64 archive.

Author: Ruck
Generated: 2026-07-30 10:13:00 Asia/Shanghai

本地 `_staging`、Python 缓存和临时 partial 文件不会进入公开总包。
Local staging, Python caches, and partial files are excluded from the public bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path


STORED_SUFFIXES = {".zip", ".pt", ".exe", ".dll", ".pyd", ".png", ".jpg", ".jpeg"}


# 中文：流式 SHA256。/ English: Streaming SHA256.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


# 中文：判断是否为公开总包应忽略的路径。/ English: Decide whether a path is excluded.
def excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        "_staging" in relative.parts
        or "__pycache__" in relative.parts
        or path.suffix == ".pyc"
        or ".partial-" in path.name
    )


# 中文：创建总包并写外部清单。/ English: Create the bundle and its external manifest.
def package(root: Path, output: Path, manifest: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and not excluded(path, root)
    )
    total = sum(path.stat().st_size for path in files)
    start = time.time()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        archive.comment = b"NFE MXene Studio GitHub Bundle | Author: Ruck | 2026-07-30"
        processed = 0
        for index, path in enumerate(files, start=1):
            arcname = (Path(root.name) / path.relative_to(root)).as_posix()
            compression = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in STORED_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, arcname, compress_type=compression, compresslevel=1)
            processed += path.stat().st_size
            if index % 20 == 0 or index == len(files):
                print(
                    json.dumps(
                        {
                            "files": index,
                            "total_files": len(files),
                            "percent": round(100 * processed / max(total, 1), 2),
                        }
                    ),
                    flush=True,
                )
    result = {
        "project": "NFE MXene Studio",
        "version": "1.0",
        "author": "Ruck",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(root),
        "archive": str(output),
        "file_count": len(files),
        "uncompressed_bytes": total,
        "archive_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "excluded": ["_staging", "__pycache__", "*.pyc", "*.partial-*"],
        "elapsed_seconds": round(time.time() - start, 2),
    }
    manifest.write_text(
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
