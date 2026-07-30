"""
中文：将最终 Windows application 1.0 onedir 程序打包为支持大于 4 GiB 的通用 ZIP64。
English: Package the final Windows application 1.0 onedir application as a portable ZIP64 archive.

Author: Ruck
Generated: 2026-07-30 09:31:00 Asia/Shanghai

脚本只读取发布目录并创建新的时间戳归档，不删除或覆盖既有文件。
The script only reads the release directory and creates a new timestamped archive;
it never deletes or overwrites existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


# 中文：以流式方式计算大文件哈希，避免一次性占用内存。
# English: Hash large files as a stream to keep memory use bounded.
def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


# 中文：生成 ZIP64，并把最小发布元数据直接写入归档。
# English: Create ZIP64 and embed minimal release metadata directly in the archive.
def package(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    started = time.time()
    partial = output.with_suffix(output.suffix + f".partial-{os.getpid()}")
    manifest = {
        "product": "NFE MXene Studio",
        "version": "1.0",
        "author": "Ruck",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layout": "PyInstaller onedir",
        "source_directory": source.name,
        "file_count": len(files),
        "uncompressed_bytes": total_bytes,
        "entry_exe": f"{source.name}/{source.name}.exe",
    }
    with zipfile.ZipFile(
        partial,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        archive.comment = (
            b"NFE MXene Studio 1.0 | Author: Ruck | Portable Windows ZIP64"
        )
        archive.writestr(
            f"{source.name}/RELEASE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        processed = 0
        for index, path in enumerate(files, start=1):
            archive.write(path, (Path(source.name) / path.relative_to(source)).as_posix())
            processed += path.stat().st_size
            if index % 100 == 0 or index == len(files):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "files": index,
                            "file_total": len(files),
                            "bytes": processed,
                            "byte_total": total_bytes,
                            "percent": round(100.0 * processed / max(total_bytes, 1), 2),
                            "MiB_per_s": round(processed / elapsed / (1024**2), 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    partial.replace(output)
    manifest["zip_bytes"] = output.stat().st_size
    manifest["sha256"] = sha256_file(output)
    manifest["elapsed_seconds"] = round(time.time() - started, 2)
    return manifest


# 中文：命令行入口。/ English: Command-line entry point.
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = package(args.source.resolve(), args.output.resolve())
    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, **result}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
