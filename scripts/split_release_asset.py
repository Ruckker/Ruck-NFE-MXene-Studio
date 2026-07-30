"""Split a ZIP64 release into GitHub-compatible ordered assets.

中文：GitHub 单个 Release 附件存在大小限制。本工具把完整 ZIP 按字节顺序
切为 `.part01`、`.part02` 等文件，并生成包含各分片和完整文件 SHA256 的清单。
它不会删除或覆盖任何已有文件。

English: GitHub limits the size of each Release asset. This utility splits a
complete ZIP into ordered `.part01`, `.part02`, ... files and writes a manifest
containing per-part and whole-file SHA256 values. It never deletes or overwrites
existing files.

Author: Ruck
Generated: 2026-07-30 22:41:00 Asia/Shanghai
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Compute an uppercase streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def split_file(source: Path, part_size: int) -> list[Path]:
    """Create ordered parts while refusing partial or accidental overwrites."""
    if part_size <= 0:
        raise ValueError("part size must be positive")
    expected_parts = max(1, (source.stat().st_size + part_size - 1) // part_size)
    targets = [
        source.with_name(f"{source.name}.part{index:02d}")
        for index in range(1, expected_parts + 1)
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing split assets: "
            + ", ".join(str(path) for path in existing)
        )
    with source.open("rb") as input_stream:
        for target in targets:
            remaining = part_size
            with target.open("xb") as output_stream:
                while remaining > 0:
                    block = input_stream.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        break
                    output_stream.write(block)
                    remaining -= len(block)
    return targets


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--part-size-mib",
        type=int,
        default=1900,
        help="maximum part size in MiB (default: 1900)",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else source.with_name(f"{source.name}.parts.json")
    )
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}"
        )
    parts = split_file(source, args.part_size_mib * 1024 * 1024)
    payload = {
        "source": source.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "ordered_parts": [
            {
                "name": part.name,
                "bytes": part.stat().st_size,
                "sha256": sha256(part),
            }
            for part in parts
        ],
        "reassembly": (
            "python scripts/reassemble_release_parts.py "
            + " ".join(part.name for part in parts)
            + f" --output {source.name}"
        ),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
