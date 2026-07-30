"""
中文：按顺序流式合并 GitHub Release 分卷，并可校验最终 SHA256。
English: Stream-concatenate GitHub Release parts and optionally verify SHA256.

Author: Ruck
Generated: 2026-07-30 Asia/Shanghai

安全策略 / Safety policy:
- 拒绝覆盖已有输出文件。/ Refuse to overwrite an existing output.
- 不删除输入分卷或失败输出。/ Never delete input parts or failed output.
- 使用固定大小缓冲区，不把大型分卷整体载入内存。/ Use bounded memory.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


# 中文：流式拼接分卷并同步计算最终哈希。/ Stream parts and hash the output.
def reassemble(parts: list[Path], output: Path, buffer_size: int) -> str:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    missing = [str(part) for part in parts if not part.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release parts: {missing}")

    digest = hashlib.sha256()
    with output.open("xb") as destination:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(buffer_size):
                    destination.write(chunk)
                    digest.update(chunk)
    return digest.hexdigest()


# 中文：命令行入口。/ Command-line entry point.
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reassemble ordered GitHub Release parts without overwriting files."
    )
    parser.add_argument("parts", nargs="+", type=Path, help="ordered .part01, .part02, ...")
    parser.add_argument("--output", required=True, type=Path, help="output ZIP path")
    parser.add_argument("--sha256", help="expected SHA256 of the reassembled file")
    parser.add_argument("--buffer-mib", type=int, default=8)
    args = parser.parse_args()

    if args.buffer_mib <= 0:
        raise ValueError("--buffer-mib must be positive")
    actual = reassemble(
        [part.resolve() for part in args.parts],
        args.output.resolve(),
        args.buffer_mib * 1024 * 1024,
    )
    print(f"SHA256  {actual}  {args.output}")
    if args.sha256 and actual.lower() != args.sha256.lower():
        raise ValueError(
            "SHA256 mismatch. The output is retained for inspection and is not deleted: "
            f"{actual} != {args.sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
