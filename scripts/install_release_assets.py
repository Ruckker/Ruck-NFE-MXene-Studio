"""
中文：把 GitHub Release 中的服务器数据、模型和环境归档安装到仓库约定位置。
English: Install server dataset, model, and environment archives from GitHub Releases
into the repository's conventional locations.

Author: Ruck
Generated: 2026-07-30 09:37:00 Asia/Shanghai

安全策略 / Safety policy:
- 默认拒绝向非空目标目录写入。/ Refuse to write into non-empty targets.
- 检查 ZIP 路径穿越。/ Reject ZIP path traversal.
- 不删除、不覆盖已有文件。/ Never delete or overwrite an existing file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ARCHIVES = {
    "dataset": (
        "nfe_server_dataset_20260730_090526.zip",
        Path("data/full"),
        "d21e3184cb2a8b26fd1e4beedc41526bff51970305221aba7efbbe39fccb9cd2",
    ),
    "models": (
        "nfe_server_models_1.0_20260730.zip",
        Path("models/server"),
        "cbd941dc070cb5bf68fdbb1efd68821619f976e67c45e9850cb2cbf06f058b36",
    ),
    "environment": (
        "nfe_server_environment_20260730_090526.zip",
        Path("environment/server"),
        "afe8be42e6d0a4ca5ba5cdbd242fb860ebfe62541824054e864e2d0c6b2b6538",
    ),
}


# 中文：流式计算 SHA256。/ English: Compute SHA256 as a stream.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# 中文：确认归档成员不会逃逸目标目录。/ English: Ensure members cannot escape the target directory.
def safe_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    root = destination.resolve()
    members = archive.infolist()
    for member in members:
        candidate = (destination / member.filename).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    return members


# 中文：安装一个经过哈希校验的归档。/ English: Install one hash-verified archive.
def install_one(archive_path: Path, destination: Path, expected_hash: str) -> dict[str, object]:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    actual_hash = sha256(archive_path)
    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            f"SHA256 mismatch for {archive_path.name}: {actual_hash} != {expected_hash}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive, destination)
        archive.extractall(destination, members)
    return {
        "archive": str(archive_path),
        "destination": str(destination),
        "sha256": actual_hash,
        "members": len(members),
    }


# 中文：命令行入口。/ English: Command-line entry point.
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("release_assets/server"),
        help="directory containing the server ZIP files",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=sorted(ARCHIVES),
        default=sorted(ARCHIVES),
    )
    args = parser.parse_args()
    results = []
    for part in args.parts:
        filename, relative_destination, expected = ARCHIVES[part]
        results.append(
            install_one(
                (args.assets / filename).resolve(),
                (args.root / relative_destination).resolve(),
                expected,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
