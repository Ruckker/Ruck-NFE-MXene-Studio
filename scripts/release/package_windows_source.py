"""
中文：创建可学习、可重建的 Windows application 1.0 源码包，并包含运行所需模型与元数据。
English: Create a learning-friendly, rebuildable Windows application 1.0 source package
including the required models and metadata.

Author: Ruck
Generated: 2026-07-30 09:52:00 Asia/Shanghai

源码来自已添加双语导读的 GitHub 副本；模型来自最终本地 App 资源。
The source comes from bilingual annotated GitHub copies; models come from the
final local application resources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path


# 中文：计算文件 SHA256。/ English: Compute a file SHA256.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


# 中文：向归档添加目录树并统一目标前缀。/ English: Add a directory tree under a target prefix.
def add_tree(
    archive: zipfile.ZipFile,
    source: Path,
    target: Path,
    records: list[dict[str, object]],
) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        arcname = (target / path.relative_to(source)).as_posix()
        archive.write(path, arcname)
        records.append(
            {
                "archive_path": arcname,
                "source_bytes": path.stat().st_size,
                "source_mtime": datetime.fromtimestamp(path.stat().st_mtime)
                .astimezone()
                .isoformat(timespec="seconds"),
            }
        )


# 中文：构建源码 ZIP。/ English: Build the source ZIP.
def build(repo: Path, app_assets: Path, output: Path, manifest_output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    prefix = Path("NFE_MXene_Studio_1_0_Source")
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.comment = b"NFE MXene Studio 1.0 Source | Author: Ruck"
        add_tree(archive, repo / "src" / "nfe_model", prefix / "nfe_model", records)
        add_tree(
            archive,
            repo / "app" / "windows" / "nfe_mxene_studio",
            prefix / "windows_app",
            records,
        )
        add_tree(
            archive,
            repo / "app" / "windows" / "packaging",
            prefix / "windows_app",
            records,
        )
        add_tree(archive, app_assets / "models", prefix / "windows_app" / "models", records)
        add_tree(
            archive,
            app_assets / "resources",
            prefix / "windows_app" / "resources",
            records,
        )
        add_tree(
            archive,
            repo / "examples" / "structures",
            prefix / "windows_app" / "samples",
            records,
        )
        add_tree(archive, repo / "tests", prefix / "tests", records)
        for relative in (
            Path("README.md"),
            Path("LICENSE"),
            Path("CITATION.cff"),
            Path("pyproject.toml"),
            Path("docs/WINDOWS_APP.md"),
            Path("docs/SCIENTIFIC_OVERVIEW.md"),
            Path("docs/TECH_STACK.md"),
            Path("docs/ARCHITECTURE.md"),
            Path("models/MODEL_CARD.md"),
        ):
            source = repo / relative
            arcname = (prefix / relative).as_posix()
            archive.write(source, arcname)
            records.append(
                {"archive_path": arcname, "source_bytes": source.stat().st_size}
            )
        embedded = {
            "product": "NFE MXene Studio",
            "version": "1.0",
            "author": "Ruck",
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "purpose": "annotated rebuildable Windows source",
            "file_count": len(records),
            "files": records,
        }
        archive.writestr(
            (prefix / "SOURCE_MANIFEST.json").as_posix(),
            json.dumps(embedded, ensure_ascii=False, indent=2) + "\n",
        )
    result = {
        "product": "NFE MXene Studio Source",
        "version": "1.0",
        "author": "Ruck",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "zip": str(output),
        "zip_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "source_file_count": len(records),
    }
    manifest_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


# 中文：命令行入口。/ English: Command-line entry point.
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--app-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            build(
                args.repo.resolve(),
                args.app_assets.resolve(),
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
