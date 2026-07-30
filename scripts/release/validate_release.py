"""
中文：验证公开源码注释、归档完整性、关键文件、哈希与 Markdown 相对链接。
English: Validate public-source annotations, archive integrity, required files,
hashes, and Markdown relative links.

Author: Ruck
Generated: 2026-07-30 10:06:00 Asia/Shanghai

只执行读取和校验，不删除或修改任何发行文件。
This validator is read-only and never deletes or modifies release files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


PUBLIC_CODE_ROOTS = ("src", "training", "data_tools", "app", "tests", "scripts")


# 中文：计算 SHA256。/ English: Compute SHA256.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


# 中文：验证所有 Python 源码都有双语元数据。/ English: Check bilingual metadata in every Python source.
def validate_annotations(root: Path) -> dict[str, object]:
    files = [
        path
        for name in PUBLIC_CODE_ROOTS
        for path in (root / name).rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    errors = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        missing = [
            token
            for token in ("中文", "English", "Author: Ruck", "Generated:")
            if token not in text
        ]
        if missing:
            errors.append({"file": str(path.relative_to(root)), "missing": missing})
    return {"count": len(files), "errors": errors}


# 中文：测试 ZIP 的 CRC 并返回成员数。/ English: Test ZIP CRC and report member count.
def validate_zip(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        return {"file": path.name, "members": len(archive.infolist()), "bad": bad}


# 中文：解析 Markdown 相对链接并检查本地目标。/ English: Check local targets of Markdown relative links.
def validate_markdown_links(root: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    errors: list[dict[str, str]] = []
    for source in root.rglob("*.md"):
        if "_staging" in source.parts or "_internal" in source.parts:
            continue
        for target in pattern.findall(source.read_text(encoding="utf-8")):
            clean = target.strip().split("#", 1)[0]
            if not clean or "://" in clean or clean.startswith("#"):
                continue
            destination = (source.parent / clean).resolve()
            if not destination.exists():
                errors.append(
                    {
                        "source": str(source.relative_to(root)),
                        "target": target,
                    }
                )
    return errors


# 中文：执行所有验证。/ English: Run every release validation.
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path / 可选 JSON 验证报告路径",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    annotations = validate_annotations(root)
    link_errors = validate_markdown_links(root)
    server_dir = root / "release_assets" / "server"
    expected_server: dict[str, str] = {}
    for line in (
        server_dir / "nfe_server_archives_1.0.sha256"
    ).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+(.+)$", line)
        if match:
            expected_server[Path(match.group(2)).name] = match.group(1).upper()
    hash_errors = []
    for filename, expected in expected_server.items():
        actual = sha256(server_dir / filename)
        if actual != expected:
            hash_errors.append({"file": filename, "expected": expected, "actual": actual})

    windows_dir = root / "release_assets" / "windows"
    windows_expected: dict[str, str] = {}
    for line in (windows_dir / "SHA256SUMS_1.0.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        match = re.match(
            r"^([0-9a-fA-F]{64})\s+(.+\.(?:zip|exe))$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            relative = match.group(2).strip().replace("\\", "/")
            windows_expected[relative] = match.group(1).upper()
    for filename, expected in windows_expected.items():
        actual = sha256(windows_dir / filename)
        if actual != expected:
            hash_errors.append({"file": filename, "expected": expected, "actual": actual})

    zip_results = [
        validate_zip(server_dir / filename)
        for filename in expected_server
    ] + [
        validate_zip(windows_dir / filename)
        for filename in windows_expected
        if filename.lower().endswith(".zip")
    ]
    required = [
        root / "README.md",
        root / "docs" / "TECH_STACK.md",
        root / "data" / "DATA_DICTIONARY.md",
        root / "models" / "MODEL_CARD.md",
        root / "app" / "windows" / "nfe_mxene_studio" / "structure_preview.py",
        root / "release_assets" / "windows" / "SHA256SUMS_1.0.txt",
        root
        / "release_assets"
        / "windows"
        / "NFE_MXene_Studio_1_0"
        / "NFE_MXene_Studio_1_0.exe",
        root
        / "release_assets"
        / "windows"
        / "NFE_MXene_Studio_1_0"
        / "_internal"
        / "python39.dll",
    ]
    missing_required = [str(path.relative_to(root)) for path in required if not path.exists()]
    data_root = root / "data" / "full"
    with (data_root / "nfe_dataset.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.reader(stream)
        data_columns = len(next(reader))
        data_rows = sum(1 for _ in reader)
    data_validation = {
        "rows": data_rows,
        "columns": data_columns,
        "clean_structure_files": sum(
            1 for path in (data_root / "data").iterdir() if path.is_file()
        ),
        "dirty_structure_files": sum(
            1 for path in (data_root / "dirty").iterdir() if path.is_file()
        ),
    }
    result = {
        "author": "Ruck",
        "annotations": annotations,
        "markdown_link_errors": link_errors,
        "hash_errors": hash_errors,
        "zip_results": zip_results,
        "missing_required": missing_required,
        "installed_data": data_validation,
    }
    success = not (
        annotations["errors"]
        or link_errors
        or hash_errors
        or any(item["bad"] for item in zip_results)
        or missing_required
        or data_validation
        != {
            "rows": 15206,
            "columns": 118,
            "clean_structure_files": 15206,
            "dirty_structure_files": 72,
        }
    )
    result["success"] = success
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
