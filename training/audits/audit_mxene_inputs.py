# ==============================================================================
# 中文概述：用 Windows 程序的 MXene 输入判定逐个检查数据集中的结构，确认训练集没有被误判为不合法，
#           并统计训练域提示与脏数据的判定结果。
# English overview: Run the Windows application's MXene input check over every structure of a dataset to
#           confirm that no training structure is rejected, and summarize domain notes and dirty structures.
#
# 中文输入：数据集安装目录（其下 data/ 为清洁结构、dirty/ 为隔离结构）、并行进程数。
# English inputs: Dataset install directory (data/ holds clean structures, dirty/ quarantined ones), worker count.
# 中文输出：各子目录的合法/不合法计数、原因直方图、训练域提示计数与不合法文件清单（JSON）。
# English outputs: Valid/invalid counts per folder, reason histogram, domain-note counts and invalid files (JSON).
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DIGITS = re.compile(r"[0-9]+(\.[0-9]+)?")


# 中文：子进程中检查一个文件。/ English: Check one file inside a worker process.
def _check(path_text: str) -> dict[str, Any]:
    from nfe_model.mxene_validation import check_mxene_file

    check = check_mxene_file(path_text)
    return {
        "file": Path(path_text).name,
        "valid": check.valid,
        "reasons": list(check.reasons),
        "n": check.n,
        "in_training_domain": check.in_training_domain,
        "domain_notes": list(check.domain_notes),
        "vacuum_A": check.vacuum_A,
        "thickness_A": check.thickness_A,
    }


def _reason_key(reason: str) -> str:
    return DIGITS.sub("#", reason)


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/full_v1_1")
    parser.add_argument("--folders", nargs="+", default=["data", "dirty"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="models/metadata/mxene_input_validation_audit.json")
    args = parser.parse_args(argv)

    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    started = time.time()
    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/audit_mxene_inputs.py",
        "validator": "src/nfe_model/mxene_validation.py",
        "validator_sha256": hashlib.sha256((ROOT / "src/nfe_model/mxene_validation.py").read_bytes()).hexdigest().upper(),
        "root": args.root,
        "folders": {},
    }
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for folder in args.folders:
            files = sorted(str(path) for path in (root / folder).iterdir() if path.is_file())
            results = list(pool.map(_check, files, chunksize=64))
            invalid = [item for item in results if not item["valid"]]
            valid = [item for item in results if item["valid"]]
            report["folders"][folder] = {
                "files": len(results),
                "valid": len(valid),
                "invalid": len(invalid),
                "in_training_domain": sum(1 for item in valid if item["in_training_domain"]),
                "n_counts": dict(Counter(str(item["n"]) for item in valid)),
                "domain_note_counts": dict(Counter(note for item in valid for note in item["domain_notes"]).most_common(20)),
                "invalid_reason_counts": dict(Counter(_reason_key(reason) for item in invalid for reason in item["reasons"]).most_common(20)),
                "vacuum_A_min": min((item["vacuum_A"] for item in valid), default=None),
                "thickness_A_max": max((item["thickness_A"] for item in valid), default=None),
                "invalid_files": invalid[:200],
            }
            print(folder, json.dumps({k: v for k, v in report["folders"][folder].items() if k != "invalid_files"}, ensure_ascii=False), flush=True)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
