from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nfe_model.data_contract import DATA_IMPLEMENTATION_SCHEMA, data_implementation_sha256
from nfe_model.data_v2 import (
    CACHE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    NEIGHBOR_POLICY,
    STRUCTURE_MANIFEST_SCHEMA,
    TARGET_SCHEMA,
    target_schema_sha256,
)
from nfe_model.prediction_manifest import load_prediction_manifest
from nfe_model.provenance_v2 import NORMALIZER_SCHEMA, git_repository_state


EXPECTED = {
    "structure_manifest_schema": STRUCTURE_MANIFEST_SCHEMA,
    "target_schema": TARGET_SCHEMA,
    "target_schema_sha256": target_schema_sha256(),
    "data_implementation_schema": DATA_IMPLEMENTATION_SCHEMA,
    "data_implementation_sha256": data_implementation_sha256(),
    "normalizer_schema": NORMALIZER_SCHEMA,
    "cache_schema": CACHE_SCHEMA,
    "global_feature_schema": GLOBAL_FEATURE_SCHEMA,
    "neighbor_policy": NEIGHBOR_POLICY,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast final gate before using NFE benchmark artifacts in paper tables/statistics."
    )
    parser.add_argument("results", nargs="+", help="result.json or final_metrics.json files")
    parser.add_argument(
        "--allow-cache-skips",
        action="store_true",
        help="exploratory only; formal paper tables should retain the default zero-skip requirement",
    )
    return parser.parse_args(argv)


def _prediction_paths(result_path: Path) -> list[tuple[str, Path]]:
    return [
        ("validation", result_path.with_name("validation_predictions.csv")),
        ("test", result_path.with_name("test_predictions.csv")),
    ]


def _validate_result(path: Path, allow_cache_skips: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{path} has no formal provenance mapping")
    for key, expected in EXPECTED.items():
        observed = provenance.get(key)
        if str(observed) != str(expected):
            raise RuntimeError(
                f"{path} uses stale/incompatible {key}: observed={observed!r} expected={expected!r}"
            )
    for key in (
        "dataset_table_sha256",
        "structure_manifest_sha256",
        "cache_records_sha256",
        "normalizer_sha256",
        "split_manifest_sha256",
        "git_commit",
    ):
        value = str(provenance.get(key, ""))
        if not value or value == "unknown":
            raise RuntimeError(f"{path} is missing formal provenance field {key}")
    if provenance.get("git_dirty") is not False:
        raise RuntimeError(f"{path} was produced from a dirty/unknown Git worktree")
    skipped = int(provenance.get("skipped_cache_records", payload.get("skipped_cache_records", -1)))
    if skipped < 0:
        raise RuntimeError(f"{path} does not record skipped_cache_records")
    if skipped and not allow_cache_skips:
        raise RuntimeError(
            f"{path} skipped {skipped} dataset rows; formal paper-ready results require zero cache skips"
        )

    manifests = []
    for split, prediction_path in _prediction_paths(path):
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"paper-ready result {path} is missing {split} prediction CSV: {prediction_path}"
            )
        manifest = load_prediction_manifest(prediction_path, expected_split=split)
        identity = manifest["data_identity"]
        for key, expected in EXPECTED.items():
            if str(identity.get(key)) != str(expected):
                raise RuntimeError(
                    f"{prediction_path} manifest uses stale {key}: "
                    f"observed={identity.get(key)!r} expected={expected!r}"
                )
        for key in (
            "dataset_table_sha256",
            "structure_manifest_sha256",
            "cache_records_sha256",
            "normalizer_sha256",
            "split_manifest_sha256",
            "git_commit",
        ):
            if str(identity.get(key, "")) != str(provenance.get(key, "")):
                raise RuntimeError(
                    f"{prediction_path} manifest/result mismatch for {key}: "
                    f"manifest={identity.get(key)!r} result={provenance.get(key)!r}"
                )
        manifests.append(manifest)
    return {
        "path": str(path),
        "git_commit": provenance["git_commit"],
        "dataset_table_sha256": provenance["dataset_table_sha256"],
        "cache_records_sha256": provenance["cache_records_sha256"],
        "normalizer_sha256": provenance["normalizer_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "skipped_cache_records": skipped,
        "prediction_data_identity_sha256": manifests[0]["data_identity_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = git_repository_state()
    if runtime.get("git_dirty") is not False:
        raise RuntimeError("paper preflight requires a clean current Git worktree")
    runtime_commit = str(runtime.get("git_commit", "unknown"))
    if runtime_commit == "unknown":
        raise RuntimeError("paper preflight requires a resolvable current Git commit")

    reports = [
        _validate_result(Path(value).resolve(), bool(args.allow_cache_skips))
        for value in args.results
    ]
    commits = {report["git_commit"] for report in reports}
    if commits != {runtime_commit}:
        raise RuntimeError(
            f"paper-ready artifacts must be produced by current commit {runtime_commit}; found={sorted(commits)}"
        )
    for key in (
        "dataset_table_sha256",
        "cache_records_sha256",
        "normalizer_sha256",
        "split_manifest_sha256",
        "prediction_data_identity_sha256",
    ):
        values = {str(report[key]) for report in reports}
        if len(values) != 1:
            raise RuntimeError(f"paper-ready result set mixes {key}: {sorted(values)}")
    print(json.dumps({"paper_ready": True, "results": reports}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
