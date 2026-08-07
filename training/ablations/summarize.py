from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ABLATION_ORDER = {
    "full": 0,
    "no_vector": 1,
    "no_global": 2,
    "no_masked_pretrain": 3,
    "no_denoise": 4,
    "no_self_supervision": 5,
    "no_auxiliary_regression": 6,
    "matched_supervision": 7,
    "classification_only": 8,
}
DISPLAY_NAMES = {
    "full": "Full model",
    "no_vector": "− vector information (and denoise objective)",
    "no_global": "− global slab information",
    "no_masked_pretrain": "− masked-atom objective",
    "no_denoise": "− coordinate denoising",
    "no_self_supervision": "− all self-supervision",
    "no_auxiliary_regression": "− auxiliary regression (score only, SSL kept)",
    "matched_supervision": "Full architecture; class + score only; no SSL",
    "classification_only": "Classification only",
}
PAPER_METRICS = (
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_macro_roc_auc",
    "test_macro_average_precision",
    "test_high_average_precision",
    "test_high_precision_at_5pct",
    "test_high_recall_at_5pct",
    "test_high_enrichment_at_5pct",
    "test_low_f1",
    "test_medium_f1",
    "test_high_f1",
    "test_low_recall",
    "test_high_recall",
    "test_NFE_Pseudo_Score_mae",
    "test_NFE_Pseudo_Score_rmse",
    "test_NFE_Pseudo_Score_spearman",
    "test_NFE_Pseudo_Score_r2",
    "test_calibrated_ece",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate audited NFE predictor ablations.")
    parser.add_argument("--runs-root", default="runs/ablations")
    parser.add_argument("--output-dir", default="training/ablations/results")
    parser.add_argument("--minimum-seeds", type=int, default=5)
    return parser.parse_args(argv)


def mean_std_text(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return ""
    if len(array) == 1:
        return f"{array[0]:.5f}"
    return f"{array.mean():.5f} ± {array.std(ddof=1):.5f}"


def flatten_metrics(
    ablation: str, seed: int, payload: dict[str, Any], path: Path
) -> dict[str, Any]:
    provenance = payload.get("provenance", {})
    ablation_config = payload.get("ablation_config", {})
    row: dict[str, Any] = {
        "ablation": ablation,
        "seed": seed,
        "best_epoch": payload.get("best_epoch"),
        "classification_temperature": payload.get("classification_temperature"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "experiment_protocol_sha256": payload.get("experiment_protocol_sha256"),
        "training_protocol_sha256": payload.get("training_protocol_sha256"),
        "ablation_config_name": ablation_config.get("name"),
        "result_path": str(path),
        "dataset_table_sha256": provenance.get("dataset_table_sha256"),
        "structure_manifest_schema": provenance.get("structure_manifest_schema"),
        "structure_manifest_sha256": provenance.get("structure_manifest_sha256"),
        "split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "cache_schema": provenance.get("cache_schema"),
        "global_feature_schema": provenance.get("global_feature_schema"),
        "neighbor_policy": provenance.get("neighbor_policy"),
        "graph_radius_A": provenance.get("graph_radius_A"),
        "max_neighbors": provenance.get("max_neighbors"),
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
        "git_state_sha256": provenance.get("git_state_sha256"),
    }
    for split in ("validation", "test"):
        for key, value in payload.get(split, {}).items():
            row[f"{split}_{key}"] = value
        for key, value in payload.get(f"{split}_calibrated", {}).items():
            row[f"{split}_calibrated_{key}"] = value
    return row


def load_runs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/seed_*/final_metrics.json")):
        ablation = path.parents[1].name
        if ablation not in ABLATION_ORDER or not path.parent.name.startswith("seed_"):
            continue
        try:
            seed = int(path.parent.name.removeprefix("seed_"))
        except ValueError:
            continue
        rows.append(
            flatten_metrics(
                ablation,
                seed,
                json.loads(path.read_text(encoding="utf-8")),
                path,
            )
        )
    return rows


def assert_common_provenance(frame: pd.DataFrame) -> None:
    keys = (
        "dataset_table_sha256",
        "structure_manifest_schema",
        "structure_manifest_sha256",
        "split_manifest_sha256",
        "cache_schema",
        "global_feature_schema",
        "neighbor_policy",
        "graph_radius_A",
        "max_neighbors",
        "git_commit",
    )
    for key in keys:
        if key not in frame or frame[key].isna().any() or (frame[key].astype(str).str.len() == 0).any():
            raise RuntimeError(f"ablation results contain missing provenance: {key}")
        values = set(frame[key].astype(str))
        if len(values) != 1:
            raise RuntimeError(f"ablation results mix incompatible {key}: {sorted(values)}")
    commit = str(frame["git_commit"].iloc[0])
    if commit == "unknown" or len(commit) != 40:
        raise RuntimeError("formal ablation aggregation requires a resolvable Git commit")
    if "git_dirty" not in frame or frame["git_dirty"].isna().any():
        raise RuntimeError("ablation results contain unknown git_dirty provenance")
    if frame["git_dirty"].astype(bool).any():
        raise RuntimeError("formal ablation aggregation refuses dirty-worktree results")


def assert_protocol_matrix(frame: pd.DataFrame) -> None:
    """Require each ablation to be one protocol varied only by independent seed."""
    for ablation, group in frame.groupby("ablation", sort=False):
        names = set(group["ablation_config_name"].dropna().astype(str))
        if names != {ablation}:
            raise RuntimeError(
                f"ablation directory/checkpoint contract mismatch for {ablation}: {sorted(names)}"
            )
        training = group["training_protocol_sha256"]
        if training.isna().any() or len(set(training.astype(str))) != 1:
            raise RuntimeError(
                f"ablation {ablation} mixes training protocols across seeds"
            )
        experiments = [
            str(value)
            for value in group["experiment_protocol_sha256"].tolist()
            if pd.notna(value) and str(value)
        ]
        if len(experiments) != len(group) or len(set(experiments)) != len(experiments):
            raise RuntimeError(
                f"ablation {ablation} requires a distinct seed-specific experiment protocol per run"
            )


def assert_complete_seed_matrix(frame: pd.DataFrame, minimum_seeds: int) -> None:
    seed_sets: dict[str, tuple[int, ...]] = {}
    for ablation, group in frame.groupby("ablation", sort=False):
        if group["seed"].duplicated().any():
            raise RuntimeError(f"duplicate seed rows for ablation {ablation}")
        seeds = tuple(sorted(int(value) for value in group["seed"].tolist()))
        if len(seeds) < int(minimum_seeds):
            raise RuntimeError(
                f"ablation {ablation} requires at least {minimum_seeds} seeds; found {len(seeds)}"
            )
        seed_sets[ablation] = seeds
        hashes = [
            str(value)
            for value in group.get("checkpoint_sha256", pd.Series(dtype=object)).tolist()
            if pd.notna(value) and str(value)
        ]
        if len(hashes) != len(group) or len(set(hashes)) != len(hashes):
            raise RuntimeError(
                f"ablation {ablation} must contain one distinct checkpoint SHA256 per seed"
            )
    if "full" not in seed_sets:
        raise RuntimeError("formal ablation table requires the full model reference")
    reference = seed_sets["full"]
    mismatched = {name: seeds for name, seeds in seed_sets.items() if seeds != reference}
    if mismatched:
        raise RuntimeError(
            f"all ablations must use the same seed set as full={reference}; mismatched={mismatched}"
        )


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "ablation",
        "result_path",
        "checkpoint_sha256",
        "experiment_protocol_sha256",
        "training_protocol_sha256",
        "ablation_config_name",
        "dataset_table_sha256",
        "structure_manifest_schema",
        "structure_manifest_sha256",
        "split_manifest_sha256",
        "cache_schema",
        "global_feature_schema",
        "neighbor_policy",
        "git_commit",
        "git_dirty",
        "git_state_sha256",
    }
    numeric = [
        column
        for column in frame
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows = []
    for ablation, group in frame.groupby("ablation", sort=False):
        row: dict[str, Any] = {"ablation": ablation, "n_runs": len(group)}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values):
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_order"] = result["ablation"].map(ABLATION_ORDER).fillna(999)
        result = result.sort_values(["_order", "ablation"]).drop(columns="_order")
    return result


def _paired_delta(group: pd.DataFrame, full: pd.DataFrame, metric: str) -> str:
    left = group[["seed", metric]].copy() if metric in group else pd.DataFrame()
    right = full[["seed", metric]].copy() if metric in full else pd.DataFrame()
    if left.empty or right.empty:
        return ""
    joined = left.merge(right, on="seed", suffixes=("_ablation", "_full"), validate="one_to_one")
    a = pd.to_numeric(joined[f"{metric}_ablation"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(joined[f"{metric}_full"], errors="coerce").to_numpy(float)
    delta = a - b
    return mean_std_text(delta[np.isfinite(delta)])


def paper_table(frame: pd.DataFrame) -> pd.DataFrame:
    full = frame[frame["ablation"] == "full"].copy()
    rows = []
    for ablation, group in frame.groupby("ablation", sort=False):
        row: dict[str, Any] = {
            "Ablation": DISPLAY_NAMES.get(ablation, ablation),
            "key": ablation,
            "Seeds": int(group["seed"].nunique()),
        }
        for metric in PAPER_METRICS:
            values = pd.to_numeric(
                group.get(metric, pd.Series(dtype=float)), errors="coerce"
            ).dropna().tolist()
            row[metric] = mean_std_text(values)
        row["Δ macro F1 vs full (paired)"] = _paired_delta(group, full, "test_macro_f1")
        if ablation == "classification_only":
            row["Δ score MAE vs full (paired)"] = "N/A"
            for metric in (
                "test_NFE_Pseudo_Score_mae",
                "test_NFE_Pseudo_Score_rmse",
                "test_NFE_Pseudo_Score_spearman",
                "test_NFE_Pseudo_Score_r2",
            ):
                row[metric] = "N/A"
        else:
            row["Δ score MAE vs full (paired)"] = _paired_delta(
                group, full, "test_NFE_Pseudo_Score_mae"
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_order"] = result["key"].map(ABLATION_ORDER).fillna(999)
        result = result.sort_values(["_order", "key"]).drop(columns=["_order", "key"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.runs_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_runs(root)
    if not rows:
        raise SystemExit(f"no ablation final_metrics.json files found under {root}")
    per_seed = pd.DataFrame(rows)
    assert_common_provenance(per_seed)
    assert_protocol_matrix(per_seed)
    assert_complete_seed_matrix(per_seed, args.minimum_seeds)
    per_seed["_order"] = per_seed["ablation"].map(ABLATION_ORDER).fillna(999)
    per_seed = per_seed.sort_values(["_order", "ablation", "seed"]).drop(columns="_order")
    summary = numeric_summary(per_seed)
    paper = paper_table(per_seed)
    per_seed.to_csv(output / "ablation_per_seed.csv", index=False)
    summary.to_csv(output / "ablation_summary.csv", index=False)
    paper.to_csv(output / "ablation_paper_table.csv", index=False)
    print(paper.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
