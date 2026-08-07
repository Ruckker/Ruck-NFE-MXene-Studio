from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "dummy": "Dummy prior/median",
    "xgboost": "XGBoost (structure-only)",
    "cgcnn": "CGCNN-style (controlled)",
    "schnet": "SchNet-style (controlled)",
    "alignn": "ALIGNN-style (legacy controlled key)",
    "m3gnet": "M3GNet-style (legacy controlled key)",
    "cgcnn_controlled": "CGCNN-style (controlled)",
    "schnet_controlled": "SchNet-style (controlled)",
    "angle_moment": "Angle-moment GNN (controlled)",
    "state_threebody": "State/three-body-moment GNN (controlled)",
    "painn": "Ruck-NFE backbone (matched)",
    "cgcnn_official": "CGCNN (official backbone)",
    "schnet_official": "SchNetPack SchNet (official backbone)",
    "alignn_official": "ALIGNN (official backbone)",
    "m3gnet_official": "MatGL M3GNet (official backbone)",
    "ours_full": "Ruck-NFE Full",
}
MODEL_ORDER = {name: index for index, name in enumerate(DISPLAY_NAMES)}
TRACK_ORDER = {"architecture": 0, "official-upstream": 1, "full-system": 2}
NEURAL_BENCHMARK_MODELS = {
    "cgcnn_controlled",
    "schnet_controlled",
    "angle_moment",
    "state_threebody",
    "painn",
    "cgcnn_official",
    "schnet_official",
    "alignn_official",
    "m3gnet_official",
}

PAPER_METRICS = [
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_macro_roc_auc",
    "test_macro_average_precision",
    "test_high_average_precision",
    "test_high_precision_at_5pct",
    "test_high_recall_at_5pct",
    "test_high_enrichment_at_5pct",
    "test_NFE_Pseudo_Score_mae",
    "test_NFE_Pseudo_Score_rmse",
    "test_NFE_Pseudo_Score_spearman",
    "test_NFE_Pseudo_Score_r2",
    "test_low_f1",
    "test_medium_f1",
    "test_high_f1",
    "test_low_recall",
    "test_high_recall",
    "test_ece",
]

PROVENANCE_KEYS = (
    "dataset_table_sha256",
    "structure_manifest_schema",
    "structure_manifest_sha256",
    "target_schema",
    "target_schema_sha256",
    "data_implementation_schema",
    "data_implementation_sha256",
    "split_manifest_sha256",
    "cache_schema",
    "global_feature_schema",
    "neighbor_policy",
    "graph_radius_A",
    "max_neighbors",
    "git_commit",
)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    provenance = result.get("provenance", {})
    details = result.get("details", {})
    row: dict[str, Any] = {
        "track": result.get("track", "architecture"),
        "model": result.get("model"),
        "seed": result.get("seed"),
        "parameter_count": result.get("parameter_count"),
        "training_seconds": result.get("training_seconds"),
        "evaluation_seconds": result.get("evaluation_seconds"),
        "temperature": result.get("temperature"),
        "benchmark_common_protocol_sha256": result.get("benchmark_common_protocol_sha256"),
        "model_protocol_sha256": result.get("model_protocol_sha256"),
        "git_dirty": provenance.get("git_dirty"),
        "git_state_sha256": provenance.get("git_state_sha256"),
        "checkpoint_sha256": details.get("checkpoint_sha256"),
        "checkpoint_seed": details.get("checkpoint_seed"),
        "checkpoint_training_git_commit": details.get("checkpoint_training_git_commit"),
        "checkpoint_training_git_dirty": details.get("checkpoint_training_git_dirty"),
    }
    for key in PROVENANCE_KEYS:
        row[key] = provenance.get(key)
    for split in ("validation", "test"):
        for key, value in result.get(f"{split}_metrics", {}).items():
            row[f"{split}_{key}"] = value
    return row


def mean_std_text(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return ""
    if len(array) == 1:
        return f"{array[0]:.5f}"
    return f"{array.mean():.5f} ± {array.std(ddof=1):.5f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate audited NFE benchmark tracks.")
    parser.add_argument("--results-root", default="training/baselines/results")
    parser.add_argument("--output-dir", default="training/baselines/results")
    parser.add_argument("--minimum-full-seeds", type=int, default=5)
    parser.add_argument("--minimum-model-seeds", type=int, default=5)
    return parser.parse_args(argv)


def load_results(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/*/seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "nfe-baseline-result-2.1":
            continue
        row = flatten_result(payload)
        row["result_path"] = str(path)
        rows.append(row)
    return rows


def assert_common_provenance(frame: pd.DataFrame) -> None:
    for key in PROVENANCE_KEYS:
        if key not in frame:
            raise RuntimeError(f"audited result table is missing provenance column {key}")
        if frame[key].isna().any() or (frame[key].astype(str).str.len() == 0).any():
            raise RuntimeError(f"cannot aggregate results with missing {key}")
        values = set(frame[key].astype(str))
        if len(values) != 1:
            raise RuntimeError(f"cannot aggregate mixed {key}: {sorted(values)}")
    commit = str(frame["git_commit"].iloc[0])
    if commit == "unknown" or len(commit) != 40:
        raise RuntimeError("formal benchmark aggregation requires a resolvable Git commit")
    if "git_dirty" not in frame or frame["git_dirty"].isna().any():
        raise RuntimeError("formal benchmark aggregation requires known git_dirty provenance")
    if frame["git_dirty"].astype(bool).any():
        raise RuntimeError("formal benchmark aggregation refuses results produced from a dirty worktree")


def assert_seed_coverage(frame: pd.DataFrame, minimum_model_seeds: int) -> None:
    stochastic = frame[frame["model"] != "dummy"]
    if stochastic.empty:
        return
    seed_sets: dict[tuple[str, str], tuple[int, ...]] = {}
    for (track, model), group in stochastic.groupby(["track", "model"], sort=False):
        if group["seed"].duplicated().any():
            raise RuntimeError(f"duplicate seed rows for {track}/{model}")
        seeds = tuple(sorted(int(value) for value in group["seed"].tolist()))
        if len(seeds) < int(minimum_model_seeds):
            raise RuntimeError(
                f"{track}/{model} requires at least {minimum_model_seeds} seeds; found {len(seeds)}"
            )
        seed_sets[(track, model)] = seeds
    unique_sets = {value for value in seed_sets.values()}
    if len(unique_sets) != 1:
        formatted = {f"{track}/{model}": seeds for (track, model), seeds in seed_sets.items()}
        raise RuntimeError(f"formal model comparisons must use the same seed set: {formatted}")


def _one_nonempty_value(group: pd.DataFrame, column: str, label: str) -> str:
    if column not in group or group[column].isna().any():
        raise RuntimeError(f"{label} is missing {column}")
    values = {str(value) for value in group[column].tolist() if str(value)}
    if len(values) != 1:
        raise RuntimeError(f"{label} mixes {column}: {sorted(values)}")
    return next(iter(values))


def assert_training_protocols(frame: pd.DataFrame) -> None:
    neural = frame[frame["model"].isin(NEURAL_BENCHMARK_MODELS)].copy()
    if not neural.empty:
        common_values = set()
        for (track, model), group in neural.groupby(["track", "model"], sort=False):
            common_values.add(_one_nonempty_value(group, "benchmark_common_protocol_sha256", f"{track}/{model}"))
            _one_nonempty_value(group, "model_protocol_sha256", f"{track}/{model}")
            hashes = [str(value) for value in group["checkpoint_sha256"].tolist() if pd.notna(value) and str(value)]
            if len(hashes) != len(group) or len(set(hashes)) != len(hashes):
                raise RuntimeError(f"{track}/{model} must contain one distinct checkpoint SHA256 per seed")
        if len(common_values) != 1:
            raise RuntimeError(
                "controlled/matched/official neural baselines do not share one training/capacity "
                f"budget fingerprint: {sorted(common_values)}"
            )
    full = frame[(frame["track"] == "full-system") & (frame["model"] == "ours_full")]
    if not full.empty:
        _one_nonempty_value(full, "model_protocol_sha256", "full-system/ours_full")


def assert_independent_full_system(frame: pd.DataFrame, minimum_seeds: int) -> None:
    full = frame[(frame["track"] == "full-system") & (frame["model"] == "ours_full")]
    if full.empty:
        return
    if int(full["seed"].nunique()) < int(minimum_seeds):
        raise RuntimeError(
            f"full-system summary requires at least {minimum_seeds} independent seeds; found {full['seed'].nunique()}"
        )
    hashes = [str(value) for value in full["checkpoint_sha256"].tolist() if pd.notna(value) and str(value)]
    if len(hashes) != len(full) or len(set(hashes)) != len(hashes):
        raise RuntimeError("full-system rows must use distinct checkpoint SHA256 values")
    execution_commit = str(full["git_commit"].iloc[0])
    for _, row in full.iterrows():
        if pd.notna(row.get("checkpoint_seed")) and int(row["checkpoint_seed"]) != int(row["seed"]):
            raise RuntimeError("full-system checkpoint seed does not match result seed")
        training_commit = str(row.get("checkpoint_training_git_commit") or "")
        if training_commit != execution_commit:
            raise RuntimeError(
                "full-system checkpoint was not trained from the same formal code revision used for "
                f"the benchmark: training={training_commit or 'missing'} execution={execution_commit}"
            )
        if row.get("checkpoint_training_git_dirty") is not False:
            raise RuntimeError("full-system checkpoint was trained from a dirty or unknown worktree")


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "track",
        "model",
        "result_path",
        "benchmark_common_protocol_sha256",
        "model_protocol_sha256",
        *PROVENANCE_KEYS,
        "git_dirty",
        "git_state_sha256",
        "checkpoint_sha256",
        "checkpoint_training_git_commit",
        "checkpoint_training_git_dirty",
    }
    numeric = [column for column in frame if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]
    rows = []
    for (track, model), group in frame.groupby(["track", "model"], sort=False):
        row: dict[str, Any] = {"track": track, "model": model, "n_runs": len(group)}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values):
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    subset = frame[frame["track"] == track]
    rows = []
    for model, group in subset.groupby("model", sort=False):
        parameter_values = pd.to_numeric(group["parameter_count"], errors="coerce").dropna()
        row: dict[str, Any] = {
            "Track": track,
            "Model": DISPLAY_NAMES.get(model, model),
            "Seeds": int(group["seed"].nunique()),
            "Parameters_mean": float(parameter_values.mean()) if len(parameter_values) else np.nan,
        }
        for metric in PAPER_METRICS:
            values = pd.to_numeric(group.get(metric, pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            row[metric] = mean_std_text(values)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        reverse = {value: key for key, value in DISPLAY_NAMES.items()}
        result["_key"] = result["Model"].map(reverse).fillna(result["Model"])
        result["_order"] = result["_key"].map(MODEL_ORDER).fillna(999)
        result = result.sort_values(["_order", "Model"]).drop(columns=["_key", "_order"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_results(root)
    if not rows:
        raise SystemExit(
            f"no audited nfe-baseline-result-2.1 files found under {root}; legacy result schemas are intentionally excluded"
        )
    per_seed = pd.DataFrame(rows)
    assert_common_provenance(per_seed)
    assert_seed_coverage(per_seed, args.minimum_model_seeds)
    assert_training_protocols(per_seed)
    assert_independent_full_system(per_seed, args.minimum_full_seeds)
    per_seed["_track"] = per_seed["track"].map(TRACK_ORDER).fillna(999)
    per_seed["_model"] = per_seed["model"].map(MODEL_ORDER).fillna(999)
    per_seed = per_seed.sort_values(["_track", "_model", "seed"]).drop(columns=["_track", "_model"])
    summary = numeric_summary(per_seed)
    tables = {track: paper_table(per_seed, track) for track in TRACK_ORDER}
    combined = pd.concat([tables[track] for track in TRACK_ORDER], ignore_index=True)
    per_seed.to_csv(output / "benchmark_per_seed.csv", index=False)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    tables["architecture"].to_csv(output / "architecture_paper_table.csv", index=False)
    tables["official-upstream"].to_csv(output / "official_upstream_paper_table.csv", index=False)
    tables["full-system"].to_csv(output / "full_system_paper_table.csv", index=False)
    combined.to_csv(output / "benchmark_paper_table.csv", index=False)
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
