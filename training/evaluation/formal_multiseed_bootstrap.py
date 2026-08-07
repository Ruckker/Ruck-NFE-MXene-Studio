from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nfe_model.prediction_manifest import (
    assert_same_prediction_data_identity,
    load_prediction_manifest,
)
from nfe_model.utils import save_json
from training.evaluation.paired_bootstrap import _prepare, _values


@dataclass
class SeedPair:
    seed: int
    frame: pd.DataFrame
    groups: np.ndarray
    by_group: dict[str, np.ndarray]
    observed: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested paired bootstrap over independent training seeds and Split_Group blocks."
    )
    parser.add_argument("--a", nargs="+", required=True, help="model A signed test prediction CSVs")
    parser.add_argument("--b", nargs="+", required=True, help="model B signed test prediction CSVs")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027, help="bootstrap RNG seed")
    parser.add_argument("--minimum-training-seeds", type=int, default=5)
    parser.add_argument(
        "--output", default="training/evaluation/results/formal_multiseed_bootstrap.json"
    )
    return parser.parse_args()


def _manifest_seed(manifest: dict, path: str) -> int:
    value = manifest.get("run_identity", {}).get("seed")
    if value is None:
        raise ValueError(f"signed prediction manifest has no training seed: {path}")
    return int(value)


def _pair_frames(a_path: str, b_path: str) -> SeedPair:
    manifest_a = load_prediction_manifest(a_path, expected_split="test")
    manifest_b = load_prediction_manifest(b_path, expected_split="test")
    assert_same_prediction_data_identity(manifest_a, manifest_b)
    seed_a = _manifest_seed(manifest_a, a_path)
    seed_b = _manifest_seed(manifest_b, b_path)
    if seed_a != seed_b:
        raise RuntimeError(
            f"paired prediction seeds disagree: A={seed_a} ({a_path}) B={seed_b} ({b_path})"
        )

    a = _prepare(a_path, "A")
    b = _prepare(b_path, "B")
    if "Record_Index" in a and "Record_Index" in b:
        if set(a["Record_Index"].astype(int)) != set(b["Record_Index"].astype(int)):
            raise RuntimeError(f"seed {seed_a}: A/B Record_Index sets differ")
        keys = ["Record_Index"]
    else:
        if set(a["Structure_Name"].astype(str)) != set(b["Structure_Name"].astype(str)):
            raise RuntimeError(f"seed {seed_a}: A/B Structure_Name sets differ")
        keys = ["Structure_Name"]
    if len(a) != len(b):
        raise RuntimeError(f"seed {seed_a}: A/B row counts differ")

    joined = a.merge(b, on=keys, how="inner", validate="one_to_one", suffixes=("_a", "_b"))
    if len(joined) != len(a):
        raise RuntimeError(f"seed {seed_a}: paired merge lost rows")
    if keys == ["Record_Index"]:
        if (joined["Structure_Name_a"].astype(str) != joined["Structure_Name_b"].astype(str)).any():
            raise RuntimeError(f"seed {seed_a}: Record_Index rows disagree on Structure_Name")
        joined["Structure_Name"] = joined["Structure_Name_a"]
    group_a = joined["Split_Group_a"].fillna("").astype(str).str.strip()
    group_b = joined["Split_Group_b"].fillna("").astype(str).str.strip()
    if (group_a == "").any() or (group_a != group_b).any():
        raise RuntimeError(f"seed {seed_a}: A/B Split_Group values are missing or inconsistent")
    joined["Split_Group"] = group_a

    label_a = joined["A_True_Label"].fillna("").astype(str).str.strip().str.lower()
    label_b = joined["B_True_Label"].fillna("").astype(str).str.strip().str.lower()
    if (label_a != label_b).any():
        raise RuntimeError(f"seed {seed_a}: A/B True_Label differs")
    joined["True_Label"] = label_a
    truth_a = pd.to_numeric(joined["A_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    truth_b = pd.to_numeric(joined["B_True_NFE_Pseudo_Score"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(truth_a)) or not np.allclose(truth_a, truth_b, atol=1e-10, rtol=0):
        raise RuntimeError(f"seed {seed_a}: A/B true NFE pseudo-score differs or is non-finite")
    joined["True_NFE_Pseudo_Score"] = truth_a

    groups = joined["Split_Group"].unique()
    if len(groups) < 2:
        raise RuntimeError(f"seed {seed_a}: at least two Split_Group blocks are required")
    group_values = joined["Split_Group"].to_numpy()
    by_group = {str(group): np.flatnonzero(group_values == group) for group in groups}
    observed = _values(joined)
    return SeedPair(seed_a, joined, groups.astype(str), by_group, observed)


def _observed_seed_mean_delta(
    pairs: list[tuple[int, pd.DataFrame, str, str]],
    metric: str,
) -> tuple[float, list[dict[str, object]]]:
    """Return the mean A-minus-B point estimate across complete training seeds.

    The observed formal estimand is seed-level: compute one paired delta for
    each independently trained seed, then average those deltas. Do not pool
    rows across seeds before computing the point estimate.
    """

    rows: list[dict[str, object]] = []
    for seed, frame, source_a, source_b in pairs:
        if metric != "NFE_Pseudo_Score_mae":
            raise ValueError(
                "_observed_seed_mean_delta currently exposes the preregistered "
                "continuous MAE estimand only"
            )

        def column(*names: str) -> np.ndarray:
            for name in names:
                if name in frame:
                    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
                    if not np.all(np.isfinite(values)):
                        raise ValueError(
                            f"seed {seed}: non-finite values in observed-estimate column {name}"
                        )
                    return values
            raise ValueError(f"seed {seed}: missing observed-estimate columns; tried={names}")

        truth_a = column(
            "True_NFE_Pseudo_Score_a",
            "A_True_NFE_Pseudo_Score",
            "True_NFE_Pseudo_Score",
        )
        truth_b = column(
            "True_NFE_Pseudo_Score_b",
            "B_True_NFE_Pseudo_Score",
            "True_NFE_Pseudo_Score",
        )
        pred_a = column(
            "Predicted_NFE_Pseudo_Score_a",
            "A_Predicted_NFE_Pseudo_Score",
        )
        pred_b = column(
            "Predicted_NFE_Pseudo_Score_b",
            "B_Predicted_NFE_Pseudo_Score",
        )
        if not np.allclose(truth_a, truth_b, atol=1e-10, rtol=0.0):
            raise RuntimeError(f"seed {seed}: A/B truth differs")
        delta = float(
            np.mean(np.abs(pred_a - truth_a))
            - np.mean(np.abs(pred_b - truth_b))
        )
        rows.append(
            {
                "seed": int(seed),
                "source_a": str(source_a),
                "source_b": str(source_b),
                "delta_a_minus_b": delta,
            }
        )

    deltas = np.asarray([float(row["delta_a_minus_b"]) for row in rows], dtype=float)
    if not len(deltas) or not np.all(np.isfinite(deltas)):
        raise RuntimeError("observed seed-mean point estimate requires complete finite seed deltas")
    return float(np.mean(deltas)), rows


def _sample_seed_pair(pair: SeedPair, rng: np.random.Generator) -> dict[str, float]:
    selected = rng.choice(pair.groups, size=len(pair.groups), replace=True)
    indices = np.concatenate([pair.by_group[str(group)] for group in selected])
    return _values(pair.frame.iloc[indices])


def main() -> int:
    args = parse_args()
    if len(args.a) != len(args.b):
        raise ValueError("--a and --b must provide the same number of seed prediction files")
    if len(args.a) < args.minimum_training_seeds:
        raise RuntimeError(
            f"formal multi-seed comparison requires at least {args.minimum_training_seeds} training seeds; "
            f"found {len(args.a)}"
        )
    if args.iterations < 100:
        raise ValueError("--iterations must be at least 100")

    pairs = [_pair_frames(a, b) for a, b in zip(args.a, args.b)]
    seeds = [pair.seed for pair in pairs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"duplicate training seeds in paired comparison: {seeds}")

    # Every seed/model file must refer to the same fixed benchmark identity.
    all_manifests = [
        load_prediction_manifest(path, expected_split="test") for path in [*args.a, *args.b]
    ]
    reference = all_manifests[0]
    for manifest in all_manifests[1:]:
        assert_same_prediction_data_identity(reference, manifest)

    metric_names = tuple(pairs[0].observed)
    observed = {}
    for metric in metric_names:
        values = np.asarray([pair.observed.get(metric, np.nan) for pair in pairs], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) != len(values):
            observed[metric] = np.nan
        else:
            observed[metric] = float(np.mean(finite))

    rng = np.random.default_rng(args.seed)
    samples = {metric: [] for metric in metric_names}
    n_seeds = len(pairs)
    for _ in range(args.iterations):
        chosen = rng.integers(0, n_seeds, size=n_seeds)
        per_seed = [_sample_seed_pair(pairs[int(index)], rng) for index in chosen]
        for metric in metric_names:
            values = np.asarray([item.get(metric, np.nan) for item in per_seed], dtype=float)
            samples[metric].append(
                float(np.mean(values)) if np.all(np.isfinite(values)) else np.nan
            )

    result = {
        "comparison": f"{args.name_a} - {args.name_b}",
        "training_seeds": sorted(seeds),
        "training_seed_count": len(seeds),
        "bootstrap_unit": "nested training-seed x Split_Group",
        "iterations": args.iterations,
        "data_identity_sha256": reference["data_identity_sha256"],
        "metrics": {},
    }
    for metric in metric_names:
        array = np.asarray(samples[metric], dtype=float)
        finite = array[np.isfinite(array)]
        valid_fraction = float(len(finite) / len(array)) if len(array) else 0.0
        if not len(finite):
            result["metrics"][metric] = {
                "observed_mean_improvement": observed[metric],
                "ci95_low": None,
                "ci95_high": None,
                "probability_A_better": None,
                "valid_iterations": 0,
                "valid_iteration_fraction": valid_fraction,
            }
        else:
            result["metrics"][metric] = {
                "observed_mean_improvement": observed[metric],
                "ci95_low": float(np.quantile(finite, 0.025)),
                "ci95_high": float(np.quantile(finite, 0.975)),
                "probability_A_better": float(
                    np.mean(finite > 0) + 0.5 * np.mean(finite == 0)
                ),
                "valid_iterations": int(len(finite)),
                "valid_iteration_fraction": valid_fraction,
            }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())