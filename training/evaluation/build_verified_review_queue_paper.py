from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfe_model.data_v2 import INDEX_TO_LABEL, load_or_build_cache, split_indices
from nfe_model.formal_config import validate_formal_config
from nfe_model.provenance_v2 import build_provenance, canonical_sha256
from nfe_model.utils import load_config


SELECTION_SCHEMA = "verified-nfe-review-selection-2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic prediction-blind verified-NFE review queue for paper analysis."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument(
        "--mode",
        choices=("class-balanced-group-diverse", "test-prevalence-random"),
        default="class-balanced-group-diverse",
    )
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--total", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default="training/evaluation/results/verified_review_queue.csv")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def _resolve_config(path: Path) -> dict[str, Any]:
    config = load_config(path)
    base = path.resolve().parent
    for key in ("table", "root", "cache"):
        value = Path(config["data"][key])
        if not value.is_absolute():
            value = base / value
        config["data"][key] = str(value.resolve())
    validate_formal_config(config)
    return config


def _group_diverse_order(frame: pd.DataFrame, rng: np.random.Generator) -> list[int]:
    groups = []
    for _, group in frame.groupby("Split_Group", sort=True):
        values = group.index.to_numpy(dtype=int).copy()
        rng.shuffle(values)
        groups.append(values.tolist())
    rng.shuffle(groups)
    ordered: list[int] = []
    depth = 0
    while True:
        added = False
        for values in groups:
            if depth < len(values):
                ordered.append(values[depth])
                added = True
        if not added:
            return ordered
        depth += 1


def _balanced_group_diverse(
    frame: pd.DataFrame, per_class: int, rng: np.random.Generator
) -> pd.DataFrame:
    if per_class <= 0:
        raise ValueError("--per-class must be > 0")
    selected: list[int] = []
    for label in (0, 1, 2):
        subset = frame[frame["Label_Index"] == label]
        if len(subset) < per_class:
            raise RuntimeError(
                f"test split class {INDEX_TO_LABEL[label]!r} has {len(subset)} rows, "
                f"fewer than requested {per_class}"
            )
        selected.extend(_group_diverse_order(subset, rng)[:per_class])
    selected_array = np.asarray(selected, dtype=int)
    rng.shuffle(selected_array)
    return frame.loc[selected_array].copy()


def _test_prevalence_random(
    frame: pd.DataFrame, total: int, rng: np.random.Generator
) -> pd.DataFrame:
    if total <= 0 or total > len(frame):
        raise ValueError(f"--total must be in 1..{len(frame)}")
    selected = rng.choice(frame.index.to_numpy(dtype=int), size=total, replace=False)
    return frame.loc[selected].copy()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = _resolve_config(config_path)
    data = config["data"]
    cache = load_or_build_cache(
        data["table"],
        data["root"],
        data["cache"],
        radius=float(data["radius"]),
        max_neighbors=int(data["max_neighbors"]),
        rebuild=bool(args.rebuild_cache),
    )
    records = cache["records"]
    splits = split_indices(records)
    provenance = build_provenance(cache=cache, records=records, splits=splits)

    rows = []
    for record_index in splits["test"]:
        record = records[int(record_index)]
        label = int(record["label"])
        rows.append(
            {
                "Record_Index": int(record_index),
                "Structure_Name": str(record["id"]),
                "Split_Group": str(record["split_group"]),
                "Label_Index": label,
                "Pseudo_Label_Stratum": INDEX_TO_LABEL[label],
                "Source_File_SHA256": str(record.get("source_file_sha256", "")),
            }
        )
    frame = pd.DataFrame(rows).set_index("Record_Index", drop=False)
    rng = np.random.default_rng(args.seed)
    if args.mode == "class-balanced-group-diverse":
        selected = _balanced_group_diverse(frame, int(args.per_class), rng)
    else:
        selected = _test_prevalence_random(frame, int(args.total), rng)

    queue = selected[
        ["Record_Index", "Structure_Name", "Split_Group", "Pseudo_Label_Stratum", "Source_File_SHA256"]
    ].reset_index(drop=True)
    for column in (
        "Charge_Localization_Reviewed",
        "Charge_Localization_Confirmed",
        "Parabolic_Dispersion_Reviewed",
        "Parabolic_Dispersion_Confirmed",
        "Effective_Mass_Reviewed",
        "Effective_Mass_Consistent",
        "Verified_NFE_Label",
        "Verified_NFE_Score",
        "Verified_NFE_Score_Definition",
        "Reviewer_Confidence",
        "Reviewer_ID",
        "Review_Notes",
    ):
        queue[column] = ""

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False)
    queue_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    protocol = {
        "schema": SELECTION_SCHEMA,
        "mode": args.mode,
        "selection_seed": int(args.seed),
        "per_class": int(args.per_class) if args.mode == "class-balanced-group-diverse" else None,
        "total": int(args.total) if args.mode == "test-prevalence-random" else None,
        "dataset_table_sha256": provenance["dataset_table_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
    }
    manifest = {
        **protocol,
        "rows": int(len(queue)),
        "queue_sha256": queue_hash,
        "selection_protocol_sha256": canonical_sha256(protocol),
        "selected_class_support": queue["Pseudo_Label_Stratum"].value_counts().sort_index().to_dict(),
        "selected_split_group_count": int(queue["Split_Group"].nunique()),
        "provenance": provenance,
        "interpretation": (
            "class-balanced-group-diverse is a designed balanced verification sample; "
            "test-prevalence-random is an unweighted simple random sample of the fixed test split"
        ),
        "prediction_blinding": "selection is generated without reading model predictions",
    }
    manifest_path = output.with_name(f"{output.stem}.selection.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
