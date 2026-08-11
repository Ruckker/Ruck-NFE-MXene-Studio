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


SELECTION_SCHEMA = "verified-nfe-review-selection-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic prediction-blind review queue from the fixed test split."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--mode", choices=("balanced-class", "natural"), default="balanced-class")
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
    result = []
    depth = 0
    while True:
        added = False
        for values in groups:
            if depth < len(values):
                result.append(values[depth])
                added = True
        if not added:
            break
        depth += 1
    return result


def _select_balanced(frame: pd.DataFrame, per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    if per_class <= 0:
        raise ValueError("--per-class must be > 0")
    selected = []
    for label_index in (0, 1, 2):
        subset = frame[frame["Label_Index"] == label_index]
        if len(subset) < per_class:
            raise RuntimeError(
                f"test split has only {len(subset)} rows for class {INDEX_TO_LABEL[label_index]!r}; "
                f"cannot select {per_class}"
            )
        order = _group_diverse_order(subset, rng)
        selected.extend(order[:per_class])
    result = frame.loc[selected].copy()
    return result.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))


def _select_natural(frame: pd.DataFrame, total: int, rng: np.random.Generator) -> pd.DataFrame:
    if total <= 0 or total > len(frame):
        raise ValueError(f"--total must be in 1..{len(frame)}")
    order = _group_diverse_order(frame, rng)
    return frame.loc[order[:total]].copy()


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
    if args.mode == "balanced-class":
        selected = _select_balanced(frame, int(args.per_class), rng)
    else:
        selected = _select_natural(frame, int(args.total), rng)

    queue = selected[
        [
            "Record_Index",
            "Structure_Name",
            "Split_Group",
            "Pseudo_Label_Stratum",
            "Source_File_SHA256",
        ]
    ].reset_index(drop=True)
    queue["Charge_Localization_Reviewed"] = ""
    queue["Charge_Localization_Confirmed"] = ""
    queue["Parabolic_Dispersion_Reviewed"] = ""
    queue["Parabolic_Dispersion_Confirmed"] = ""
    queue["Effective_Mass_Reviewed"] = ""
    queue["Effective_Mass_Consistent"] = ""
    queue["Verified_NFE_Label"] = ""
    queue["Verified_NFE_Score"] = ""
    queue["Reviewer_Confidence"] = ""
    queue["Reviewer_ID"] = ""
    queue["Review_Notes"] = ""

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False)
    queue_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    selection = {
        "schema": SELECTION_SCHEMA,
        "mode": args.mode,
        "selection_seed": int(args.seed),
        "rows": int(len(queue)),
        "per_class": int(args.per_class) if args.mode == "balanced-class" else None,
        "requested_total": int(args.total) if args.mode == "natural" else None,
        "selected_class_support": queue["Pseudo_Label_Stratum"].value_counts().sort_index().to_dict(),
        "selected_split_group_count": int(queue["Split_Group"].nunique()),
        "queue_sha256": queue_sha,
        "provenance": provenance,
        "selection_protocol_sha256": canonical_sha256(
            {
                "schema": SELECTION_SCHEMA,
                "mode": args.mode,
                "selection_seed": int(args.seed),
                "per_class": int(args.per_class) if args.mode == "balanced-class" else None,
                "total": int(args.total) if args.mode == "natural" else None,
                "dataset_table_sha256": provenance["dataset_table_sha256"],
                "split_manifest_sha256": provenance["split_manifest_sha256"],
            }
        ),
        "prediction_blinding": (
            "queue is generated without reading any model prediction file; reviewers should freeze this table "
            "before predictions are exposed"
        ),
    }
    manifest = output.with_name(f"{output.stem}.selection.json")
    manifest.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
