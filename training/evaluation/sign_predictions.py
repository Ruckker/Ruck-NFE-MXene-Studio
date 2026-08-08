from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from nfe_model.prediction_manifest import write_prediction_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cryptographically bind an audited prediction CSV to its sibling run result/provenance."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result", help="result.json/final_metrics.json; auto-detected from prediction directory")
    parser.add_argument("--split", choices=("validation", "test"))
    return parser.parse_args()


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("provenance"), Mapping):
        raise ValueError(f"run result has no formal provenance mapping: {path}")
    return payload


def _result_path(prediction_path: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        prediction_path.with_name("result.json"),
        prediction_path.with_name("final_metrics.json"),
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(
            "prediction signer requires exactly one sibling result.json or final_metrics.json; "
            f"found={[str(path) for path in existing]}"
        )
    return existing[0]


def _split(prediction_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    stem = prediction_path.stem.lower()
    if stem.startswith("validation_") or stem == "validation_predictions":
        return "validation"
    if stem.startswith("test_") or stem == "test_predictions":
        return "test"
    raise ValueError("cannot infer validation/test split from prediction filename; pass --split")


def _seed(payload: Mapping[str, Any], result_path: Path) -> int | None:
    value = payload.get("seed")
    if value is None and isinstance(payload.get("config"), Mapping):
        value = payload["config"].get("seed")
    if value is None and result_path.parent.name.startswith("seed_"):
        try:
            value = int(result_path.parent.name.removeprefix("seed_"))
        except ValueError:
            value = None
    return None if value is None else int(value)


def _track_model(payload: Mapping[str, Any]) -> tuple[str, str]:
    track = str(payload.get("track") or "")
    model = str(payload.get("model") or "")
    ablation = payload.get("ablation_config")
    if isinstance(ablation, Mapping) and ablation.get("name"):
        name = str(ablation["name"])
        return track or "ablation", model or name
    return track or "predictor", model or "ours_full"


def _checkpoint_hash(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("checkpoint_sha256")
    if value:
        return str(value)
    details = payload.get("details")
    if isinstance(details, Mapping) and details.get("checkpoint_sha256"):
        return str(details["checkpoint_sha256"])
    return None


def _expected_rows(payload: Mapping[str, Any], split: str) -> int | None:
    sizes = payload.get("split_sizes")
    if isinstance(sizes, Mapping) and split in sizes:
        return int(sizes[split])
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        coverage = provenance.get("primary_target_coverage")
        if isinstance(coverage, Mapping) and isinstance(coverage.get(split), Mapping):
            rows = coverage[split].get("rows")
            if rows is not None:
                return int(rows)
    return None


def main() -> int:
    args = parse_args()
    prediction_path = Path(args.predictions).resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    result_path = _result_path(prediction_path, args.result)
    split = _split(prediction_path, args.split)
    payload = _load_result(result_path)

    frame = pd.read_csv(prediction_path)
    required = {
        "Structure_Name",
        "True_Label",
        "Probability_Low",
        "Probability_Medium",
        "Probability_High",
        "True_NFE_Pseudo_Score",
        "Predicted_NFE_Pseudo_Score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction CSV is missing formal columns: {sorted(missing)}")
    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (identifiers == "").any() or identifiers.duplicated().any():
        raise ValueError("prediction CSV requires unique non-empty Structure_Name values before signing")
    expected_rows = _expected_rows(payload, split)
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(
            f"prediction row count {len(frame)} does not match formal {split} support {expected_rows}"
        )

    track, model = _track_model(payload)
    output = write_prediction_manifest(
        prediction_path,
        split=split,
        provenance=payload["provenance"],
        track=track,
        model=model,
        seed=_seed(payload, result_path),
        checkpoint_sha256=_checkpoint_hash(payload),
        training_protocol_sha256=(
            payload.get("training_protocol_sha256")
            or payload.get("benchmark_common_protocol_sha256")
            or payload.get("model_protocol_sha256")
        ),
        model_protocol_sha256=payload.get("model_protocol_sha256"),
        temperature=(
            payload.get("classification_temperature")
            if payload.get("classification_temperature") is not None
            else payload.get("temperature")
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
