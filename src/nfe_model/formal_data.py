from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FORMAL_CLASS_COUNT = 3


def _scalar_at(value: Any, index: int, *, name: str, record_id: str) -> Any:
    try:
        return value[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"formal dataset record {record_id!r} has no {name}[{index}] primary target entry"
        ) from exc


def assert_formal_primary_target_coverage(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, Any]]:
    """Require a well-defined class+NFE-score benchmark on every fixed split.

    OOD/verified *slices* are intentionally allowed to miss classes and are
    evaluated by ``metrics_v2`` with NaN-aware macros. The canonical fixed
    train/validation/test benchmark is different: every retained row must have
    both primary targets, and every split must contain all three classes. This
    keeps checkpoint selection and paper metrics comparable across runs.
    """

    summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        indices = [int(index) for index in splits.get(split, ())]
        if not indices:
            raise RuntimeError(f"formal benchmark split {split!r} is empty")

        support = [0] * FORMAL_CLASS_COUNT
        missing_labels: list[str] = []
        missing_scores: list[str] = []
        invalid_scores: list[str] = []
        for record_index in indices:
            record = records[record_index]
            record_id = str(record.get("id", record_index))
            label = int(record.get("label", -1))
            if label < 0:
                missing_labels.append(record_id)
            elif label >= FORMAL_CLASS_COUNT:
                raise RuntimeError(
                    f"formal dataset record {record_id!r} has invalid class label {label}; "
                    f"expected 0..{FORMAL_CLASS_COUNT - 1}"
                )
            else:
                support[label] += 1

            mask_value = _scalar_at(
                record.get("target_mask"), 0, name="target_mask", record_id=record_id
            )
            if not bool(mask_value):
                missing_scores.append(record_id)
                continue
            score_value = _scalar_at(
                record.get("targets"), 0, name="targets", record_id=record_id
            )
            try:
                score = float(score_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"formal dataset record {record_id!r} has a non-numeric NFE score"
                ) from exc
            if not math.isfinite(score):
                invalid_scores.append(record_id)

        if missing_labels:
            raise RuntimeError(
                f"formal {split} split contains rows without NFE class labels; "
                f"examples={missing_labels[:5]}"
            )
        if missing_scores or invalid_scores:
            raise RuntimeError(
                f"formal {split} split requires a finite NFE_Pseudo_Score on every row; "
                f"missing_examples={missing_scores[:5]} invalid_examples={invalid_scores[:5]}"
            )
        missing_classes = [index for index, count in enumerate(support) if count == 0]
        if missing_classes:
            raise RuntimeError(
                f"formal {split} split does not contain all three NFE classes; "
                f"support={support}, missing_class_indices={missing_classes}"
            )
        summary[split] = {
            "rows": len(indices),
            "class_support": tuple(int(value) for value in support),
            "primary_score_support": len(indices),
        }
    return summary
