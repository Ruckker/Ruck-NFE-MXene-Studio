from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import formal_data_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


VACUUM_CUTOFF_SAFETY_MARGIN_A = 0.10


def assert_graph_vacuum_adequacy(
    record: Mapping[str, Any], radius: float, *, record_id: str | None = None
) -> float:
    """Require atom-free normal vacuum to exceed cutoff by a safety margin."""

    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("graph radius must be finite and > 0 for slab-vacuum auditing")
    vacuum = _core.graph_normal_vacuum_A(record)
    minimum = radius + VACUUM_CUTOFF_SAFETY_MARGIN_A
    if vacuum <= minimum + 1e-6:
        name = record_id or str(record.get("id", "structure"))
        raise RuntimeError(
            f"formal slab {name!r} has only {vacuum:.6f} Å normal vacuum; "
            f"the {radius:.6f} Å graph cutoff requires >{minimum:.6f} Å including "
            "the formal safety margin, otherwise 3D PBC can create cross-vacuum neighbors"
        )
    return vacuum


def assert_formal_slab_vacuum(
    records: Sequence[Mapping[str, Any]], radius: float
) -> float:
    if not records:
        raise RuntimeError("formal slab-vacuum audit received no records")
    minimum = float("inf")
    for index, record in enumerate(records):
        vacuum = assert_graph_vacuum_adequacy(
            record, radius, record_id=str(record.get("id", index))
        )
        minimum = min(minimum, vacuum)
    return float(minimum)


def _has_relative_energy_slot(record: Mapping[str, Any]) -> bool:
    targets = record.get("targets")
    mask = record.get("target_mask")
    try:
        return len(targets) > 1 and len(mask) > 1
    except TypeError:
        return False


def assert_formal_primary_target_coverage(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, Any]]:
    """Require class + finite primary NFE score coverage on every fixed split.

    Full production records also carry the relative-energy slot used by the
    pseudo-label rule; when that evidence slot exists, its consistency audit is
    retained. Minimal fixtures or downstream datasets that intentionally expose
    only the primary score are still valid inputs to this *coverage* function.
    """

    summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        indices = [int(index) for index in splits.get(split, ())]
        if not indices:
            raise RuntimeError(f"formal benchmark split {split!r} is empty")

        support = [0] * _core.FORMAL_CLASS_COUNT
        missing_labels: list[str] = []
        missing_scores: list[str] = []
        invalid_scores: list[str] = []
        for record_index in indices:
            record = records[record_index]
            record_id = str(record.get("id", record_index))
            label = int(record.get("label", -1))
            if label < 0:
                missing_labels.append(record_id)
            elif label >= _core.FORMAL_CLASS_COUNT:
                raise RuntimeError(
                    f"formal dataset record {record_id!r} has invalid class label {label}; "
                    f"expected 0..{_core.FORMAL_CLASS_COUNT - 1}"
                )
            else:
                support[label] += 1

            mask_value = _core._scalar_at(
                record.get("target_mask"), 0, name="target_mask", record_id=record_id
            )
            if not bool(mask_value):
                missing_scores.append(record_id)
                continue
            score_value = _core._scalar_at(
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
                continue
            if label >= 0 and _has_relative_energy_slot(record):
                _core.assert_pseudo_label_consistency(record, record_id=record_id)

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
            "pseudo_label_schema": _core.PSEUDO_LABEL_SCHEMA,
        }
    return summary


# Ensure helpers defined inside the preserved module resolve the stricter
# vacuum gate as well.
_core.assert_graph_vacuum_adequacy = assert_graph_vacuum_adequacy
_core.assert_formal_slab_vacuum = assert_formal_slab_vacuum
_core.assert_formal_primary_target_coverage = assert_formal_primary_target_coverage


def __getattr__(name: str):
    return getattr(_core, name)
