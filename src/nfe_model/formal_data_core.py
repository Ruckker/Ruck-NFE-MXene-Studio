from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


FORMAL_CLASS_COUNT = 3
PSEUDO_LABEL_SCHEMA = "nfe-pseudo-label-v1:low<0.48;medium>=0.48;high>=0.70&abs(relative_E)<=1.0"
PSEUDO_LABEL_ROUNDING_TOL = 1e-7


def _scalar_at(value: Any, index: int, *, name: str, record_id: str) -> Any:
    try:
        return value[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"formal dataset record {record_id!r} has no {name}[{index}] target entry"
        ) from exc


def _finite_scalar(value: Any, *, name: str, record_id: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"formal dataset record {record_id!r} has a non-numeric {name}"
        ) from exc
    if not math.isfinite(numeric):
        raise RuntimeError(
            f"formal dataset record {record_id!r} has a non-finite {name}"
        )
    return numeric


def assert_pseudo_label_consistency(record: Mapping[str, Any], *, record_id: str) -> dict[str, float | int]:
    """Check necessary implications of the dataset builder's pseudo-label rule.

    The builder computes labels from unrounded values, while the CSV stores
    values rounded to eight decimal places. A small tolerance therefore avoids
    false failures exactly at 0.48, 0.70 and |relative_E|=1.0 without allowing a
    materially inconsistent label/score pair.
    """

    label = int(record.get("label", -1))
    if label not in (0, 1, 2):
        raise RuntimeError(
            f"formal dataset record {record_id!r} has invalid class label {label}; expected 0..2"
        )

    score_mask = bool(
        _scalar_at(record.get("target_mask"), 0, name="target_mask", record_id=record_id)
    )
    if not score_mask:
        raise RuntimeError(
            f"formal dataset record {record_id!r} has a masked NFE_Pseudo_Score"
        )
    score = _finite_scalar(
        _scalar_at(record.get("targets"), 0, name="targets", record_id=record_id),
        name="NFE_Pseudo_Score",
        record_id=record_id,
    )
    tol = PSEUDO_LABEL_ROUNDING_TOL
    if score < -tol or score > 1.0 + tol:
        raise RuntimeError(
            f"formal dataset record {record_id!r} has NFE_Pseudo_Score={score:.10g} outside [0,1]"
        )

    energy_mask = bool(
        _scalar_at(record.get("target_mask"), 1, name="target_mask", record_id=record_id)
    )
    relative_energy: float | None = None
    if energy_mask:
        relative_energy = _finite_scalar(
            _scalar_at(record.get("targets"), 1, name="targets", record_id=record_id),
            name="NFE_Energy_Relative_EF_eV",
            record_id=record_id,
        )

    if label == 0:
        if score > 0.48 + tol:
            raise RuntimeError(
                f"formal dataset record {record_id!r} is labelled low but score={score:.10g} is clearly >=0.48"
            )
    elif label == 1:
        if score < 0.48 - tol:
            raise RuntimeError(
                f"formal dataset record {record_id!r} is labelled medium but score={score:.10g} is clearly <0.48"
            )
        # A medium example clearly above the high score threshold requires the
        # energy gate to explain why it is not high. If the stored energy is
        # clearly inside the high window, the row contradicts the builder rule.
        if score > 0.70 + tol:
            if relative_energy is None:
                raise RuntimeError(
                    f"formal dataset record {record_id!r} is medium with score>0.70 but lacks relative-energy evidence"
                )
            if abs(relative_energy) < 1.0 - tol:
                raise RuntimeError(
                    f"formal dataset record {record_id!r} is medium although score={score:.10g} and "
                    f"relative_energy={relative_energy:.10g} satisfy the high-label rule"
                )
    else:
        if score < 0.70 - tol:
            raise RuntimeError(
                f"formal dataset record {record_id!r} is labelled high but score={score:.10g} is clearly <0.70"
            )
        if relative_energy is None:
            raise RuntimeError(
                f"formal dataset record {record_id!r} is labelled high but lacks NFE_Energy_Relative_EF_eV"
            )
        if abs(relative_energy) > 1.0 + tol:
            raise RuntimeError(
                f"formal dataset record {record_id!r} is labelled high but |relative_energy|={abs(relative_energy):.10g} >1.0 eV"
            )

    return {
        "label": label,
        "score": float(score),
        "relative_energy_eV": float(relative_energy) if relative_energy is not None else float("nan"),
    }


def graph_normal_vacuum_A(record: Mapping[str, Any]) -> float:
    """Return the largest atom-free fractional-z gap in Cartesian slab-normal Å."""
    frac = record.get("frac_pos")
    lattice = record.get("lattice")
    if not torch.is_tensor(frac) or frac.ndim != 2 or frac.shape[1] != 3 or len(frac) == 0:
        raise RuntimeError("formal slab record has invalid frac_pos tensor")
    if not torch.is_tensor(lattice) or tuple(lattice.shape) != (3, 3):
        raise RuntimeError("formal slab record has invalid lattice tensor")
    z = torch.remainder(frac[:, 2].detach().cpu().double(), 1.0).sort().values
    if len(z) == 1:
        vacuum_fraction = 1.0
    else:
        gaps = torch.cat((z[1:] - z[:-1], (z[:1] + 1.0) - z[-1:]))
        vacuum_fraction = float(gaps.max().item())
    cell = lattice.detach().cpu().double()
    area = float(torch.linalg.vector_norm(torch.cross(cell[0], cell[1], dim=0)).item())
    volume = abs(float(torch.linalg.det(cell).item()))
    if not math.isfinite(area) or not math.isfinite(volume) or area <= 1e-12 or volume <= 1e-12:
        raise RuntimeError("formal slab record has singular/non-finite lattice geometry")
    normal_repeat = volume / area
    return float(vacuum_fraction * normal_repeat)


def assert_graph_vacuum_adequacy(
    record: Mapping[str, Any], radius: float, *, record_id: str | None = None
) -> float:
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("graph radius must be finite and > 0 for slab-vacuum auditing")
    vacuum = graph_normal_vacuum_A(record)
    if vacuum <= radius + 1e-6:
        name = record_id or str(record.get("id", "structure"))
        raise RuntimeError(
            f"formal slab {name!r} has only {vacuum:.6f} Å normal vacuum, not greater than "
            f"the {radius:.6f} Å graph cutoff; 3D PBC would create cross-vacuum neighbors"
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


def assert_formal_primary_target_coverage(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, Any]]:
    """Require a well-defined class+NFE-score benchmark on every fixed split.

    OOD/verified *slices* are intentionally allowed to miss classes and are
    evaluated by ``metrics_v2`` with NaN-aware macros. The canonical fixed
    train/validation/test benchmark is different: every retained row must have
    both primary targets, every score must obey the pseudo-label definition, and
    every split must contain all three classes. This keeps checkpoint selection
    and paper metrics comparable across runs.
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
                continue
            if label >= 0:
                assert_pseudo_label_consistency(record, record_id=record_id)

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
            "pseudo_label_schema": PSEUDO_LABEL_SCHEMA,
        }
    return summary
