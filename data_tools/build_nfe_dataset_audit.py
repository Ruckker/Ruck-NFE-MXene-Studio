from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Composition, Structure

try:
    from . import build_nfe_dataset_legacy as _legacy
except ImportError:  # direct script execution from data_tools/
    import build_nfe_dataset_legacy as _legacy  # type: ignore


_ORIGINAL_NFE_BAND_FEATURES = _legacy.nfe_band_features
_ORIGINAL_INSPECT_ONE = _legacy.inspect_one


def _assert_indexed_gamma_vectors(band_data: Any) -> None:
    nk = int(band_data.energies.shape[1])
    if nk < 15 or nk % 3 != 0:
        return  # the legacy implementation raises the canonical path error
    segment = nk // 3
    indices = (2 * segment - 1, 2 * segment)
    vectors = np.asarray(band_data.kpoints[list(indices)], dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms > 1e-8):
        raise ValueError(
            "band-path index contract does not land on Gamma k-vectors: "
            f"indices={indices} vectors={vectors.tolist()}"
        )


def nfe_band_features(*args: Any, **kwargs: Any) -> dict[str, Any]:
    band_data = args[0] if args else kwargs.get("band_data")
    if band_data is None:
        raise TypeError("nfe_band_features requires band_data")
    _assert_indexed_gamma_vectors(band_data)
    return _ORIGINAL_NFE_BAND_FEATURES(*args, **kwargs)


def _termination_elements(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "bare", "clean", "null", "nan"}:
        return set()
    try:
        return {str(element) for element in Composition(text).elements}
    except Exception as exc:
        raise ValueError(f"cannot parse termination chemistry {text!r}") from exc


def _expected_name_elements(row: dict[str, Any]) -> set[str]:
    if not bool(row.get("Name_Parse_OK")):
        raise ValueError("structure_name_chemistry_unparseable")
    expected = {
        str(row.get("Metal_Top") or "").strip(),
        str(row.get("Metal_Bottom") or "").strip(),
        str(row.get("X_Element") or "").strip(),
    }
    expected.discard("")
    expected.update(_termination_elements(row.get("Termination_Top")))
    expected.update(_termination_elements(row.get("Termination_Bottom")))
    if not expected:
        raise ValueError("structure_name_chemistry_empty")
    return expected


def _append_chemistry_audit(result: dict[str, Any]) -> dict[str, Any]:
    row = result.get("row")
    if not isinstance(row, dict):
        return result
    reasons = list(result.get("hard_reasons", []))
    try:
        expected = _expected_name_elements(row)
    except ValueError as exc:
        reasons.append(str(exc))
        result["hard_reasons"] = reasons
        result["status"] = "dirty"
        return result

    source = str(row.get("_structure_source") or "").strip()
    if not source:
        return result
    path = Path(source)
    if not path.is_file():
        return result
    try:
        structure = Structure.from_file(path)
        actual = {str(element) for element in structure.composition.elements}
    except Exception:
        # The legacy parser already owns structure parse errors.
        return result
    if actual != expected:
        reasons.append(
            "structure_name_chemistry_mismatch:"
            f"expected={sorted(expected)}:actual={sorted(actual)}"
        )
        result["hard_reasons"] = reasons
        result["status"] = "dirty"
    return result


def inspect_one(task: tuple[str, bool]) -> dict[str, Any]:
    return _append_chemistry_audit(_ORIGINAL_INSPECT_ONE(task))


# Patch the preserved implementation so internal calls and ProcessPoolExecutor
# jobs use the audited entrypoints as well.
_legacy.nfe_band_features = nfe_band_features
_legacy.inspect_one = inspect_one

for _name in dir(_legacy):
    if _name.startswith("_") or _name in {"nfe_band_features", "inspect_one"}:
        continue
    globals()[_name] = getattr(_legacy, _name)


def main() -> int:
    return _legacy.main()


__all__ = [
    *[
        name
        for name in dir(_legacy)
        if not name.startswith("_") and name not in {"nfe_band_features", "inspect_one"}
    ],
    "inspect_one",
    "nfe_band_features",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
