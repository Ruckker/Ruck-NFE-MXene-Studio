from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pymatgen.core import Element

from nfe_model.prediction_manifest import prediction_data_identity
from nfe_model.provenance_v2 import canonical_sha256, file_sha256
from training.baselines.common import load_benchmark_data


OOD_MANIFEST_SCHEMA = "nfe-ood-manifest-1.0"


def _norm(value) -> str:
    return str(value).strip() if pd.notna(value) else ""


def _pair(row: pd.Series, first: str, second: str) -> str:
    values = [_norm(row[first]), _norm(row[second])]
    if any(not value for value in values):
        raise ValueError(
            f"OOD chemistry metadata requires non-empty {first}/{second} for {row['Structure_Name']}"
        )
    return "|".join(sorted(values))


def _parse_bool(value, *, name: str) -> bool:
    text = _norm(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{name} contains an unparseable boolean value {value!r}")


def _elements(row: pd.Series) -> frozenset[str]:
    text = _norm(row["Elements"])
    if not text:
        raise ValueError(f"blank Elements metadata for {row['Structure_Name']}")
    values = [value.strip() for value in text.split("|") if value.strip()]
    if not values:
        raise ValueError(f"empty Elements metadata for {row['Structure_Name']}")
    invalid = []
    for value in values:
        try:
            Element(value)
        except Exception:
            invalid.append(value)
    if invalid:
        raise ValueError(
            f"unrecognized element symbols for {row['Structure_Name']}: {invalid[:5]}"
        )
    return frozenset(values)


def ood_manifest_sidecar_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    return path.with_name(f"{path.stem}.manifest.json")


def load_ood_manifest_sidecar(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    sidecar_path = ood_manifest_sidecar_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"formal OOD manifest has no identity sidecar: {sidecar_path}"
        )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if payload.get("schema") != OOD_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported OOD manifest sidecar schema: {payload.get('schema')!r}"
        )
    if payload.get("manifest_filename") != path.name:
        raise ValueError("OOD sidecar filename does not match the supplied manifest CSV")
    observed_hash = file_sha256(path)
    if payload.get("manifest_file_sha256") != observed_hash:
        raise ValueError(
            "OOD manifest CSV bytes do not match its sidecar: "
            f"sidecar={payload.get('manifest_file_sha256')} current={observed_hash}"
        )
    identity = payload.get("data_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("OOD sidecar has no data_identity mapping")
    canonical_identity = prediction_data_identity(identity)
    observed_identity_hash = canonical_sha256(canonical_identity)
    if observed_identity_hash != payload.get("data_identity_sha256"):
        raise ValueError("OOD sidecar data identity hash is inconsistent")
    construction = payload.get("construction")
    if not isinstance(construction, Mapping):
        raise ValueError("OOD sidecar has no construction mapping")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build chemistry and large-cell-representation OOD strata from the exact audited "
            "benchmark table/cache identity."
        )
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--output", default="training/evaluation/ood_manifest.csv")
    parser.add_argument("--cell-size-quantile", type=float, default=0.95)
    return parser.parse_args()


def _assert_table_cache_alignment(frame: pd.DataFrame, data) -> None:
    if len(frame) != len(data.records):
        raise RuntimeError(
            "formal OOD construction requires zero cache skips and one cache record per dataset row: "
            f"table_rows={len(frame)} cache_records={len(data.records)}"
        )
    for position, (_, row) in enumerate(frame.iterrows()):
        record = data.records[position]
        identifier = _norm(row["Structure_Name"])
        if identifier != str(record.get("id", "")):
            raise RuntimeError(
                f"OOD table/cache row identity mismatch at row {position}: "
                f"table={identifier!r} cache={record.get('id')!r}"
            )
        group = _norm(row["Split_Group"])
        if group != str(record.get("split_group", "")):
            raise RuntimeError(
                f"OOD table/cache Split_Group mismatch for {identifier}: "
                f"table={group!r} cache={record.get('split_group')!r}"
            )
        n_atoms = pd.to_numeric(row["N_Atoms"], errors="coerce")
        if pd.isna(n_atoms) or int(n_atoms) != int(record["z"].numel()):
            raise RuntimeError(
                f"OOD table/cache N_Atoms mismatch for {identifier}: "
                f"table={n_atoms!r} cache={int(record['z'].numel())}"
            )


def main() -> int:
    args = parse_args()
    if not 0.5 <= float(args.cell_size_quantile) < 1.0:
        raise ValueError("--cell-size-quantile must be in [0.5, 1.0)")

    data = load_benchmark_data(args.config, rebuild_cache=False)
    if data.skipped_cache_records != 0:
        raise RuntimeError(
            "formal OOD manifest construction requires zero skipped cache records; "
            f"observed={data.skipped_cache_records}"
        )
    frame = pd.read_csv(data.table_path)
    required = {
        "Structure_Name",
        "Suggested_Split",
        "Split_Group",
        "Metal_Top",
        "Metal_Bottom",
        "X_Element",
        "Termination_Top",
        "Termination_Bottom",
        "Elements",
        "N_Atoms",
        "Name_Parse_OK",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing OOD metadata columns: {sorted(missing)}")

    identifiers = frame["Structure_Name"].fillna("").astype(str).str.strip()
    if (identifiers == "").any() or identifiers.duplicated().any():
        raise ValueError("OOD manifest requires unique non-empty Structure_Name values")
    groups = frame["Split_Group"].fillna("").astype(str).str.strip()
    if (groups == "").any():
        raise ValueError("OOD manifest requires non-empty Split_Group values")

    parse_ok = frame["Name_Parse_OK"].map(
        lambda value: _parse_bool(value, name="Name_Parse_OK")
    )
    if not bool(parse_ok.all()):
        examples = identifiers[~parse_ok].tolist()[:5]
        raise ValueError(
            "formal chemistry OOD analysis requires successfully parsed structure-name metadata; "
            f"examples={examples}"
        )

    split = frame["Suggested_Split"].astype(str).str.strip().str.lower().replace(
        {"val": "validation", "valid": "validation"}
    )
    invalid_split = ~split.isin({"train", "validation", "test"})
    if invalid_split.any():
        raise ValueError(
            f"unrecognized Suggested_Split values: {sorted(split[invalid_split].unique())}"
        )
    train = frame.loc[split == "train"].copy()
    if train.empty:
        raise RuntimeError("no training rows found")

    _assert_table_cache_alignment(frame, data)

    for column in (
        "Metal_Top",
        "Metal_Bottom",
        "X_Element",
        "Termination_Top",
        "Termination_Bottom",
        "Elements",
    ):
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            examples = identifiers[
                frame[column].fillna("").astype(str).str.strip().eq("")
            ].tolist()[:5]
            raise ValueError(f"formal OOD metadata column {column} contains blanks; examples={examples}")

    train_metal = {_pair(row, "Metal_Top", "Metal_Bottom") for _, row in train.iterrows()}
    train_term = {
        _pair(row, "Termination_Top", "Termination_Bottom")
        for _, row in train.iterrows()
    }
    train_x = {_norm(value) for value in train["X_Element"]}
    train_elements: set[str] = set()
    for _, row in train.iterrows():
        train_elements.update(_elements(row))

    n_atoms_all = pd.to_numeric(frame["N_Atoms"], errors="coerce")
    if n_atoms_all.isna().any() or (n_atoms_all <= 0).any():
        examples = identifiers[n_atoms_all.isna() | (n_atoms_all <= 0)].tolist()[:5]
        raise ValueError(f"N_Atoms must be positive for formal OOD analysis; examples={examples}")
    n_atoms_train = n_atoms_all[split == "train"].to_numpy(float)
    cell_threshold = float(np.quantile(n_atoms_train, args.cell_size_quantile))

    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        elements = _elements(row)
        metal = _pair(row, "Metal_Top", "Metal_Bottom")
        termination = _pair(row, "Termination_Top", "Termination_Bottom")
        x_element = _norm(row["X_Element"])
        n_atoms = float(n_atoms_all.iloc[position])
        unseen_metal = metal not in train_metal
        unseen_termination = termination not in train_term
        unseen_x = x_element not in train_x
        unseen_element = bool(set(elements) - train_elements)
        large_cell_representation = n_atoms > cell_threshold
        chemistry_ood = unseen_metal or unseen_termination or unseen_x or unseen_element
        rows.append(
            {
                "Dataset_Row_Index": position,
                "Structure_Name": _norm(row["Structure_Name"]),
                "Split_Group": _norm(row["Split_Group"]),
                "Suggested_Split": split.iloc[position],
                "OOD_Unseen_Metal_Pair": unseen_metal,
                "OOD_Unseen_Termination_Pair": unseen_termination,
                "OOD_Unseen_X_Element": unseen_x,
                "OOD_Unseen_Element": unseen_element,
                "OOD_Large_Cell_Representation": large_cell_representation,
                "OOD_Any_Chemistry": chemistry_ood,
                "OOD_Any": chemistry_ood or large_cell_representation,
                "Cell_Size_Threshold_NAtoms": cell_threshold,
                "Cell_Size_Quantile": float(args.cell_size_quantile),
                "Cell_Size_Slice_Interpretation": (
                    "representation-size stress test; exact supercell invariance is audited separately"
                ),
            }
        )

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    identity = prediction_data_identity(data.provenance)
    sidecar = {
        "schema": OOD_MANIFEST_SCHEMA,
        "manifest_filename": out.name,
        "manifest_file_sha256": file_sha256(out),
        "data_identity": identity,
        "data_identity_sha256": canonical_sha256(identity),
        "construction": {
            "cell_size_quantile": float(args.cell_size_quantile),
            "cell_size_threshold_n_atoms": cell_threshold,
            "train_only_reference_statistics": True,
            "chemistry_reference_split": "train",
            "large_cell_reference_split": "train",
            "rows": len(rows),
        },
    }
    sidecar_path = ood_manifest_sidecar_path(out)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} rows to {out} and identity sidecar {sidecar_path}; "
        f"train N_Atoms q={args.cell_size_quantile}: {cell_threshold:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
