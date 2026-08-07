from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chemistry and large-cell-representation OOD strata from the fixed split."
    )
    parser.add_argument("--dataset", default="data/full/nfe_dataset.csv")
    parser.add_argument("--output", default="training/evaluation/ood_manifest.csv")
    parser.add_argument("--cell-size-quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.5 <= float(args.cell_size_quantile) < 1.0:
        raise ValueError("--cell-size-quantile must be in [0.5, 1.0)")
    frame = pd.read_csv(args.dataset)
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
                "Cell_Size_Slice_Interpretation": "representation-size stress test; exact supercell invariance is audited separately",
            }
        )
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        f"wrote {len(rows)} rows to {out}; train N_Atoms q={args.cell_size_quantile}: "
        f"{cell_threshold:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
