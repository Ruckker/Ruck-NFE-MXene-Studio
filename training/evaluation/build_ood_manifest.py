from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


def _norm(value) -> str:
    return str(value).strip() if pd.notna(value) else ""


def _pair(row: pd.Series, first: str, second: str) -> str:
    return "|".join(sorted(value for value in (_norm(row.get(first)), _norm(row.get(second))) if value))


def _elements(row: pd.Series) -> frozenset[str]:
    text = _norm(row.get("Composition"))
    if text:
        try:
            return frozenset(Composition(text).as_dict())
        except Exception:
            pass
    values = []
    for key in ("Metal_Top", "Metal_Bottom", "X_Element"):
        value = _norm(row.get(key))
        if value:
            values.append(value)
    return frozenset(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chemistry and cell-size OOD strata from the fixed split.")
    parser.add_argument("--dataset", default="data/full/nfe_dataset.csv")
    parser.add_argument("--output", default="training/evaluation/ood_manifest.csv")
    parser.add_argument("--cell-size-quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.5 <= float(args.cell_size_quantile) < 1.0:
        raise ValueError("--cell-size-quantile must be in [0.5, 1.0)")
    frame = pd.read_csv(args.dataset)
    required = {"Structure_Name", "Suggested_Split", "Split_Group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing OOD columns: {sorted(missing)}")
    if frame["Structure_Name"].astype(str).duplicated().any():
        raise ValueError("OOD manifest requires unique Structure_Name values")
    split = frame["Suggested_Split"].astype(str).str.strip().str.lower().replace(
        {"val": "validation", "valid": "validation"}
    )
    invalid_split = ~split.isin({"train", "validation", "test"})
    if invalid_split.any():
        raise ValueError(f"unrecognized Suggested_Split values: {sorted(split[invalid_split].unique())}")
    train = frame.loc[split == "train"].copy()
    if train.empty:
        raise RuntimeError("no training rows found")

    train_metal = {_pair(row, "Metal_Top", "Metal_Bottom") for _, row in train.iterrows()}
    train_term = {_pair(row, "Termination_Top", "Termination_Bottom") for _, row in train.iterrows()}
    train_x = {_norm(value) for value in train.get("X_Element", pd.Series(dtype=str)) if _norm(value)}
    train_elements: set[str] = set()
    for _, row in train.iterrows():
        train_elements.update(_elements(row))
    n_atoms_train = pd.to_numeric(train.get("N_Atoms"), errors="coerce").dropna()
    cell_threshold = (
        float(np.quantile(n_atoms_train, args.cell_size_quantile))
        if len(n_atoms_train)
        else np.inf
    )

    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        elements = _elements(row)
        metal = _pair(row, "Metal_Top", "Metal_Bottom")
        termination = _pair(row, "Termination_Top", "Termination_Bottom")
        x_element = _norm(row.get("X_Element"))
        n_atoms = pd.to_numeric(row.get("N_Atoms"), errors="coerce")
        unseen_metal = bool(metal and metal not in train_metal)
        unseen_termination = bool(termination and termination not in train_term)
        unseen_x = bool(x_element and x_element not in train_x)
        unseen_element = bool(set(elements) - train_elements)
        cell_ood = bool(pd.notna(n_atoms) and float(n_atoms) > cell_threshold)
        chemistry_ood = unseen_metal or unseen_termination or unseen_x or unseen_element
        rows.append(
            {
                "Dataset_Row_Index": position,
                "Structure_Name": _norm(row.get("Structure_Name")),
                "Split_Group": _norm(row.get("Split_Group")),
                "Suggested_Split": split.iloc[position],
                "OOD_Unseen_Metal_Pair": unseen_metal,
                "OOD_Unseen_Termination_Pair": unseen_termination,
                "OOD_Unseen_X_Element": unseen_x,
                "OOD_Unseen_Element": unseen_element,
                "OOD_Cell_Size": cell_ood,
                "OOD_Any_Chemistry": chemistry_ood,
                "OOD_Any": chemistry_ood or cell_ood,
                "Cell_Size_Threshold_NAtoms": cell_threshold,
                "Cell_Size_Quantile": float(args.cell_size_quantile),
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
