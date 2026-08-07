from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


def _norm(value) -> str:
    return str(value).strip() if pd.notna(value) else ""


def _pair(row: pd.Series, a: str, b: str) -> str:
    return "|".join(sorted(x for x in (_norm(row.get(a)), _norm(row.get(b))) if x))


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
    p = argparse.ArgumentParser(description="Build chemistry and cell-size OOD strata from the fixed split.")
    p.add_argument("--dataset", default="data/full/nfe_dataset.csv")
    p.add_argument("--output", default="training/evaluation/ood_manifest.csv")
    p.add_argument("--cell-size-quantile", type=float, default=0.95)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.dataset)
    split = frame["Suggested_Split"].astype(str).str.lower().replace({"val": "validation", "valid": "validation"})
    train = frame[split == "train"].copy()
    if train.empty:
        raise RuntimeError("no training rows found")

    train_metal = {_pair(row, "Metal_Top", "Metal_Bottom") for _, row in train.iterrows()}
    train_term = {_pair(row, "Termination_Top", "Termination_Bottom") for _, row in train.iterrows()}
    train_x = {_norm(x) for x in train.get("X_Element", pd.Series(dtype=str)) if _norm(x)}
    train_elements: set[str] = set()
    for _, row in train.iterrows():
        train_elements.update(_elements(row))
    n_atoms_train = pd.to_numeric(train.get("N_Atoms"), errors="coerce").dropna()
    cell_threshold = float(np.quantile(n_atoms_train, args.cell_size_quantile)) if len(n_atoms_train) else np.inf

    rows = []
    for index, row in frame.iterrows():
        elements = _elements(row)
        metal = _pair(row, "Metal_Top", "Metal_Bottom")
        term = _pair(row, "Termination_Top", "Termination_Bottom")
        x = _norm(row.get("X_Element"))
        n_atoms = pd.to_numeric(row.get("N_Atoms"), errors="coerce")
        unseen_metal = bool(metal and metal not in train_metal)
        unseen_term = bool(term and term not in train_term)
        unseen_x = bool(x and x not in train_x)
        unseen_element = bool(set(elements) - train_elements)
        cell_ood = bool(pd.notna(n_atoms) and float(n_atoms) > cell_threshold)
        rows.append(
            {
                "Structure_Name": _norm(row.get("Structure_Name")),
                "Split_Group": _norm(row.get("Split_Group")),
                "Suggested_Split": split.iloc[index],
                "OOD_Unseen_Metal_Pair": unseen_metal,
                "OOD_Unseen_Termination_Pair": unseen_term,
                "OOD_Unseen_X_Element": unseen_x,
                "OOD_Unseen_Element": unseen_element,
                "OOD_Cell_Size": cell_ood,
                "OOD_Any_Chemistry": unseen_metal or unseen_term or unseen_x or unseen_element,
                "OOD_Any": unseen_metal or unseen_term or unseen_x or unseen_element or cell_ood,
                "Cell_Size_Threshold_NAtoms": cell_threshold,
            }
        )
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}; train N_Atoms q={args.cell_size_quantile}: {cell_threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
