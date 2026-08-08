from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from nfe_model.data_v2 import load_or_build_cache, split_indices
from nfe_model.formal_config import validate_formal_config
from nfe_model.utils import load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Require reverse-pair closure of every retained periodic edge in the formal common graph."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor.yaml")
    parser.add_argument("--output", default="training/evaluation/results/neighbor_pair_symmetry.json")
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


def _edge_key(source: int, destination: int, shift: torch.Tensor) -> tuple[int, int, int, int, int]:
    return (
        int(source),
        int(destination),
        int(round(float(shift[0]))),
        int(round(float(shift[1]))),
        int(round(float(shift[2]))),
    )


def main() -> int:
    args = parse_args()
    config = _resolve_config(Path(args.config).resolve())
    data = config["data"]
    cap = int(data["max_neighbors"])
    cache = load_or_build_cache(
        data["table"],
        data["root"],
        data["cache"],
        radius=float(data["radius"]),
        max_neighbors=cap,
        rebuild=False,
    )
    records = cache["records"]
    split_indices(records)

    asymmetric = []
    above_nominal_cap = []
    maximum_degree = 0
    for record in records:
        source, destination = record["edge_index"].detach().cpu().long()
        shift = record["edge_shift"].detach().cpu()
        n_atoms = int(record["z"].numel())
        degree = torch.bincount(destination, minlength=n_atoms)
        maximum_degree = max(maximum_degree, int(degree.max()))
        if torch.any(degree > cap):
            above_nominal_cap.append(
                {
                    "id": str(record["id"]),
                    "max_degree": int(degree.max()),
                    "atoms_above_cap": int(torch.sum(degree > cap)),
                }
            )
        edge_set = {
            _edge_key(source[index], destination[index], shift[index])
            for index in range(source.numel())
        }
        missing = []
        for key in edge_set:
            s, d, sx, sy, sz = key
            reverse = (d, s, -sx, -sy, -sz)
            if reverse not in edge_set:
                missing.append(key)
                if len(missing) >= 5:
                    break
        if missing:
            asymmetric.append(
                {
                    "id": str(record["id"]),
                    "missing_reverse_examples": missing,
                    "edge_count": int(source.numel()),
                }
            )

    result = {
        "pass": not asymmetric,
        "records": len(records),
        "configured_soft_cap": cap,
        "maximum_realized_degree_after_shell_completion": maximum_degree,
        "records_with_degree_above_nominal_cap": len(above_nominal_cap),
        "records_with_missing_reverse_edges": len(asymmetric),
        "above_cap_examples": above_nominal_cap[:20],
        "asymmetry_examples": asymmetric[:20],
        "interpretation": (
            "shell completion may legitimately exceed the nominal cap; however the formal common graph must be "
            "pair symmetric before using it as a shared bond graph for line-graph/angular baselines"
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if asymmetric:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
