from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nfe_model.formal_config import validate_formal_config
from nfe_model.utils import load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit generator candidate geometry/input contract before formal predictor screening."
    )
    parser.add_argument("--config", default="training/configs/nfe_predictor_v2_4.yaml")
    parser.add_argument(
        "--output", default="training/evaluation/results/generator_predictor_contract.json"
    )
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


def main() -> int:
    args = parse_args()
    config = _resolve_config(Path(args.config).resolve())
    predictor_cutoff = float(config["data"]["radius"])
    model_cutoff = float(config["model"]["cutoff"])
    generator = config.get("generator_model")
    generation = config.get("generation")
    if not isinstance(generator, dict) or not isinstance(generation, dict):
        raise ValueError("config must contain generator_model and generation sections")
    generator_cutoff = float(generator.get("cutoff", -1))
    minimum_vacuum = float(generation.get("minimum_vacuum_A", -1))
    if generator_cutoff <= 0 or minimum_vacuum <= 0:
        raise ValueError("generator cutoff and generation.minimum_vacuum_A must be positive")
    required = max(predictor_cutoff, model_cutoff, generator_cutoff)
    passed = minimum_vacuum > required
    result = {
        "predictor_graph_cutoff_A": predictor_cutoff,
        "predictor_model_cutoff_A": model_cutoff,
        "generator_graph_cutoff_A": generator_cutoff,
        "generation_minimum_vacuum_A": minimum_vacuum,
        "required_strictly_greater_than_A": required,
        "pass": passed,
        "interpretation": (
            "formal generated slabs must have an atom-free normal vacuum gap larger than every periodic graph cutoff "
            "they will encounter before screening"
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
