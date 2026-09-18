# ==============================================================================
# 中文概述：确定性表示探针。把同一 slab 改写为不同真空厚度、γ = 60° 晶格设定、2×2×1 超胞和平移后分别建图，
#           比较预测器给出的 P(high)；分别在开启与关闭输入规范化时统计极差与改档比例。
# English overview: Deterministic representation probe. The same slab is re-embedded with a different vacuum,
#           a gamma = 60 deg cell setting, a 2x2x1 supercell and a translation; P(high) spread and label flips are
#           reported with and without input canonicalization.
#
# 中文输入：数据表、结构目录、若干 名称=检查点路径、每档抽样数与随机种子。
# English inputs: Dataset table, structure directory, one or more name=checkpoint pairs, samples per label and seed.
# 中文输出：每个检查点、每种模式的 P(high) 极差统计与改档数（JSON）。
# English outputs: Per-checkpoint, per-mode P(high) spread statistics and label-flip counts (JSON).
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nfe_model.data import LABEL_TO_INDEX, move_batch, structure_to_graph  # noqa: E402
from nfe_model.predict import load_checkpoint_model, prediction_batch  # noqa: E402

VARIANTS = ("as-is", "c=20", "c=40", "gamma=60", "2x2x1", "translated")
HIGH = LABEL_TO_INDEX["high"]


# 中文：流式计算 SHA256。/ English: Compute SHA256 as a stream.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


# 中文：生成同一结构的六种等价表示。/ English: Build six equivalent representations of one structure.
def variants(base: Structure) -> dict[str, Structure]:
    matrix = np.array(base.lattice.matrix)
    cartesian = base.cart_coords
    species = [site.specie for site in base]
    result: dict[str, Structure] = {"as-is": base}
    for c_new in (20.0, 40.0):
        new_matrix = matrix.copy()
        new_matrix[2] = [0.0, 0.0, c_new]
        shifted = cartesian.copy()
        shifted[:, 2] += (c_new - base.lattice.c) / 2
        result[f"c={c_new:.0f}"] = Structure(Lattice(new_matrix), species, shifted, coords_are_cartesian=True)
    result["gamma=60"] = Structure(
        Lattice(np.array([matrix[0], matrix[0] + matrix[1], matrix[2]])),
        species,
        cartesian,
        coords_are_cartesian=True,
        to_unit_cell=True,
    )
    supercell = base.copy()
    supercell.make_supercell([2, 2, 1])
    result["2x2x1"] = supercell
    translated = base.copy()
    translated.translate_sites(list(range(len(translated))), [0.3, 0.1, 0.2], frac_coords=True)
    result["translated"] = translated
    return result


# 中文：确定性前向求 P(high)。/ English: Deterministic forward pass for P(high).
def probability_high(model: Any, checkpoint: dict[str, Any], graphs: list[dict[str, Any]], device: torch.device) -> np.ndarray:
    normalizers = {key: value.cpu() for key, value in checkpoint["normalizers"].items()}
    temperature = float(checkpoint.get("classification_temperature", 1.0))
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(graphs), 64):
            batch = move_batch(prediction_batch(graphs[start : start + 64], normalizers), device)
            logits = model(batch)["class_logits"].float() / temperature
            probabilities.append(torch.softmax(logits, dim=-1)[:, HIGH].cpu().numpy())
    return np.concatenate(probabilities)


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/full_v1_1/nfe_dataset.csv")
    parser.add_argument("--root", default="data/full_v1_1", help="directory whose data/ holds the structure copies")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=3, help="per-structure detail kept for the first N structures")
    parser.add_argument("--output", default="models/metadata/representation_probe_v1_1.json")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    table = (ROOT / args.table) if not Path(args.table).is_absolute() else Path(args.table)
    structure_dir = ((ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)) / "data"
    frame = pd.read_csv(table)
    frame = frame[frame["Suggested_Split"] == args.split].reset_index(drop=True)
    picked = pd.concat(
        [
            frame[frame["NFE_Pseudo_Label"] == label].sample(
                min(args.per_label, int((frame["NFE_Pseudo_Label"] == label).sum())), random_state=args.seed
            )
            for label in ("low", "medium", "high")
        ]
    ).reset_index(drop=True)
    started = time.time()
    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/representation_probe.py",
        "table": args.table,
        "table_sha256": sha256(table),
        "split": args.split,
        "structures": int(len(picked)),
        "per_label": args.per_label,
        "seed": args.seed,
        "variants": list(VARIANTS),
        "mc_dropout": False,
        "note": "deterministic forward pass; the run-time predictor averages MC-dropout samples, whose spread is reported as uncertainty, not as input sensitivity",
        "models": {},
    }
    bases = {
        row.Structure_Name: (row.NFE_Pseudo_Label, Structure.from_file(str(structure_dir / Path(row.File_Path).name)))
        for row in picked.itertuples()
    }
    for name, path in (value.split("=", 1) for value in args.checkpoint):
        checkpoint_path = Path(path) if Path(path).is_absolute() else ROOT / path
        model, checkpoint = load_checkpoint_model(str(checkpoint_path), device)
        data_config = checkpoint["config"]["data"]
        complete_shells = bool(data_config.get("complete_shells", False))
        threshold = float(checkpoint.get("high_probability_threshold", 0.5))
        entry: dict[str, Any] = {
            "checkpoint": Path(path).as_posix(),
            "checkpoint_sha256": sha256(checkpoint_path),
            "global_features": checkpoint["model_config"].get("global_features"),
            "complete_shells": complete_shells,
            "high_threshold": threshold,
            "modes": {},
        }
        for canonicalize in (True, False):
            keys, graphs = [], []
            for structure_name, (label, base) in bases.items():
                for variant_name, structure in variants(base).items():
                    keys.append((structure_name, label, variant_name))
                    graphs.append(
                        structure_to_graph(
                            structure,
                            float(data_config["radius"]),
                            int(data_config["max_neighbors"]),
                            identifier=variant_name,
                            canonicalize=canonicalize,
                            complete_shells=complete_shells,
                        )
                    )
            values = probability_high(model, checkpoint, graphs, device)
            per_structure: dict[str, dict[str, Any]] = {}
            for (structure_name, label, variant_name), value in zip(keys, values):
                per_structure.setdefault(structure_name, {"true": label, "p_high": {}})["p_high"][variant_name] = round(float(value), 4)
            spreads = np.array([max(d["p_high"].values()) - min(d["p_high"].values()) for d in per_structure.values()])
            flips = np.array(
                [(max(d["p_high"].values()) >= threshold) != (min(d["p_high"].values()) >= threshold) for d in per_structure.values()]
            )
            mode = {
                "mean_spread": round(float(spreads.mean()), 4),
                "median_spread": round(float(np.median(spreads)), 4),
                "p95_spread": round(float(np.percentile(spreads, 95)), 4),
                "max_spread": round(float(spreads.max()), 4),
                "fraction_label_flips": round(float(flips.mean()), 4),
                "n_flips": int(flips.sum()),
                "examples": {key: per_structure[key] for key in list(per_structure)[: args.examples]},
            }
            entry["modes"]["canonicalized" if canonicalize else "raw"] = mode
            print(name, "canonicalized" if canonicalize else "raw", json.dumps({k: v for k, v in mode.items() if k != "examples"}), flush=True)
        report["models"][name] = entry
    report["elapsed_seconds"] = round(time.time() - started, 1)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
