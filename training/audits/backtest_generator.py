# ==============================================================================
# 中文概述：用验证/测试集里已有 DFT 弛豫结构的组成回测生成器：流 + 投影 与 模板 + 投影（无流）对比。
# English overview: Back-test the generator on held-out compositions that already have DFT-relaxed
#                   structures: flow + projection versus template + projection (no flow).
#
# 中文输入：surface generator 检查点（含训练模板目录）、NFE 预测器检查点、数据表与结构目录。
# English inputs: Surface-generator checkpoint (with the training template catalog), NFE predictor
#                 checkpoint, dataset table and structure directory.
# 中文输出：每个目标结构在各模式下的几何 RMSD、晶格误差、预测档位一致性，以及汇总 JSON/CSV。
# English outputs: Per-target geometry RMSD, lattice error and label agreement for every mode,
#                  plus a JSON/CSV summary.
#
# 关键约束 / Key invariants:
# - 只使用训练集模板；目标组成来自 validation/test，按 Split_Group 与训练集不重叠。
#   Only training templates are used; target compositions come from validation/test and never overlap training.
# - 没有 CHGNet 预弛豫，比较的是“投影后的原始候选”与 DFT 几何。
#   No CHGNet pre-relaxation; projected raw candidates are compared with DFT geometry.
# - 主要接口 / Main APIs: dft_reference, substitute_template, rmsd_angstrom, main
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from nfe_model.data import LABEL_TO_INDEX, structure_to_graph
from nfe_model.generator_data import center_slab_fractional
from nfe_model.manifold_generation import project_structure_to_template_manifold
from nfe_model.predict import infer_chunk, load_checkpoint_model
from nfe_model.strict_generation import (
    load_generator,
    sample_structures,
    template_entry_from_structure,
)
from nfe_model.utils import save_json

TARGET_TO_SCORE = {"low": 0.25, "medium": 0.58, "high": 0.85}


def stacking_of(name: str) -> tuple[str, str]:
    """Stacking labels from a canonical structure name M1-M2-X-T1-T2-S1-S2."""
    tokens = str(name).split("-")
    return (tokens[5], tokens[6]) if len(tokens) >= 7 else ("", "")


def topology_key(entry: dict[str, Any], match_stacking: bool = True) -> tuple:
    key = (
        len(entry["z"]),
        tuple(int(v) for v in entry["surface_side"]),
        tuple(int(v) for v in entry["group_type"]),
        str(entry["termination_motif"]),
    )
    if match_stacking:
        # The training pools ignore stacking, so a template may put the
        # terminations on the other hollow site (1.8 A away); the manifold
        # projection can never undo that, so an unmatched stacking makes the
        # DFT target unreachable by construction.
        key = key + stacking_of(entry.get("id", ""))
    return key


# 中文：顶层接口 `dft_reference`；把 DFT 结构整理成与模板同序（按高度排序）的参考记录。
# English: Top-level function `dft_reference`; turn the DFT structure into a height-ordered reference entry.
def dft_reference(structure: Structure, identifier: str) -> dict[str, Any]:
    return template_entry_from_structure(structure, identifier)


# 中文：顶层接口 `substitute_template`；把训练模板的物种逐位替换为目标组成（同拓扑、同端基）。
# English: Top-level function `substitute_template`; position-wise species substitution into a training template.
def substitute_template(template: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    result = dict(template)
    result["z"] = [int(v) for v in target["z"]]
    result["id"] = f"{template.get('id', 'template')}->{target['id']}"
    result["formula"] = target["formula"]
    return result


# 中文：顶层接口 `rmsd_angstrom`；同序原子的最小像 RMSD（Å），比较前各自固定平移规范。
# English: Top-level function `rmsd_angstrom`; minimum-image RMSD in angstrom for same-order atoms.
def rmsd_angstrom(candidate: Structure, reference: dict[str, Any]) -> tuple[float, float]:
    entry = template_entry_from_structure(candidate, "candidate")
    if entry["z"] != [int(v) for v in reference["z"]]:
        return float("nan"), float("nan")
    frac_candidate = center_slab_fractional(np.asarray(entry["frac_pos"], dtype=np.float64))
    frac_reference = center_slab_fractional(np.asarray(reference["frac_pos"], dtype=np.float64))
    lattice = np.asarray(reference["lattice"], dtype=np.float64)
    delta = frac_candidate - frac_reference
    delta[:, :2] = (delta[:, :2] + 0.5) % 1.0 - 0.5
    cartesian = delta @ lattice
    distances = np.linalg.norm(cartesian, axis=1)
    return float(np.sqrt(np.mean(distances**2))), float(np.max(distances))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Back-test the generator on held-out DFT compositions.")
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--table", default="data/full/nfe_dataset.csv")
    parser.add_argument("--root", default="data/full")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--templates-per-target", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--any-stacking",
        action="store_true",
        help="draw templates regardless of fcc/hcp stacking (the training-pool behaviour)",
    )
    parser.add_argument("--output", default="models/metadata/generator_backtest.json")
    parser.add_argument("--csv", default="runs/generator_backtest/per_target.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    started = time.time()
    generator, generator_checkpoint = load_generator(args.generator_checkpoint, device)
    predictor = load_checkpoint_model(args.predictor_checkpoint, device)
    predictor_config = predictor[1]["config"]
    radius = float(predictor_config["data"]["radius"])
    max_neighbors = int(predictor_config["data"]["max_neighbors"])
    complete_shells = bool(predictor_config["data"].get("complete_shells", False))

    catalog = generator_checkpoint["surface_template_catalog"]
    match_stacking = not args.any_stacking
    pools: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for entry in catalog:
        pools[topology_key(entry, match_stacking)].append(entry)
    frame = pd.read_csv(args.table)
    held_out = frame[frame["Suggested_Split"] == args.split].reset_index(drop=True)
    if args.limit:
        held_out = held_out.sample(n=min(args.limit, len(held_out)), random_state=args.seed).reset_index(drop=True)
    root = Path(args.root)

    modes = ("flow_projected", "template_projected", "flow_raw")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for target_index, row in held_out.iterrows():
        structure = Structure.from_file(root / "data" / Path(row["File_Path"]).name)
        reference = dft_reference(structure, str(row["Structure_Name"]))
        pool = pools.get(topology_key(reference, match_stacking), [])
        if not pool:
            skipped += 1
            continue
        chosen = random.sample(pool, min(args.templates_per_target, len(pool)))
        templates = [substitute_template(template, reference) for template in chosen]
        label = str(row["NFE_Pseudo_Label"])
        label_index = LABEL_TO_INDEX[label]
        target_score = TARGET_TO_SCORE[label]
        samples = {
            "flow_projected": sample_structures(
                generator, generator_checkpoint, templates, target_label=label_index,
                target_score=target_score, steps=args.steps, guidance_scale=args.guidance_scale, device=device,
            ),
            "template_projected": sample_structures(
                generator, generator_checkpoint, templates, target_label=label_index,
                target_score=target_score, steps=0, guidance_scale=1.0, device=device,
            ),
        }
        samples["flow_raw"] = list(samples["flow_projected"])
        samples["flow_projected"] = [
            project_structure_to_template_manifold(s, t) for s, t in zip(samples["flow_projected"], templates)
        ]
        samples["template_projected"] = [
            project_structure_to_template_manifold(s, t) for s, t in zip(samples["template_projected"], templates)
        ]
        for mode in modes:
            graphs = []
            for candidate in samples[mode]:
                try:
                    graphs.append(
                        structure_to_graph(candidate, radius, max_neighbors, identifier=mode, canonicalize=True, complete_shells=complete_shells)
                    )
                except Exception:
                    graphs.append(None)
            valid = [g for g in graphs if g is not None]
            predictions = infer_chunk(valid, [predictor], device, args.mc_samples) if valid else []
            prediction_iter = iter(predictions)
            for template, candidate, graph in zip(templates, samples[mode], graphs):
                rmsd, max_distance = rmsd_angstrom(candidate, reference)
                lattice_error = abs(float(candidate.lattice.a) - float(structure.lattice.a)) / float(structure.lattice.a)
                prediction = next(prediction_iter) if graph is not None else None
                rows.append(
                    {
                        "target": row["Structure_Name"],
                        "target_label": label,
                        "target_score": float(row["NFE_Pseudo_Score"]),
                        "template": template["id"],
                        "mode": mode,
                        "rmsd_A": rmsd,
                        "max_displacement_A": max_distance,
                        "lattice_a_relative_error": lattice_error,
                        "predicted_label": prediction["Predicted_NFE_Label"] if prediction else "",
                        "probability_target": (
                            float(prediction[f"Probability_{label.capitalize()}"]) if prediction else float("nan")
                        ),
                        "label_agreement": bool(prediction and prediction["Predicted_NFE_Label"] == label),
                        "ood_risk": prediction["OOD_Risk"] if prediction else "",
                    }
                )
        if (target_index + 1) % 25 == 0:
            print(f"processed {target_index + 1}/{len(held_out)} targets", flush=True)

    table = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.csv, index=False)
    summary: dict[str, Any] = {}
    for mode in modes:
        subset = table[table["mode"] == mode]
        finite = subset[np.isfinite(subset["rmsd_A"])]
        summary[mode] = {
            "candidates": int(len(subset)),
            "rmsd_A_mean": float(finite["rmsd_A"].mean()) if len(finite) else None,
            "rmsd_A_median": float(finite["rmsd_A"].median()) if len(finite) else None,
            "max_displacement_A_median": float(finite["max_displacement_A"].median()) if len(finite) else None,
            "lattice_a_relative_error_median": float(subset["lattice_a_relative_error"].median()),
            "label_agreement": float(subset["label_agreement"].mean()),
            "probability_target_mean": float(subset["probability_target"].mean()),
            "label_agreement_by_target_label": {
                label: float(subset[subset["target_label"] == label]["label_agreement"].mean())
                for label in ("low", "medium", "high")
                if int((subset["target_label"] == label).sum())
            },
        }
    # Baseline reference: how far is a raw training template (different metals) from the DFT target?
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": args.split,
        "targets": int(len(held_out)) - skipped,
        "skipped_no_template": skipped,
        "templates_per_target": args.templates_per_target,
        "match_stacking": match_stacking,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "relaxer": "none",
        "generator_checkpoint": str(args.generator_checkpoint),
        "predictor_checkpoint": str(args.predictor_checkpoint),
        "elapsed_seconds": round(time.time() - started, 1),
        "summary": summary,
        "note": (
            "flow_projected = conditional flow + manifold projection; template_projected = noised training "
            "template + manifold projection (no flow); flow_raw = flow without projection. RMSD is measured "
            "against the DFT-relaxed structure of the same composition; no CHGNet relaxation was applied."
        ),
    }
    save_json(args.output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {args.output} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
