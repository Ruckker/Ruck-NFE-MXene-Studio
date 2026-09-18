# ==============================================================================
# 中文概述：枚举尚未计算的 M1-M2-X-T1-T2-堆垛 组合，用最近的已弛豫模板替换物种，再用 NFE 预测器排序。
# English overview: Enumerate every uncomputed M1-M2-X-T1-T2-stacking configuration, build it from the
#                   nearest relaxed template by species substitution, and rank it with the NFE predictor.
#
# 中文输入：nfe_dataset.csv（已计算配置）、dirty_manifest.csv、结构目录、预测器检查点。
# English inputs: nfe_dataset.csv (computed configurations), dirty_manifest.csv, structure directory,
#                 predictor checkpoint.
# 中文输出：按 P(high) 排序的候选表（CSV）与汇总 JSON；候选几何来自模板，未经弛豫。
# English outputs: A CSV ranked by P(high) plus a JSON summary; candidate geometry is the template's,
#                  not relaxed.
#
# 关键约束 / Key invariants:
# - 当前设计空间只有 11 金属 × 2 核心 × 7 端基 × 4 堆垛，可以穷举；这是 1.0 的主线筛选路径，
#   生成器保留给多层/混合端基/缺陷等无法穷举的空间。
#   The current space (11 metals x 2 cores x 7 terminations x 4 stackings) is enumerable; this is the
#   primary screening path, the generator is kept for spaces that cannot be enumerated.
# - 主要接口 / Main APIs: enumerate_configurations, nearest_template, substitute, main
#
# Author: Ruck
# Generated: 2026-09-15
# ==============================================================================

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element, Structure

from nfe_model.data import structure_to_graph
from nfe_model.manifold_generation import _substitution_weight
from nfe_model.predict import infer_chunk, load_checkpoint_model
from nfe_model.utils import save_json

METALS = ("Sc", "Ti", "V", "Cr", "Y", "Zr", "Nb", "Mo", "Hf", "Ta", "W")
CORES = ("C", "N")
TERMINATIONS = ("F", "Cl", "Br", "I", "S", "Se", "OH")
STACKINGS = ("T_fcc", "T_hcp")


# 中文：顶层接口 `enumerate_configurations`；47,432 个有序配置名。
# English: Top-level function `enumerate_configurations`; the 47,432 ordered configuration names.
def enumerate_configurations() -> list[tuple[str, ...]]:
    return list(itertools.product(METALS, METALS, CORES, TERMINATIONS, TERMINATIONS, STACKINGS, STACKINGS))


def configuration_name(config: Sequence[str]) -> str:
    return "-".join(config)


# 中文：顶层接口 `nearest_template`；同 X/端基/堆垛的已计算结构中金属替换代价最小者。
# English: Top-level function `nearest_template`; the computed structure with the same X/terminations/stacking
#          and the cheapest metal substitution.
def nearest_template(
    config: Sequence[str], groups: dict[tuple, list[dict[str, Any]]]
) -> tuple[dict[str, Any] | None, float, str]:
    m1, m2, x, t1, t2, s1, s2 = config
    for level, key in (
        ("same_terminations_stacking", (x, t1, t2, s1, s2)),
        ("same_terminations", (x, t1, t2)),
        ("same_termination_pair_any_core", (t1, t2)),
    ):
        candidates = groups.get(key, [])
        if not candidates:
            continue
        best, best_weight = None, -1.0
        for candidate in candidates:
            weight = _substitution_weight(Element(candidate["Metal_Top"]).Z, Element(m1).Z) * _substitution_weight(
                Element(candidate["Metal_Bottom"]).Z, Element(m2).Z
            )
            if weight > best_weight:
                best, best_weight = candidate, weight
        return best, best_weight, level
    return None, 0.0, "none"


# 中文：顶层接口 `substitute`；上金属/下金属/核心按物理高度替换，端基与氢保持模板。
# English: Top-level function `substitute`; replace the upper metal, lower metal and core by physical height.
def substitute(structure: Structure, config: Sequence[str]) -> Structure:
    m1, m2, x, *_ = config
    metal_symbols = set(METALS)
    z = np.asarray([site.frac_coords[2] for site in structure])
    sorted_z = np.sort(z % 1.0)
    gaps = np.diff(np.r_[sorted_z, sorted_z[0] + 1.0])
    start = sorted_z[(int(np.argmax(gaps)) + 1) % len(sorted_z)]
    height = (z - start) % 1.0
    species = [site.specie.symbol for site in structure]
    metals = [(height[i], i) for i, symbol in enumerate(species) if symbol in metal_symbols]
    cores = [i for i, symbol in enumerate(species) if symbol in CORES]
    if len(metals) != 2 or len(cores) != 1:
        raise ValueError("template is not an M2X slab")
    metals.sort()
    new_species = list(species)
    new_species[metals[1][1]] = m1  # upper metal = Metal_Top
    new_species[metals[0][1]] = m2  # lower metal = Metal_Bottom
    new_species[cores[0]] = x
    return Structure(structure.lattice, new_species, structure.frac_coords, coords_are_cartesian=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate and rank uncomputed MXene configurations.")
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--table", default="data/full/nfe_dataset.csv")
    parser.add_argument("--dirty-table", default="data/full/dirty_manifest.csv")
    parser.add_argument("--root", default="data/full")
    parser.add_argument("--output-dir", default="runs/enumeration")
    parser.add_argument("--summary", default="models/metadata/enumeration_summary.json")
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="score only the first N missing configurations (smoke runs)")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    started = time.time()
    frame = pd.read_csv(args.table)
    computed = set(frame["Structure_Name"])
    dirty = set()
    if Path(args.dirty_table).is_file():
        dirty = set(pd.read_csv(args.dirty_table)["Structure_Name"])
    training_compositions = {
        (tuple(sorted((r.Metal_Top, r.Metal_Bottom))), r.X_Element, tuple(sorted((r.Termination_Top, r.Termination_Bottom))))
        for r in frame.itertuples()
        if r.Suggested_Split == "train"
    }
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in frame.to_dict(orient="records"):
        groups[(record["X_Element"], record["Termination_Top"], record["Termination_Bottom"], record["Stacking_Top"], record["Stacking_Bottom"])].append(record)
        groups[(record["X_Element"], record["Termination_Top"], record["Termination_Bottom"])].append(record)
        groups[(record["Termination_Top"], record["Termination_Bottom"])].append(record)

    all_configs = enumerate_configurations()
    missing = [c for c in all_configs if configuration_name(c) not in computed and configuration_name(c) not in dirty]
    if args.limit:
        missing = missing[: args.limit]
    print(
        json.dumps(
            {"enumerable": len(all_configs), "computed_clean": len(computed), "dirty": len(dirty), "missing": len(missing)},
            ensure_ascii=False,
        ),
        flush=True,
    )

    model = load_checkpoint_model(args.predictor_checkpoint, device)
    config = model[1]["config"]
    radius = float(config["data"]["radius"])
    max_neighbors = int(config["data"]["max_neighbors"])
    complete_shells = bool(config["data"].get("complete_shells", False))
    root = Path(args.root)
    structure_cache: dict[str, Structure] = {}
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    level_counts: Counter[str] = Counter()

    def flush() -> None:
        if not graphs:
            return
        for start in range(0, len(graphs), args.batch_size):
            predictions = infer_chunk(graphs[start : start + args.batch_size], [model], device, args.mc_samples)
            for offset, prediction in enumerate(predictions):
                rows[len(rows) - len(graphs) + start + offset].update(
                    {
                        "Predicted_NFE_Label": prediction["Predicted_NFE_Label"],
                        "Probability_Low": prediction["Probability_Low"],
                        "Probability_Medium": prediction["Probability_Medium"],
                        "Probability_High": prediction["Probability_High"],
                        "Predicted_High_vs_Rest": prediction["Predicted_High_vs_Rest"],
                        "Predicted_NFE_Score": prediction["Predicted_NFE_Score"],
                        "NFE_Score_Std": prediction["NFE_Score_Std"],
                        "OOD_Risk": prediction["OOD_Risk"],
                        "Predicted_Work_Function_Mean_eV": prediction["Predicted_Work_Function_Mean_eV"],
                    }
                )
        graphs.clear()

    for index, config_tuple in enumerate(missing, start=1):
        template, weight, level = nearest_template(config_tuple, groups)
        level_counts[level] += 1
        if template is None:
            continue
        name = configuration_name(config_tuple)
        path = root / "data" / Path(template["File_Path"]).name
        if template["Structure_Name"] not in structure_cache:
            structure_cache[template["Structure_Name"]] = Structure.from_file(path)
        try:
            candidate = substitute(structure_cache[template["Structure_Name"]], config_tuple)
            graph = structure_to_graph(candidate, radius, max_neighbors, identifier=name, canonicalize=True, complete_shells=complete_shells)
        except Exception as exc:
            level_counts[f"build_error:{type(exc).__name__}"] += 1
            continue
        m1, m2, x, t1, t2, s1, s2 = config_tuple
        rows.append(
            {
                "Structure_Name": name,
                "Metal_Top": m1,
                "Metal_Bottom": m2,
                "X_Element": x,
                "Termination_Top": t1,
                "Termination_Bottom": t2,
                "Stacking_Top": s1,
                "Stacking_Bottom": s2,
                "Template": template["Structure_Name"],
                "Template_Match_Level": level,
                "Substitution_Weight": weight,
                "Composition_In_Training": (tuple(sorted((m1, m2))), x, tuple(sorted((t1, t2)))) in training_compositions,
                "Lattice_a_A": float(candidate.lattice.a),
            }
        )
        graphs.append(graph)
        if len(graphs) >= args.batch_size * 4:
            flush()
        if index % 2000 == 0:
            print(f"built {index}/{len(missing)}", flush=True)
    flush()

    table = pd.DataFrame(rows)
    table = table.sort_values(["Probability_High", "Predicted_NFE_Score"], ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "enumeration_ranked.csv"
    table.to_csv(csv_path, index=False)
    low_ood = table[table["OOD_Risk"] == "low"]
    high_pairs = Counter()
    for record in low_ood[low_ood["Predicted_High_vs_Rest"]].itertuples():
        high_pairs["|".join(sorted((record.Termination_Top, record.Termination_Bottom)))] += 1
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "predictor_checkpoint": str(args.predictor_checkpoint),
        "enumerable": len(all_configs),
        "computed_clean": len(computed),
        "dirty": len(dirty),
        "missing": len(missing),
        "scored": int(len(table)),
        "template_match_levels": dict(level_counts),
        "predicted_label_counts": table["Predicted_NFE_Label"].value_counts().to_dict(),
        "ood_risk_counts": table["OOD_Risk"].value_counts().to_dict(),
        "predicted_high_low_ood": int(len(low_ood[low_ood["Predicted_High_vs_Rest"]])),
        "predicted_high_low_ood_by_termination_pair": dict(high_pairs.most_common()),
        "top_50_low_ood_by_probability_high": low_ood.head(50)[
            ["Rank", "Structure_Name", "Probability_High", "Predicted_NFE_Score", "NFE_Score_Std", "Template", "Composition_In_Training"]
        ].to_dict(orient="records"),
        "ranked_csv": str(csv_path),
        "elapsed_seconds": round(time.time() - started, 1),
        "note": (
            "Candidate geometry is the substituted nearest relaxed template (no relaxation); predictions use "
            "canonicalized inputs. Treat this as a DFT priority queue, not as a materials claim."
        ),
    }
    save_json(args.summary, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "top_50_low_ood_by_probability_high"}, ensure_ascii=False, indent=2))
    print(f"Saved {csv_path} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
