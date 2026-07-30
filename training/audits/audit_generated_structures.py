# ==============================================================================
# 中文概述：提供 NFE MXene 项目中的单一、可复用源码职责。
# English overview: Provide one reusable source-code responsibility in the NFE MXene project.
#
# 中文输入：请结合类型标注、命令行帮助和调用方查看输入。
# English inputs: Read type hints, CLI help, and callers for the expected inputs.
# 中文输出：返回值或生成文件由公开接口和命令行参数定义。
# English outputs: Return values or generated files are defined by public APIs and CLI arguments.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: parse_args, main
#
# Author: Ruck
# Generated: 2026-07-30 01:17:05 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from nfe_model.data import build_periodic_graph, torch_load_compat
from nfe_model.strict_generation import (
    composition_key,
    prediction_matches_target,
    reference_structures,
    safe_structure_match,
    validate_structure,
)
from nfe_model.predict import infer_chunk, load_checkpoint_model
from nfe_model.surface_geometry import validate_surface_topology


# 中文：顶层接口 `parse_args`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `parse_args`; review type hints and callers before extending it.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently re-audit strict manifold-generated CIF files."
    )
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", action="append", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--min-target-probability", type=float, default=0.50)
    parser.add_argument("--max-score-std", type=float, default=0.08)
    parser.add_argument("--max-force", type=float, default=0.05)
    return parser.parse_args(argv)


# 中文：顶层接口 `main`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `main`; review type hints and callers before extending it.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit directory: {output}")
    output.mkdir(parents=True)

    source_rows = []
    for directory in args.input:
        summary = Path(directory).resolve() / "generation_summary.csv"
        frame = pd.read_csv(summary)
        frame["_Source_Summary"] = str(summary)
        source_rows.append(frame)
    source = pd.concat(source_rows, ignore_index=True)
    if source.empty:
        raise ValueError("no generated candidates found")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    generator_checkpoint = torch_load_compat(
        args.generator_checkpoint, map_location="cpu"
    )
    predictors = [
        load_checkpoint_model(path, device)
        for path in args.predictor_checkpoint
    ]
    predictor_config = predictors[0][1]["config"]
    radius = float(predictor_config["data"]["radius"])
    max_neighbors = int(predictor_config["data"]["max_neighbors"])
    generation_config = generator_checkpoint["config"]["generation"]
    profile_path = Path(generation_config["surface_profile"])
    if not profile_path.is_absolute():
        profile_path = (
            Path(args.generator_checkpoint).resolve().parent.parent / profile_path
        )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    minimum_factor = float(generation_config["minimum_distance_factor"])
    minimum_vacuum = float(generation_config["minimum_vacuum_A"])
    maximum_slab_thickness = float(
        generation_config["maximum_slab_thickness_A"]
    )
    maximum_nearest_ratio = float(
        generation_config["maximum_nearest_radius_ratio"]
    )
    minimum_layers = int(generation_config["minimum_atomic_layers"])
    training_keys = set(generator_checkpoint.get("novelty_reference", {}))
    matcher = StructureMatcher(
        ltol=0.15,
        stol=0.20,
        angle_tol=4.0,
        primitive_cell=False,
        scale=True,
        attempt_supercell=False,
    )

    audited_structures: list[Structure] = []
    rows = []
    for candidate_index, source_row in source.iterrows():
        cif_path = Path(str(source_row["CIF_Path"])).resolve()
        structure = Structure.from_file(cif_path)
        geometry_valid, geometry = validate_structure(
            structure,
            minimum_factor,
            minimum_vacuum,
            maximum_slab_thickness,
            maximum_nearest_ratio,
            minimum_layers,
        )
        surface_valid, surface = validate_surface_topology(structure, profile)
        graph = build_periodic_graph(
            structure,
            radius,
            max_neighbors,
            identifier=f"audit_{candidate_index:04d}",
        )
        torch.manual_seed(41000 + int(candidate_index))
        predictions = infer_chunk(
            [graph],
            predictors,
            device,
            int(args.mc_samples),
        )
        prediction = predictions[0]
        target = str(source_row["Target_Label"])
        target_matches, target_probability = prediction_matches_target(
            prediction,
            target,
            float(args.min_target_probability),
        )
        z_values = [int(site.specie.Z) for site in structure]
        unseen_composition = composition_key(z_values) not in training_keys
        matches_training = any(
            safe_structure_match(matcher, structure, reference)
            for reference in reference_structures(
                generator_checkpoint, structure
            )
        )
        duplicate_generated = any(
            safe_structure_match(matcher, structure, prior)
            for prior in audited_structures
        )
        audited_structures.append(structure)
        force = float(source_row["CHGNet_Max_Force_eV_A"])
        force_valid = bool(math.isfinite(force) and force < args.max_force)
        centered = bool(
            abs(float(geometry["Slab_Center_Fractional_Z"]) - 0.5) < 1e-6
            and float(geometry["Slab_Center_Offset_A"]) < 1e-4
        )
        stable = bool(
            target_matches
            and float(prediction["NFE_Score_Std"]) <= args.max_score_std
            and str(prediction["OOD_Risk"]) == "low"
        )
        reasons = []
        if not geometry_valid:
            reasons.append(f"geometry:{geometry['Geometry_Reasons']}")
        if not surface_valid:
            reasons.append(
                f"surface:{surface['Surface_Topology_Reasons']}"
            )
        if not centered:
            reasons.append("not_centered")
        if not force_valid:
            reasons.append("chgnet_force")
        if not unseen_composition:
            reasons.append("composition_seen_in_training")
        if matches_training:
            reasons.append("matches_training_structure")
        if duplicate_generated:
            reasons.append("duplicate_generated_structure")
        if not stable:
            reasons.append("nfe_or_ood_not_stable")
        passed = not reasons
        rows.append(
            {
                "CIF_Path": str(cif_path),
                "Formula": str(source_row["Formula"]),
                "Target_Label": target,
                "Audit_Passed": passed,
                "Audit_Reasons": "|".join(reasons),
                "CIF_Reparsed": True,
                "Geometry_Valid": geometry_valid,
                "Surface_Topology_Valid": surface_valid,
                "Centered": centered,
                "Atomic_Layer_Count": geometry["Atomic_Layer_Count"],
                "Surface_Layer_Count": surface["Surface_Layer_Count"],
                "Surface_Group_Count": surface["Surface_Group_Count"],
                "Termination_Motif": surface["Termination_Motif"],
                "OH_Bond_Lengths_A": surface["OH_Bond_Lengths_A"],
                "Surface_Anchor_Details": surface[
                    "Surface_Anchor_Details"
                ],
                "Slab_Center_Fractional_Z": geometry[
                    "Slab_Center_Fractional_Z"
                ],
                "Slab_Center_Offset_A": geometry[
                    "Slab_Center_Offset_A"
                ],
                "CHGNet_Max_Force_eV_A": force,
                "CHGNet_Force_Valid": force_valid,
                "Unseen_Composition": unseen_composition,
                "Matches_Training_Structure": matches_training,
                "Duplicate_Generated_Structure": duplicate_generated,
                "Independent_Target_Matched": target_matches,
                "Independent_Target_Probability": target_probability,
                "Independent_Predicted_Label": prediction[
                    "Predicted_NFE_Label"
                ],
                "Independent_Probability_Low": prediction["Probability_Low"],
                "Independent_Probability_Medium": prediction[
                    "Probability_Medium"
                ],
                "Independent_Probability_High": prediction[
                    "Probability_High"
                ],
                "Independent_NFE_Score": prediction[
                    "Predicted_NFE_Score"
                ],
                "Independent_NFE_Score_Std": prediction["NFE_Score_Std"],
                "Independent_OOD_Risk": prediction["OOD_Risk"],
                "Independent_OOD_Embedding_Z_RMS": prediction[
                    "OOD_Embedding_Z_RMS"
                ],
                "Independent_OOD_Nearest_Embedding_Distance": prediction[
                    "OOD_Nearest_Embedding_Distance"
                ],
            }
        )

    report = pd.DataFrame(rows)
    report_path = output / "generated_structures_audit.csv"
    report.to_csv(report_path, index=False)
    by_target = {
        target: {
            "count": int(len(group)),
            "passed": int(group["Audit_Passed"].sum()),
            "minimum_independent_target_probability": float(
                group["Independent_Target_Probability"].min()
            ),
            "maximum_independent_score_std": float(
                group["Independent_NFE_Score_Std"].max()
            ),
            "maximum_chgnet_force_eV_A": float(
                group["CHGNet_Max_Force_eV_A"].max()
            ),
        }
        for target, group in report.groupby("Target_Label")
    }
    summary = {
        "audited": int(len(report)),
        "passed": int(report["Audit_Passed"].sum()),
        "all_passed": bool(report["Audit_Passed"].all()),
        "mc_samples": int(args.mc_samples),
        "surface_profile": str(profile_path),
        "report": str(report_path),
        "by_target": by_target,
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        report[
            [
                "Formula",
                "Target_Label",
                "Audit_Passed",
                "Independent_Target_Probability",
                "Independent_NFE_Score",
                "Independent_NFE_Score_Std",
                "Independent_OOD_Risk",
                "CHGNet_Max_Force_eV_A",
                "Termination_Motif",
            ]
        ].to_string(index=False)
    )
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
