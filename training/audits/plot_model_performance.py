# ==============================================================================
# 中文概述：把 models/metadata 中的评测结果画成两张图：预测器诊断（分档 F1、混淆矩阵、表示不变性）与
#           生成器 1.3.0 对 1.0 的对比（监督指标、回测几何误差、严格生成成功率）。逐指标的模型对比另见
#           training/audits/plot_benchmark_panels.py。
# English overview: Turn the evaluations in models/metadata into two figures: predictor diagnostics (per-class F1,
#           confusion matrix, representation invariance) and generator 1.3.0 versus 1.0 (supervised metrics,
#           backtest geometry error, strict generation success). The metric-by-metric model comparison lives in
#           training/audits/plot_benchmark_panels.py.
#
# 中文输入：models/metadata 下的 JSON 评测结果。
# English inputs: The evaluation JSON files under models/metadata.
# 中文输出：docs/images/ 下的 PNG 图，以及每张图的数值清单（JSON）。
# English outputs: PNG figures under docs/images/ plus the plotted numbers as JSON.
#
# 关键约束 / Key invariants:
# - 只画已有评测文件里的数字，不重新计算，也不外推；单 seed 的模型在图中标注。
#   Only numbers already present in the evaluation files are plotted; single-seed models are marked.
# - 条形图从 0 起画，避免视觉放大差异。
#   Bar charts start at zero so differences are not visually exaggerated.
# - 主要接口 / Main APIs: main
#
# Author: Ruck
# Generated: 2026-09-16
# ==============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASELINE_COLOR = "#9db4cc"
STRONG_BASELINE_COLOR = "#5b82a8"
CONTROL_COLOR = "#7e8a97"
OURS_COLOR = "#e0703a"
OLD_COLOR = "#9aa5b1"
GRID = {"color": "#d8dee6", "linewidth": 0.8}


def style() -> None:
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#8a95a1",
        "axes.labelcolor": "#26323d",
        "text.color": "#26323d",
        "xtick.color": "#4a5563",
        "ytick.color": "#4a5563",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "font.size": 9.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def bars(axis: plt.Axes, labels: Sequence[str], values: Sequence[float], errors: Sequence[float], colors: Sequence[str], fmt: str = "{:.3f}", limit: float | None = None) -> None:
    positions = np.arange(len(labels))
    handles = axis.barh(positions, values, xerr=errors, color=colors, height=0.66, error_kw={"ecolor": "#43505d", "elinewidth": 1.0, "capsize": 3})
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, limit or max(values) * 1.26)
    axis.xaxis.grid(True, **GRID)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    for rectangle, value, error in zip(handles, values, errors):
        axis.text(value + (error or 0) + axis.get_xlim()[1] * 0.015, rectangle.get_y() + rectangle.get_height() / 2, fmt.format(value), va="center", fontsize=9)


def grouped(axis: plt.Axes, groups: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]], fmt: str = "{:.3f}", limit: float | None = None) -> None:
    positions = np.arange(len(groups))
    width = 0.8 / len(series)
    for index, (name, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        handles = axis.bar(positions + offset, values, width=width * 0.92, label=name, color=color)
        for rectangle, value in zip(handles, values):
            axis.text(rectangle.get_x() + rectangle.get_width() / 2, value + (limit or 1.0) * 0.012, fmt.format(value), ha="center", fontsize=8.2)
    axis.set_xticks(positions, groups)
    axis.set_ylim(0, limit or 1.0)
    axis.yaxis.grid(True, **GRID)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left", ncols=len(series))


def seed_entries(evaluation: dict[str, Any]) -> list[str]:
    names = sorted(key for key, value in evaluation.items()
                   if isinstance(value, dict) and "local_test" in value and key.startswith("v1_1_seed"))
    return names or ["v1_1"]


def diagnostics_figure(meta: Path, output: Path) -> dict[str, Any]:
    baselines = json.loads((meta / "baseline_metrics_v1_1.json").read_text(encoding="utf-8"))
    evaluation = json.loads((meta / "predictor_v1_1_evaluation.json").read_text(encoding="utf-8"))
    probe = json.loads((meta / "representation_probe_v1_1.json").read_text(encoding="utf-8"))
    summary = {name: item["summary"] for name, item in baselines["results"].items()}
    released = seed_entries(evaluation)[0]
    ours = evaluation[released]["local_test"]
    strong = "MLP-128（组成 + 几何）"
    mine = "等变 GNN 1.1.0（发布检查点）"

    figure = plt.figure(figsize=(13.4, 4.4))
    grid = figure.add_gridspec(1, 3, wspace=0.30, left=0.055, right=0.985, top=0.775, bottom=0.115)

    axis = figure.add_subplot(grid[0, 0])
    classes = ["low", "medium", "high"]
    grouped(axis, classes, [
        (strong, [summary["mlp128_composition_geometry"][f"{name}_f1"]["mean"] for name in classes], STRONG_BASELINE_COLOR),
        (mine, [ours[f"{name}_f1"] for name in classes], OURS_COLOR),
    ], limit=1.12)
    axis.set_title("分档 F1：low 仍是最难的一档")

    axis = figure.add_subplot(grid[0, 1])
    matrix = np.asarray(evaluation[released]["confusion"], dtype=float)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    image = axis.imshow(normalized, cmap="Oranges", vmin=0, vmax=1)
    axis.set_xticks(range(3), classes)
    axis.set_yticks(range(3), classes)
    axis.set_xlabel("预测")
    axis.set_ylabel("真实")
    for i in range(3):
        for j in range(3):
            axis.text(j, i, f"{int(matrix[i, j])}\n{normalized[i, j] * 100:.0f}%", ha="center", va="center",
                      fontsize=9, color="white" if normalized[i, j] > 0.55 else "#26323d")
    axis.set_title("发布检查点的混淆矩阵（1,514 条测试）")
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03).ax.tick_params(labelsize=8)

    axis = figure.add_subplot(grid[0, 2])
    spreads = [probe["models"]["v1_0"]["modes"]["raw"]["mean_spread"],
               probe["models"]["v1_0"]["modes"]["canonicalized"]["mean_spread"],
               probe["models"]["v1_1"]["modes"]["canonicalized"]["mean_spread"]]
    flips = [probe["models"]["v1_0"]["modes"]["raw"]["fraction_label_flips"],
             probe["models"]["v1_0"]["modes"]["canonicalized"]["fraction_label_flips"],
             probe["models"]["v1_1"]["modes"]["canonicalized"]["fraction_label_flips"]]
    handles = axis.bar(["1.0 未规范化", "1.0 规范化", "1.1.0（本模型）"], spreads,
                       color=[OLD_COLOR, CONTROL_COLOR, OURS_COLOR], width=0.6)
    for rectangle, spread, flip in zip(handles, spreads, flips):
        axis.text(rectangle.get_x() + rectangle.get_width() / 2, spread + 0.02,
                  f"{spread:.3f}\n改档 {flip * 100:.0f}%", ha="center", fontsize=8.5)
    axis.set_ylim(0, 0.62)
    axis.set_ylabel("P(high) 平均极差")
    axis.yaxis.grid(True, **GRID)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("同一结构换 6 种表示后的预测波动")

    figure.suptitle("NFE 预测器诊断 · nfe-v1.1 测试集（1,514 个结构）", fontsize=14, fontweight="bold", y=0.965)
    figure.text(0.5, 0.885, "逐指标的模型对比见 docs/images/benchmark_panels_dark.png；此图只放对比图没有的三项诊断。",
                ha="center", fontsize=9.5, color="#4a5563")
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return {"released_entry": released, "per_class_f1": {name: ours[f"{name}_f1"] for name in classes},
            "confusion": matrix.tolist(), "spreads": spreads, "label_flips": flips}


def generator_figure(meta: Path, output: Path) -> dict[str, Any]:
    new = json.loads((meta / "generator_final_metrics_v1_1.json").read_text(encoding="utf-8"))["test"]
    old = json.loads((meta / "generator_final_metrics.json").read_text(encoding="utf-8"))["test"]
    back_new = json.loads((meta / "generator_backtest_gen_v1_1.json").read_text(encoding="utf-8"))["summary"]
    back_old = json.loads((meta / "generator_backtest_v1_1.json").read_text(encoding="utf-8"))["summary"]
    bench = json.loads((meta / "generator_strict_generation_benchmark.json").read_text(encoding="utf-8"))["summary"]
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))
    figure.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.18, wspace=0.28)

    grouped(axes[0], ["端点 RMSE", "内核 MAE", "表面 MAE"], [
        ("1.0 生成器", [old["endpoint_rmse_A"], old["core_mae_A"], old["surface_mae_A"]], OLD_COLOR),
        ("1.3.0 生成器（v1.1 标签）", [new["endpoint_rmse_A"], new["core_mae_A"], new["surface_mae_A"]], OURS_COLOR),
    ], limit=0.64)
    axes[0].set_ylabel("Å（越低越好）")
    axes[0].set_title("监督测试集几何误差")

    modes = ["流 + 投影", "模板 + 投影", "流，不投影"]
    grouped(axes[1], modes, [
        ("1.0 生成器", [back_old[k]["rmsd_A_median"] for k in ("flow_projected", "template_projected", "flow_raw")], OLD_COLOR),
        ("1.3.0 生成器", [back_new[k]["rmsd_A_median"] for k in ("flow_projected", "template_projected", "flow_raw")], OURS_COLOR),
    ], limit=0.36)
    axes[1].set_ylabel("相对 DFT 几何的 RMSD 中位数 (Å)")
    axes[1].set_title("回测：同堆垛模板，149 个 test 组成")

    grouped(axes[2], ["成功组数（共 12 组）", "导出候选数"], [
        ("1.0 生成器", [bench["gen_1_0"]["succeeded"], bench["gen_1_0"]["candidates"]], OLD_COLOR),
        ("1.3.0 生成器", [bench["gen_v1_1"]["succeeded"], bench["gen_v1_1"]["candidates"]], OURS_COLOR),
    ], fmt="{:.0f}", limit=28)
    axes[2].set_title("严格生成：4 种骨架 × 3 个档位")
    figure.text(
        0.5, 0.04,
        f"严格生成的平均目标概率 {bench['gen_1_0']['mean_target_probability']:.3f} → {bench['gen_v1_1']['mean_target_probability']:.3f}；"
        f"平均 CHGNet 最大力 {bench['gen_1_0']['mean_chgnet_fmax_eV_A']:.3f} → {bench['gen_v1_1']['mean_chgnet_fmax_eV_A']:.3f} eV/Å（1.0 → 1.3.0）",
        ha="center", fontsize=9.2, color="#4a5563",
    )

    figure.suptitle("表面生成器：nfe-v1.1 重训（1.3.0）与 1.0 版对比", fontsize=13.5, fontweight="bold", y=0.955)
    figure.text(0.5, 0.875, "回测与严格生成都使用同一个 1.1.0 预测器；两个生成器各训练一次，严格生成只有 12 组，差别在波动范围内。", ha="center", fontsize=9.5, color="#4a5563")
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return {"test_new": new, "test_old": old, "benchmark": bench}


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="models/metadata")
    parser.add_argument("--output-dir", default="docs/images")
    parser.add_argument("--summary", default="models/metadata/performance_figures.json")
    args = parser.parse_args(argv)
    style()
    meta = ROOT / args.metadata
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/plot_model_performance.py",
        "diagnostics_figure": str(Path(args.output_dir) / "predictor_diagnostics_v1_1.png"),
        "generator_figure": str(Path(args.output_dir) / "generator_v1_1_vs_1_0.png"),
        "diagnostics": diagnostics_figure(meta, output / "predictor_diagnostics_v1_1.png"),
        "generator": generator_figure(meta, output / "generator_v1_1_vs_1_0.png"),
    }
    (ROOT / args.summary).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("diagnostics_figure", "generator_figure")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
