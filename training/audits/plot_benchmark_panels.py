# ==============================================================================
# 中文概述：以分节卡片的形式绘制基准对比图：每个指标一组行，圆点为多次训练的均值，横线为 ± 标准差，
#           右侧写出数值，最好的模型标星，指标下方给出该指标的坐标范围。
# English overview: Draw the benchmark comparison as sectioned cards: one row group per metric, a dot for the mean
#           over repeated trainings, a bar for the standard deviation, the value on the right, a star on the best
#           model and the axis range under each metric.
#
# 中文输入：组成基线结果 JSON（已含均值与标准差）与预测器检查点复评 JSON（每个 seed 一条）。
# English inputs: The composition-baseline JSON (means and standard deviations included) and the predictor
#           checkpoint evaluation JSON (one entry per seed).
# 中文输出：深色与浅色两版 PNG，以及图中全部数值（JSON）。
# English outputs: A dark and a light PNG plus every plotted number as JSON.
#
# 关键约束 / Key invariants:
# - 只画评测文件里已经算出的数字；本模型的均值 ± 标准差来自同一组 seed，别的模型按各自的重复次数给出。
#   Only numbers already present in the evaluation files are drawn; the mean and standard deviation of this work
#   come from one seed group, and every other model reports its own number of repeats.
# - 模型用各自的正式名称；没有在本项目训练过的公开模型名不会出现在图里。
#   Models carry their own proper names; names of published models never trained here do not appear.
# - 所有卡片共用同一纵向刻度，行高一致。
#   Every card shares one vertical scale so the rows keep the same height.
# - 主要接口 / Main APIs: main
#
# Author: Ruck
# Generated: 2026-09-16
# ==============================================================================

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# 中文：模型行的定义（来源、键、名称、深色与浅色的颜色）。
# English: Model rows (source, key, label, colour on the dark and on the light theme).
MODELS = [
    ("baseline", "rule_has_OH_is_high", "OH-termination rule", ("#6c7683", "#69737f")),
    ("baseline", "logistic_regression_composition", "Logistic regression", ("#9aa5b4", "#8d97a5")),
    ("baseline", "mlp128_composition", "MLP-128 (composition)", ("#f2994a", "#d97706")),
    ("baseline", "mlp128_composition_geometry", "MLP-128 (+ geometry)", ("#b58bf2", "#7c3aed")),
    ("baseline", "mlp128_score_regression_composition_geometry", "MLP-128 regressor", ("#f2994a", "#d97706")),
    ("official", "cgcnn_official", "CGCNN (official)", ("#4c9aff", "#2563eb")),
    ("official", "schnet_official", "SchNet (SchNetPack)", ("#5ad1e6", "#0891b2")),
    ("official", "alignn_official", "ALIGNN (official)", ("#f2c14e", "#b45309")),
    ("official", "m3gnet_official", "M3GNet (MatGL)", ("#ef7b8e", "#db2777")),
    ("ours", "", "This work", ("#5ad19a", "#0f9d6b")),
]
TONES = {"dark": 0, "light": 1}
# 中文：本仓库与基准套件的同义指标名。/ English: Metric names that the two code bases spell differently.
ALIASES = {"high_average_precision": "high_vs_rest_average_precision"}
REGRESSION_ONLY = {"mlp128_score_regression_composition_geometry"}
THEMES = {
    "dark": {
        "page": "#0b0e12", "card": "#141920", "edge": "#222a34", "text": "#e8edf3",
        "muted": "#8d99a8", "faint": "#68727f", "axis": "#333c47", "star": "#f6c445",
    },
    "light": {
        "page": "#ffffff", "card": "#f6f8fb", "edge": "#dde4ed", "text": "#16202b",
        "muted": "#5d6a78", "faint": "#7a8695", "axis": "#c3ccd8", "star": "#c8860d",
    },
}
SECTIONS = [
    ("Overall classification", [
        ("Macro F1", "higher is better", "macro_f1", "{:.3f}", ""),
        ("Macro average precision", "higher is better", "macro_average_precision", "{:.3f}", ""),
        ("Macro ROC-AUC", "higher is better", "macro_roc_auc", "{:.3f}", ""),
    ]),
    ("High-value screening", [
        ("High-class average precision", "higher is better", "high_average_precision", "{:.3f}", ""),
        ("High-class F1", "higher is better", "high_f1", "{:.3f}", ""),
        ("Enrichment at top 5 %", "higher is better", "high_enrichment_at_5pct", "{:.2f}", "x"),
    ]),
    ("Score regression", [
        ("Mean absolute error", "lower is better", "NFE_Pseudo_Score_mae", "{:.4f}", ""),
        ("Spearman rank correlation", "higher is better", "NFE_Pseudo_Score_spearman", "{:.3f}", ""),
        ("Coefficient of determination", "higher is better", "NFE_Pseudo_Score_r2", "{:.3f}", ""),
    ]),
    ("Calibration", [
        ("Expected calibration error", "lower is better", "ece", "{:.4f}", ""),
    ]),
]
CARDS = [[0], [1], [2, 3]]
# 中文：卡片内的横向分栏（占卡片宽度的比例）。/ English: Column layout inside a card, as a fraction of its width.
NAME_LEFT = 0.028
PLOT_LEFT, PLOT_RIGHT = 0.350, 0.645
MEAN_RIGHT, DEVIATION_LEFT, HINT_RIGHT, STAR_X = 0.790, 0.806, 0.938, 0.972
ROW, TITLE_GAP, AXIS_GAP, SECTION_GAP, TOP_GAP = 1.0, 1.3, 2.35, 1.0, 1.2


# 中文：均值与样本标准差。/ English: Mean and sample standard deviation.
def summarize(values: Sequence[float]) -> tuple[float, float, int]:
    numbers = [float(value) for value in values]
    if len(numbers) == 1:
        return numbers[0], 0.0, 1
    return statistics.fmean(numbers), statistics.stdev(numbers), len(numbers)


# 中文：取出一个指标下的所有模型行。/ English: Collect every model row for one metric.
def collect(metric: str, baselines: dict[str, Any], evaluation: dict[str, Any], official: dict[str, Any],
            seeds: Sequence[str]) -> list[dict[str, Any]]:
    regression = metric.startswith("NFE_Pseudo_Score")
    local = ALIASES.get(metric, metric)
    rows: list[dict[str, Any]] = []
    for source, key, label, colour in MODELS:
        if source == "baseline" and (key in REGRESSION_ONLY) != regression:
            continue
        if source == "baseline":
            entry = baselines["results"].get(key, {}).get("summary", {}).get(local)
            if entry is None:
                continue
            mean, deviation = float(entry["mean"]), float(entry["std"])
            repeats = 1 if key == "rule_has_OH_is_high" else len(baselines.get("seeds", []))
        elif source == "official":
            entry = official["models"].get(key, {}).get("summary", {}).get(metric)
            if entry is None:
                continue
            mean, deviation, repeats = float(entry["mean"]), float(entry["std"]), int(entry["count"])
        else:
            values = [evaluation[name]["local_test"][local] for name in seeds
                      if name in evaluation and local in evaluation[name]["local_test"]]
            if not values:
                continue
            mean, deviation, repeats = summarize(values)
        rows.append({"model": label, "colour": colour, "mean": mean, "std": deviation,
                     "repeats": repeats, "ours": source == "ours"})
    return rows


# 中文：一个指标块需要的纵向单位数。/ English: Vertical units needed by one metric block.
def metric_units(rows: int) -> float:
    return TITLE_GAP + ROW * (rows - 1) + AXIS_GAP


# 中文：一张卡片需要的纵向单位数。/ English: Vertical units needed by one card.
def card_units(sections: Sequence[int], data: dict[str, list]) -> float:
    total = TOP_GAP
    for index in sections:
        total += 1.9 + SECTION_GAP + 0.35
        for _name, _hint, key, _fmt, _suffix in SECTIONS[index][1]:
            total += metric_units(len(data[key]))
    return total + 1.1


# 中文：画一张卡片。/ English: Draw one card.
def draw_card(axis: plt.Axes, sections: Sequence[int], data: dict[str, list], theme: dict[str, str], height: float, tone: int) -> None:
    axis.set_facecolor(theme["card"])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(height, 0.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(theme["edge"])
        spine.set_linewidth(1.0)

    cursor = TOP_GAP
    for index in sections:
        title, metrics = SECTIONS[index]
        axis.text(NAME_LEFT, cursor, title, color=theme["text"], fontsize=14.5, fontweight="bold", va="center")
        axis.plot([NAME_LEFT, STAR_X + 0.015], [cursor + 0.78] * 2, color=theme["edge"], linewidth=1.2)
        cursor += 1.9 + SECTION_GAP
        for name, hint, key, fmt, suffix in metrics:
            rows = data[key]
            axis.text(NAME_LEFT, cursor, name, color=theme["text"], fontsize=10.5, fontweight="bold", va="center")
            axis.text(HINT_RIGHT, cursor, hint, color=theme["faint"], fontsize=9, va="center", ha="right")
            cursor += TITLE_GAP
            edges = [row["mean"] + row["std"] for row in rows] + [row["mean"] - row["std"] for row in rows]
            low, high = min(edges), max(edges)
            pad = (high - low) * 0.16 or max(abs(high) * 0.05, 1e-3)
            low, high = low - pad, high + pad
            if low < 0.0 <= min(row["mean"] - row["std"] for row in rows):
                low = 0.0
            best = min(row["mean"] for row in rows) if "lower" in hint else max(row["mean"] for row in rows)
            best_text = fmt.format(best)

            def place(value: float) -> float:
                return PLOT_LEFT + (value - low) / (high - low) * (PLOT_RIGHT - PLOT_LEFT)

            for row in rows:
                colour = row["colour"][tone]
                axis.text(NAME_LEFT, cursor, row["model"], color=theme["text"] if row["ours"] else theme["muted"],
                          fontsize=9.5, fontweight="bold" if row["ours"] else "normal", va="center")
                if row["std"] > 0:
                    axis.plot([place(row["mean"] - row["std"]), place(row["mean"] + row["std"])], [cursor, cursor],
                              color=colour, linewidth=1.5, alpha=0.9, solid_capstyle="butt", zorder=2)
                    for edge in (row["mean"] - row["std"], row["mean"] + row["std"]):
                        axis.plot([place(edge)] * 2, [cursor - 0.2, cursor + 0.2], color=colour, linewidth=1.3, zorder=2)
                axis.plot([place(row["mean"])], [cursor], marker="o", markersize=7.0 if row["ours"] else 5.8,
                          color=colour, markeredgecolor=theme["card"], markeredgewidth=0.9, zorder=3)
                weight = "bold" if row["ours"] else "normal"
                colour_text = theme["text"] if row["ours"] else theme["muted"]
                axis.text(MEAN_RIGHT, cursor, fmt.format(row["mean"]) + suffix, color=colour_text,
                          fontsize=9.5, fontweight=weight, va="center", ha="right")
                if row["std"] > 0:
                    axis.text(DEVIATION_LEFT, cursor, "± " + fmt.format(row["std"]).lstrip(),
                              color=theme["faint"] if not row["ours"] else theme["muted"],
                              fontsize=9.0, fontweight=weight, va="center", ha="left")
                if fmt.format(row["mean"]) == best_text:
                    axis.text(STAR_X, cursor, "★", color=theme["star"], fontsize=11, va="center", ha="center")
                cursor += ROW
            baseline = cursor - ROW + 0.70
            axis.plot([PLOT_LEFT, PLOT_RIGHT], [baseline] * 2, color=theme["axis"], linewidth=1.0, zorder=1)
            for position, align in ((PLOT_LEFT, "left"), (PLOT_RIGHT, "right")):
                axis.plot([position] * 2, [baseline - 0.12, baseline + 0.12], color=theme["axis"], linewidth=1.0)
            axis.text(PLOT_LEFT, baseline + 0.48, fmt.format(low) + suffix, color=theme["faint"], fontsize=8.5, va="center", ha="left")
            axis.text(PLOT_RIGHT, baseline + 0.48, fmt.format(high) + suffix, color=theme["faint"], fontsize=8.5, va="center", ha="right")
            cursor += AXIS_GAP - ROW
        cursor += 0.35


# 中文：生成一个主题的图。/ English: Render the figure for one theme.
def build(theme_name: str, data: dict[str, list], caption: str, footer: Sequence[str], output: Path) -> None:
    theme = THEMES[theme_name]
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": theme["page"],
        "savefig.facecolor": theme["page"],
    })
    height = max(card_units(card, data) for card in CARDS)
    inches = round(height * 0.235 + 2.0, 2)
    head = 0.62 / inches
    foot = (0.30 + 0.21 * len(footer)) / inches
    figure = plt.figure(figsize=(17.4, inches))
    figure.subplots_adjust(left=0.010, right=0.990, top=1.0 - head, bottom=foot, wspace=0.028)
    figure.text(0.010, 1.0 - head * 0.42, "NFE MXene predictor versus published backbones on the same split",
                color=theme["text"], fontsize=17, fontweight="bold", va="center")
    figure.text(0.990, 1.0 - head * 0.42, caption, color=theme["muted"], fontsize=10.5, va="center", ha="right")
    for position, card in enumerate(CARDS):
        axis = figure.add_subplot(1, len(CARDS), position + 1)
        draw_card(axis, card, data, theme, height, TONES[theme_name])
    for offset, line in enumerate(footer):
        figure.text(0.010, foot - (0.30 + 0.21 * offset) / inches + 0.105 / inches, line,
                    color=theme["faint"], fontsize=9.5, va="center")
    figure.text(0.990, foot - 0.195 / inches, "★ best value of the metric",
                color=theme["faint"], fontsize=9.5, va="center", ha="right")
    figure.savefig(output, dpi=170)
    plt.close(figure)


# 中文：命令行入口。/ English: Command-line entry point.
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", default="models/metadata/baseline_metrics_v1_1.json")
    parser.add_argument("--evaluation", default="models/metadata/predictor_v1_1_evaluation.json")
    parser.add_argument("--official", default="models/metadata/official_baselines_v1_1.json")
    parser.add_argument("--seeds", nargs="+",
                        default=["v1_1_seed2027", "v1_1_seed2028", "v1_1_seed2029",
                                 "v1_1_seed2030", "v1_1_seed2031"])
    parser.add_argument("--output-dir", default="docs/images")
    parser.add_argument("--summary", default="models/metadata/benchmark_panels.json")
    args = parser.parse_args(argv)

    baselines = json.loads((ROOT / args.baselines).read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / args.evaluation).read_text(encoding="utf-8"))
    official = json.loads((ROOT / args.official).read_text(encoding="utf-8"))
    missing = [name for name in args.seeds if name not in evaluation]
    if missing:
        raise SystemExit(f"{args.evaluation} has no entry for {missing}")
    if official["table_sha256"].upper() != evaluation["table_sha256"].upper():
        raise SystemExit("the published backbones and this work were evaluated on different dataset tables")
    data = {key: collect(key, baselines, evaluation, official, args.seeds)
            for _title, metrics in SECTIONS for _n, _h, key, _f, _s in metrics}
    empty = [key for key, rows in data.items() if not rows]
    if empty:
        raise SystemExit(f"no model reports {empty}")

    rows = int(evaluation.get("rows", baselines["split_counts"]["test"]))
    caption = f"nfe-v1.1 held-out test split · {rows:,} structures · group-aware split"
    footer = [
        f"Dot = mean, bar = ± 1 standard deviation over {len(args.seeds)} trainings that differ only in the random seed "
        "(the released checkpoint is one of them); the OH-termination rule is deterministic and has no bar.",
        "Published backbones: CGCNN (txie-93 checkout), SchNet (SchNetPack 2.2.0), ALIGNN (2026.5.20 with DGL 2.1.0), "
        "M3GNet (MatGL 4.0.3) - upstream message-passing operators with this project's split, task head, optimiser,",
        "graph budget and calibration, not untouched upstream training pipelines. They are trained purely supervised on "
        "the class and the score, without the auxiliary electronic targets, masked-atom objective, coordinate denoising "
        "or self-supervised schedule of this work.",
    ]
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "training/audits/plot_benchmark_panels.py",
        "baselines": args.baselines,
        "evaluation": args.evaluation,
        "seeds": list(args.seeds),
        "test_rows": rows,
        "figures": {},
        "values": {key: [{k: v for k, v in row.items() if k != "colour"} for row in rows_]
                   for key, rows_ in data.items()},
    }
    for theme in ("dark", "light"):
        path = output / f"benchmark_panels_{theme}.png"
        build(theme, data, caption, footer, path)
        report["figures"][theme] = (Path(args.output_dir) / path.name).as_posix()
    (ROOT / args.summary).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["figures"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
