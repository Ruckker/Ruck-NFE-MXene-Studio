# 推理与生成 / Inference and Generation

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 输入结构 NFE 预测 / Predict NFE for input structures

```bash
python training/entrypoints/predict.py \
  --checkpoint models/server_v1_1/nfe_predictor_v1_1/best.pt \
  examples/structures/sample_low_ScTaCSeBr.cif \
  examples/structures/sample_medium_TiNbCSeCl.cif \
  examples/structures/sample_high_ZrTiHSNO.cif \
  --mc-samples 30 \
  --output predictions.csv
```

关键输出：

- `Predicted_NFE_Label`；
- `Probability_Low/Medium/High`；
- `Predicted_High_vs_Rest` 与 `High_vs_Rest_Threshold`：high 对非 high 的二分类判定，阈值由
  训练时在验证集上拟合（1.0 检查点没有该字段，回退到 0.5）；
- `Predicted_NFE_Score`；
- MC dropout 标准差；
- embedding OOD distance/risk；
- 辅助物性；
- `Recommended_Low/Medium/High_NFE`；
- `Class_Probability_Ranking`，始终保留三档完整排序；
- `Canonicalized`：输入是否经过规范化。

用户界面显示 low/medium/high 全部概率，而不是只显示最大类，便于发现边界样本。

Windows 程序在导入时还会检查输入是否为 MXene 片层，不合法的文件不会进入预测，详见
[`WINDOWS_APP.md`](WINDOWS_APP.md) 的“输入校验”；命令行 `predict.py` 不做这项检查。

### 输入规范化 / Input canonicalization

预测前每个结构都会被 `nfe_model.canonical.canonicalize_slab` 规范化：约化为原胞、面内晶格取
最短基并采用 γ = 120° 约定、真空轴替换为长度 30 Å 的法向、slab 居中到 z = 0.5、固定面内平移
规范、原子按物种与坐标排序。训练集全部是 c = 30 Å 的六方 1×1 胞，预测器 1.0 的 11 维全局
特征里有 4 维在训练集内方差为零，不规范化的输入（例如 20 Å 真空、γ = 60° 设定、2×2 超胞）
会把这些特征推到饱和值并改变档位。`--no-canonicalize` 只用于调试。

规范化后 1.0 检查点在 test 集上的指标略有提高（macro F1 0.735 → 0.738，ECE 0.013 → 0.008），
同一结构在不同表示下的 P(high) 变化从最大 0.9 降到 0.3。1.1.0 检查点（`complete_shells: true`、
无全局特征）在确定性前向下对 90 个 test 结构的真空厚度（20/40 Å）、γ = 60° 设定、2×2×1 超胞
和平移变体给出完全相同的 P(high)（极差 0，关闭规范化亦为 0；
`models/metadata/representation_probe_v1_1.json`）。运行时预测取 MC dropout 样本的均值，
样本间差异作为不确定性报告，不是输入敏感性。

## 条件生成 / Conditional generation

最终生成应使用 manifold generator 入口。先查看参数：

```bash
python training/entrypoints/generate_mxene.py --help
```

`--sampler template` 跳过条件流、直接用加噪模板作为原始候选，是生成器消融的无流基线；
`training/audits/backtest_generator.py` 用 validation/test 中已有 DFT 结构的组成对比
"流 + 投影"与"模板 + 投影"。从 test 抽样 200 个组成，其中 149 个有同堆垛训练模板，每个取
3 个模板，无 CHGNet；目标档位取自 v1.1 表，档位一致率由 1.1.0 预测器判定
（`models/metadata/generator_backtest_v1_1.json`）：

| 模式 | RMSD 中位 (Å) | 最大位移中位 (Å) | 晶格 a 误差中位 | 档位一致率 |
|---|---:|---:|---:|---:|
| 流 + 流形投影 | 0.141 | 0.211 | 1.1% | 0.884 |
| 模板 + 流形投影（无流） | 0.252 | 0.368 | 2.7% | 0.846 |
| 流，不投影 | 0.106（均值 0.619） | 0.162 | 1.1% | 0.886 |

1.0 预测器与 v1.0 表下的同一回测（`generator_backtest.json`）为 RMSD 中位 0.152 / 0.258 /
0.150 Å、一致率 0.787 / 0.765 / 0.794。两次回测的几何数值略有差别，原因是目标档位取自不同的表，
且预测器的 MC dropout 在目标之间消耗的随机数不同；结论不变。

结论：流在投影允许的范围内确实把模板几何向 DFT 几何精修了约 45%，投影则消除了少数
大偏差样本（不投影时均值 0.62 Å）；但若模板的 fcc/hcp 堆垛与目标不同，三种模式的 RMSD
都约 1.1 Å（`generator_backtest_v1_1_any_stacking.json`），因为投影不允许端基换 hollow 位点，
而训练时的模板池也不区分堆垛。生成器应被理解为"同拓扑模板精修器"，新颖性来自组成替换
与模板选择，不来自流本身。

1.3.0 起生成器在 v1.1 标签上重训。同一回测（`models/metadata/generator_backtest_gen_v1_1.json`）中，
流 + 投影的 RMSD 中位数为 0.140 Å，档位一致率 0.886；low 目标的一致率由 0.556 升到 0.593。
`training/audits/benchmark_strict_generation.py` 在同一预测器、4 种骨架 × 3 个档位上比较严格生成：

| 生成器 | 成功组数 | 导出候选 | 平均目标概率 | 平均 CHGNet 最大力 (eV/Å) |
|---|---:|---:|---:|---:|
| 1.3.0（v1.1 标签） | 12/12 | 22 | 0.922 | 0.0428 |
| 1.0（v1.0 标签） | 11/12 | 20 | 0.953 | 0.0423 |

样本量小，两者差别在波动范围内；重训的主要意义是生成与筛选使用同一套 v1.1 标签。

概念性参数包括：

- target: low/medium/high；
- core: C/N；
- top/bottom inner metal；
- predictor checkpoint；
- generator checkpoint；
- dataset/table/template root；
- surface geometry profile；
- oversample、sampling steps、guidance；
- CHGNet relax steps/fmax；
- target probability 和 MC samples；
- 输出目录。

实际命令以当前 `--help` 名称为准，避免教程与代码版本参数漂移。

## 为什么端基不能由用户指定？

最终产品约束用户只能指定：

- 目标 NFE 档位；
- 核心 C/N；
- 两种内层金属。

表面基团由训练模板分布和生成/筛选系统决定，因为端基与 NFE、表面配位、键长和
结构稳定性强耦合。任意强制端基会把模型推到训练分布外并提高塌缩风险。

## 严格接受条件 / Strict acceptance

一个候选只有同时通过才导出：

1. CIF 可重新解析；
2. slab 中心 `z≈0.5`；
3. 原子距合理；
4. 5/6/7 层且层序正确；
5. 上下端基完整；
6. 端基属于训练分布；
7. 三配位 hollow；
8. OH/金属–端基键在训练分位；
9. CHGNet 固定晶胞达到目标最大力；
10. 不与训练结构重复；
11. OOD 风险可接受；
12. 独立预测器 MC 复评与目标档位一致；
13. 若要求，金属组合在训练集中未见。

禁止用 `allow_target_mismatch` 伪装成功。若某档没有候选，应报告 0 和拒绝原因。

## 输出文件 / Outputs

- `rank_*.cif`；
- `POSCAR_rank_*`；
- `generation_summary.csv`；
- `generation_summary_with_poscar.csv`；
- `run_info.json`；
- 每次尝试日志。

`run_info.json` 是最重要的诊断文件。应先统计主要拒绝原因，再决定是否增加 oversample、
调整模板或改进模型；不能先放宽物理标准。

## 推荐 VASP 验证顺序 / Recommended VASP validation

1. 结构和对称性人工检查；
2. 使用一致赝势、ENCUT、k 网格进行离子弛豫；
3. 确认力、应力和电子收敛；
4. 高精度静态计算；
5. 自旋分辨能带与 DOS；
6. Γ 附近更密 k 点拟合有效质量；
7. band-decomposed partial charge density；
8. 真空厚度与偶极修正收敛；
9. 必要时声子、AIMD、形成能/凸包稳定性。

模型只负责缩小搜索空间，不能替代该验证链。

