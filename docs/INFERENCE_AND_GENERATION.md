# 推理与生成 / Inference and Generation

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 输入结构 NFE 预测 / Predict NFE for input structures

```bash
python training/entrypoints/predict.py \
  --checkpoint models/server/ruck_dp/nfe_predictor/best.pt \
  examples/structures/sample_low_ScTaCSeBr.cif \
  examples/structures/sample_medium_TiNbCSeCl.cif \
  examples/structures/sample_high_ZrTiHSNO.cif \
  --mc-samples 30 \
  --output predictions.csv
```

关键输出：

- `Predicted_NFE_Label`；
- `Probability_Low/Medium/High`；
- `Predicted_NFE_Pseudo_Score`；
- MC dropout 标准差；
- embedding OOD distance/risk；
- 辅助物性；
- `Recommended_Low/Medium/High_NFE`；
- `Class_Probability_Ranking`，始终保留三档完整排序。

用户界面显示 low/medium/high 全部概率，而不是只显示最大类，便于发现边界样本。

## 条件生成 / Conditional generation

最终生成应使用 manifold generator 入口。先查看参数：

```bash
python training/entrypoints/manifold_generation.py --help
```

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

