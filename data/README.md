# 数据集 / Dataset

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

完整数据位于：

`../release_assets/server/nfe_server_dataset_20260730_090526.zip`

SHA256:

`d21e3184cb2a8b26fd1e4beedc41526bff51970305221aba7efbbe39fccb9cd2`

安全安装：

```bash
python scripts/install_release_assets.py --parts dataset
```

解压后结构：

```text
data/full/
├─ nfe_dataset.csv        # 15,206 × 118，训练主表
├─ data/                  # 15,206 个清洗后的 .vasp 结构
├─ dirty/                 # 72 个脏数据结构副本
├─ dirty_manifest.csv     # 脏数据字段与硬失败原因
├─ extraction_audit.csv   # 所有被检查任务的审计
├─ extraction_summary.json
└─ nfe_extraction.log
```

原始 `static_calc/` 是上游 VASP 计算目录，体积大且不由训练脚本直接读取，因此没有在
训练数据 ZIP 内再次复制。`Source_Directory` 保留每条记录的原始计算路径；
`build_nfe_dataset.py` 可以从拥有原始计算的环境重新生成数据。

The large `static_calc/` tree is upstream provenance rather than a direct
training input, so it is not duplicated in the dataset ZIP. Source paths remain
in `Source_Directory`, and the extractor can rebuild the dataset where raw VASP
calculations are available.

## 数量与划分 / Counts and splits

| 项目 | 数量 |
|---|---:|
| 审计计算 | 15,278 |
| 清洗记录 | 15,206 |
| 脏记录 | 72 |
| low / medium / high | 764 / 12,383 / 2,059 |
| train / validation / test | 12,193 / 1,499 / 1,514 |

`Suggested_Split` 是确定性分组划分。训练代码还会检查同一 `Split_Group` 不跨集合，
降低同骨架/同结构家族泄漏。

## 为什么有 12,246 条 warnings？

`extraction_summary.json` 的 `warnings=12246` 不是“12,246 个错误结构”，而是
**通过硬质量门但存在一个或多个软质量提示的记录数**。例如功函数真空平台不够平、
偶极修正可能缺失、某个辅助网格特征不可用。软 warning 可让结构参与主要 NFE/图学习，
但相应辅助字段通过 mask 或 `Work_Function_Reliable` 降权/屏蔽。

只有 `Hard_Failure_Reasons` 非空的 72 条进入 `dirty/`：

- 62 条：计算目录内容缺失/空，静态计算未完成、SCF 未收敛或关键能量不可读；
- 8 条：静态 SCF 未收敛；
- 2 条：目录几乎为空且多个关键文件缺失。

因此：

- `Quality_Warnings`：可审计的软提示；
- `Hard_Failure_Reasons`：禁止进入训练主表的硬失败；
- `Data_Quality_Score`：记录级综合质量分；
- warning 数绝不能直接等同于脏数据数。

## NFE 相关核心字段 / Core NFE fields

最直接的目标是：

- `NFE_Pseudo_Label`: low/medium/high 分类目标；
- `NFE_Pseudo_Score`: 0–1 连续强度目标；
- `NFE_Label_Is_Ground_Truth`: 当前均表明是伪标签，不是实验/人工真值。

物理解释字段：

- 候选带：`NFE_Candidate_Spin`, `NFE_Candidate_Band_Index`,
  `NFE_Energy_at_Gamma_eV`, `NFE_Energy_Relative_EF_eV`,
  `NFE_Occupation_at_Gamma`;
- 原子投影：`NFE_Atomic_Projection_Total/s/p/d`;
- 有效质量：`NFE_Effective_Mass_KG_me`, `NFE_Effective_Mass_GM_me`,
  `NFE_Effective_Mass_Geomean_me`, `NFE_Mass_Anisotropy`;
- 抛物线质量：`NFE_Parabolic_R2_*`, `NFE_Parabolic_RMSE_*`;
- 打分分解：`NFE_Score_Projection/Parabola/Energy/Mass/Isotropy_Component`;
- 候选数：`NFE_Candidate_Count`。

完整字段字典见 [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md)。

## 使用纪律 / Usage discipline

1. 不用 `Total_Energy_eV` 在不同组成之间直接比较稳定性；应使用形成能/凸包。
2. 不把伪标签当作实验真值。
3. 不随机按行重新切分后宣称独立测试；应保持 `Split_Group`。
4. 不用测试集反复挑模型或调阈值。
5. 所有生成结构先经过几何与 CHGNet，再做 VASP；CHGNet 通过也不保证 DFT 稳定。

