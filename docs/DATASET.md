# 数据构建教程 / Dataset Construction Tutorial

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 从原始计算重新抽取 / Rebuild from raw calculations

```bash
python data_tools/build_nfe_dataset.py \
  --source /path/to/static_calc \
  --output-root /path/to/new_dataset \
  --workers 32
```

请先运行 `--help` 查看当前参数名。默认策略是非破坏性的：复制而不移动结构，不修改
源计算目录，不静默覆盖已有输出。任何显式覆盖表格选项也只应替换表，不删除结构。

The extractor is deliberately non-destructive: it copies structures, leaves raw
calculations untouched, and refuses silent overwrites.

## 已知问题：nfe-v1.0 表的自旋解析 / Known issue in the nfe-v1.0 table

2026-09-15 审查发现，v1.0 提取器的 `parse_gamma_projections` 依赖一行 VASP 从不
写入 PROCAR 的 `spin component` 标记，因此 ISPIN=2 的两个自旋块都被记到 spin=0，
第二块覆盖第一块，spin=1 的能带因取不到投影被跳过。已发布的
`nfe_dataset.csv`（`Extraction_Schema_Version = nfe-v1.0`）表现为：

- 15,206 条 `NFE_Candidate_Spin` 全部为 `up`，而 `N_Spin_Channels` 全为 2，
  31.3% 的结构 |M| > 0.1 μB；
- 磁性结构的自旋向上能带配对的是自旋向下的投影，`NFE_Atomic_Projection_*` 与
  权重最大的 projection 分量可能错配；非磁结构两块相同，不受影响；
- 自旋向下通道里的 NFE 候选带被系统性排除。

提取器已修复（`nfe-v1.1`，以重复的 `# of k-points` 头行计数自旋块，与 pymatgen
`Procar` 相同），并有 `tests/test_dataset_extractor.py` 回归测试。

**v1.1 表已于 2026-09-15 在集群上重新抽取**（`--fit-points 12`，与 v1.0 只差自旋修复），
安装后位于 `data/full_v1_1/`，发行归档 `nfe_server_dataset_v1_1_20260915.zip`。与 v1.0
逐行比较（`training/audits/compare_label_versions.py`，
`models/metadata/label_version_comparison.json`）：

| v1.0 \ v1.1 | low | medium | high |
|---|---:|---:|---:|
| low | 582 | 181 | 1 |
| medium | 3 | 11,721 | 659 |
| high | 0 | 20 | 2,039 |

- 标签变化 864 条（5.7%）；候选带来自自旋向下通道的有 7,259 条（v1.0 为 0）；
- 磁性结构（|M| > 0.1 μB，4,766 条）中 848 条改档（17.8%），非磁结构只有 16 条（0.15%）；
  磁性结构的分数平均绝对变化 0.035，非磁 0.0005；
- 新标签分布 low / medium / high = 585 / 11,922 / 2,699；划分（`Split_Group`）不变，硬失败仍为 72 条。

v1.0 表保留在 `data/full/` 供复现 1.0 检查点。重新抽取命令（如需再次执行）：

```bash
python data_tools/build_nfe_dataset.py \
  --root /path/to/new-strcutre --source static_calc \
  --output nfe_dataset_v1_1.csv --dirty-output dirty_manifest_v1_1.csv \
  --audit-output extraction_audit_v1_1.csv --summary-output extraction_summary_v1_1.json \
  --fit-points 40 --workers 32 --write-outputs
```

`--fit-points` 是新参数：v1.0 每侧只用 12 个 k 点（150 点/段，约 0.1 Å⁻¹），
对 m* = 1 的带只覆盖约 35 meV，任何光滑带的 R² 都接近 1，parabola 与 isotropy
分量因而几乎是常数。建议 30–40。重新抽取后请比较新旧表的标签迁移矩阵，再决定是否重训。

The v1.0 extractor never separated PROCAR spin blocks. The fixed extractor
(`nfe-v1.1`) counts blocks from the repeated `# of k-points` preamble. The
nfe-v1.1 table was re-extracted on 2026-09-15 and is the training table of the
1.1.0 predictor and, since 1.3.0, of the surface generator; the v1.0 table and the 1.0 predictor
and generator are kept for reproduction.

## 质量门 / Quality gates

硬失败示例：

- 缺失/空 `CONTCAR`、`OUTCAR` 或关键能带文件；
- 静态任务未完成；
- 电子未收敛；
- 总能或费米能不可读；
- 结构无法由 pymatgen 重新解析；
- 原子数/晶格/最小距离明显异常。

软 warning 示例：

- 真空平台不够平，功函数不可靠；
- 某一辅助 ELF/charge 特征缺失；
- 非致命 VASP warning；
- 一个辅助目标超出可靠范围。

训练器使用目标掩码处理缺失辅助标签；分类主目标只有通过硬门的记录参与。

## 防止数据泄漏 / Prevent leakage

不要执行简单 `train_test_split(rows)`。结构名解析得到核心、两种金属、端基和堆垛，
再构造 `Split_Group`。相同家族必须全部进入同一 split。训练时
`assert_disjoint_split_groups` 会再次验证。

## 伪标签再校准 / Recalibrate pseudo-labels

建议从 low/medium/high 各分层抽样，补做 band-decomposed partial charge density，
由人工判定真 NFE 与非 NFE；随后：

1. 保留当前伪标签用于预训练；
2. 增加人工真值列和证据路径；
3. 用少量真值微调或训练校准层；
4. 分别报告伪标签测试和人工真值测试；
5. 对生成结构只把伪标签作为筛选，不作为最终科学结论。

## 可追溯性 / Traceability

- `Source_Directory`：原始 VASP 位置；
- `Extraction_UTC`：抽取时间；
- `Extraction_Schema_Version`：字段规则版本；
- `extraction_audit.csv`：所有候选，包括失败；
- `dirty_manifest.csv`：硬失败原因；
- `nfe_extraction.log`：运行过程；
- 数据 ZIP SHA256：防止传输损坏。

