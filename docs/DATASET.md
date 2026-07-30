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

