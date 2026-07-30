# 数据工具 / Data Tools

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

`build_nfe_dataset.py` 从上游 VASP 计算构建 118 列 NFE 数据集。它设计为非破坏性：
复制结构、保留源目录、记录每条审计、把硬失败放入 dirty。

Read [`../docs/DATASET.md`](../docs/DATASET.md) before rebuilding. The extractor
copies rather than moves structures and keeps complete audit provenance.

