# 训练入口 / Training Entrypoints

作者 / Author: Ruck

## Paper-ready / 论文正式入口

最终 benchmark、消融和正式统计统一从：

```bash
python -m training.paper <alias> [arguments...]
```

进入。`training.paper` 固定 v2.4 pair-symmetric 数据/图语义、paper-ready YAML、训练预算、单进程单 GPU、clean-Git provenance，并对最终 benchmark/ablation 使用 closed-set 汇总器。完整顺序见 [`../docs/FINAL_PAPER_WORKFLOW.md`](../docs/FINAL_PAPER_WORKFLOW.md)。

4×GPU 正式并行脚本：

- `ablations/run_4gpu.sh`：固定 9 个消融 × 5 seeds，每个 Python 进程只使用 1 张 GPU；
- `baselines/run_4gpu.sh`：正式 architecture track，可选 full-system evaluation；official-upstream 需在各自 pinned isolated environment 中运行。

## Development / Smoke / Archive

- `formal_v2_4.py`：v2.4 开发/短 smoke dispatcher；允许显式模块或实验性参数，因此其 altered-budget 输出不能自动进入论文表；
- `entrypoints/train.py`：通用 predictor/base generator launcher；
- `entrypoints/predict.py`：CIF/POSCAR 推理；
- `entrypoints/manifold_generation.py`：流形投影与生成；
- `baselines/`：受控、matched 与 official-upstream benchmark 实现；
- `ablations/`：vector/global/SSL/multitask 消融；
- `evaluation/`：cache、split、representation、verified-NFE、OOD、prediction manifest 与统计审计；
- `configs/`：predictor/generator 配置；
- `audits/`：环境、训练集表面和生成候选审计。

不要把旧 v2.3 cache/result、直接调用低层 runner 的结果、或缩短 epoch/batch 的 smoke run 与 `training.paper` 结果混合。

基线说明见 [`baselines/README.md`](baselines/README.md)，消融说明见 [`ablations/README.md`](ablations/README.md)。
