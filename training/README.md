# 训练入口 / Training Entrypoints

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

- `entrypoints/train.py`：选择 1–4 GPU 并启动 predictor / surface generator；
- `entrypoints/predict.py`：CIF/POSCAR 推理（默认先规范化为原胞、γ=120°、c=30 Å、居中）；
- `entrypoints/generate_mxene.py`：manifold generator，最终流形投影与未见组合生成；
- `entrypoints/enumerate_candidates.py`：穷举尚未计算的 M1-M2-X-T1-T2-堆垛 组合并用预测器排序，
  这是 1.0 设计空间的主线筛选路径；
- `baselines/composition_baselines.py`：组成基线（规则 / 逻辑回归 / MLP），报告 GNN 时必须并列；
- `configs/`：`nfe_predictor.yaml`（1.0 检查点的配置）、`nfe_predictor_v1_1.yaml`（规范化输入、
  完整壳层、无全局特征）、`surface_generator.yaml`、`manifold_generation.yaml`；
- `audits/`：环境、训练集表面、生成候选审计、多 seed 汇总（`aggregate_seed_runs.py`）与
  生成器回测（`backtest_generator.py`）。

完整命令与监控指标见 [`../docs/TRAINING.md`](../docs/TRAINING.md)。
