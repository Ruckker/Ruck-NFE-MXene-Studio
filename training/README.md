# 训练入口 / Training Entrypoints

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

- `entrypoints/train.py`：选择 1–4 GPU 并启动 predictor/base generator；
- `entrypoints/predict.py`：CIF/POSCAR 推理；
- `entrypoints/manifold_generation.py`：最终流形投影与未见组合生成；
- `configs/`：NFE predictor、surface generator、manifold generator inference；
- `audits/`：环境、训练集表面和生成候选审计。

完整命令与监控指标见 [`../docs/TRAINING.md`](../docs/TRAINING.md)。
