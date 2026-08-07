# 训练入口 / Training Entrypoints

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

- `entrypoints/train.py`：选择 1–4 GPU 并启动 predictor/base generator；
- `entrypoints/predict.py`：CIF/POSCAR 推理；
- `entrypoints/manifold_generation.py`：最终流形投影与未见组合生成；
- `baselines/`：固定 Split_Group 的 Dummy/XGBoost/受控晶体 GNN 基线比较；
- `ablations/`：主 NFE predictor 的 vector、global slab、自监督与多任务消融；
- `configs/`：NFE predictor、surface generator、manifold generator inference；
- `audits/`：环境、训练集表面和生成候选审计。

完整训练命令与监控指标见 [`../docs/TRAINING.md`](../docs/TRAINING.md)。  
基线比较见 [`baselines/README.md`](baselines/README.md)，内部消融见
[`ablations/README.md`](ablations/README.md)。
