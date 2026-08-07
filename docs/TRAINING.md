# 训练教程 / Training Guide

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 1. 前置条件 / Prerequisites

- Linux x86_64；
- Python 3.10；
- 1–4 张 NVIDIA GPU；最终训练使用 4 × RTX 3090；
- CUDA-enabled PyTorch，最终参考为 `2.6.0+cu118`；
- 完整数据已安装到 `data/full/`；
- `python -m pip install -e .` 已使 `nfe_model` 可导入。

检查：

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

## 2. 配置定位 / Config paths

公开仓库的配置相对于 `training/configs/`：

- 表格：`data/full/nfe_dataset.csv`；
- 结构根：`data/full/`，CSV 的 `File_Path` 再追加 `data/...`；
- 图缓存：`cache/nfe_graphs.pt`；
- 表面几何先验：`models/metadata/surface_geometry_summary.json`。

首次训练可传 `--rebuild-cache`。缓存包含表格 SHA256，表格更换后不应复用旧缓存。

## 3. NFE 预测器 / NFE predictor

单卡：

```bash
python training/entrypoints/train.py \
  --gpus 1 \
  --task predictor \
  --config training/configs/nfe_predictor.yaml \
  --rebuild-cache
```

四卡：

```bash
python training/entrypoints/train.py \
  --gpus 4 \
  --devices 0,1,2,3 \
  --task predictor \
  --config training/configs/nfe_predictor.yaml
```

核心超参数：

- 220 epochs，35 epoch 自监督预训练；
- 每卡 batch 96；
- AdamW learning rate `3e-4`；
- 8 epoch warmup + cosine；
- AMP；
- early stopping patience 35；
- score loss 1.5、auxiliary 0.45、masked atom 0.35、denoise 0.65。

选择检查点不是只看 accuracy，而是综合 macro ROC-AUC、
macro F1、ECE 与回归误差的 `selection_score`。

## 4. surface generator 表面生成器 / surface generator

最终入口：

```bash
torchrun --standalone --nproc-per-node=4 \
  -m nfe_model.train_surface_generator \
  --config training/configs/surface_generator.yaml
```

核心超参数：

- 260 epochs、每卡 batch 32；
- learning rate `2e-4`；
- sampling cutoff 12 Å、max neighbors 24；
- coordinate/lattice weights 1.0/0.35；
- repulsion 0.30、endpoint 1.0、pair 0.80、layer 0.50；
- surface anchor 0.80、OH geometry 1.50；
- early stopping patience 35。

监控字段：

- `val_endpoint_rmse_A`：重建端点总 RMSE；
- `val_core_mae_A`：内核层误差；
- `val_surface_mae_A`：表面端基误差；
- `val_pair_loss`：关键成对距离；
- `val_anchor_loss`：hollow 锚点；
- `val_oh_loss`：O–H 几何；
- `val_layer_loss`：层序违反。

若总 loss 降低而某一物理分解指标恶化，不应接受该 checkpoint。

## 5. manifold generator 的含义 / What manifold generator means

manifold generator 不是另一次从零训练。它继承 surface generator 权重，在生成时加入：

- 按原子角色限制位移；
- 模板流形投影；
- 晶格应变上限；
- OH 键投影；
- 未见金属组合替换；
- 更严格的目标、重复、中心、表面拓扑检查。

因此复现 manifold generator 必须同时保留：

1. surface/manifold checkpoint；
2. `manifold_generation.py`；
3. `surface_geometry.py`；
4. 表面几何 summary；
5. NFE predictor；
6. CHGNet。

## 6. 恢复与不覆盖 / Resume without overwriting

服务器入口支持 `--resume`。建议每次实验设置新的 `checkpoint_dir` 和日志名：

```yaml
training:
  checkpoint_dir: runs/predictor_ablation_001
```

不要把消融实验写入最终 `nfe_predictor`。恢复前检查 checkpoint
配置、表格 SHA256、world size 和优化器状态是否匹配。

## 7. 训练健康检查 / Health checks

立即停止并诊断：

- `NaN/Inf`；
- CUDA OOM 持续出现；
- NCCL timeout/collective mismatch；
- 某一 rank 提前退出；
- train loss 降低而 validation 全面恶化；
- 缓存跳过结构比例超过配置；
- split-group 重叠；
- 端点误差下降但层序/anchor/OH 违反升高。

可以安全尝试：

- 降低每卡 batch，增加 gradient accumulation；
- 检查 AMP scaler 与异常样本；
- 固定 seed 后复现实验；
- 单卡复现数据错误；
- 保留旧配置/检查点，用新目录运行修正版。

## 8. 评价报告最低要求 / Minimum evaluation report

预测器：

- 每类 precision/recall/F1/support/ROC-AUC；
- accuracy、balanced accuracy、macro F1、macro AUC；
- 3×3 混淆矩阵；
- ECE 与温度；
- NFE score MAE/RMSE；
- 按 OOD 分层结果。

生成器：

- 坐标、晶格、内核、表面、端点、anchor、OH 分解 loss；
- low/medium/high 各自严格接受率；
- CHGNet 最大力分布；
- 中心偏移、层数、端基、hollow、键长通过率；
- 训练集重复率和 OOD；
- 最终 VASP 成功率。

