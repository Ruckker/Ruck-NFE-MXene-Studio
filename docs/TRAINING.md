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

两份配置：`nfe_predictor.yaml` 复现 1.0 检查点；`nfe_predictor_v1_1.yaml` 是 1.1.0 发布
预测器的配置，差别在 `data.canonicalize: true`（训练与推理同一规范化表示）、
`data.complete_shells: true`（邻居截断不切断配位壳层）、`model.global_features: 0`
（移除全局特征）、`training.primary_task: high_vs_rest`（以 high 对非 high 选择检查点），
以及数据表指向 `data/full_v1_1/`（nfe-v1.1 标签）和新的图缓存
（`cache/nfe_graphs_v1_1_canonical.pt`）。

1.1.0 的训练记录（2026-09-15，集群 GPU 节点单卡 RTX 3090，torch 2.6.0+cu118，
`python -m nfe_model.train --config ... --rebuild-cache`，patience 35 早停）：

| 运行 | 配置 | epoch（最佳） | 用时 | test macro F1 | test high-vs-rest F1 |
|---|---|---:|---:|---:|---:|
| `nfe_predictor_v1_1` | `nfe_predictor_v1_1.yaml`（seed 2027，发布检查点） | 117（81） | 10 分钟，5.2 s/epoch | 0.8017 | 0.8706 |
| `nfe_predictor_v1_1_seed2028` | 同上，仅改 `seed` | 139（103） | 18 分钟，7.8 s/epoch | 0.7865 | 0.8760 |
| `nfe_predictor_v1_1_seed2029` | 同上，仅改 `seed` | 150（114） | 19 分钟，7.8 s/epoch | 0.8002 | 0.8702 |
| `nfe_predictor_v1_0arch_on_v1_1` | 1.0 架构与超参，v1.1 表 | 147（111） | 12 分钟，5.0 s/epoch | 0.7963 | 0.8614 |

两个补 seed 与发布检查点共用同一个图缓存，在同一 GPU 节点上并行训练（因此每 epoch 比单独训练慢），
只用于给 1.1.0 架构的指标加上误差棒：三者均值 macro F1 0.7961 ± 0.0084、high 对非 high F1
0.8723 ± 0.0032。它们的检查点不随发布分发，可用同一配置改 `seed` 复现。

检查点、`final_metrics.json`、`history.jsonl`、服务器端配置与日志在
`nfe_server_models_1.3.0_20260916.zip`（安装到 `models/server_v1_1/`）；本机复评由
`training/audits/evaluate_predictor_checkpoints.py` 生成 `models/metadata/predictor_v1_1_evaluation.json`，
表示探针由 `training/audits/representation_probe.py` 生成 `models/metadata/representation_probe_v1_1.json`。表面生成器的 v1.1 重训记录见第 4 节。

单卡：

```bash
python training/entrypoints/train.py \
  --gpus 1 \
  --task predictor \
  --config training/configs/nfe_predictor_v1_1.yaml \
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

选择检查点不是只看 accuracy，而是综合 macro ROC-AUC、balanced accuracy、
macro F1、ECE 与回归误差的 `selection_score`；`primary_task: high_vs_rest` 时改用
high-vs-rest F1 与 AUC。训练结束会在验证集上拟合 P(high) 阈值并写入检查点。

评估在多卡下按数据集行号去重：`DistributedSampler(drop_last=False)` 会把每个 rank 补齐到
相同长度，1.0 发布的 test 指标因此包含 2 个重复样本（1,516 对 1,514）。`final_metrics.json`
新增 `evaluated_samples` 字段用于核对。

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

### 1.3.0 的 nfe-v1.1 重训记录 / nfe-v1.1 retraining for 1.3.0

`training/configs/surface_generator_v1_1.yaml` 与 `surface_generator.yaml` 只差数据表、图缓存与检查点目录。
2026-09-15 至 16 日在集群 GPU 节点的 4 张 RTX 3090 上以 torchrun 训练，启动脚本与服务器配置随模型归档发布
（`configs/run_surface_generator.sh`、`configs/server_surface_generator_v1_1.yaml`）：

| 项目 | 1.3.0（nfe-v1.1） | 1.0（nfe-v1.0） |
|---|---:|---:|
| 训练 epoch（最佳） | 245（209） | 251（215） |
| 每 epoch 用时 | 31.8 s | 30.5 s |
| 测试 endpoint RMSE | 0.4317 Å | 0.4472 Å |
| 测试 core / surface MAE | 0.2644 / 0.1932 Å | 0.2779 / 0.1981 Å |

Windows 程序使用的检查点由服务器检查点加 `windows_app` 元数据、表面几何摘要改为相对路径得到，权重不变。
严格生成对比用 `training/audits/benchmark_strict_generation.py`，结果在 `models/metadata/generator_strict_generation_benchmark.json`。

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
- high-vs-rest precision/recall/F1/AUC/average precision 与所用阈值；
- 3×3 混淆矩阵；
- ECE 与温度；
- NFE score MAE/RMSE；
- 按 OOD 分层结果；
- **至少 3 个 seed 的均值 ± 标准差**（`training/audits/aggregate_seed_runs.py`），
  单个 seed 的数字只能作为示例；
- **组成基线并列**（`training/baselines/composition_baselines.py`）：规则、逻辑回归、
  MLP、MLP + 几何标量，GNN 必须与它们同表比较。

生成器：

- 坐标、晶格、内核、表面、端点、anchor、OH 分解 loss；
- 与无流基线（`--sampler template`，模板 + 投影）在同一门控下的接受率、CHGNet 力和
  DFT 存活率对比；
- 在 validation/test 已有 DFT 结构的组成上的回测（`training/audits/backtest_generator.py`）：
  几何 RMSD、晶格误差、目标档位一致性；
- low/medium/high 各自严格接受率；
- CHGNet 最大力分布；
- 中心偏移、层数、端基、hollow、键长通过率；
- 训练集重复率和 OOD；
- 最终 VASP 成功率。

