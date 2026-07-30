# 可复现性 / Reproducibility

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 已归档证据 / Archived evidence

| 归档 | 内容 |
|---|---|
| `nfe_server_training_source_1.0_20260730.zip` | 清理旧入口后的最终源码、配置和全部最终测试 |
| `nfe_server_dataset_*.zip` | 完整清洗/脏数据、表格、审计、日志 |
| `nfe_server_models_*.zip` | predictor、surface/manifold generator、历史、指标、表面先验 |
| `nfe_server_environment_*.zip` | Python/pip/conda、CPU/GPU/OS、数据统计、源码 mtime |
| Windows source ZIP | 双语注释 App、核心模型源码、模型资源、构建脚本、样例、测试 |
| Windows ZIP64 | 最终 onedir 程序 |

服务器四包已在服务器和本地分别校验 SHA256，并通过 `unzip -t`。
归档通过 GitHub Release 附件发布；下载链接和下载后的本地放置位置见
[`DOWNLOADS.md`](DOWNLOADS.md)。

## 复现实验级别 / Levels of reproduction

1. **推理复现**：使用最终 checkpoint 和同一结构，输出应在浮点误差内接近。
2. **训练趋势复现**：相同 seed/环境下验证曲线接近，但多 GPU 浮点规约不保证逐位一致。
3. **完全逐位复现**：未承诺；CUDA、NCCL、驱动和并行顺序可能产生差异。
4. **科学复现**：必须拥有原始 VASP 输入/输出或重新计算；清洗数据 ZIP不是原始计算的替代。

## 种子与确定性 / Seeds and determinism

- NFE predictor seed 2027；
- surface generator seed 2029；
- Python、NumPy、PyTorch 和 rank 派生 seed 由 `seed_everything` 设置；
- 数据划分确定；
- DDP sampler 每 epoch 设置 seed；
- CHGNet/ASE 优化和部分 CUDA kernel 可能非严格确定。

## 校验命令 / Verification commands

先按 [`DOWNLOADS.md`](DOWNLOADS.md) 下载附件并放入本地
`release_assets/server/` 或 `release_assets/windows/`，再执行：

PowerShell:

```powershell
Get-FileHash release_assets\server\*.zip -Algorithm SHA256
Get-FileHash release_assets\windows\*.zip -Algorithm SHA256
```

Linux:

```bash
sha256sum -c release_assets/server/nfe_server_archives_1.0.sha256
unzip -t release_assets/server/nfe_server_dataset_20260730_090526.zip
```

源码：

```bash
python -m compileall src training data_tools app tests scripts
python -m unittest discover -s tests
```

## 为什么把源码与数据/模型分包 / Why source, data, and models are separate

final-only 源码包用于阅读、训练和重建；数据、模型和环境包用于大文件分发与复现。
源码已删除旧基础生成器、旧训练入口、旧配置和旧测试逻辑，同时保留 NFE 预测器、
surface generator 表面训练骨干、manifold generator 最终推理层及三套最终测试。数据/模型/环境 ZIP 的服务器下载
字节保持不变。
