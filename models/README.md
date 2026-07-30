# 模型文件 / Model Files

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

Git 仓库只保留小型指标和几何元数据；完整 `.pt` 在
`release_assets/server/nfe_server_models_1.0_20260730.zip`。

安装：

```bash
python scripts/install_release_assets.py --parts models
```

安装后路径以 `models/server/ruck_dp/` 开头。不要把未知来源 checkpoint 传给
`torch.load`。详细适用范围见 [`MODEL_CARD.md`](MODEL_CARD.md)。

