# 模型文件 / Model Files

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

Git 仓库只保留小型指标和几何元数据；完整 `.pt` 通过 GitHub Release 发布：

[下载 nfe_server_models_1.0_20260730.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_models_1.0_20260730.zip)

全部大文件入口及本地放置方法见
[`../docs/DOWNLOADS.md`](../docs/DOWNLOADS.md)。

下载后把 ZIP 放入本地 `release_assets/server/`，再安装：

```bash
python scripts/install_release_assets.py --parts models
```

安装后路径以 `models/server/ruck_dp/` 开头。不要把未知来源 checkpoint 传给
`torch.load`。详细适用范围见 [`MODEL_CARD.md`](MODEL_CARD.md)。
