# 发行资产 / Release Assets

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30 15:39:00 Asia/Shanghai  
正式版本 / Release version: 1.0

大型 ZIP 不属于 Git 仓库目录树。本目录只保存索引、manifest、许可证与 SHA256；
实际附件请从 [`../docs/DOWNLOADS.md`](../docs/DOWNLOADS.md) 或
[GitHub Releases](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases) 下载。

## server/

- [`nfe_server_training_source_1.0_20260730.zip`](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_training_source_1.0_20260730.zip)：训练与推理源码。
- [`nfe_server_dataset_20260730_090526.zip`](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_dataset_20260730_090526.zip)：完整清洁数据、脏数据及审计记录。
- [`nfe_server_models_1.0_20260730.zip`](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_models_1.0_20260730.zip)：NFE 预测器、表面约束生成器和流形生成器。
- [`nfe_server_environment_20260730_090526.zip`](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_environment_20260730_090526.zip)：服务器环境与依赖记录。
- `nfe_server_archives_1.0.sha256`：服务器交付包校验值。

The server assets contain the function-named source, full dataset, trained
models, and environment records required to reproduce release 1.0.

## windows/

- [`NFE_MXene_Studio_1.0_Source_20260730_2310.zip`](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/NFE_MXene_Studio_1.0_Source_20260730_2310.zip)：可学习、可重建源码及必要模型。
- [`NFE_MXene_Studio_1.0_Windows_20260730_2300.zip.part01`](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/NFE_MXene_Studio_1.0_Windows_20260730_2300.zip.part01) 与
  [`part02`](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/NFE_MXene_Studio_1.0_Windows_20260730_2300.zip.part02)：完整 Windows onedir ZIP64 的两个 GitHub Release 分卷；合并方法见
  [`../docs/DOWNLOADS.md`](../docs/DOWNLOADS.md)。
- `NFE_MXene_Studio_1_0/`：解压 Windows ZIP 后得到的本地 onedir，不存储在普通 Git 历史中。
- `SHA256SUMS_1.0.txt`：Windows 源码包、程序包和入口 EXE 校验值。

这些大型文件通过 GitHub Releases 发布，避免直接写入普通 Git 历史。

Publish these large artifacts through GitHub Releases instead of ordinary Git
history.

项目首页、科学定位、技术栈、架构和模型卡已经按学术研究软件标准整理，明确说明
MXene NFE 研究空白、系统级创新、证据层级和 DFT 验证边界。
