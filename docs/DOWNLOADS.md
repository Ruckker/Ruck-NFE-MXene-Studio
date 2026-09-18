# 下载与 GitHub Release 附件 / Downloads and GitHub Release Assets

作者 / Author: Ruck  
更新时间 / Updated: 2026-09-18

## 重要说明 / Important note

GitHub Release 附件不属于 Git 仓库目录树，因此大型 ZIP **不会**实际出现在
`release_assets/server/` 或 `release_assets/windows/` 中。仓库中的这两个目录只保存
README、许可证、manifest 和 SHA256 校验文件。

GitHub Release assets are not part of the Git repository tree. The
`release_assets/` directory contains only indexes, manifests, licenses, and
checksums; large ZIP files are distributed from the Releases page.

- Release 页面 / Release page:
  <https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases>
- 最新正式版 / Latest published release:
  <https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest>


## 当前公开状态 / Current publication status

本次同步更新代码与项目文档。GitHub 当前已发布的 `release` 仍提供 1.0 二进制、数据与模型附件；1.3.0 及 nfe-v1.1 的本地归档尚未作为 Release 附件上传。下表中这些新归档标记为待发布，不能通过旧 Release 下载。论文与论文图表不包含在本次同步中。

This update publishes source code and project documentation. The existing GitHub release still contains version 1.0 assets; newer local archives are pending publication.

## 直接下载 / Direct downloads

下列 `latest/download` 链接会自动指向最新的**已发布、非草稿** Release。附件名称必须
与表中完全一致。

| 内容 / Asset | 下载 / Download | SHA256 |
|---|---|---|
| 训练与推理源码 1.3.0 | `nfe_server_training_source_1.3.0_20260916.zip`（本地归档，待发布） | `DC1E39BC34DB2D0E6A00418147C2139AB8055CE9A96A44DFABBC22D5B6A4218D` |
| 数据集 nfe-v1.1（自旋修复后重抽，发布模型的训练表） | `nfe_server_dataset_v1_1_20260915.zip`（本地归档，待发布） | `854789B7550D1DC58A86E309BAC993B447049333066AD17B7EC152C89AF05C6A` |
| 数据集 nfe-v1.0（1.0 模型的训练表，仅供复现） | [nfe_server_dataset_20260730_090526.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_dataset_20260730_090526.zip) | `D21E3184CB2A8B26FD1E4BEEDC41526BFF51970305221ABA7EFBBE39FCCB9CD2` |
| v1.1 模型 1.3.0（1.1.0 预测器、1.0 架构对照、v1.1 表面生成器；指标、训练历史与服务器配置） | `nfe_server_models_1.3.0_20260916.zip`（本地归档，待发布） | `23E511215D6B5D6ED5E304BF98E26338C601766EAB593D6A3D5487E711762A80` |
| 1.0 模型权重（1.0 预测器与生成器，仅供复现） | [nfe_server_models_1.0_20260730.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_models_1.0_20260730.zip) | `CBD941DC070CB5BF68FDBB1EFD68821619F976E67C45E9850CB2CBF06F058B36` |
| 服务器环境记录（与 1.0 相同） | [nfe_server_environment_20260730_090526.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_environment_20260730_090526.zip) | `AFE8BE42E6D0A4CA5BA5CDBD242FB860EBFE62541824054E864E2D0C6B2B6538` |
| Windows 可重建源码包 1.3.0 | `NFE_MXene_Studio_1.3.0_Source_20260916.zip`（本地归档，待发布） | `C85D870719E7A76AFC4E48184873AB695EC52D03F5DCEC8ADDE3AD1E98AB6C02` |
| Windows 完整程序 1.3.0 分卷 1 | `NFE_MXene_Studio_1.3.0_Windows_20260916.zip.part01`（本地归档，待发布） | `1770E6558902A7BF8BE0EAB719BA83F112F0E489AEC2C2AC743293F2B90B7D02` |
| Windows 完整程序 1.3.0 分卷 2 | `NFE_MXene_Studio_1.3.0_Windows_20260916.zip.part02`（本地归档，待发布） | `C5F0239B6A8F37646D3486653A09F7D25BA75C656B70DC6D2DB17BF41A3C36BE` |

1.3.0 更换了表面生成器：它在 nfe-v1.1 标签上重训，测试集端点 RMSE 0.432 Å；严格生成对比中
12/12 组成功（1.0 生成器 11/12）。v1.1 模型归档同时包含预测器与生成器，预测和条件生成只需要它；
1.0 模型归档仅供复现。三维预览与 MXene 输入校验与 1.2.0 相同。

本地归档中，1.2.0、1.1.1、1.1.0、1.0.1 与 1.0 的附件由 1.3.0 取代；这不表示远端 Release 已更新，其哈希保留在 `release_assets/windows/SHA256SUMS_*.txt` 与
`release_assets/server/nfe_server_archives_*.sha256` 中供核对；`nfe_server_models_1.1.0_20260915.zip` 由 `nfe_server_models_1.3.0_20260916.zip` 取代。

## 下载后的本地放置位置 / Local placement after download

为了继续使用仓库中的安装和校验命令，把下载文件放到以下**本地目录**：

```text
release_assets/
├─ server/
│  ├─ nfe_server_training_source_1.3.0_20260916.zip
│  ├─ nfe_server_dataset_v1_1_20260915.zip
│  ├─ nfe_server_dataset_20260730_090526.zip
│  ├─ nfe_server_models_1.3.0_20260916.zip
│  ├─ nfe_server_models_1.0_20260730.zip
│  └─ nfe_server_environment_20260730_090526.zip
└─ windows/
   ├─ NFE_MXene_Studio_1.3.0_Source_20260916.zip
   ├─ NFE_MXene_Studio_1.3.0_Windows_20260916.zip.part01
   └─ NFE_MXene_Studio_1.3.0_Windows_20260916.zip.part02
```

这些 ZIP 会被 `.gitignore` 排除，不应再次提交到普通 Git 历史。放置完成后可运行：

```bash
python scripts/install_release_assets.py
```

安装位置：`dataset` 到 `data/full/`，`dataset_v1_1` 到 `data/full_v1_1/`，`models` 到
`models/server/`，`models_v1_1` 到 `models/server_v1_1/`，`environment` 到 `environment/server/`。
只使用发布模型时可用 `--parts dataset_v1_1 models_v1_1`；安装脚本拒绝写入非空目录。

Windows 程序原始 ZIP 为 2,851,414,704 字节，超过 GitHub 单个 Release 附件的
2 GiB 限制，因此发布为两个分卷。下载两个分卷后，在仓库根目录运行：

```bash
python scripts/reassemble_release_parts.py \
  release_assets/windows/NFE_MXene_Studio_1.3.0_Windows_20260916.zip.part01 \
  release_assets/windows/NFE_MXene_Studio_1.3.0_Windows_20260916.zip.part02 \
  --output release_assets/windows/NFE_MXene_Studio_1.3.0_Windows_20260916.zip \
  --sha256 6AF1A07B12BAA06F7FCE3107E197B7540399B21D26A47D9C3FAAC4182FD37715
```

校验通过后再解压生成的 ZIP。脚本不会覆盖已有 ZIP，也不会删除任何分卷。解压得到
`NFE_MXene_Studio_1_3_0/`，入口 `NFE_MXene_Studio_1_3_0.exe`（SHA256
`1E06107CE6C22DFDEE973A8F833EAAF4786B40BDDFE2EF24A2CC0CEE2F777024`）。

1.3.0 构建的变化见 [`WINDOWS_APP.md`](WINDOWS_APP.md)：内置生成器换为 nfe-v1.1 表面生成器。冻结自检报告见
`release_assets/windows/frozen_self_test_1_3_0.json`。

## 链接返回 404 时 / If a link returns 404

依次检查：

1. Release 是否仍是 `Draft`；草稿附件不能被公开下载；
2. 是否已经点击 `Publish release`；
3. 附件名称是否与上表完全一致，包括大小写、下划线和时间戳；
4. Release 是否被标记为最新正式版；若为 prerelease，优先使用带标签的固定链接：
   `https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/download/<TAG>/<FILE>`。
