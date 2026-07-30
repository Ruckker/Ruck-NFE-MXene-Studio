# 下载与 GitHub Release 附件 / Downloads and GitHub Release Assets

作者 / Author: Ruck  
更新时间 / Updated: 2026-07-30

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

## 直接下载 / Direct downloads

下列 `latest/download` 链接会自动指向最新的**已发布、非草稿** Release。附件名称必须
与表中完全一致。

| 内容 / Asset | 下载 / Download | SHA256 |
|---|---|---|
| 服务器训练与推理源码 | [nfe_server_training_source_1.0_20260730.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_training_source_1.0_20260730.zip) | `50E5A9416C496B99C29705CE08E4D8DD728BAEEFF4C98DCC25EEEECAEBE20243` |
| 完整数据集 | [nfe_server_dataset_20260730_090526.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_dataset_20260730_090526.zip) | `D21E3184CB2A8B26FD1E4BEEDC41526BFF51970305221ABA7EFBBE39FCCB9CD2` |
| 训练模型权重 | [nfe_server_models_1.0_20260730.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_models_1.0_20260730.zip) | `CBD941DC070CB5BF68FDBB1EFD68821619F976E67C45E9850CB2CBF06F058B36` |
| 服务器环境记录 | [nfe_server_environment_20260730_090526.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/nfe_server_environment_20260730_090526.zip) | `AFE8BE42E6D0A4CA5BA5CDBD242FB860EBFE62541824054E864E2D0C6B2B6538` |
| Windows 可重建源码包 | [NFE_MXene_Studio_1.0_Source_20260730.zip](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/NFE_MXene_Studio_1.0_Source_20260730.zip) | `8EB5CBA11F2461915C06477FFF8CE6AA2BDF1EC3611546F214F3CB3AC2FE9E03` |
| Windows 完整程序分卷 1 | [NFE_MXene_Studio_1.0_Windows_20260730.zip.part01](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/NFE_MXene_Studio_1.0_Windows_20260730.zip.part01) | `D66E0035B79B6F356ECD8C825276B89B5650DAAB3CA9DA36A0220EBB8988CA14` |
| Windows 完整程序分卷 2 | [NFE_MXene_Studio_1.0_Windows_20260730.zip.part02](https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/latest/download/NFE_MXene_Studio_1.0_Windows_20260730.zip.part02) | `0D071DC78DF2D8E22DC611C0AB93DBD4AD174775AEFF9AB88969555B5603DFE8` |

## 下载后的本地放置位置 / Local placement after download

为了继续使用仓库中的安装和校验命令，把下载文件放到以下**本地目录**：

```text
release_assets/
├─ server/
│  ├─ nfe_server_training_source_1.0_20260730.zip
│  ├─ nfe_server_dataset_20260730_090526.zip
│  ├─ nfe_server_models_1.0_20260730.zip
│  └─ nfe_server_environment_20260730_090526.zip
└─ windows/
   ├─ NFE_MXene_Studio_1.0_Source_20260730.zip
   ├─ NFE_MXene_Studio_1.0_Windows_20260730.zip.part01
   └─ NFE_MXene_Studio_1.0_Windows_20260730.zip.part02
```

这些 ZIP 会被 `.gitignore` 排除，不应再次提交到普通 Git 历史。放置完成后可运行：

```bash
python scripts/install_release_assets.py
```

Windows 程序原始 ZIP 为 2,852,507,046 字节，超过 GitHub 单个 Release 附件的
2 GiB 限制，因此发布为两个分卷。下载两个分卷后，在仓库根目录运行：

```bash
python scripts/reassemble_release_parts.py \
  release_assets/windows/NFE_MXene_Studio_1.0_Windows_20260730.zip.part01 \
  release_assets/windows/NFE_MXene_Studio_1.0_Windows_20260730.zip.part02 \
  --output release_assets/windows/NFE_MXene_Studio_1.0_Windows_20260730.zip \
  --sha256 BB4D1B76B9C8B007B6E534C587E85F7C74475A69B1D7DF865EF11F0B649039EF
```

校验通过后再解压生成的 ZIP。脚本不会覆盖已有 ZIP，也不会删除任何分卷。

## 链接返回 404 时 / If a link returns 404

依次检查：

1. Release 是否仍是 `Draft`；草稿附件不能被公开下载；
2. 是否已经点击 `Publish release`；
3. 附件名称是否与上表完全一致，包括大小写、下划线和时间戳；
4. Release 是否被标记为最新正式版；若为 prerelease，优先使用带标签的固定链接：
   `https://github.com/Ruckker/Ruck-NFE-MXene-Studio/releases/download/<TAG>/<FILE>`。
