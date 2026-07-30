# 安全说明 / Security

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

- 不要在 issue、日志或配置中公开 SSH 密码、令牌、私钥或集群内部地址。
- CIF/POSCAR 属于外部输入；批量运行前应限制文件大小并在非特权账户中执行。
- 本项目只加载可信来源的 PyTorch `.pt` 文件。PyTorch 反序列化可能执行恶意对象，
  不要加载未知检查点。
- `scripts/install_release_assets.py` 会验证 SHA256、拒绝路径穿越并拒绝写入非空目录。
- Windows 程序按 `onedir` 分发；不要从不可信来源替换 `_internal/` 中 DLL 或模型。

Do not disclose credentials, load untrusted PyTorch checkpoints, or replace
bundled DLLs/models with files from unknown sources. Run external structures
under a non-privileged account and verify release checksums.

