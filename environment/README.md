# 环境说明 / Environment

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

- `requirements.txt`：Linux 训练普通依赖；GPU torch 单独安装。
- `requirements-relax.txt`：CHGNet/ASE 预弛豫扩展。
- `server/`：运行 `scripts/install_release_assets.py --parts environment` 后得到完整
  pip freeze、conda explicit、OS/CPU/GPU 快照、数据统计和源码时间。
- Windows 依赖：`app/windows/packaging/requirements-windows.txt`。

不要用普通镜像命令覆盖已经安装好的 CUDA PyTorch。清华/阿里镜像适合 NumPy、
pandas、pymatgen、PyYAML 等普通包；特殊 CUDA wheel 应由已验证的离线 wheelhouse 提供。

Do not let a general mirror replace a working CUDA PyTorch build. Use mirrors
for ordinary packages and verified local wheels for CUDA-specific builds.

