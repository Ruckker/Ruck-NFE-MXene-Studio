# 贡献指南 / Contributing

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

感谢改进本项目。提交前请遵循以下约定：

1. 新算法放入 `src/nfe_model/`，命令行入口放入 `training/entrypoints/`。
2. 保持中英双语模块导读，写明输入、输出、单位和周期边界假设。
3. 禁止把随机结构直接标记为“物理合理”；必须给出几何、OOD、重复和松弛证据。
4. 数据切分必须以 `Split_Group` 为单位，不能让同一结构家族跨 train/validation/test。
5. 新增模型结果必须报告逐类别指标、macro 指标、混淆矩阵和类别支持数。
6. 不提交 VASP 商业程序、许可受限文件、密码、主机地址或个人凭据。
7. 大模型、完整数据和 Windows 程序使用 GitHub Releases 或 Git LFS，不直接进入普通 Git 历史。

Before opening a pull request, preserve bilingual module guides, document units
and periodic assumptions, keep split groups disjoint, report class-wise metrics,
add tests, and never commit credentials or licensed VASP binaries.

建议检查 / Suggested checks:

```bash
python -m compileall src training data_tools app tests scripts
python -m unittest discover -s tests
```

