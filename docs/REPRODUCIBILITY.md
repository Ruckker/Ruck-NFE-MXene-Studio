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

## 2026-09-15 修订后的复核证据 / Re-verification evidence

- `python -m unittest discover -s tests`：26 项，在 Python 3.12 + torch 2.6 与 Python 3.9 +
  torch 2.0.1 两个环境中均通过；
- 用 Windows 包内的 `nfe_predictor.pt` 在 1,514 条 test 记录上重算：accuracy 0.879、
  macro F1 0.735，与检查点内存储的指标一致（后者含 2 个 DDP 补齐重复样本）；
- `models/metadata/baseline_metrics.json`（组成基线，3 seed）、`ablation_seed_summary.json`
  （作者 5 seed 消融）、`generator_backtest.json`（生成器回测）、`enumeration_summary.json`
  （穷举筛选）均由仓库内脚本生成，文件中记录了表格 SHA256、检查点路径与时间。

## 1.1.0 复核证据 / Re-verification evidence for 1.1.0

- 1.1.0 预测器（集群单卡 RTX 3090 训练）在本机（RTX 4070 SUPER，torch 2.6）对 1,514 条
  test 记录重算：accuracy 0.9122、macro F1 0.8017、high-vs-rest F1 0.8706、ECE 0.0184，与
  检查点内存储的指标一致（`training/audits/evaluate_predictor_checkpoints.py` 生成
  `models/metadata/predictor_v1_1_evaluation.json`）；
- 同表的 1.0 架构对照：macro F1 0.7963、high-vs-rest F1 0.8614；
- `baseline_metrics_v1_1.json`（v1.1 标签组成基线，3 seed）、`representation_probe_v1_1.json`
  （`training/audits/representation_probe.py`，确定性表示探针，90 个 test 结构 × 6 种表示，
  P(high) 极差 0）、
  `enumeration_summary_v1_1.json`（v1.1 穷举筛选）、`generator_backtest_v1_1*.json`
  （以 1.1.0 预测器评估的生成器回测）均由仓库内脚本生成；
- Windows 1.1.0 冻结程序自检通过（`release_assets/windows/frozen_self_test_1_1_0.json`）。

## 1.1.1 复核证据 / Re-verification evidence for 1.1.1

- 44 个单元测试在 Windows 构建环境（Python 3.9 + torch 2.0.1）中全部通过，其中 3 个 Tk GUI 测试在真实窗口中
  验证单文件报错、多文件只排除不合法文件与重复导入不弹窗；研究环境（Python 3.12 + torch 2.6）缺少
  tkinterdnd2，跳过这 3 个，其余 41 个通过；
- `training/audits/audit_mxene_inputs.py` 生成 `models/metadata/mxene_input_validation_audit.json`：15,206 个
  清洁结构与 72 个隔离结构全部判为合法。首版规则曾误判 69 个偏心的 W/Mo/Cr 氮化物，已按训练集
  几何极值重新设定阈值；
- Windows 1.1.1 冻结程序自检通过，新增输入校验项：四个样例被接受，岩盐 TiC 体相与文本文件被排除
  （`release_assets/windows/frozen_self_test_1_1_1.json`）。

## 1.2.0 复核证据 / Re-verification evidence for 1.2.0

- 53 个单元测试在 Windows 构建环境中全部通过，其中新增的三维预览测试覆盖场景构建（超胞、镜像、分层、片层晶胞框、
  跨边界片层）、四种模式与三种主题的渲染、拾取、测量、视角与适配，以及 Tk 组件的点击、拖动、缩放、导出和窄布局；
  研究环境跳过 3 个需要 tkinterdnd2 的测试，其余 50 个通过；
- Windows 1.2.0 冻结自检通过，新增一帧离屏渲染：45 个原子、131 根键，
  800×450，用时 95.3 ms（`release_assets/windows/frozen_self_test_1_2_0.json`）。

## 1.3.0 复核证据 / Re-verification evidence for 1.3.0

- 表面生成器在集群 GPU 节点 4 张 RTX 3090 上以 `training/configs/surface_generator_v1_1.yaml` 的服务器副本重训，
  245 个 epoch 后早停，最佳 epoch 209；检查点、历史、配置与日志在 `nfe_server_models_1.3.0_20260916.zip`；
- `training/audits/backtest_generator.py` 以新生成器与 1.1.0 预测器回测（`generator_backtest_gen_v1_1*.json`），
  结果与 1.0 生成器一致；
- `training/audits/benchmark_strict_generation.py` 在本机 RTX 4070 SUPER 上以同一预测器、同样 12 组骨架与目标比较
  严格生成：新生成器 12/12 组成功，1.0 生成器 11/12 组
  （`generator_strict_generation_benchmark.json`）；
- Windows 1.3.0 冻结自检通过，内置新生成器导出 ScTaCSeBr low 候选，目标概率 0.96，CHGNet 最大力
  0.042 eV/Å（`release_assets/windows/frozen_self_test_1_3_0.json`）。
- `training/audits/evaluate_predictor_checkpoints.py` 用同一份代码复评 4 个检查点（1.1.0 架构的
  seed 2027/2028/2029 与 1.0 架构对照），结果写入 `models/metadata/predictor_v1_1_evaluation.json`；
  seed 2028、2029 用 `training/configs/nfe_predictor_v1_1.yaml` 只改 `seed` 在 GPU 节点重训，
  各自早停于第 139、150 个 epoch，检查点不随发布分发。
- `training/audits/plot_benchmark_panels.py` 把上面两个结果画成逐指标对比图
  （`docs/images/benchmark_panels_dark.png` 与 `_light.png`），数值写入
  `models/metadata/benchmark_panels.json`。
- `training/audits/plot_model_performance.py` 生成 `docs/images/` 下的预测器诊断图与生成器对比图，
  并把图中数值写入 `models/metadata/performance_figures.json`。

## 种子与确定性 / Seeds and determinism

- NFE predictor seed 2027（发布检查点），复训对照 seed 2028、2029；
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
