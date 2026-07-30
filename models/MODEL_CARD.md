# 模型卡 / Model Card

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 模型清单 / Model inventory

| 模型 | 归档路径 | 用途 |
|---|---|---|
| NFE predictor | `ruck_dp/nfe_predictor/best.pt` | 三分类、多物性、OOD |
| surface generator | `ruck_dp/surface_generator/best_generator.pt` | 表面模板条件流 |
| manifold generator derived generator | `ruck_dp/manifold_generator/best_generator.pt` | 与 manifold generator 推理/流形投影配合 |

完整模型归档：

`release_assets/server/nfe_server_models_1.0_20260730.zip`

SHA256:

`CBD941DC070CB5BF68FDBB1EFD68821619F976E67C45E9850CB2CBF06F058B36`

该值对应完整服务器模型 ZIP，而不是单个 `.pt` 文件。

## 预期用途 / Intended use

- 筛选 MXene NFE 候选；
- 对输入 CIF/POSCAR 预测 low/medium/high 与连续分数；
- 在训练化学空间附近提出由 VASP 验证的新结构；
- 教学和模型消融。

### 研究决策位置 / Position in the research workflow

该模型适合作为 **surrogate screener and hypothesis generator**：

\[
\text{large candidate space}
\rightarrow \text{ML ranking/generation}
\rightarrow \text{strict pre-screening}
\rightarrow \text{small DFT queue}.
\]

它不处于最终结论层。论文中应把模型输出描述为 predicted/pseudo-NFE candidate，
只有完成收敛电子结构分析后才能描述为 DFT-supported NFE material。

不适用于：

- 直接宣称实验可合成；
- 替代 VASP/DFT；
- 对完全不同材料家族做无验证外推；
- 计算热力学/动力学稳定性；
- 将伪标签当作实验标签。

## 训练数据 / Training data

15,206 个已弛豫后静态计算的 MXene，包含 C/N 核心、多种金属、F/Cl/Br/I/S/Se/OH
端基和不同堆垛。类别不均衡，medium 占主导。

## 预测性能 / Predictor performance

测试集：

- accuracy 0.87797；
- balanced accuracy 0.74838；
- macro F1 0.73404；
- macro ROC-AUC 0.92006；
- low/medium/high F1 = 0.5000/0.92314/0.77899；
- calibrated ECE 0.01373；
- NFE score MAE 0.03487。

low 类只有 85 个测试样本，召回约 0.494，因此 low/medium 边界仍是主要短板。
不能用总体 accuracy 掩盖这一点。

## 生成性能 / Generator performance

surface generator 测试集：

- endpoint RMSE 0.44717 Å；
- core MAE 0.27788 Å；
- surface MAE 0.19813 Å；
- lattice loss 0.11376；
- layer loss 0；
- OH loss 0.001325。

这些是训练目标重建指标，不等价于 DFT 稳定率。最终 manifold generator 依赖严格生成后处理。

## 创新定位 / Novelty statement

模型组件所借鉴的等变消息传递、条件流匹配和 CHGNet 均有独立来源。本项目的贡献是：

- 为 MXene NFE 建立包含物理分量和证据来源的专用学习目标；
- 在同一语义下联合正向 NFE 预测与目标条件结构生成；
- 将端基、上下表面、hollow 配位、OH 和层序纳入生成表示；
- 使用独立预测、OOD、重复检测与 ML 势构成生成接受协议。

因此，推荐将创新描述为 **domain-specific formulation and integrated
physics-gated workflow**，而非新的通用 GNN、通用流模型或通用原子势。

## 局限与风险 / Limitations and risks

- 伪标签偏差会被模型继承；
- medium 类占比 81.4%，类别边界受不平衡影响；
- 对训练集中未覆盖的元素、层数、端基或大晶胞 OOD；
- CHGNet 的训练域与目标 MXene/端基不完全一致；
- manifold generator 的未见组合仍复用已见局部模板，不代表真正自由生成；
- PyTorch checkpoint 只应从可信来源加载；
- 模型生成结果可能在 VASP 弛豫中重构或坍塌。

## 正确解释 / Correct interpretation

“high 概率 0.9”表示模型在其训练分布和伪标签体系下更像 high，不表示 90% 的实验成功率。
应同时查看 MC 不确定性、OOD、几何、CHGNet、重复情况及最终 DFT。

## 建议报告规范 / Recommended reporting

使用模型发表结果时，至少报告：

1. 检查点哈希、数据 schema 和 group-aware split；
2. 三档全部概率、校准方式、MC 次数和 OOD 风险；
3. 生成数量、过采样数及每类拒绝原因；
4. CHGNet 优化设置、最终最大力及其仅作为预筛选的声明；
5. StructureMatcher 参数和训练集重复结果；
6. VASP 赝势、泛函、ENCUT、k 点、真空、偶极修正和收敛标准；
7. 最终 band-decomposed charge density、有效质量和 ELF/电荷证据。
