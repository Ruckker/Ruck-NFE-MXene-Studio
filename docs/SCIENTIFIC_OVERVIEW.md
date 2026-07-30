# MXene 近自由电子态智能设计：科学定位与方法论
# Scientific Positioning and Methodology for NFE-Aware MXene Design

作者 / Author: **Ruck**  
生成时间 / Generated: **2026-07-30 15:30:00 Asia/Shanghai**

## 摘要 / Abstract

近自由电子态（nearly free electron, NFE）为研究二维表面离域电子、低散射输运和
表面电子结构调控提供了独特平台。MXene 的金属层、C/N 核心、层数、堆垛和表面端基
共同决定表面偶极、真空势、功函数及 NFE 候选能带相对费米能级的位置，使其成为具有
明确物理意义但组合空间复杂的逆向设计问题。

NFE MXene Studio 建立了一个以第一性原理数据为起点、以受约束候选生成为中间环节、
以新一轮 DFT 验证为终点的闭环框架。项目首先从已弛豫结构的 VASP 静态、能带、DOS、
LOCPOT、ELFCAR 和 CHGCAR 等文件中提取 118 个字段，并以投影、抛物线色散、能量位置、
有效质量和各向同性构造可审计的 NFE 伪标签。随后，周期旋转等变图网络完成结构级
三分类、多任务回归、概率校准和 OOD 识别；表面感知条件流模型根据目标 NFE 档位和
MXene 骨架生成结构；最终通过表面拓扑、流形投影、CHGNet、重复检测和独立预测器复评
形成 VASP 前物理门控。

The central contribution is not a replacement for density-functional theory,
but an NFE-specific hypothesis engine that reduces the number of indiscriminate
calculations while preserving a traceable path from source calculations to every
prediction and generated candidate.

## 1. 物理背景 / Physical background

在理想自由电子模型中，电子色散近似抛物线。材料中的 NFE 态虽然受到周期势调制，
但仍表现出较弱的离子核局域化、较小且较各向同性的面内有效质量，以及近抛物线的能带。
对二维材料而言，具有研究价值的 NFE 态还应满足两个空间特征：

- 电子密度主要位于原子层外侧或层间真空区域；
- 波函数沿二维表面方向延展，而非局域于特定原子轨道。

MXene 表面端基能够改变表面偶极和功函数，因此可显著移动 NFE 候选带相对费米能级的
位置。经典第一性原理研究指出，部分 OH 终止 MXene 的 NFE 态可接近甚至穿过费米能级，
并可能形成低核散射的面内传输通道。由此产生的科学问题不是“是否存在一个 NFE MXene”
这么简单，而是：

> 在多金属、多核心、多层数、多堆垛和多端基空间中，哪些弛豫结构具有目标强度的 NFE
> 特征，以及如何以最低第一性原理成本找到它们？

## 2. 领域研究空白 / Domain research gap

### 2.1 从个案机理到规模化设计之间的缺口

已有 NFE MXene 研究以高质量 DFT 个案和机理分析为主。这类工作适合确认特定结构的
电子态性质，但难以直接扩展到数万结构的统一比较，因为：

- 不同工作可能采用不同的候选带判定、k 点路径、真空层和投影证据；
- 只报告最终“有/无 NFE”会丢失有效质量、投影、能量位置与各向同性等连续信息；
- 原始计算文件、收敛状态、警告与失败原因通常不在统一机器可读模式中。

### 2.2 从通用 MXene 机器学习到 NFE 专属学习之间的缺口

近期 MXene 机器学习已覆盖稳定性、催化、力学、合成和若干通用电子性质，但 NFE 任务
同时要求能带曲率、表面外电子分布、功函数/真空势和端基几何。仅使用组成描述符或通用
表格回归很难表达这些耦合，也不能自然处理 CIF/POSCAR 的旋转与周期对称性。

### 2.3 从正向预测到目标导向逆向设计之间的缺口

结构性质模型回答的是

\[
\text{structure}\longrightarrow p(y_\mathrm{NFE}\mid\text{structure}),
\]

而材料设计真正希望求解：

\[
\text{target NFE}+\text{chemical constraints}
\longrightarrow \{\text{candidate structures}\}.
\]

后一问题是一对多、带几何约束且存在化学分布偏移的逆问题。单纯对数据库排序无法提出
数据库以外的候选；不受约束的通用生成器又容易产生层塌缩、端基缺失或错误配位。

### 2.4 从“CIF 可写出”到“值得做 VASP”之间的缺口

生成模型的数学有效样本不等于物理有效材料。对 MXene，还必须检查 slab 中心、真空层、
上下表面、层序、端基种类、hollow 位点、OH 键和金属–端基距离。即使几何合理，仍可能
与训练结构重复、位于高 OOD 区域、与目标 NFE 档位不一致，或在快速弛豫中产生大残余力。

### 2.5 项目对空白的定位边界

本项目在所覆盖的公开文献中未发现同时公开以下完整链路的 NFE MXene 工具：

1. NFE 证据因子化的可审计大规模数据；
2. CIF/POSCAR 到校准 NFE 概率、连续分数和 OOD 的正向模型；
3. 目标 NFE 档位到带端基 MXene 结构的逆向生成；
4. 面向二维端基拓扑的硬约束和 ML 势预弛豫；
5. 从 HPC 训练到桌面批量使用的可复现交付。

这是一项**项目定位陈述**，不是经过系统综述、专利检索和全部私有工作的“全球首创”
断言。更严格的优先权主张需要独立文献计量和同行评议。

## 3. 任务形式化 / Problem formulation

### 3.1 NFE 伪标签

对每个已通过质量门的结构 \(X\)，构造五个归一化物理分量：

\[
s_\mathrm{proj},\quad
s_\mathrm{para},\quad
s_\mathrm{energy},\quad
s_\mathrm{mass},\quad
s_\mathrm{iso}.
\]

它们分别表示低原子投影、Γ 附近抛物线质量、相对费米能位置、自由电子质量接近度和
面内各向同性。连续伪分数可抽象写成：

\[
S_\mathrm{NFE}(X)=
\sum_k w_k s_k(X),\qquad S_\mathrm{NFE}\in[0,1].
\]

具体权重和阈值由数据提取配置定义；表中保留每个 \(s_k\)，从而使最终标签可以追溯，
也允许未来使用 band-decomposed charge density 真值重新校准。

### 3.2 正向多任务学习

预测器学习：

\[
f_\theta(X)=
\left[
p_\theta(y_\mathrm{low},y_\mathrm{medium},y_\mathrm{high}\mid X),
\hat S_\mathrm{NFE},
\hat{\boldsymbol q},
\boldsymbol z
\right],
\]

其中 \(\boldsymbol q\) 包括相对费米能、原子投影、有效质量、功函数、带隙、DOS、
表面 ELF 和电荷分数等辅助目标，\(\boldsymbol z\) 是用于 OOD 距离的图嵌入。

多任务训练的目的不只是提高单一分数，而是促使共享表示同时解释 NFE 相关电子证据。
辅助标签缺失时使用掩码损失，不把缺失值错误当作零。

### 3.3 表面感知条件流

生成器在噪声状态 \(x_0\) 与真实结构状态 \(x_1\) 之间构造概率路径，例如：

\[
x_t=(1-t)x_0+t x_1,\qquad
v^\*(x_t,t)=x_1-x_0.
\]

网络 \(v_\phi\) 通过条件流匹配学习速度场：

\[
\mathcal L_\mathrm{FM}
=
\mathbb E_{t,x_0,x_1}
\left\|v_\phi(x_t,t,c)-v^\*(x_t,t)\right\|^2,
\]

条件 \(c\) 包含 NFE 档位、连续分数、组成、层/表面角色和模板信息。生成时积分 ODE，
并用 classifier-free guidance 调整条件强度。实际损失还包含 endpoint、core、surface、
pair、layer、anchor、OH 和 repulsion 项。

### 3.4 严格接受集合

最终导出结构不是所有生成样本，而是交集：

\[
\mathcal A=
\mathcal G_\mathrm{geometry}
\cap\mathcal G_\mathrm{surface}
\cap\mathcal G_\mathrm{force}
\cap\mathcal G_\mathrm{target}
\cap\mathcal G_\mathrm{OOD}
\cap\mathcal G_\mathrm{novelty}.
\]

这里分别表示基础几何、MXene 表面拓扑、CHGNet 最大力、目标 NFE 匹配、分布风险和
训练集/候选去重。任一条件失败都会写入 `run_info.json`，而不是静默输出。

## 4. 方法贡献 / Methodological contributions

### 4.1 可审计 NFE 表型而非单一标签

118 字段把结构、收敛、能带、投影、有效质量、功函数、DOS、ELF、电荷和质量标记放在
同一数据记录中。该模式支持模型训练，也支持材料学家回到原始证据审查异常样本。

### 4.2 适用于二维周期 slab 的结构表示

预测器保留三维周期晶体的等变局部环境；生成器则显式使用二维周期 XY 邻域和真空方向
层信息。两种图定义服务于不同任务，避免用一个通用邻居假设处理所有物理问题。

### 4.3 端基被视为结构自由度而非后处理标签

端基影响 NFE、功函数和表面配位，因此不允许用户任意强制。模型将上下端基、OH 内部
键和吸附锚点纳入状态及损失，并在导出前验证端基是否处于训练分布。

### 4.4 生成多样性与物理可信度解耦

条件流负责提出多样候选；流形投影和表面规则约束结构；CHGNet 估计局部力学合理性；
NFE 预测器只负责目标电子表型复评。不同模型各自承担清晰职责，减少“一个网络既生成
又自我证明”的闭环偏差。

### 4.5 证据分层

| 证据层 | 来源 | 可以支持的结论 |
|---|---|---|
| L0 | CIF/POSCAR 可解析 | 文件语法有效 |
| L1 | 几何与表面拓扑规则 | 与训练 MXene 层/端基模式一致 |
| L2 | CHGNet 固定晶胞弛豫 | 在该 ML 势近似下残余力较低 |
| L3 | NFE predictor + MC/OOD | 在当前伪标签与训练分布下像目标 NFE 档位 |
| L4 | 收敛 VASP 能带、投影、ELF/电荷 | 第一性原理电子结构证据 |
| L5 | 稳定性与实验表征 | 可合成性、稳定性和器件性质证据 |

项目自动化到 L3；L4–L5 必须由后续研究完成。

## 5. 定量结果 / Quantitative evidence

### 5.1 数据

- 15,206 个通过硬质量门的已弛豫 MXene；
- 72 个隔离脏结构；
- 118 个字段；
- low/medium/high = 764/12,383/2,059；
- group-aware train/validation/test = 12,193/1,499/1,514。

### 5.2 预测器独立测试集

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.8780 |
| Balanced accuracy | 0.7484 |
| Macro F1 | 0.7340 |
| Macro ROC-AUC | 0.9201 |
| Calibrated ECE | 0.0137 |
| NFE pseudo-score MAE / RMSE | 0.0349 / 0.0467 |
| Low F1 / recall | 0.5000 / 0.4941 |
| Medium F1 / recall | 0.9231 / 0.9114 |
| High F1 / recall | 0.7790 / 0.8396 |

总体 accuracy 受 medium 类占比影响，因此 balanced accuracy、macro F1 和逐类召回更适合
衡量科学筛选能力。low 类仍是主要误差来源。

### 5.3 表面生成器独立测试集

| 指标 | 数值 |
|---|---:|
| Endpoint RMSE | 0.4472 Å |
| Core MAE | 0.2779 Å |
| Surface MAE | 0.1981 Å |
| Lattice loss | 0.1138 |
| Layer loss | 0 |
| OH loss | 0.001325 |

这些是监督重建/速度学习相关指标，不是 DFT 稳定率或实验成功率。最终输出依赖流形投影、
CHGNet 和严格筛选，后续研究还应报告“候选进入 DFT 后的弛豫存活率”。

## 6. 能解决的问题 / Problems addressed

1. **高通量 NFE 初筛：** 对批量 CIF/POSCAR 进行一致、快速的 NFE 优先级排序。
2. **计算队列压缩：** 把有限 VASP 资源集中到低 OOD、高目标概率且几何合理的候选。
3. **目标驱动结构假设：** 针对 low/medium/high 和指定 MXene 骨架提出新 CIF/POSCAR。
4. **端基与结构耦合建模：** 避免把端基当作与几何无关的离散标签。
5. **数据治理：** 识别未收敛、缺文件、解析异常和不可靠辅助性质。
6. **可重复研究：** 保存 split、配置、检查点、指标、拒绝原因、环境和归档哈希。

## 7. 不能解决的问题 / Limitations

1. NFE 标签是物理启发式伪标签，而非 band-decomposed charge density 人工真值。
2. 数据类别不平衡，low 边界分类能力有限。
3. group-aware split 降低家族泄漏，但不等于所有组成/拓扑外推都可靠。
4. CHGNet 并非针对本数据集重新拟合的 MXene 专用势。
5. 固定晶胞预弛豫不能证明晶格应力、声子、热力学或动力学稳定性。
6. 生成器在已弛豫模板附近探索，创新性与物理合理性之间存在主动折衷。
7. 实际样品常含混合端基、缺陷、吸附、氧化和环境效应，当前理想结构模型未完全覆盖。

## 8. 可检验研究假设 / Testable hypotheses

- **H1：** 分解后的 NFE 辅助物理量能够比单一分类标签提供更可迁移的结构表示。
- **H2：** 表面模板损失和流形投影能够提高生成候选的 DFT 弛豫存活率。
- **H3：** 校准概率、MC 不确定性和 embedding OOD 的联合阈值能够提高 VASP 队列的命中率。
- **H4：** 在固定骨架下，端基/表面偶极的选择对目标 NFE 档位的贡献高于小幅核心坐标扰动。
- **H5：** 对少量 band-decomposed charge density 真值进行主动学习校准，可显著改善 low 类召回。

这些假设应通过消融、主动学习和盲测 DFT 队列验证，而不应仅用当前训练指标证明。

## 9. 推荐后续研究 / Recommended research program

1. 从三档和 OOD 区间分层抽样，建立 band-decomposed charge density 人工真值集；
2. 对生成候选执行统一参数 VASP 全弛豫，报告存活率、重构类型和力/能误差；
3. 增加形成能、凸包、声子和 AIMD，建立“电子目标 + 稳定性”的多目标设计；
4. 显式建模混合端基、缺陷、覆盖度和实验环境；
5. 用 ensemble/深度核或 conformal prediction 建立外推覆盖保证；
6. 进行 predictor、auxiliary targets、surface losses、manifold projection 和 CHGNet 的消融；
7. 将新 DFT 结果回流数据集，形成不确定性驱动的主动学习循环。

## 10. 关键文献 / Selected references

1. Khazaei, M. *et al.* “Nearly free electron states in MXenes.”
   *Physical Review B* **93**, 205125 (2016).
   [DOI: 10.1103/PhysRevB.93.205125](https://doi.org/10.1103/PhysRevB.93.205125)
2. “Modulation of nearly free electron states in hydroxyl-functionalized
   MXenes: a first-principles study.” *Journal of Materials Chemistry C* (2020).
   [DOI: 10.1039/C9TC06837F](https://doi.org/10.1039/C9TC06837F)
3. “First-principles and machine-learning approaches for interpreting and
   predicting the properties of MXenes.” *npj 2D Materials and Applications*
   (2025). [DOI: 10.1038/s41699-025-00529-5](https://doi.org/10.1038/s41699-025-00529-5)
4. Schütt, K. T., Unke, O. T. & Gastegger, M. “Equivariant message passing
   for the prediction of tensorial properties and molecular spectra.”
   *ICML/PMLR* **139**, 9377–9388 (2021).
   [PMLR](https://proceedings.mlr.press/v139/schutt21a.html)
5. Lipman, Y. *et al.* “Flow Matching for Generative Modeling.” *ICLR* (2023).
   [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
6. Deng, B. *et al.* “CHGNet as a pretrained universal neural network
   potential for charge-informed atomistic modelling.”
   *Nature Machine Intelligence* **5**, 1031–1041 (2023).
   [DOI: 10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3)

## 11. 正确引用本项目 / Citing this project

请使用仓库根目录 [`CITATION.cff`](../CITATION.cff)，并在论文中分别说明：

- 数据标签为 NFE pseudo-label；
- 预测器用于候选排序；
- 生成结构经过哪些 L0–L3 门控；
- 最终材料结论采用何种 DFT/实验方法验证。

