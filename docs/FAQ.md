# 常见问题 / FAQ

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 1. 为什么不用 CPU 版 PyTorch？

最终训练需要 1–4 张 GPU，服务器参考是 `torch 2.6.0+cu118`。CPU 版不能提供 CUDA
训练。安装后检查 `torch.version.cuda`、`torch.cuda.is_available()` 和 GPU 数。

## 2. 驱动显示 CUDA 11.6，而 PyTorch 是 cu118，会冲突吗？

`nvidia-smi` 的 CUDA 字段是驱动接口能力提示，PyTorch wheel 自带 CUDA runtime。
是否实际兼容应由 CUDA 张量测试决定。最终环境已成功识别 4 张 RTX 3090 并完成训练。

## 3. warnings=12246 是不是数据几乎都坏了？

不是。它统计软质量提示，72 条硬失败才进入 dirty。参见 `data/README.md`。

## 4. 哪个字段直接对应 NFE？

分类：`NFE_Pseudo_Label`；连续强度：`NFE_Pseudo_Score`。相关物理解释字段以
`NFE_` 开头。

## 5. 为什么模型生成后还要流形投影和 CHGNet？

生成器测试端点 RMSE 仍约 0.447 Å。流形投影保持层和端基，CHGNet 降低局部力，
二者能减少 VASP 前明显不合理结构，但不能保证 DFT 稳定。

## 6. 为什么不允许用户指定端基？

端基与 NFE、表面配位和稳定性强耦合。任意指定会导致 OOD。用户只指定核心和内层金属，
端基由模型/模板与严格筛选决定。

## 7. 可以生成训练集中没有的结构吗？

manifold generator 支持训练集中未见的金属组合替换，并检查结构不与训练集重复。但局部几何仍来自
训练模板先验，因此属于受约束的新颖性，不是无条件探索整个晶体空间。

## 8. Windows 为什么有 4 GB 多？

onedir 包含 PyTorch、CUDA DLL、pymatgen、CHGNet、Matplotlib、模型和权重。
ZIP 压缩后约 2.85 GB。不能只复制 EXE。

## 9. 预览能替代 VESTA 吗？

不能。预览面向快速旋转、周期键和晶胞检查。体数据、轨道、对称性和高级显示仍使用 VESTA。

## 10. 如何提高 low 类效果？

优先补充/人工确认 low 与 low-medium 边界样本，采用分组主动学习和真值微调；
不要只通过调整 class weight 追求表面指标。

## 11. 如何提高生成结构的 VASP 成功率？

记录每个拒绝原因和 VASP 失败模式；按端基/金属/层数分层分析；加入成功弛豫结构回流；
训练力/能量辅助模型或等变弛豫器；保持物理标准，不以允许 target mismatch 换取数量。

## 12. 可以直接发表模型预测的 high-NFE 结构吗？

不可以。必须完成 DFT 弛豫、收敛、能带、分波电荷密度、真空/偶极修正和稳定性验证。

