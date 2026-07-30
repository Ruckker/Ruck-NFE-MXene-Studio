# 118 字段字典 / 118-Column Data Dictionary

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

布尔字段可由 CSV 解析为 bool；空值表示源文件缺失、该物理量不适用或解析失败。
Boolean fields may be parsed as bool. Missing values mean unavailable,
inapplicable, or failed-to-parse source information.

## A. 标识、组成与划分（1–16）

| 字段 | 含义 / Meaning |
|---|---|
| `Structure_Name` | 规范结构名 / canonical structure name |
| `Source_Directory` | 上游 VASP 目录 / upstream VASP directory |
| `Metal_Top`, `Metal_Bottom` | 两种金属位点 / two metal-site elements |
| `X_Element` | 核心 C/N 元素 / core C or N element |
| `Termination_Top`, `Termination_Bottom` | 上下表面端基 / top and bottom terminations |
| `Stacking_Top`, `Stacking_Bottom` | 上下堆垛标签 / top and bottom stacking labels |
| `Name_Parse_OK` | 名称能否按规则解析 / whether name parsing succeeded |
| `Split_Group` | 防泄漏结构家族键 / leakage-safe structure-family key |
| `Suggested_Split` | train/validation/test |
| `Formula`, `Reduced_Formula` | 完整与约化化学式 / full and reduced formula |
| `Elements` | 元素集合 / element set |
| `N_Elements`, `N_Atoms` | 元素数与原子数 / element and atom counts |

## B. 晶格、slab 与真空（17–31）

| 字段 | 单位 | 含义 |
|---|---:|---|
| `Lattice_a_A`, `Lattice_b_A`, `Lattice_c_A` | Å | 晶格长度 |
| `Lattice_alpha_deg`, `Lattice_beta_deg`, `Lattice_gamma_deg` | degree | 晶格角 |
| `Cell_Volume_A3` | Å³ | 晶胞体积 |
| `InPlane_Area_A2` | Å² | 面内面积 |
| `Min_Interatomic_Distance_A` | Å | 周期最小原子距 |
| `slab_low_frac`, `slab_high_frac` | fraction | 解包后的 slab z 边界 |
| `vacuum_fraction` | fraction | c 方向最大真空间隙比例 |
| `slab_thickness_A` | Å | slab 厚度 |
| `vacuum_thickness_A` | Å | 真空厚度 |

## C. 静态计算质量与全局物性（32–43）

| 字段 | 含义 |
|---|---|
| `Total_Energy_eV`, `Energy_per_Atom_eV` | 总能与每原子能；不同组成不可直接比较稳定性 |
| `Fermi_Level_eV` | OUTCAR 费米能级 |
| `NELECT`, `OUTCAR_NIONS`, `NELM` | 电子数、离子数、最大电子步 |
| `Run_Complete` | 静态计算正常结束 |
| `Electronic_Converged` | 电子收敛 |
| `Severe_VASP_Errors` | 严重 VASP 错误摘要 |
| `VASP_Warning_Count` | 静态 warning 数 |
| `Total_Mag_muB` | 总磁矩 |
| `SCF_Steps` | SCF 步数 |

## D. 能带与带边（44–60）

| 字段 | 含义 |
|---|---|
| `Band_Run_Complete`, `Band_Electronic_Converged` | 能带任务完成/收敛 |
| `Band_Severe_VASP_Errors`, `Band_VASP_Warning_Count` | 能带错误与 warning |
| `VBM_Up_eV`, `CBM_Up_eV`, `Band_Gap_Up_eV` | 自旋上带边 |
| `VBM_Down_eV`, `CBM_Down_eV`, `Band_Gap_Down_eV` | 自旋下带边 |
| `Band_Gap_eV`, `Is_Metal` | 汇总带隙与金属标志 |
| `VBM_Relative_EF_eV`, `CBM_Relative_EF_eV` | 相对费米能带边 |
| `Static_NKPoints`, `Static_NBands` | 静态任务 k 点/能带数 |
| `N_Spin_Channels` | 自旋通道数 |

## E. DOS（61–66）

| 字段 | 含义 |
|---|---|
| `DOSCAR_Fermi_Level_eV` | DOSCAR 费米能 |
| `DOS_at_EF_States_per_eV` | EF 总 DOS |
| `DOS_at_EF_per_Atom` | 每原子 EF DOS |
| `DOS_at_EF_Up`, `DOS_at_EF_Down` | 自旋分辨 EF DOS |
| `DOS_Spin_Polarization_at_EF` | EF 自旋极化 |

## F. NFE 标签、候选带与分数（67–94）

| 字段 | 含义 |
|---|---|
| `NFE_Pseudo_Score` | 0–1 连续 NFE 伪分数，主要回归目标 |
| `NFE_Pseudo_Label` | low/medium/high，主要分类目标 |
| `NFE_Label_Is_Ground_Truth` | 是否真值；当前任务强调为伪标签 |
| `NFE_Candidate_Spin` | 最佳候选带自旋 |
| `NFE_Candidate_Band_Index` | 候选带索引 |
| `NFE_Energy_at_Gamma_eV` | Γ 点候选带能量 |
| `NFE_Energy_Relative_EF_eV` | 候选带相对 EF |
| `NFE_Occupation_at_Gamma` | Γ 点占据 |
| `NFE_Atomic_Projection_Total` | 总原子轨道投影 |
| `NFE_Atomic_Projection_s/p/d` | s/p/d 分解投影 |
| `NFE_Effective_Mass_KG_me`, `NFE_Effective_Mass_GM_me` | Γ–K、Γ–M 有效质量（`m_e`） |
| `NFE_Effective_Mass_Geomean_me` | 两方向几何平均 |
| `NFE_Mass_Anisotropy` | 有效质量各向异性 |
| `NFE_Parabolic_R2_KG`, `NFE_Parabolic_R2_GM` | 抛物线拟合 R² |
| `NFE_Parabolic_RMSE_KG_eV`, `NFE_Parabolic_RMSE_GM_eV` | 抛物线拟合 RMSE |
| `NFE_Score_Projection_Component` | 低原子投影分量 |
| `NFE_Score_Parabola_Component` | 抛物线色散分量 |
| `NFE_Score_Energy_Component` | 能量位置分量 |
| `NFE_Score_Mass_Component` | 自由电子质量接近度分量 |
| `NFE_Score_Isotropy_Component` | 面内各向同性分量 |
| `NFE_Candidate_Count` | 通过预筛的候选带数 |
| `Band_NKPoints`, `Band_NBands` | 能带路径 k 点数和能带数 |

## G. 真空势与功函数（95–103）

| 字段 | 单位/含义 |
|---|---|
| `Vacuum_Level_Top_eV`, `Vacuum_Level_Bottom_eV` | 上下真空平台 |
| `Work_Function_Top_eV`, `Work_Function_Bottom_eV` | 上下功函数 |
| `Work_Function_Mean_eV` | 平均功函数 |
| `Vacuum_Potential_Asymmetry_eV` | 两侧真空势差 |
| `Vacuum_Field_Top_eV_per_A`, `Vacuum_Field_Bottom_eV_per_A` | 真空残余场 |
| `Work_Function_Reliable` | 真空平台是否足够可靠 |

## H. ELF 与电荷空间分布（104–112）

| 字段 | 含义 |
|---|---|
| `ELF_Surface_Top_Mean/Max` | 上表面 ELF 均值/最大值 |
| `ELF_Surface_Bottom_Mean/Max` | 下表面 ELF 均值/最大值 |
| `ELF_Deep_Vacuum_Mean` | 深真空 ELF |
| `Charge_Surface_Top_Fraction` | 上表面电荷比例 |
| `Charge_Surface_Bottom_Fraction` | 下表面电荷比例 |
| `Charge_Deep_Vacuum_Fraction` | 深真空电荷比例 |
| `Charge_Surface_Total_Fraction` | 两表面总电荷比例 |

## I. 质量、版本与文件（113–118）

| 字段 | 含义 |
|---|---|
| `Data_Quality_Score` | 记录级质量分 |
| `Quality_Warnings` | 软提示，非硬失败 |
| `Hard_Failure_Reasons` | 硬失败；主表应为空，dirty 表记录原因 |
| `Extraction_Schema_Version` | 当前 `nfe-v1.0` |
| `Extraction_UTC` | 抽取时间 |
| `File_Path` / dirty 中的 `Dirty_File_Path` | 复制后的训练/脏结构路径 |

注意：主表共 118 列，最后一项在主表为 `File_Path`；`dirty_manifest.csv` 使用
`Dirty_File_Path` 替代它，因此同样是 118 列。
