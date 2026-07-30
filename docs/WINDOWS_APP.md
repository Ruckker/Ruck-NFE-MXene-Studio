# Windows application 1.0 使用与构建 / Windows application 1.0 Usage and Build

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

## 用户功能 / End-user features

### 批量 NFE 预测

- 拖入一个或多个 CIF/POSCAR；
- 文件选择器支持多选；
- 目录导入会收集可识别结构文件；
- 结果同时给出 low/medium/high 三档概率和连续 NFE 分数；
- 显示置信度、MC 不确定性、OOD 和辅助物性；
- 结果可导出 CSV。

### 条件生成

用户只选择：

- 目标 NFE：低/中/高；
- 核心：C/N；
- 两种内层金属；
- 候选数等运行选项。

端基不可手工强制，由模型与训练模板分布决定。输出 CIF 和 POSCAR。

生成页提供 0–100% 确定型进度条，依次展示模板选择、流生成采样、几何/表面拓扑
筛选、周期图构建、初始 NFE 预测、CHGNet 固定晶胞预弛豫、弛豫后 NFE 复评、
训练集/候选去重和 CIF/POSCAR 导出。若第一次严格筛选不足而自动增加过采样，
总进度仍保持单调递增。

### 三维预览

- 批量导入默认预览第一个文件；
- 下拉框切换当前预览；
- 生成后下拉框切换候选；
- 鼠标拖动自由旋转；
- 滚轮缩放；
- 按钮重置视角；
- CPK 元素颜色和图例；
- 显示共价半径判定的键；
- 显示跨边界周期键与幽灵像；
- 显示 12 条晶胞边；
- 可切换完整晶胞显示。

它用于快速检查，不是 VESTA 的完整替代品；对称性、轨道、体数据和复杂超胞仍建议使用 VESTA。

## 安装最终程序 / Install the final program

解压：

`release_assets/windows/NFE_MXene_Studio_1.0_Windows_20260730.zip`

运行：

`NFE_MXene_Studio_1_0/NFE_MXene_Studio_1_0.exe`

注意：

- 仓库同时包含完整目录
  `release_assets/windows/NFE_MXene_Studio_1_0/`，可直接运行其中的同名 EXE；
- 保留整个目录；
- `_internal/` 不能移动或删除；
- 首次加载 CUDA/PyTorch/CHGNet 可能较慢；
- 没有可用 GPU 时部分预测可退回 CPU，但生成会显著变慢；
- 杀毒软件可能对大型未签名 PyInstaller 程序提示，请先核对 SHA256。

最终 ZIP64 SHA256 见同目录 `.manifest.json` 和 `SHA256SUMS.txt`。

## 源码结构 / Source layout

```text
app/windows/
├─ nfe_mxene_studio/
│  ├─ app.py                 # GUI、任务线程、表格、预览联动
│  ├─ backend.py             # 模型加载、预测、生成与导出
│  ├─ structure_preview.py   # 3D 场景与 Tk 画布
│  └─ smoke_test.py          # 无 GUI 冒烟测试
└─ packaging/
   ├─ build_windows.ps1
   ├─ NFE_MXene_Studio_1_0.spec
   └─ requirements-windows.txt
```

可重建源码包 `NFE_MXene_Studio_1.0_Source_20260730.zip` 额外包含
最终 `src/nfe_model/`、模型、元数据、样例与全部最终测试，不含旧生成器和旧测试。

## 重新构建 / Rebuild

参考最终兼容环境为 Python 3.9、PyTorch 2.0.1+cu118。普通依赖可用清华源：

```powershell
python -m pip install -r app\windows\packaging\requirements-windows.txt `
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

CUDA PyTorch 与 PyG 扩展的 `+cu118` wheel 通常不在普通 PyPI 镜像，应使用已经验证的
本地 wheel；不要让 pip 自动换成 CPU torch。

源码包布局就绪后：

```powershell
powershell -ExecutionPolicy Bypass -File app\windows\packaging\build_windows.ps1 `
  -Python C:\path\to\python.exe
```

最终 spec 收集：

- app 与 `nfe_model`；
- predictor/generator `.pt`；
- predictor/generator metrics 和表面 profile；
- pymatgen 数据；
- CHGNet 权重；
- Matplotlib/Tkinter/tkinterdnd2；
- PyTorch/CUDA DLL。

## 冒烟测试 / Smoke test

源码：

```powershell
python -m app.windows.nfe_mxene_studio.smoke_test
```

公开仓库采用 `app.windows.nfe_mxene_studio`，可重建源码 ZIP 为保持 PyInstaller
原始布局则采用 `windows_app`；代码中的兼容导入支持两种形式。

冻结程序支持自动 self-test，最终 1.0 已验证：

- low/medium/high 三个样例预测；
- 3D 场景原子、周期键、幽灵像和晶胞；
- 生成 ScTaCSeBr low 候选；
- CIF/POSCAR 再解析一致；
- slab center 0.5；
- CHGNet 最大力低于 0.05 eV/Å；
- 非训练集重复、低 OOD。

这些是软件冒烟证据，不是该材料的最终 VASP 证据。
