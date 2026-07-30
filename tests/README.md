# 测试 / Tests

作者 / Author: Ruck  
生成时间 / Generated: 2026-07-30

- `test_smoke.py`：图、模型、训练/生成关键路径；
- `test_manifold_generation.py`：流形投影和未见组合；
- `test_windows_preview.py`：原子、周期键、幽灵像和晶胞场景。

```bash
python -m unittest discover -s tests
```

部分测试需要安装 PyTorch、pymatgen、CHGNet 或模型资源；缺失可选依赖时应明确 skip，
不能把未运行当作通过。

