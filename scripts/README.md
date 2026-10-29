# 数据可视化

在项目根目录、已安装 NumPy 和 Matplotlib 的环境中运行。不需要 PyTorch、GPU 或图形桌面。

```bash
# 默认读取 /data/jl/data/neusis，预览所有数据集
python scripts/visualize_data.py

# 只预览一个数据集
python scripts/visualize_data.py --dataset 14deg_submarine

# 使用项目 data 目录，并指定预览帧数和输出目录
python scripts/visualize_data.py --data-root ./data --samples 6 --output-dir ./outputs/data_preview
```

每个数据集在 `outputs/data_preview/<dataset>/` 下生成：

- `sonar_samples.png`：均匀抽取的声呐图像，上行为原始数据，下行为与 `load_data.py` 一致的训练预处理（小于 0.2 置零、最后 200 行置零）。横轴为方位角，纵轴为距离，各子图共用强度色标。
- `trajectory.png`：所有帧的传感器位置，包含三维视图和 XY 投影。按文件名自然排序连接，颜色表示帧序号；该连线表示文件顺序，不是额外估计的运动轨迹。
- `summary.json`：帧数、图像尺寸、声呐参数、抽样文件名及位置范围。

脚本逐帧读取，仅保留抽样图像和位置，不修改原始数据，也不调用训练数据加载器。原始格式为 pickle，只应读取可信数据。可以直接在 VS Code 中打开生成的 PNG。
