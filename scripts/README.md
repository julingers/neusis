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

## 极坐标扇形显示

```bash
python scripts/visualize_polar.py --dataset 14deg_submarine
python scripts/visualize_polar.py --dataset 14deg_planeFull --frame 100.pkl
python scripts/visualize_polar.py --dataset all --samples 4
```

默认输出到 `outputs/polar_preview/<dataset>/polar_samples.png`，指定单帧时为 `polar_<文件名>.png`。
每行对应一帧，三列依次是原始 r-theta 矩形图、原始扇形图、训练预处理后的扇形图，共用强度色标。
默认按文件名自然排序等间隔选择 4 帧（不是随机抽样），也支持 `--data-root` 和 `--output-dir`。

扇形采用配置中的水平方位角和距离范围，以 `x=r*sin(theta), y=r*cos(theta)` 绘制每个像素单元，保持米制等比例，不对强度插值。青色三角标记声呐位置，灰色线为距离圆弧和方位参考线。14°/28° 是垂直视场角，不是扇形的水平张角。
该图以零俯仰角展示距离和方位，未应用 PoseSensor，不能当作真实世界水平截面或三维点云。
