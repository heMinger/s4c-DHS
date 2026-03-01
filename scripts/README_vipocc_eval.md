# ViPOcc-Style 3D Occupancy Evaluation for DINO-DA3

本文档说明如何使用 ViPOcc 的评估协议来评估 DINO-DA3 模型的 3D 占用预测能力。

## 📊 评估指标说明

| 指标 | 含义 |
|------|------|
| **Scene O_acc** | 场景整体占用准确率 (Overall Accuracy) |
| **Scene IE_acc** | 不可见区域占用准确率 (Invisible Empty Accuracy) |
| **Scene IE_rec** | 不可见空白区域召回率 (Invisible Empty Recall) |

## 🎯 Baseline 对比 (ViPOcc 论文结果)

| 方法 | Scene O_acc | Scene IE_acc | Scene IE_rec |
|------|------------|--------------|--------------|
| **ViPOcc** | 92.7% | 71.3% | 68.6% |
| **BTS** | 92.3% | 69.0% | 64.4% |
| **DINO-DA3** | ? | ? | ? |

> 评估配置: z_range=[20, 4], x_range=[-4, 4], ppm=10

## 🚀 快速开始

### 方法1: 使用独立脚本 (推荐)

```bash
cd /home/lmh/dino-da3-adaptive-sampling1

# 基础评估
python scripts/eval_vipocc_style.py \
    --checkpoint out/kitti_360/kitti_360_backend-None-1_20260210-203842/training_checkpoint_151000-bp.pt \
    --data_path /data/lmh_data/KITTI360 \
    --z_range 20 4 \
    --ppm 10

# 带可视化的评估
python scripts/eval_vipocc_style.py \
    --checkpoint out/kitti_360/kitti_360_backend-None-1_20260210-203842/training_checkpoint_151000-bp.pt \
    --data_path /data/lmh_data/KITTI360 \
    --z_range 20 4 \
    --save_vis \
    --save_dir visualization/dino_da3_occ

# 使用更远的评估范围 (与 ViPOcc 50m 设置对比)
python scripts/eval_vipocc_style.py \
    --checkpoint out/kitti_360/kitti_360_backend-None-1_20260210-203842/training_checkpoint_151000-bp.pt \
    --data_path /data/lmh_data/KITTI360 \
    --z_range 50 4 \
    --ppm 10
```

### 方法2: 使用配置文件

```bash
cd /home/lmh/dino-da3-adaptive-sampling1

# 修改 configs/eval_vipocc_occ.yaml 中的 checkpoint 路径后运行
python eval.py -cn eval_vipocc_occ
```

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | (必需) | 模型检查点路径 |
| `--data_path` | `/data/lmh_data/KITTI360` | KITTI-360 数据集路径 |
| `--z_range` | `[20, 4]` | 深度范围 (远, 近) 米 |
| `--x_range` | `[-4, 4]` | 水平范围 米 |
| `--ppm` | `10` | 每米采样点数 (分辨率 = 1/ppm) |
| `--aggregate_timesteps` | `300` | GT 聚合的 LiDAR 帧数 |
| `--save_vis` | `False` | 是否保存可视化 |
| `--gt_occ_path` | `""` | 预计算 GT 路径 (可选加速) |

## 📐 评估范围配置

### 标准配置 (与 ViPOcc 对齐)
```yaml
z_range: [20, 4]    # 4-20m 深度范围
x_range: [-4, 4]    # ±4m 水平范围
ppm: 10             # 0.1m 分辨率
```

### 远距离配置
```yaml
z_range: [50, 4]    # 4-50m 深度范围
x_range: [-4, 4]    # ±4m 水平范围
ppm: 10             # 0.1m 分辨率
```

## 🔍 可视化输出

启用 `--save_vis` 后，会在 `save_dir` 目录下生成:
- `{seq}_{frame_id}_occ.png`: 包含输入图像、GT占用、预测占用、可见性图

## 📁 输出目录结构

```
visualization/
└── occ_eval/
    ├── occ_results.txt          # 评估结果汇总
    ├── 0000_0000000xxx_occ.png  # 可视化图像
    └── ...
```

## ⚠️ 注意事项

1. **首次运行较慢**: 需要从 300 帧 LiDAR 数据生成 GT 占用图
2. **预计算 GT**: 可以先生成并保存 GT 占用图加速后续评估
3. **内存消耗**: `query_batch_size` 可调整以适应 GPU 内存

## 🔧 预计算 GT 占用图 (可选)

如果需要多次评估，可以先预计算 GT:

```python
# 在 eval_vipocc_style.py 中设置
save_gt_occ_map_path = "data/KITTI-360/GT_Occ"  # GT 保存路径
```

之后评估时使用:
```bash
python scripts/eval_vipocc_style.py \
    --checkpoint your_checkpoint.pt \
    --gt_occ_path data/KITTI-360/GT_Occ
```

## 📖 参考

- [ViPOcc Paper](https://arxiv.org/abs/2412.11210)
- [ViPOcc Code](https://github.com/mias-group/ViPOcc)
