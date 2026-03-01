# 📊 S4C 项目架构设计报告

## 1. 项目概述

这是 **S4C (Self-Supervised Semantic Scene Completion with Neural Fields)** 项目，发表于 **3DV 2024 (Spotlight)**。该项目实现了一种自监督的语义场景补全方法，基于隐式神经场表示，能够从单张图像重建三维场景并预测语义分割。

### 核心创新点
- 使用 **DINOv2 ViT-L/14** 作为视觉特征提取骨干网络
- 集成 **Depth Anything 3 (DA3)** 深度先验进行自适应采样
- 通过 **体积渲染** 进行自监督训练，无需稠密 3D 标注

---

## 2. 项目目录结构

```
18-dino-da3/
├── configs/                    # 配置文件 (Hydra)
│   ├── default.yaml           # 默认训练参数
│   ├── exp_kitti_360.yaml     # KITTI-360 实验配置
│   └── data/
│       └── kitti_360.yaml     # KITTI-360 数据集配置
├── models/                     # 模型定义
│   ├── bts/                   # BTS (Behind The Scenes) 核心模型
│   │   ├── model/
│   │   │   ├── models_bts.py  # BTSNet 主网络 + DINOv2 backbone
│   │   │   ├── loss.py        # 损失函数
│   │   │   ├── ray_sampler.py # 光线采样器
│   │   │   └── image_processor.py
│   │   ├── trainer.py         # 训练逻辑
│   │   ├── evaluator.py       # 评估逻辑
│   │   └── evaluator_*.py     # 特定评估器
│   └── common/
│       ├── backbones/         # 编码器 backbone
│       ├── model/             # 通用模型组件 (MLP, 位置编码等)
│       └── render/
│           └── nerf.py        # NeRF 体积渲染器
├── datasets/                   # 数据集加载器
│   ├── kitti_360/             # KITTI-360 数据集
│   ├── kitti_raw/             # KITTI-Raw 数据集
│   └── ...                    # 其他数据集
├── utils/                      # 工具函数
│   ├── base_trainer.py        # 基础训练框架
│   ├── metrics.py             # 评估指标
│   └── projection_operations.py
├── scripts/                    # 脚本工具
│   ├── benchmarks/            # 基准测试
│   └── images/                # 图像推理
├── train.py                    # 训练入口
├── eval.py                     # 评估入口
└── dinov2_vitl14_reg4_pretrain.pth  # DINOv2 预训练权重
```

---

## 3. 核心架构设计

### 3.1 整体流程图

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 训练流程                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────┐    ┌────────────────┐    ┌──────────────────┐                     │
│  │ 输入图像 │───>│ DINOv2 ViT-L/14│───>│ PixelShuffle     │                     │
│  │(N,V,3,H,W)│   │ (Frozen)       │    │ Adapter + ResBlock│                     │
│  └──────────┘    └────────────────┘    └────────┬─────────┘                     │
│                                                  │                               │
│                                                  ▼                               │
│                                        ┌────────────────┐                        │
│                                        │ Feature Map    │                        │
│                                        │ (N,V,64,H,W)   │                        │
│                                        └───────┬────────┘                        │
│                                                │                                 │
│  ┌───────────┐    ┌─────────────────┐         │                                 │
│  │DA3 Depth  │───>│ 自适应采样       │         │                                 │
│  │Prior      │    │ z_samp 深度位置  │         │                                 │
│  └───────────┘    └────────┬────────┘         │                                 │
│                            │                   │                                 │
│                            ▼                   │                                 │
│                   ┌─────────────────┐         │                                 │
│                   │ 3D 采样点 (xyz) │<────────┘                                 │
│                   │ ray_o + z*ray_d │                                           │
│                   └────────┬────────┘                                           │
│                            │                                                     │
│         ┌──────────────────┼──────────────────┐                                 │
│         │                  │                  │                                 │
│         ▼                  ▼                  ▼                                 │
│  ┌───────────┐      ┌───────────┐      ┌────────┐                               │
│  │ 位置编码  │      │ 特征采样  │      │ Color  │                               │
│  │ (6 freqs) │      │grid_sample│      │ Sample │                               │
│  └─────┬─────┘      └─────┬─────┘      └───┬────┘                               │
│        │                  │                │                                     │
│        └────────┬─────────┘                │                                     │
│                 │ concat                   │                                     │
│                 ▼                          │                                     │
│        ┌─────────────────┐                 │                                     │
│        │   mlp_input     │                 │                                     │
│        │[features+code]  │                 │                                     │
│        └────────┬────────┘                 │                                     │
│     ┌───────────┴───────────┐              │                                     │
│     │                       │              │                                     │
│     ▼                       ▼              │                                     │
│ ┌──────────────┐   ┌───────────────┐       │                                     │
│ │  MLP Coarse  │   │MLP Segmentation│      │     ← 并行执行，共享输入            │
│ │  (σ density) │   │ (19 classes)   │      │                                     │
│ └──────┬───────┘   └───────┬───────┘       │                                     │
│        │                   │               │                                     │
│        └─────────┬─────────┴───────────────┘                                     │
│                  │                                                               │
│                  ▼                                                               │
│           ┌──────────────┐                                                       │
│           │ NeRF Volume  │  C(r) = Σ T_i · α_i · c_i                            │
│           │ Rendering    │  S(r) = Σ T_i · α_i · s_i                            │
│           └──────┬───────┘                                                       │
│                  │                                                               │
│       ┌──────────┼──────────┐                                                    │
│       ▼          ▼          ▼                                                    │
│ ┌──────────┐ ┌──────────┐ ┌──────────────┐                                      │
│ │ RGB 重建 │ │ 深度预测 │ │ 语义分割预测 │                                      │
│ └────┬─────┘ └────┬─────┘ └──────┬───────┘                                      │
│      │            │              │                                               │
│      ▼            ▼              ▼                                               │
│ ┌──────────────────────────────────────────┐                                    │
│ │            Loss Computation              │                                    │
│ │  • L1 + SSIM (RGB)                       │                                    │
│ │  • Cross Entropy (Segmentation)          │                                    │
│ │  • Edge-aware Smoothness                 │                                    │
│ └──────────────────────────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**流程说明**：
1. **DA3 自适应采样**：根据 DA3 深度先验决定沿光线的采样深度位置 `z_samp`
2. **3D 采样点生成**：`xyz = ray_origin + z_samp × ray_direction`
3. **特征获取**：xyz 坐标同时用于位置编码和从 Feature Map 中采样特征
4. **MLP 并行执行**：MLP Coarse (密度) 和 MLP Segmentation (分割) 共享输入，独立输出
5. **体积渲染**：使用 NeRF 公式聚合所有采样点的颜色、密度、分割结果

---

### 3.2 DINOv2 Backbone 架构

**核心代码位置**: `models/bts/model/models_bts.py`

```python
class BTSNet(torch.nn.Module):
    def __init__(self, conf):
        # ============== DINOv2 Backbone ==============
        self._init_dino_backbone()
        
        # ============== Complex Adapter with PixelShuffle ==============
        # PixelShuffle 进行 2x 上采样：1024 -> 256 -> 64
        self.dino_adapter = nn.Sequential(
            nn.Conv2d(DINO_EMBED_DIM, 256 * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),  # 1024 -> 256, 空间分辨率 2x
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),  # 残差块细化
            nn.Conv2d(256, DINO_CONTEXT_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(DINO_CONTEXT_DIM),
            nn.ReLU(inplace=True),
        )
```

**关键设计点**：

| 组件 | 配置 | 说明 |
|------|------|------|
| DINOv2 Model | `dinov2_vitl14_reg` | ViT-Large, patch=14, 带 register tokens |
| Embed Dim | 1024 | DINOv2 输出维度 |
| Adapter Output | 64 | 与原始 encoder 兼容 |
| 上采样 | PixelShuffle 2x | 从 patch 分辨率上采样 |

---

### 3.3 DA3 引导的自适应采样 (含置信度过滤)

**核心代码位置**: `models/common/render/nerf.py`, `models/da3_depth_generator.py`

```python
def sample_adaptive(self, rays, da3_depths, n_surface=16, n_global=8, 
                    absrel_prior=0.12, min_thickness=0.5, low_conf_mask=None):
    """
    Adaptive sampling based on DA3 depth prior with confidence filtering.
    
    Algorithm:
    1. Compute dynamic std: σ(d) = max(min_thickness, absrel_prior * d)
    2. Sample N_surface points in [d - 2σ, d + 2σ]
    3. Sample N_global points in [z_near, z_far] for background/foreground
    4. Merge and sort all samples
    5. For low-confidence rays (bottom 20%), use uniform sampling instead
    """
```

**采样策略**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DA3 置信度过滤采样策略                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  高置信度 (80%): 自适应采样围绕 DA3 深度                                     │
│                                                                             │
│         z_near                   DA3 深度 d                    z_far        │
│           │    ┌────────────────────┼────────────────────┐      │          │
│           │    │    表面采样区间     │                    │      │          │
│           │    │   [d-2σ, d+2σ]     │                    │      │          │
│           │    └────────────────────┼────────────────────┘      │          │
│           ▼         16 samples      ▼                           ▼          │
│    ───●───●───●───●───●───●───●───●─●─●───●───●───●───●───●───●───●───     │
│       └──────────────────────────────────────────────────────────────┘     │
│                              8 global samples                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  低置信度 (20%): 均匀采样                                                    │
│                                                                             │
│         z_near                                                 z_far        │
│           ▼                                                      ▼          │
│    ───●───●───●───●───●───●───●───●───●───●───●───●───●───●───●───●───     │
│       └──────────────────────────────────────────────────────────────┘     │
│                        24 stratified uniform samples                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**配置参数** (`configs/exp_kitti_360.yaml`):

```yaml
adaptive_sampling: true
n_surface_samples: 16    # Samples around DA3 depth
n_global_samples: 8      # Global background/foreground samples
absrel_prior: 0.12       # DA3 relative error prior
min_thickness: 0.5       # Minimum sampling thickness (meters)

# Online DA3 depth generation
depth_mode: "online"
da3_conf_percentile: 20.0  # Bottom 20% confidence uses uniform sampling
```

---

### 3.4 损失函数设计

**配置** (`configs/exp_kitti_360.yaml`):

```yaml
loss:
    criterion: "l1+ssim"
    invalid_policy: weight_guided
    lambda_edge_aware_smoothness: 0.001
    lambda_segmentation: 0.02
```

| 损失项 | 权重 | 说明 |
|--------|------|------|
| L1 + SSIM | 0.85 + 0.15 | RGB 重建损失 |
| Cross Entropy | 0.02 | 语义分割损失 |
| Edge-aware Smoothness | 0.001 | 深度平滑正则化 |

**分割类别权重** (Cityscapes 19 类):

```yaml
# 重要类别增加权重
'sidewalk': 10, 'pole': 10, 'traffic sign': 10
'traffic light': 5, 'person': 5
```

---

### 3.5 渲染器架构

**核心代码位置**: `models/common/render/nerf.py`

```python
class NeRFRenderer(torch.nn.Module):
    def __init__(
        self,
        n_coarse=128,      # 粗采样点数 (24 for adaptive)
        n_fine=0,          # 细采样点数
        n_fine_depth=0,    # 深度细采样
        noise_std=0.0,     # sigma 噪声
        depth_std=0.01,    # 深度采样噪声
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,     # 线性视差采样
        hard_alpha_cap=False
    ):
```

**体积渲染公式**：

```
C(r) = Σ T_i · α_i · c_i
D(r) = Σ T_i · α_i · z_i
T_i = Π(1 - α_j) for j < i
```

---

## 4. 训练流程

### 4.1 数据流

**核心代码位置**: `models/bts/trainer.py` - `BTSWrapper.forward()`

```python
def forward(self, data):
    # 1. 数据准备
    images = torch.stack(data["imgs"], dim=1)     # (n, v, c, h, w)
    poses = torch.stack(data["poses"], dim=1)      # (n, v, 4, 4)
    projs = torch.stack(data["projs"], dim=1)      # (n, v, 4, 4)
    segs = torch.stack(data["segs_gt"], dim=1)     # (n, n_segs, h, w)
    
    # 2. DINOv2 特征提取
    self.renderer.net.encode(data, images, projs, poses, ...)
    
    # 3. DA3 深度引导采样
    da3_depths = torch.stack(data["depths"], dim=1)
    
    # 4. 光线采样 & 渲染
    render_dict = self.renderer(all_rays, da3_depths=all_da3_depths, ...)
```

### 4.2 帧采样策略 (KITTI-360)

```python
frame_sample_mode: "kitti360-mono"
```

对于 4 个相机视角，交替选择用于损失计算和渲染的帧：

```python
for cam in range(4):
    ids_loss += [cam * steps + i for i in range(start_from, steps, 2)]
    ids_render += [cam * steps + i for i in range(1 - start_from, steps, 2)]
```

---

## 5. 数据集配置

### 5.1 KITTI-360 数据配置

**配置文件**: `configs/data/kitti_360.yaml`

```yaml
type: "KITTI_360"
data_path: "/data/lmh_data/KITTI360"
data_segmentation_path: "/data/lmh_data/KITTI360/mask2former_cityscapes_swinl_in21k_384_90k_original_reso"
# 使用原始分辨率 1408 x 376
image_size: [376, 1408]

# DA3 Depth Configuration (在线生成模式)
depth_mode: "online"
da3_checkpoint_path: "/home/lmh/dino-da3-adaptive-sampling1/DA3/ckpt/DA3NESTED-GIANT-LARGE"
da3_conf_percentile: 20.0  # 低置信度 20% 使用均匀采样
return_depth: true
return_lidar_depth: true  # 用于评估
```

**在线深度生成模块**: `models/da3_depth_generator.py`

```python
class DA3DepthGenerator(nn.Module):
    """
    Online depth generator using Depth Anything 3 (DA3NESTED-GIANT-LARGE).
    
    Features:
    - Model: DA3NESTED-GIANT-LARGE (1.40B params, Giant + Large nested)
    - 输出: 深度图 + 置信度图
    - 置信度过滤: 低于 20% 百分位数的区域使用均匀采样
    - 鱼眼相机: 深度填充为 0 (不进行 DA3 推理)
    """
```

### 5.2 数据组织结构

```
KITTI360/
├── calibration/
├── data_2d_raw/
├── data_2d_depth_da3/          # DA3 深度图
├── data_2d_semantics/
├── data_3d_raw/                # Velodyne 点云
├── data_poses/
└── mask2former_.../            # 伪分割标签
```

---

## 5.3 在线 DA3 深度生成性能分析

**配置**: A6000 GPU (48GB VRAM), 输入分辨率 1408 × 376

| 组件 | 时间 (估计) | 备注 |
|------|------------|------|
| DA3NESTED-GIANT-LARGE 推理 | ~150-200ms/image | 1408×376 分辨率 |
| 批量处理 4 视角 | ~400-600ms/batch | 并行处理 2 个透视相机 |
| 训练 iteration 增加 | +20-30% | 相比离线深度加载 |

**显存占用**:
- DA3NESTED-GIANT-LARGE: ~12-15GB VRAM
- DINOv2 ViT-L/14: ~3GB VRAM
- 训练总计: ~30-35GB VRAM (A6000 48GB 足够)

**优化建议**:
1. DA3 模型在 `eval` 模式下运行，梯度关闭
2. 使用 `torch.no_grad()` 包裹 DA3 推理
3. 可选: 使用混合精度 (FP16) 减少显存

---

## 6. 关键技术创新

### 6.1 DINOv2 特征提取

**核心代码位置**: `models/bts/model/models_bts.py`

```python
def _extract_dino_features(self, images):
    # ImageNet 归一化
    mean = images.new_tensor([0.485, 0.456, 0.406])
    std = images.new_tensor([0.229, 0.224, 0.225])
    images_norm = (images - mean) / std
    
    # Pad to multiple of patch size (14)
    # ...
    
    # 提取 patch tokens
    with torch.no_grad():
        features_dict = self.dino.forward_features(images_norm)
        patch_tokens = features_dict["x_norm_patchtokens"]
    
    # 通过适配器 (PixelShuffle 2x + ResBlock)
    patch_map = self.dino_adapter(patch_map)
    
    # 上采样到原始分辨率
    patch_map = F.interpolate(patch_map, size=(H, W), ...)
```

### 6.2 像素对齐特征采样

```python
def sample_features(self, xyz, use_single_featuremap=True):
    # 3D 点投影到 2D
    xyz_projected = (poses_w2c @ xyz.permute(0, 1, 3, 2))
    xyz_projected = (Ks @ xyz_projected).permute(0, 1, 3, 2)
    xy = xyz_projected[:, :, :, [0, 1]] / z
    
    # 双线性插值采样特征
    sampled_features = F.grid_sample(
        feature_map, xy, mode="bilinear", 
        padding_mode="border", align_corners=False
    )
    
    # 添加位置编码
    xyz_code = self.code_xyz(xyz_projected)
    sampled_features = torch.cat((sampled_features, xyz_code), dim=-1)
```

---

## 7. 环境依赖

**配置文件**: `environment.yml`

```yaml
name: s4c
dependencies:
  - python
  - pytorch=1.13.1
  - torchvision==0.14.1
  - pytorch-cuda=11.6
  - pip:
      - pytorch-ignite        # 分布式训练
      - hydra-core           # 配置管理
      - omegaconf
      - lpips                # 感知损失
      - pytorch-msssim       # SSIM 损失
```

---

## 8. 模型评估指标

### 深度评估

| 指标 | 说明 |
|------|------|
| abs_rel | 绝对相对误差 |
| sq_rel | 平方相对误差 |
| rmse | 均方根误差 |
| a1/a2/a3 | 阈值精度 (1.25^n) |

### 语义分割评估

| 指标 | 说明 |
|------|------|
| seg_acc | 整体像素精度 |
| seg_acc_front | 前视图精度 |
| seg_acc_side | 侧视图精度 |

### 新视角合成评估 (可选)

| 指标 | 说明 |
|------|------|
| SSIM | 结构相似度 |
| PSNR | 峰值信噪比 |
| LPIPS | 感知相似度 |

---

## 9. 使用方法

### 9.1 训练

```bash
# 激活环境
conda activate s4c

# 下载预训练 backbone
./download_backbone.sh

# 开始训练
python train.py -cn exp_kitti_360
```

### 9.2 评估

```bash
python eval.py -cn exp_kitti_360
```

### 9.3 推理

```bash
python scripts/images/gen_img_custom.py \
    --img media/example/0000.png \
    --model out/kitti_360/<model-name> \
    --plot
```

---

## 10. 总结

本项目是一个**高度模块化**的自监督语义场景补全框架，核心特点：

1. **视觉骨干**: 使用冻结的 DINOv2 ViT-L/14 提取强大的视觉特征
2. **深度先验**: 集成 Depth Anything 3 进行 DA3 引导的自适应采样
3. **隐式表示**: 使用 NeRF 风格的体积渲染进行场景表示
4. **自监督学习**: 通过多视角一致性和伪标签进行训练
5. **配置驱动**: 使用 Hydra 实现灵活的实验配置管理

该架构设计使得模型能够从单张图像预测稠密的 3D 语义场景，同时保持良好的泛化能力。

---

## 附录：核心文件索引

| 文件 | 功能描述 |
|------|----------|
| `train.py` | 训练入口 |
| `eval.py` | 评估入口 |
| `models/bts/model/models_bts.py` | BTSNet 主网络 + DINOv2 backbone |
| `models/bts/model/loss.py` | 损失函数定义 |
| `models/bts/model/ray_sampler.py` | 光线采样器 (含置信度过滤支持) |
| `models/bts/trainer.py` | 训练流程 (含在线 DA3 深度生成) |
| `models/common/render/nerf.py` | NeRF 渲染器 + 置信度感知自适应采样 |
| `models/da3_depth_generator.py` | DA3 在线深度生成模块 (新增) |
| `configs/exp_kitti_360.yaml` | KITTI-360 实验配置 |
| `configs/data/kitti_360.yaml` | KITTI-360 数据集配置 |
| `DA3/ckpt/DA3NESTED-GIANT-LARGE/` | DA3 模型权重 |

---

*文档生成日期: 2026-01-27*
*更新: 添加在线 DA3 深度生成 + 置信度过滤采样*
