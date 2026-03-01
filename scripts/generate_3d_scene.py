#!/usr/bin/env python3
"""
从单张图像生成3D场景补全结果
支持输出格式: GLB, PLY, OBJ, GLTF

使用方法:
    python scripts/generate_3d_scene.py \
        --img path/to/image.png \
        --model out/kitti_360/your_model_folder \
        --output output_scene.glb \
        --format glb

作者: 基于 dino-da3-adaptive-sampling1 项目
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm import tqdm

# 尝试导入3D库
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("[警告] trimesh 未安装，GLB/GLTF/OBJ 导出功能不可用。请运行: pip install trimesh")

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("[警告] plyfile 未安装，PLY 导出功能不可用。请运行: pip install plyfile")

from models.bts.model import BTSNet
from models.common.render import NeRFRenderer

# ==================== 配置 ====================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 场景范围 (米)
DEFAULT_X_RANGE = (-9, 9)      # 左右 18米
DEFAULT_Y_RANGE = (-0.6, 1.8)  # 高度 2.4米
DEFAULT_Z_RANGE = (21, 3)      # 深度 3-21米

# 体素分辨率
DEFAULT_RES_XZ = 128
DEFAULT_RES_Y = 32

# Cityscapes 19类颜色映射
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],    # 0: road
    [244, 35, 232],    # 1: sidewalk
    [70, 70, 70],      # 2: building
    [102, 102, 156],   # 3: wall
    [190, 153, 153],   # 4: fence
    [153, 153, 153],   # 5: pole
    [250, 170, 30],    # 6: traffic light
    [220, 220, 0],     # 7: traffic sign
    [107, 142, 35],    # 8: vegetation
    [152, 251, 152],   # 9: terrain
    [70, 130, 180],    # 10: sky
    [220, 20, 60],     # 11: person
    [255, 0, 0],       # 12: rider
    [0, 0, 142],       # 13: car
    [0, 0, 70],        # 14: truck
    [0, 60, 100],      # 15: bus
    [0, 80, 100],      # 16: train
    [0, 0, 230],       # 17: motorcycle
    [119, 11, 32],     # 18: bicycle
], dtype=np.uint8)


def get_args():
    parser = argparse.ArgumentParser(description="从单张图像生成3D场景")
    parser.add_argument("--img", "-i", required=True, help="输入图像路径")
    parser.add_argument("--model", "-m", default=None, help="模型目录路径 (包含 training*.pt)")
    parser.add_argument("--checkpoint", "-c", default=None, help="直接指定权重文件路径 (.pt)")
    parser.add_argument("--output", "-o", default="scene_output", help="输出文件名 (不含扩展名)")
    parser.add_argument("--format", "-f", default="glb", choices=["glb", "gltf", "ply", "obj", "all"],
                        help="输出格式: glb, gltf, ply, obj, all (默认: glb)")
    parser.add_argument("--resolution", "-r", type=int, default=128, help="体素分辨率 (默认: 128)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="密度阈值 (默认: 0.5)")
    parser.add_argument("--semantic", "-s", action="store_true", help="使用语义分割着色")
    parser.add_argument("--show", action="store_true", help="显示预览")
    parser.add_argument("--x_range", type=float, nargs=2, default=[-9, 9], help="X范围 (左右)")
    parser.add_argument("--y_range", type=float, nargs=2, default=[-0.6, 1.8], help="Y范围 (高度)")
    parser.add_argument("--z_range", type=float, nargs=2, default=[21, 3], help="Z范围 (深度)")
    return parser.parse_args()


def load_model(model_path=None, checkpoint_path=None, config_name="exp_kitti_360"):
    """加载模型
    
    Args:
        model_path: 模型目录路径 (自动查找 training*.pt)
        checkpoint_path: 直接指定的权重文件路径
        config_name: 配置文件名
    """
    if checkpoint_path is not None:
        cp_path = Path(checkpoint_path)
        print(f"[INFO] 加载权重文件: {cp_path}")
    elif model_path is not None:
        model_path = Path(model_path)
        cp_path = next(model_path.glob("training*.pt"))
        print(f"[INFO] 加载模型目录: {model_path}")
        print(f"[INFO] 找到权重文件: {cp_path}")
    else:
        raise ValueError("必须指定 --model 或 --checkpoint 参数")
    
    # 初始化 Hydra 配置
    try:
        initialize(version_base=None, config_path="../configs", job_name="gen_3d_scene")
    except:
        pass  # 如果已经初始化过则跳过
    
    config = compose(config_name=config_name, overrides=[])
    config = dict(config)
    
    # 设置分割模式
    if "segmentation_mode" in config.keys():
        config["model_conf"] = dict(config["model_conf"])
        config["model_conf"]["segmentation_mode"] = config["segmentation_mode"]
    
    # 创建网络
    net = BTSNet(config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True
    
    # 加载权重
    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer
    
    wrapper = _Wrapper()
    cp = torch.load(cp_path, map_location=device)
    wrapper.load_state_dict(cp["model"], strict=False)
    
    renderer.to(device)
    renderer.eval()
    
    print(f"[INFO] 模型加载成功!")
    return net, renderer, config


def get_query_points(x_range, y_range, z_range, res_xz, res_y):
    """生成3D查询网格点"""
    x = torch.linspace(x_range[0], x_range[1], res_xz).view(1, 1, res_xz).expand(res_y, res_xz, -1)
    z = torch.linspace(z_range[0], z_range[1], res_xz).view(1, res_xz, 1).expand(res_y, -1, res_xz)
    y = torch.linspace(y_range[0], y_range[1], res_y).view(res_y, 1, 1).expand(-1, res_xz, res_xz)
    xyz = torch.stack((x, y, z), dim=-1).permute(2, 0, 1, 3)  # (x, y, z, 3)
    
    # KITTI-360 相机有 5° 倾斜，调整 Y 坐标
    xyz[:, :, :, 1] -= xyz[:, :, :, 2] * 0.0874886635
    
    return xyz


def query_density_and_semantics(net, query_pts, batch_size=50000):
    """查询3D点的密度和语义"""
    # 使用 contiguous() 确保张量在内存中连续，然后再 view
    query_pts = query_pts.contiguous().to(device).view(1, -1, 3)
    
    sigmas_list = []
    segs_list = []
    
    total_pts = query_pts.shape[1]
    for i in tqdm(range(0, total_pts, batch_size), desc="查询3D点"):
        pts_batch = query_pts[:, i:min(i+batch_size, total_pts), :]
        _, invalid, sigmas, segs = net.forward(pts_batch, predict_segmentation=True)
        sigmas[torch.any(invalid, dim=-1)] = 0
        sigmas_list.append(sigmas)
        segs_list.append(segs)
    
    sigmas = torch.cat(sigmas_list, dim=1)
    segs = torch.cat(segs_list, dim=1)
    
    return sigmas, segs


def generate_voxel_mesh(is_occupied, query_pts, colors, x_res, y_res, z_res):
    """生成体素网格mesh"""
    # 立方体的8个顶点偏移
    ids_offset = torch.tensor([
        [1, 1, 0], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
        [1, 1, 1], [1, 0, 1],
        [0, 0, 1], [0, 1, 1]
    ], dtype=torch.int32, device=device)
    
    # 立方体的6个面
    faces_template = torch.tensor([
        [0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3],
        [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]
    ], device=device)
    
    # 获取占用的体素索引
    ijks = is_occupied.nonzero()
    if ijks.shape[0] == 0:
        return None, None, None
    
    # 计算顶点索引
    ids = ijks.view(-1, 1, 3) + ids_offset.view(1, -1, 3)
    ids_flat = ids[..., 0] * y_res * z_res + ids[..., 1] * z_res + ids[..., 2]
    
    # 获取顶点坐标
    query_pts_flat = query_pts.contiguous().to(device).view(-1, 3)
    verts = query_pts_flat[ids_flat.reshape(-1)].cpu().numpy()
    
    # 计算面索引
    faces_off = torch.arange(0, ijks.shape[0] * 8, 8, device=device)
    faces_off = faces_off.view(-1, 1, 1) + faces_template.view(1, 6, 4)
    faces = faces_off.reshape(-1, 4).cpu().numpy()
    
    # 获取颜色
    voxel_colors = colors[ijks[:, 0].cpu(), ijks[:, 1].cpu(), ijks[:, 2].cpu()]
    # 每个体素有8个顶点，需要重复颜色
    vert_colors = np.repeat(voxel_colors[:, np.newaxis, :], 8, axis=1).reshape(-1, 3)
    
    return verts, faces, vert_colors


def save_as_ply(filepath, verts, faces, colors):
    """保存为PLY格式"""
    if not HAS_PLYFILE:
        print("[错误] plyfile 未安装，无法保存PLY文件")
        return False
    
    verts_list = list(map(tuple, verts))
    colors_list = list(map(tuple, colors))
    verts_colors = [v + c for v, c in zip(verts_list, colors_list)]
    
    verts_data = np.array(verts_colors, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    face_data = np.array(faces, dtype='i4')
    ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,))])
    ply_faces['vertex_indices'] = face_data
    
    verts_el = PlyElement.describe(verts_data, "vertex")
    faces_el = PlyElement.describe(ply_faces, "face")
    
    PlyData([verts_el, faces_el]).write(str(filepath))
    print(f"[INFO] 已保存: {filepath}")
    return True


def save_as_trimesh(filepath, verts, faces, colors, file_format="glb"):
    """保存为GLB/GLTF/OBJ格式 (使用trimesh)"""
    if not HAS_TRIMESH:
        print(f"[错误] trimesh 未安装，无法保存{file_format.upper()}文件")
        return False
    
    # 将四边形面转换为三角形面
    tri_faces = []
    tri_colors = []
    for i, face in enumerate(faces):
        # 四边形 -> 两个三角形
        tri_faces.append([face[0], face[1], face[2]])
        tri_faces.append([face[0], face[2], face[3]])
    
    tri_faces = np.array(tri_faces)
    
    # 创建mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=tri_faces)
    
    # 添加顶点颜色
    vertex_colors = np.zeros((len(verts), 4), dtype=np.uint8)
    vertex_colors[:, :3] = colors
    vertex_colors[:, 3] = 255  # Alpha
    mesh.visual.vertex_colors = vertex_colors
    
    # 保存
    mesh.export(str(filepath), file_type=file_format)
    print(f"[INFO] 已保存: {filepath}")
    return True


def main():
    args = get_args()
    
    # 检查输入图像
    if not os.path.exists(args.img):
        print(f"[错误] 找不到输入图像: {args.img}")
        return
    
    # 检查模型参数
    if args.model is None and args.checkpoint is None:
        print("[错误] 必须指定 --model 或 --checkpoint 参数")
        return
    
    # 加载模型
    net, renderer, config = load_model(
        model_path=args.model, 
        checkpoint_path=args.checkpoint
    )
    
    # 加载和预处理图像
    print(f"[INFO] 加载图像: {args.img}")
    resolution = (192, 640)
    img = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img = cv2.resize(img, (resolution[1], resolution[0]))
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) * 2 - 1
    
    # 相机参数 (KITTI-360)
    poses = torch.eye(4).view(1, 1, 4, 4).to(device)
    proj = torch.tensor([
        [0.7849, 0.0000, -0.0312, 0],
        [0.0000, 2.9391, 0.2701, 0],
        [0.0000, 0.0000, 1.0000, 0],
        [0.0000, 0.0000, 0.0000, 1],
    ], dtype=torch.float32).view(1, 1, 4, 4).to(device)[:, :, :3, :3]
    
    print("[INFO] 编码图像特征...")
    with torch.no_grad():
        net.encode({}, img_tensor, proj, poses, ids_encoder=[0], ids_render=[0])
        net.set_scale(0)
        
        # 生成查询点
        x_range = tuple(args.x_range)
        y_range = tuple(args.y_range)
        z_range = tuple(args.z_range)
        res_xz = args.resolution
        res_y = max(8, args.resolution // 4)
        
        print(f"[INFO] 生成查询网格: {res_xz}x{res_y}x{res_xz}")
        query_pts = get_query_points(x_range, y_range, z_range, res_xz, res_y)
        
        # 查询密度和语义
        print("[INFO] 查询3D密度场...")
        sigmas, segs = query_density_and_semantics(net, query_pts)
        
        # 重塑为体素网格
        sigmas = sigmas.reshape(1, 1, res_xz, res_y, res_xz)
        segs = segs.reshape(1, 1, res_xz, res_y, res_xz, -1)
        
        # 计算体素大小
        delta = np.mean([
            (x_range[1] - x_range[0]) / res_xz,
            (y_range[1] - y_range[0]) / res_y,
            abs(z_range[0] - z_range[1]) / res_xz
        ])
        
        # 计算 alpha 值
        alphas = 1 - torch.exp(-delta * sigmas)
        
        # 用密度加权语义
        weighted_segs = segs * alphas[..., None]
        
        # 平均池化平滑
        alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)
        is_occupied = alphas_mean.squeeze() > args.threshold
        
        # 语义预测
        segs_pool_list = [F.avg_pool3d(weighted_segs[..., i], kernel_size=2, stride=1, padding=0) 
                         for i in range(weighted_segs.shape[-1])]
        segs_pool = torch.stack(segs_pool_list, dim=-1)
        classes = torch.argmax(segs_pool, dim=-1).squeeze().cpu().numpy()
        
        # 生成颜色
        if args.semantic:
            # 语义着色
            colors = CITYSCAPES_COLORS[classes.astype(int) % len(CITYSCAPES_COLORS)]
        else:
            # 高度着色 (magma colormap)
            cmap = plt.colormaps.get_cmap("magma")
            y_normalized = np.linspace(0, 1, res_y - 1)
            y_colors = (np.array([cmap(1 - y)[:3] for y in y_normalized]) * 255).astype(np.uint8)
            
            colors = np.zeros((res_xz - 1, res_y - 1, res_xz - 1, 3), dtype=np.uint8)
            for y_idx in range(res_y - 1):
                colors[:, y_idx, :, :] = y_colors[y_idx]
        
        # 生成网格
        print("[INFO] 生成3D网格...")
        # 调整查询点以匹配池化后的尺寸
        query_pts_adj = query_pts[:res_xz, :res_y, :res_xz, :]
        verts, faces, vert_colors = generate_voxel_mesh(
            is_occupied, query_pts_adj, colors,
            res_xz - 1, res_y - 1, res_xz - 1
        )
        
        if verts is None:
            print("[警告] 没有检测到占用体素，尝试降低阈值")
            return
        
        print(f"[INFO] 生成了 {len(verts)} 个顶点, {len(faces)} 个面")
        
        # 保存输出
        output_base = Path(args.output)
        output_dir = output_base.parent
        output_stem = output_base.stem
        
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        formats_to_save = [args.format] if args.format != "all" else ["glb", "ply", "obj"]
        
        for fmt in formats_to_save:
            output_path = output_dir / f"{output_stem}.{fmt}" if output_dir else Path(f"{output_stem}.{fmt}")
            
            if fmt == "ply":
                save_as_ply(output_path, verts, faces, vert_colors)
            elif fmt in ["glb", "gltf", "obj"]:
                save_as_trimesh(output_path, verts, faces, vert_colors, fmt)
        
        # 显示预览
        if args.show:
            print("[INFO] 显示预览...")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 输入图像
            axes[0].imshow(img)
            axes[0].set_title("输入图像")
            axes[0].axis("off")
            
            # 俯视图密度
            density_top = sigmas.squeeze().max(dim=1).values.cpu().numpy()
            axes[1].imshow(density_top, cmap="magma")
            axes[1].set_title("俯视图密度")
            axes[1].axis("off")
            
            # 俯视图语义
            seg_top = classes.argmax(axis=1)
            seg_colors = CITYSCAPES_COLORS[seg_top.astype(int) % len(CITYSCAPES_COLORS)]
            axes[2].imshow(seg_colors)
            axes[2].set_title("俯视图语义")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.show()
        
        print("[INFO] 完成!")


if __name__ == "__main__":
    main()
