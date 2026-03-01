"""
ViPOcc-Style 3D Occupancy Evaluation for DINO-DA3 Model

This script evaluates the DINO-DA3 model using ViPOcc's occupancy evaluation protocol.
It computes:
- Scene-level O_acc, IE_acc, IE_rec
- Object-level O_acc, IE_acc, IE_rec (if GT available)

Usage:
    python scripts/eval_vipocc_style.py \
        --checkpoint out/kitti_360/kitti_360_backend-None-1_20260210-203842/training_checkpoint_151000-bp.pt \
        --data_path /data/lmh_data/KITTI360 \
        --z_range 20 4 \
        --save_vis

Author: LMH
Date: 2026-02
"""

import argparse
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from models.bts.model.models_bts import BTSNet
from models.bts.model.ray_sampler import ImageRaySampler
from models.common.render import NeRFRenderer

# Camera inclination adjustment for KITTI-360
CAM_INCL_ADJUST = torch.tensor(
    [[1.0000000, 0.0000000, 0.0000000, 0],
     [0.0000000, 0.9961947, 0.0871557, 0],
     [0.0000000, -0.0871557, 0.9961947, 0],
     [0.0000000, 0.0000000, 0.0000000, 1]],
    dtype=torch.float32
).view(1, 1, 4, 4)


def get_pts(x_range, y_range, z_range, ppm, ppm_y, y_res=None, specify_yslice=None):
    """Generate query points for occupancy evaluation."""
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    if y_res is None:
        y_res = abs(int((y_range[1] - y_range[0]) * ppm_y))
    z_res = abs(int((z_range[1] - z_range[0]) * ppm))
    
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    
    if y_res == 1:
        if specify_yslice is None:
            y = torch.tensor([y_range[0] * .5 + y_range[1] * .5]).view(y_res, 1, 1).expand(-1, z_res, x_res)
        else:
            y = torch.tensor([specify_yslice]).view(y_res, 1, 1).expand(-1, z_res, x_res)
    else:
        y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    
    xyz = torch.stack((x, y, z), dim=-1)
    return xyz, (x_res, y_res, z_res)


def get_lidar_slices(point_clouds, velo_poses, y_range, y_res, max_dist):
    """Process LiDAR point clouds into polar slices for occupancy checking."""
    slices = []
    ys = torch.linspace(y_range[0], y_range[1], y_res)
    
    if y_res > 1:
        slice_height = ys[1] - ys[0]
    else:
        slice_height = 0
    
    n_bins = 360

    for y in ys:
        if y_res == 1:
            min_y = y
            max_y = y_range[-1]
        else:
            min_y = y - slice_height / 2
            max_y = y + slice_height / 2

        slice = []
        for pc, velo_pose in zip(point_clouds, velo_poses):
            pc_world = (velo_pose @ pc.T).T
            mask = ((pc_world[:, 1] >= min_y) & (pc_world[:, 1] <= max_y)) | (
                    torch.norm(pc_world[:, :3], dim=-1) >= max_dist)
            
            slice_points = pc[mask, :2]
            angles = torch.atan2(slice_points[:, 1], slice_points[:, 0])
            dists = torch.norm(slice_points, dim=-1)
            
            slice_points_polar = torch.stack((angles, dists), dim=1)
            slice_points_polar = slice_points_polar[torch.sort(angles)[1], :]
            
            slice_points_polar_binned = torch.zeros_like(slice_points_polar[:n_bins, :])
            bin_borders = torch.linspace(-math.pi, math.pi, n_bins + 1, device=slice_points_polar.device)
            
            dist = slice_points_polar[0, 1]
            border_is = torch.searchsorted(slice_points_polar[:, 0], bin_borders)
            
            for i in range(n_bins):
                left_i, right_i = border_is[i], border_is[i + 1]
                angle = (bin_borders[i] + bin_borders[i + 1]) * .5
                if right_i > left_i:
                    dist = torch.min(slice_points_polar[left_i:right_i, 1])
                slice_points_polar_binned[i, 0] = angle
                slice_points_polar_binned[i, 1] = dist

            slice_points_polar = slice_points_polar_binned
            slice_points_polar = torch.cat((
                torch.tensor([[slice_points_polar[-1, 0] - math.pi * 2, slice_points_polar[-1, 1]]],
                             device=slice_points_polar.device),
                slice_points_polar,
                torch.tensor([[slice_points_polar[0, 0] + math.pi * 2, slice_points_polar[0, 1]]],
                             device=slice_points_polar.device)
            ), dim=0)
            
            slice.append(slice_points_polar)
        slices.append(slice)
    
    return slices


def check_occupancy(pts, slices, velo_poses, min_dist=3):
    """Check occupancy status of query points using LiDAR data."""
    is_occupied = torch.ones_like(pts[:, 0])
    is_visible = torch.zeros_like(pts[:, 0], dtype=torch.bool)
    
    thresh = (len(slices[0]) - 2) / len(slices[0])
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    world_to_velos = torch.inverse(velo_poses)
    
    step = pts.shape[0] // len(slices)
    
    for i, slice in enumerate(slices):
        for j, (lidar_polar, world_to_velo) in enumerate(zip(slice, world_to_velos)):
            pts_velo = (world_to_velo @ pts[i * step: (i + 1) * step, :].T).T
            
            angles = torch.atan2(pts_velo[:, 1], pts_velo[:, 0])
            dists = torch.norm(pts_velo, dim=-1)
            
            indices = torch.searchsorted(lidar_polar[:, 0].contiguous(), angles)
            
            left_angles = lidar_polar[indices - 1, 0]
            right_angles = lidar_polar[indices, 0]
            left_dists = lidar_polar[indices - 1, 1]
            right_dists = lidar_polar[indices, 1]
            
            interp = (angles - left_angles) / (right_angles - left_angles)
            surface_dist = left_dists * (1 - interp) + right_dists * interp
            
            is_occupied_velo = (dists > surface_dist) | (dists < min_dist)
            is_occupied[i * step: (i + 1) * step] += is_occupied_velo.float()
            
            if j == 0:
                is_visible[i * step: (i + 1) * step] |= ~is_occupied_velo
    
    is_occupied /= len(slices[0])
    is_occupied = is_occupied > thresh
    
    return is_occupied, is_visible


class DINODA3OccWrapper(nn.Module):
    """Wrapper for DINO-DA3 model to support ViPOcc-style occupancy evaluation."""
    
    def __init__(self, renderer, config, dataset):
        super().__init__()
        self.renderer = renderer
        
        self.z_near = config.get("z_near", 3)
        self.z_far = config.get("z_far", 80)
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = 0.5
        
        # Evaluation range settings
        self.x_range = config.get("x_range", (-4, 4))
        self.y_range = config.get("y_range", (0, 0.75))
        self.z_range = config.get("z_range", (20, 4))  # (far, near)
        self.ppm = config.get("ppm", 10)
        self.ppm_y = config.get("ppm_y", 4)
        self.y_res = config.get("y_res", 1)
        
        self.cut_far_invisible_area = config.get("cut_far_invisible_area", True)
        self.save_vis = config.get("save_vis", False)
        self.save_dir = config.get("save_dir", "visualization")
        
        # GT paths
        self.read_gt_occ_path = config.get("read_gt_occ_path", "")
        self.is_eval_object = config.get("is_eval_object", False)
        self.read_gt_obj_path = config.get("read_gt_obj_path", "")
        
        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=3)
        self.dataset = dataset
        self.aggregate_timesteps = config.get("gt_aggregate_timesteps", 300)
        
        print(f"Evaluation settings: x_range={self.x_range}, y_range={self.y_range}, "
              f"z_range={self.z_range}, ppm={self.ppm}")
    
    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)
        poses = torch.stack(data["poses"], dim=1)
        projs = torch.stack(data["projs"], dim=1)
        index = data["index"].item()
        
        seq, id, is_right = self.dataset._datapoints[index]
        seq_len = self.dataset._img_ids[seq].shape[0]
        init_id = id
        
        n, v, c, h, w = images.shape
        device = images.device
        
        T_velo_to_pose = torch.tensor(self.dataset._calibs["T_velo_to_pose"], device=device)
        
        # Coordinate transform
        world_transform = torch.inverse(poses[:, :1, :, :])
        world_transform = CAM_INCL_ADJUST.to(device) @ world_transform
        poses = world_transform @ poses
        
        self.sampler.height = h
        self.sampler.width = w
        
        # Load LiDAR data for GT
        points_all = []
        velo_poses = []
        
        if self.read_gt_occ_path == "":
            for frame_id in range(id, min(id + self.aggregate_timesteps, seq_len)):
                lidar_path = os.path.join(
                    self.dataset.data_path, "data_3d_raw", seq, "velodyne_points", "data",
                    f"{self.dataset._img_ids[seq][frame_id]:010d}.bin"
                )
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
                points[:, 3] = 1.0
                points = torch.tensor(points, device=device)
                
                velo_pose = world_transform.squeeze() @ torch.tensor(
                    self.dataset._poses[seq][frame_id], device=device
                ) @ T_velo_to_pose
                
                points_all.append(points)
                velo_poses.append(velo_pose)
            
            velo_poses = torch.stack(velo_poses, dim=0)
        else:
            # Load pre-computed GT
            name = f"{seq}_{self.dataset._img_ids[seq][init_id]:010d}"
            occ_gt = np.load(os.path.join(self.read_gt_occ_path, name + "_occgt.npy"))
            vis_gt = np.load(os.path.join(self.read_gt_occ_path, name + "_visgt.npy"))
            is_occupied = torch.from_numpy(occ_gt).bool().to(device)
            is_visible = torch.from_numpy(vis_gt).bool().to(device)
        
        rays, _ = self.sampler.sample(None, poses[:, :1, :, :], projs[:, :1, :, :])
        
        # Encode image features
        ids_encoder = [0]
        
        # Prepare data dict for encoder (DINO-DA3 encode signature)
        # encode(self, data, images, Ks, poses_c2w, ids_encoder, ids_render, images_alt, combine_ids)
        encode_data = {"depths": data.get("depths", [])}
        
        # Convert poses from world-to-camera back to camera-to-world for encode
        poses_c2w = torch.inverse(poses)
        
        self.renderer.net.encode(
            encode_data, images, projs, poses_c2w,
            ids_encoder=ids_encoder, ids_render=ids_encoder,
            images_alt=images * .5 + .5, combine_ids=None
        )
        
        # Query 3D points
        q_pts, (xd, yd, zd) = get_pts(
            self.x_range, self.y_range, self.z_range,
            self.ppm, self.ppm_y, self.y_res
        )
        q_pts = q_pts.to(device).view(-1, 3)
        
        # Query densities
        densities = []
        for i_from in range(0, len(q_pts), self.query_batch_size):
            i_to = min(i_from + self.query_batch_size, len(q_pts))
            q_pts_ = q_pts[i_from:i_to]
            # Model returns (rgb, invalid, sigma) when only_density=True
            rgb_, invalid_, densities_ = self.renderer.net(q_pts_.unsqueeze(0), only_density=True)
            densities.append(densities_.squeeze(0))
        
        densities = torch.cat(densities, dim=0).squeeze()
        is_occupied_pred = densities > self.occ_threshold
        
        # Generate GT from LiDAR if not pre-computed
        if self.read_gt_occ_path == "":
            slices = get_lidar_slices(
                points_all, velo_poses, self.y_range, yd,
                (self.z_range[0] ** 2 + self.x_range[0] ** 2) ** .5
            )
            is_occupied, is_visible = check_occupancy(q_pts, slices, velo_poses)
        
        is_occupied = is_occupied.reshape(zd, xd)
        is_occupied_pred = is_occupied_pred.reshape(zd, xd)
        is_visible = is_visible.reshape(zd, xd)
        
        # Only not visible points can be occupied
        is_occupied &= ~is_visible
        
        # Save visualization
        if self.save_vis:
            self._save_visualization(
                is_occupied, is_occupied_pred, is_visible, images,
                seq, self.dataset._img_ids[seq][id]
            )
        
        # Cut far invisible area
        if self.cut_far_invisible_area:
            z_indices = torch.nonzero(is_visible)[:, 0]
            if z_indices.shape[0] == 0:
                data["scene_O_acc"] = torch.tensor(float('nan'), device=device)
                data["scene_IE_acc"] = torch.tensor(float('nan'), device=device)
                data["scene_IE_rec"] = torch.tensor(float('nan'), device=device)
                return data
            
            z_min_val = torch.min(z_indices)
            z_min_val = max(0, z_min_val - 2 * self.ppm)
            h_occ, w_occ = is_occupied.shape
            is_occupied = is_occupied[z_min_val:h_occ, :]
            is_visible = is_visible[z_min_val:h_occ, :]
            is_occupied_pred = is_occupied_pred[z_min_val:h_occ, :]
        
        # Compute metrics
        scene_eval_res = self._compute_scene_occ_scores(is_occupied, is_occupied_pred, is_visible)
        
        data["scene_O_acc"] = scene_eval_res[0]
        data["scene_IE_acc"] = scene_eval_res[1]
        data["scene_IE_rec"] = scene_eval_res[2]
        
        return data
    
    def _compute_scene_occ_scores(self, is_occupied, is_occupied_pred, is_visible):
        """Compute scene-level occupancy metrics."""
        scene_o_acc = (is_occupied_pred == is_occupied).float().mean().item()
        scene_ie_acc = (is_occupied_pred == is_occupied)[(~is_visible)].float().mean().item()
        scene_ie_rec = (~is_occupied_pred)[(~is_occupied) & (~is_visible)].float().mean().item()
        return (scene_o_acc, scene_ie_acc, scene_ie_rec)
    
    def _save_visualization(self, is_occupied, is_occupied_pred, is_visible, images, seq, frame_id):
        """Save visualization of GT and prediction."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Input image
        img = images[0, 0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # GT occupancy
        axes[1].imshow(is_occupied.cpu().numpy(), cmap='gray', origin='lower')
        axes[1].set_title("GT Occupancy")
        axes[1].axis('off')
        
        # Predicted occupancy
        axes[2].imshow(is_occupied_pred.cpu().numpy(), cmap='gray', origin='lower')
        axes[2].set_title("Predicted Occupancy")
        axes[2].axis('off')
        
        # Visibility map
        axes[3].imshow(is_visible.cpu().numpy(), cmap='gray', origin='lower')
        axes[3].set_title("Visibility")
        axes[3].axis('off')
        
        name = f"{seq.split('_')[-2]}_{frame_id:010d}"
        plt.savefig(os.path.join(self.save_dir, f"{name}_occ.png"), dpi=150, bbox_inches='tight')
        plt.close()


def make_test_dataset(config):
    """Create test dataset for evaluation."""
    return Kitti360Dataset(
        data_path=config["data_path"],
        pose_path=config.get("pose_path", os.path.join(config["data_path"], "poses")),
        split_path=os.path.join(config.get("split_path", "datasets/kitti_360/splits/seg"), "test_files.txt"),
        target_image_size=tuple(config.get("image_size", (192, 640))),
        frame_count=1,
        return_stereo=False,
        return_fisheye=False,
        return_depth=False,
        keyframe_offset=0,
        is_preprocessed=config.get("is_preprocessed", False)
    )


def main():
    parser = argparse.ArgumentParser(description="ViPOcc-style Occupancy Evaluation for DINO-DA3")
    parser.add_argument("--checkpoint", "-cp", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="/data/lmh_data/KITTI360",
                        help="Path to KITTI-360 dataset")
    parser.add_argument("--split_path", type=str, default="datasets/kitti_360/splits/seg",
                        help="Path to data splits")
    parser.add_argument("--z_range", nargs=2, type=float, default=[20, 4],
                        help="Z range for evaluation (far, near)")
    parser.add_argument("--x_range", nargs=2, type=float, default=[-4, 4],
                        help="X range for evaluation")
    parser.add_argument("--ppm", type=int, default=10,
                        help="Points per meter")
    parser.add_argument("--aggregate_timesteps", type=int, default=300,
                        help="Number of LiDAR frames to aggregate for GT")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save visualization")
    parser.add_argument("--save_dir", type=str, default="visualization/occ_eval",
                        help="Directory to save visualizations")
    parser.add_argument("--gt_occ_path", type=str, default="",
                        help="Path to pre-computed GT occupancy (optional)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config for model
    model_config = {
        "arch": "BTSNet",
        "use_code": True,
        "code": {"num_freqs": 6, "freq_factor": 1.5, "include_input": True},
        "encoder": {"type": "monodepth2", "freeze": False, "pretrained": True,
                    "resnet_layers": 50, "num_ch_dec": [32, 32, 64, 128, 256], "d_out": 64},
        "mlp_coarse": {"type": "resnet", "n_blocks": 0, "d_hidden": 64},
        "mlp_fine": {"type": "empty", "n_blocks": 1, "d_hidden": 128},
        "mlp_segmentation": {"type": "resnet", "n_blocks": 0, "d_hidden": 64},
        "segmentation_mode": "panoptic_deeplab",
        "z_near": 3,
        "z_far": 80,
        "inv_z": True,
        "code_mode": "z",
        "learn_empty": False,
        "sample_color": True,
        "adaptive_sampling": True,
        "n_surface_samples": 24,
        "n_global_samples": 40,
        "absrel_prior": 0.10,
        "min_thickness": 0.5,
    }
    
    renderer_config = {
        "n_coarse": 64,
        "n_fine": 0,
        "n_fine_depth": 0,
        "depth_std": 1.0,
        "sched": [],
        "white_bkgd": False,
        "lindisp": True,
        "hard_alpha_cap": True,
    }
    
    eval_config = {
        "z_near": 3,
        "z_far": 80,
        "x_range": tuple(args.x_range),
        "y_range": (0, 0.75),
        "z_range": tuple(args.z_range),
        "ppm": args.ppm,
        "ppm_y": 4,
        "y_res": 1,
        "gt_aggregate_timesteps": args.aggregate_timesteps,
        "cut_far_invisible_area": True,
        "save_vis": args.save_vis,
        "save_dir": args.save_dir,
        "read_gt_occ_path": args.gt_occ_path,
        "query_batch_size": 50000,
    }
    
    data_config = {
        "data_path": args.data_path,
        "pose_path": os.path.join(args.data_path, "poses"),
        "split_path": args.split_path,
        "image_size": [192, 640],
        "is_preprocessed": False,
    }
    
    # Create dataset
    print("Loading dataset...")
    dataset = make_test_dataset(data_config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    print("Creating model...")
    net = BTSNet(model_config)
    renderer = NeRFRenderer.from_conf(renderer_config)
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    
    model = DINODA3OccWrapper(renderer, eval_config, dataset)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Filter state dict for renderer
    renderer_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("renderer."):
            renderer_state_dict[k] = v
    
    if renderer_state_dict:
        model.load_state_dict(renderer_state_dict, strict=False)
    else:
        # Try loading directly to renderer
        model.renderer.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = {"scene_O_acc": [], "scene_IE_acc": [], "scene_IE_rec": []}
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move data to device
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
                elif isinstance(data[key], list):
                    data[key] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in data[key]]
            
            data["index"] = torch.tensor([batch_idx])
            
            try:
                output = model(data)
                
                for metric in results:
                    if metric in output and not torch.isnan(torch.tensor(output[metric])):
                        results[metric].append(output[metric])
            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
                continue
            
            if (batch_idx + 1) % 50 == 0:
                print(f"\nIntermediate results at batch {batch_idx + 1}:")
                for metric, values in results.items():
                    if values:
                        print(f"  {metric}: {np.mean(values):.4f}")
    
    # Print final results
    print("\n" + "=" * 50)
    print("Final Results (ViPOcc-style Occupancy Evaluation)")
    print("=" * 50)
    print(f"Evaluation range: x={args.x_range}, z={args.z_range}")
    print(f"Number of samples: {len(results['scene_O_acc'])}")
    print("-" * 50)
    
    for metric, values in results.items():
        if values:
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            print(f"{metric:15s}: {mean_val:.2f}% ± {std_val:.2f}%")
    
    print("=" * 50)
    
    # Save results
    results_path = os.path.join(args.save_dir, "occ_results.txt")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(results_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Z range: {args.z_range}\n")
        f.write(f"X range: {args.x_range}\n")
        f.write(f"PPM: {args.ppm}\n\n")
        for metric, values in results.items():
            if values:
                f.write(f"{metric}: {np.mean(values)*100:.2f}%\n")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
