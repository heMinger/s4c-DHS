"""
KDBTS/ViPOcc-style Occupancy Evaluator

This evaluator computes occupancy metrics using LiDAR point clouds as ground truth,
compatible with both KDBTS and ViPOcc evaluation protocols.

Metrics:
- Scene-level: scene_O_acc, scene_O_prec, scene_O_rec, scene_IE_acc, scene_IE_prec, scene_IE_rec
- Object-level (optional): object_O_acc, object_IE_acc, object_IE_rec

Reference:
- KDBTS: https://github.com/keonhee-han/KDBTS
- ViPOcc: https://github.com/fengyi233/ViPOcc
"""

import math
import os
from datetime import datetime

import numpy as np
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.data_util import make_test_dataset
from models.common.render import NeRFRenderer
from models.bts.model.models_bts import BTSNet
from models.bts.model.ray_sampler import ImageRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.projection_operations import distance_to_z

IDX = 0
EPS = 1e-4

# KITTI-360 cameras have a 5 degrees negative inclination
cam_incl_adjust = torch.tensor(
    [[1.0000000, 0.0000000, 0.0000000, 0],
     [0.0000000, 0.9961947, 0.0871557, 0],
     [0.0000000, -0.0871557, 0.9961947, 0],
     [0.0000000, 0.0000000, 0.0000000, 1]],
    dtype=torch.float32
).view(1, 1, 4, 4)


def get_pts(x_range, y_range, z_range, ppm, ppm_y, y_res=None, specify_yslice=None):
    """Generate 3D query points in a grid."""
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    if y_res is None:
        y_res = abs(int((y_range[1] - y_range[0]) * ppm_y))
    z_res = abs(int((z_range[1] - z_range[0]) * ppm))
    
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    
    if y_res == 1:
        if specify_yslice is None:
            y = torch.tensor([y_range[0] * 0.5 + y_range[1] * 0.5]).view(y_res, 1, 1).expand(-1, z_res, x_res)
        else:
            y = torch.tensor([specify_yslice]).view(y_res, 1, 1).expand(-1, z_res, x_res)
    else:
        y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    
    xyz = torch.stack((x, y, z), dim=-1)
    return xyz, (x_res, y_res, z_res)


def get_lidar_slices(point_clouds, velo_poses, y_range, y_res, max_dist):
    """
    Convert LiDAR point clouds to polar coordinate slices for occupancy checking.
    Points are binned into 1-degree angular bins to reduce noise.
    """
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

        slice_list = []

        for pc, velo_pose in zip(point_clouds, velo_poses):
            pc_world = (velo_pose @ pc.T).T

            # Select points in y-range or beyond max distance
            mask = ((pc_world[:, 1] >= min_y) & (pc_world[:, 1] <= max_y)) | \
                   (torch.norm(pc_world[:, :3], dim=-1) >= max_dist)

            slice_points = pc[mask, :2]

            # Convert to polar coordinates
            angles = torch.atan2(slice_points[:, 1], slice_points[:, 0])
            dists = torch.norm(slice_points, dim=-1)

            slice_points_polar = torch.stack((angles, dists), dim=1)
            # Sort by angles for fast lookup
            slice_points_polar = slice_points_polar[torch.sort(angles)[1], :]

            # Bin points into 1-degree bins and take minimum distance per bin
            slice_points_polar_binned = torch.zeros_like(slice_points_polar[:n_bins, :])
            bin_borders = torch.linspace(-math.pi, math.pi, n_bins + 1, device=slice_points_polar.device)

            dist = slice_points_polar[0, 1]
            border_is = torch.searchsorted(slice_points_polar[:, 0], bin_borders)

            for i in range(n_bins):
                left_i, right_i = border_is[i], border_is[i + 1]
                angle = (bin_borders[i] + bin_borders[i + 1]) * 0.5
                if right_i > left_i:
                    dist = torch.min(slice_points_polar[left_i:right_i, 1])
                slice_points_polar_binned[i, 0] = angle
                slice_points_polar_binned[i, 1] = dist

            slice_points_polar = slice_points_polar_binned

            # Append first element to last for full 360deg coverage
            slice_points_polar = torch.cat((
                torch.tensor([[slice_points_polar[-1, 0] - math.pi * 2, slice_points_polar[-1, 1]]],
                             device=slice_points_polar.device),
                slice_points_polar,
                torch.tensor([[slice_points_polar[0, 0] + math.pi * 2, slice_points_polar[0, 1]]],
                             device=slice_points_polar.device)
            ), dim=0)

            slice_list.append(slice_points_polar)

        slices.append(slice_list)

    return slices


def check_occupancy(pts, slices, velo_poses, min_dist=3):
    """
    Check occupancy of query points using LiDAR slices.
    A point is occupied if it's behind the LiDAR surface or too close.
    """
    is_occupied = torch.ones_like(pts[:, 0])
    is_visible = torch.zeros_like(pts[:, 0], dtype=torch.bool)

    thresh = (len(slices[0]) - 2) / len(slices[0])

    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    world_to_velos = torch.inverse(velo_poses)

    step = pts.shape[0] // len(slices)

    for i, slice_list in enumerate(slices):
        for j, (lidar_polar, world_to_velo) in enumerate(zip(slice_list, world_to_velos)):
            pts_velo = (world_to_velo @ pts[i * step: (i + 1) * step, :].T).T

            # Convert query points to polar coordinates in velo space
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


def project_into_cam(pts, proj, pose):
    """Project 3D points into camera coordinates."""
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    cam_pts = (proj @ (torch.inverse(pose).squeeze()[:3, :] @ pts.T)).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist


class BTSWrapper(nn.Module):
    """
    Wrapper for occupancy evaluation with KDBTS/ViPOcc-style metrics.
    
    Supports:
    - Configurable evaluation range and resolution
    - Scene-level and object-level metrics
    - Pre-computed GT loading for faster evaluation
    - Visualization and GT saving
    """
    
    def __init__(self, renderer, config, dataset) -> None:
        super().__init__()

        self.renderer = renderer
        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = config.get("occ_threshold", 0.5)

        # Evaluation space configuration (KDBTS/ViPOcc standard)
        self.x_range = tuple(config.get("x_range", [-4, 4]))
        self.y_range = tuple(config.get("y_range", [0, 0.75]))
        self.z_range = tuple(config.get("z_range", [20, 4]))  # Note: from far to near
        self.ppm = config.get("ppm", 10)  # points per meter
        self.ppm_y = config.get("ppm_y", 4)
        self.y_res = config.get("y_res", 1)
        self.specify_yslice = config.get("specify_yslice", None)

        # GT generation settings
        self.aggregate_timesteps = config.get("gt_aggregate_timesteps", 20)  # KDBTS: 20, ViPOcc: 300
        self.cut_far_invisible_area = config.get("cut_far_invisible_area", False)
        
        # Pre-computed GT paths (optional, for faster evaluation)
        self.read_gt_occ_path = config.get("read_gt_occ_path", "")
        self.save_gt_occ_path = config.get("save_gt_occ_path", "")
        
        # Object-level evaluation (ViPOcc style)
        self.is_eval_object = config.get("is_eval_object", False)
        self.read_gt_obj_path = config.get("read_gt_obj_path", "")
        self.obj_z_expand = int(config.get("obj_z_expand", 2) * self.ppm)
        self.obj_x_expand = int(config.get("obj_x_expand", 0.5) * self.ppm)

        # Visualization settings
        self.save_vis = config.get("save_vis", False)
        self.save_dir = config.get("save_dir", "visualization/occ_eval")
        if self.save_vis:
            os.makedirs(self.save_dir, exist_ok=True)

        # Initialize sampler
        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=3)
        self.dataset = dataset
        
        # Counters for statistics
        self.count = 0
        
        print(f"[OccEvaluator] Evaluation space: x={self.x_range}, y={self.y_range}, z={self.z_range}")
        print(f"[OccEvaluator] Resolution: ppm={self.ppm}, ppm_y={self.ppm_y}, y_res={self.y_res}")
        print(f"[OccEvaluator] GT aggregate timesteps: {self.aggregate_timesteps}")

    def compute_object_occ_scores(self, obj_locs_map, is_occupied, is_occupied_pred, is_visible):
        """
        Compute object-level occupancy metrics (ViPOcc style).
        
        Args:
            obj_locs_map: Object location map (0=background, 255=occupied, 1,2,3...=object IDs)
        """
        obj_ids = torch.unique(obj_locs_map)[1:-1]  # Exclude 0 and 255
        h, w = obj_locs_map.shape

        # Get valid map where only object areas are 1
        obj_valid_map = is_occupied.new_zeros(is_occupied.shape)

        # No annotated objects
        if (obj_locs_map != 0).sum() == 0:
            nan_val = torch.tensor(float('nan'), device=obj_locs_map.device)
            return nan_val.clone(), nan_val.clone(), nan_val.clone()

        for obj_id in obj_ids:
            mask = (obj_locs_map == obj_id).nonzero()
            y_min, x_min = mask.min(0).values
            y_max, x_max = mask.max(0).values

            # Add area expansions
            y_min = max(y_min - self.obj_z_expand, 0)
            y_max = min(y_max + self.obj_z_expand, h)
            x_min = max(x_min - self.obj_x_expand, 0)
            x_max = min(x_max + self.obj_x_expand, w)

            obj_valid_map[y_min:y_max, x_min:x_max] = 1

        obj_is_occupied = is_occupied[obj_valid_map.bool()]
        obj_is_occupied_pred = is_occupied_pred[obj_valid_map.bool()]
        obj_is_visible = is_visible[obj_valid_map.bool()]

        obj_o_acc = (obj_is_occupied_pred == obj_is_occupied).float().mean().item()
        obj_ie_acc = (obj_is_occupied_pred == obj_is_occupied)[(~obj_is_visible)].float().mean().item()
        obj_ie_rec = (~obj_is_occupied_pred)[(~obj_is_occupied) & (~obj_is_visible)].float().mean().item()

        return obj_o_acc, obj_ie_acc, obj_ie_rec

    def compute_scene_occ_scores(self, is_occupied, is_occupied_pred, is_visible):
        """
        Compute scene-level occupancy metrics (KDBTS/ViPOcc style).
        
        Returns:
            - o_acc: Overall occupancy accuracy
            - o_prec: Occupancy precision
            - o_rec: Occupancy recall
            - ie_acc: Invisible-empty accuracy
            - ie_prec: Invisible-empty precision
            - ie_rec: Invisible-empty recall
        """
        # Overall metrics
        o_acc = (is_occupied_pred == is_occupied).float().mean().item()
        o_prec = is_occupied[is_occupied_pred].float().mean().item() if is_occupied_pred.any() else 0.0
        o_rec = is_occupied_pred[is_occupied].float().mean().item() if is_occupied.any() else 0.0
        
        # Invisible-empty metrics (key for scene completion)
        invisible_mask = ~is_visible
        if invisible_mask.any():
            ie_acc = (is_occupied_pred == is_occupied)[invisible_mask].float().mean().item()
            
            # IE precision: among predicted empty in invisible region, how many are truly empty
            pred_empty_invisible = (~is_occupied_pred) & invisible_mask
            if pred_empty_invisible.any():
                ie_prec = (~is_occupied)[pred_empty_invisible].float().mean().item()
            else:
                ie_prec = 0.0
            
            # IE recall: among truly empty in invisible region, how many are predicted empty
            true_empty_invisible = (~is_occupied) & invisible_mask
            if true_empty_invisible.any():
                ie_rec = (~is_occupied_pred)[true_empty_invisible].float().mean().item()
            else:
                ie_rec = 0.0
        else:
            ie_acc = ie_prec = ie_rec = 0.0
        
        return o_acc, o_prec, o_rec, ie_acc, ie_prec, ie_rec

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)  # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)  # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)  # n, v, 4, 4 (-1, 1)
        index = data["index"].item()

        self.count += 1

        seq, id, is_right = self.dataset._datapoints[index]
        seq_len = self.dataset._img_ids[seq].shape[0]
        init_id = id

        n, v, c, h, w = images.shape
        device = images.device

        T_velo_to_pose = torch.tensor(self.dataset._calibs["T_velo_to_pose"], device=device)

        # Transform to camera0 coordinate system with inclination adjustment
        world_transform = torch.inverse(poses[:, :1, :, :])
        world_transform = cam_incl_adjust.to(device) @ world_transform
        poses = world_transform @ poses

        self.sampler.height = h
        self.sampler.width = w

        # ===== Load or compute GT occupancy =====
        if self.read_gt_occ_path:
            # Load pre-computed GT
            name = f"{seq}_{self.dataset._img_ids[seq][init_id]:010d}"
            occ_gt = np.load(os.path.join(self.read_gt_occ_path, name + "_occgt.npy"))
            vis_gt = np.load(os.path.join(self.read_gt_occ_path, name + "_visgt.npy"))
            is_occupied = torch.from_numpy(occ_gt).bool().to(device)
            is_visible = torch.from_numpy(vis_gt).bool().to(device)
            
            # Handle resolution mismatch
            z_size = self.ppm * (self.z_range[0] - self.z_range[1])
            curr_gt_size = is_occupied.shape[0]
            if z_size < curr_gt_size:
                # Crop to match current evaluation range
                upper_margin = int(self.ppm * (50.0 - self.z_range[0]))  # Assuming GT is 4-50m
                lower_margin = int(self.ppm * (self.z_range[1] - 4.0))
                is_occupied = is_occupied[upper_margin:curr_gt_size - lower_margin, :]
                is_visible = is_visible[upper_margin:curr_gt_size - lower_margin, :]
            
            velo_poses = None  # Not needed when loading GT
        else:
            # Compute GT from LiDAR point clouds
            points_all = []
            velo_poses = []
            
            for frame_id in range(id, min(id + self.aggregate_timesteps, seq_len)):
                lidar_path = os.path.join(
                    self.dataset.data_path, "data_3d_raw", seq, 
                    "velodyne_points", "data",
                    f"{self.dataset._img_ids[seq][frame_id]:010d}.bin"
                )
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
                points[:, 3] = 1.0
                points = torch.tensor(points, device=device)
                
                velo_pose = (world_transform.squeeze() @ 
                            torch.tensor(self.dataset._poses[seq][frame_id], device=device) @ 
                            T_velo_to_pose)
                
                points_all.append(points)
                velo_poses.append(velo_pose)

            velo_poses = torch.stack(velo_poses, dim=0)

        # ===== Model prediction =====
        # ImageRaySampler.sample returns: (rays, rgb_gt, da3_depths, low_conf_mask)
        rays, _, _, _ = self.sampler.sample(None, poses[:, :1, :, :], projs[:, :1, :, :])

        ids_encoder = [0]
        self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
        self.renderer.net.encode({}, images, projs, poses, 
                                  ids_encoder=ids_encoder, 
                                  ids_render=ids_encoder,
                                  images_alt=images * 0.5 + 0.5)
        self.renderer.net.set_scale(0)

        # Generate query points
        q_pts, (xd, yd, zd) = get_pts(
            self.x_range, self.y_range, self.z_range, 
            self.ppm, self.ppm_y, self.y_res, self.specify_yslice
        )
        q_pts = q_pts.to(device).view(-1, 3)

        # Query densities from the model
        densities = []
        for i_from in range(0, len(q_pts), self.query_batch_size):
            i_to = min(i_from + self.query_batch_size, len(q_pts))
            q_pts_ = q_pts[i_from:i_to]
            _, _, densities_ = self.renderer.net(q_pts_.unsqueeze(0), only_density=True)
            densities.append(densities_.squeeze(0))
        
        densities = torch.cat(densities, dim=0).squeeze()
        is_occupied_pred = densities > self.occ_threshold

        # ===== Compute GT from LiDAR if not pre-loaded =====
        if not self.read_gt_occ_path:
            slices = get_lidar_slices(
                points_all, velo_poses, self.y_range, yd,
                (self.z_range[0] ** 2 + self.x_range[0] ** 2) ** 0.5
            )
            is_occupied, is_visible = check_occupancy(q_pts, slices, velo_poses)

        # Save GT if requested
        if self.save_gt_occ_path:
            name = f"{seq}_{self.dataset._img_ids[seq][init_id]:010d}"
            is_occupied_save = is_occupied.reshape(zd, xd).cpu().numpy()
            is_visible_save = is_visible.reshape(zd, xd).cpu().numpy()
            np.save(os.path.join(self.save_gt_occ_path, name + "_occgt.npy"), is_occupied_save)
            np.save(os.path.join(self.save_gt_occ_path, name + "_visgt.npy"), is_visible_save)
            print(f"Saved GT for {name}")

        # Reshape for 2D operations
        is_occupied = is_occupied.reshape(zd, xd)
        is_occupied_pred = is_occupied_pred.reshape(zd, xd)
        is_visible = is_visible.reshape(zd, xd)

        # Only invisible points can be truly "occupied" (scene completion task)
        is_occupied = is_occupied & (~is_visible)

        # ===== Optionally cut far invisible area =====
        if self.cut_far_invisible_area:
            z_indices = torch.nonzero(is_visible)[:, 0]
            if z_indices.shape[0] == 0:
                # No visible area, skip this sample
                nan_val = torch.tensor(float('nan'), device=device)
                data["scene_O_acc"] = nan_val
                data["scene_O_prec"] = nan_val
                data["scene_O_rec"] = nan_val
                data["scene_IE_acc"] = nan_val
                data["scene_IE_prec"] = nan_val
                data["scene_IE_rec"] = nan_val
                data["object_O_acc"] = nan_val
                data["object_IE_acc"] = nan_val
                data["object_IE_rec"] = nan_val
                globals()["IDX"] += 1
                return data
            
            z_min_val = max(0, torch.min(z_indices).item() - 2 * self.ppm)
            is_occupied = is_occupied[z_min_val:, :]
            is_visible = is_visible[z_min_val:, :]
            is_occupied_pred = is_occupied_pred[z_min_val:, :]

        # ===== Compute scene-level metrics =====
        o_acc, o_prec, o_rec, ie_acc, ie_prec, ie_rec = self.compute_scene_occ_scores(
            is_occupied.flatten(), is_occupied_pred.flatten(), is_visible.flatten()
        )

        # ===== Object-level evaluation (optional, ViPOcc style) =====
        if self.is_eval_object and self.read_gt_obj_path:
            name = f"{seq}_{self.dataset._img_ids[seq][init_id]:010d}"
            obj_path = os.path.join(self.read_gt_obj_path, name + "_occgt_anno.png")
            if os.path.exists(obj_path):
                obj_locs_map = Image.open(obj_path)
                obj_locs_map = torch.from_numpy(np.array(obj_locs_map).astype(np.uint8)).to(device)
                
                # Handle size mismatch
                if obj_locs_map.shape != is_occupied.shape:
                    obj_locs_map = obj_locs_map[:is_occupied.shape[0], :is_occupied.shape[1]]
                
                obj_o_acc, obj_ie_acc, obj_ie_rec = self.compute_object_occ_scores(
                    obj_locs_map, is_occupied, is_occupied_pred, is_visible
                )
            else:
                obj_o_acc = obj_ie_acc = obj_ie_rec = float('nan')
        else:
            obj_o_acc = obj_ie_acc = obj_ie_rec = 0.0

        # ===== Save visualization =====
        if self.save_vis:
            name = f"{seq.split('_')[-2]}_{self.dataset._img_ids[seq][init_id]:010d}"
            self._save_visualization(is_occupied, is_occupied_pred, images[0, 0], name)

        # ===== Store results =====
        # Scene-level metrics (KDBTS/ViPOcc compatible)
        data["scene_O_acc"] = o_acc
        data["scene_O_prec"] = o_prec
        data["scene_O_rec"] = o_rec
        data["scene_IE_acc"] = ie_acc
        data["scene_IE_prec"] = ie_prec
        data["scene_IE_rec"] = ie_rec
        
        # Object-level metrics (ViPOcc style)
        data["object_O_acc"] = obj_o_acc
        data["object_IE_acc"] = obj_ie_acc
        data["object_IE_rec"] = obj_ie_rec

        # For TensorBoard logging
        data["z_near"] = torch.tensor(self.z_near, device=device)
        data["z_far"] = torch.tensor(self.z_far, device=device)

        globals()["IDX"] += 1
        return data

    def _save_visualization(self, is_occupied, is_occupied_pred, image, name):
        """Save occupancy visualization."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input image
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # GT occupancy
        axes[1].imshow(is_occupied.cpu().numpy(), cmap='gray')
        axes[1].set_title("GT Occupancy")
        axes[1].axis('off')
        
        # Predicted occupancy
        axes[2].imshow(is_occupied_pred.cpu().numpy(), cmap='gray')
        axes[2].set_title("Predicted Occupancy")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{name}.png"), dpi=150)
        plt.close()


def evaluation(local_rank, config):
    """Main evaluation entry point."""
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    """Create test dataloader."""
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=config["num_workers"],
        shuffle=False, 
        drop_last=False
    )
    return test_loader


def get_metrics(config, device):
    """Define metrics to track during evaluation."""
    names = [
        # Scene-level metrics
        "scene_O_acc", "scene_O_prec", "scene_O_rec",
        "scene_IE_acc", "scene_IE_prec", "scene_IE_rec",
        # Object-level metrics
        "object_O_acc", "object_IE_acc", "object_IE_rec"
    ]
    metrics = {
        name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) 
        for name in names
    }
    return metrics


def initialize(config: dict, logger=None):
    """Initialize model and wrapper."""
    arch = config["model_conf"].get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(
        renderer,
        config["model_conf"],
        make_test_dataset(config["data"])
    )

    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    """Optional visualization callback (not implemented)."""
    pass
