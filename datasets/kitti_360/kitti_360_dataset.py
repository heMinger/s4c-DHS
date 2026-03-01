import os
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import omegaconf
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from datasets.kitti_360.annotation import KITTI360Bbox3D
from utils.augmentation import get_color_aug_fn

from datasets.kitti_360.labels import labels

name2label = {label.name: label for label in labels}
id2ProposedId = {label.id: label.trainId for label in labels}

PropsedId2TrainId = dict(enumerate(list(set(id2ProposedId.values()))))
PropsedId2TrainId = {v : k for k, v in PropsedId2TrainId.items()}
id2TrainId = {k : PropsedId2TrainId[v] for k, v in id2ProposedId.items()}

class FisheyeToPinholeSampler:
    def __init__(self, K_target, target_image_size, calibs, rotation=None):
        self._compute_transform(K_target, target_image_size, calibs, rotation)

    def _compute_transform(self, K_target, target_image_size, calibs, rotation=None):
        x = torch.linspace(-1, 1, target_image_size[1]).view(1, -1).expand(target_image_size)
        y = torch.linspace(-1, 1, target_image_size[0]).view(-1, 1).expand(target_image_size)
        z = torch.ones_like(x)
        xyz = torch.stack((x, y, z), dim=-1).view(-1, 3)

        # Unproject
        xyz = (torch.inverse(torch.tensor(K_target)) @ xyz.T).T

        if rotation is not None:
            xyz = (torch.tensor(rotation) @ xyz.T).T

        # Backproject into fisheye
        xyz = xyz / torch.norm(xyz, dim=-1, keepdim=True)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        xi_src = calibs["mirror_parameters"]["xi"]
        x = x / (z + xi_src)
        y = y / (z + xi_src)

        k1 = calibs["distortion_parameters"]["k1"]
        k2 = calibs["distortion_parameters"]["k2"]

        r = x*x + y*y
        factor = (1 + k1 * r + k2 * r * r)
        x = x * factor
        y = y * factor

        gamma0 = calibs["projection_parameters"]["gamma1"]
        gamma1 = calibs["projection_parameters"]["gamma2"]
        u0 = calibs["projection_parameters"]["u0"]
        v0 = calibs["projection_parameters"]["v0"]

        x = x * gamma0 + u0
        y = y * gamma1 + v0

        xy = torch.stack((x, y), dim=-1).view(1, *target_image_size, 2)
        self.sample_pts = xy

    def resample(self, img):
        img = img.unsqueeze(0)
        resampled_img = F.grid_sample(img, self.sample_pts, align_corners=True).squeeze(0)
        return resampled_img


class Kitti360Dataset(Dataset):
    def __init__(self,
                 data_path: str,
                 pose_path: str,
                 split_path: Optional[str],
                 depth_source: str = "lidar",
                 depth_da3_path: Optional[str] = None,
                 depth_da3_scale: float = 256.0,
                 target_image_size=(192, 640),
                 return_stereo=False,
                 return_depth=False,
                 return_fisheye=True,
                 return_3d_bboxes=False,
                 return_segmentation=False,
                 segmentation_mode=None,
                 data_segmentation_path=None,
                 frame_count=2,
                 keyframe_offset=0,
                 dilation=1,
                 fisheye_rotation=0,
                 fisheye_offset=0,
                 eigen_depth=True,
                 color_aug=False,
                 is_preprocessed=False,
                 load_kitti_360_segmentation_gt=False,
                 constrain_to_datapoints=False,
                 additional_random_front_offset=False,
                 online_depth_generation=False,
                 return_lidar_depth=False,
                 # New parameters for online DA3 depth generation
                 depth_mode: str = "offline",
                 da3_checkpoint_path: Optional[str] = None,
                 da3_conf_percentile: float = 20.0,
                 ):
        self.data_path = data_path
        self.data_segmentation_path = data_segmentation_path
        self.depth_source = (depth_source or "lidar").lower()
        self.depth_da3_path = depth_da3_path
        self.depth_da3_scale = float(depth_da3_scale)
        self.pose_path = pose_path
        self.split_path = split_path
        # Support None for original image size (no resizing)
        self.target_image_size = tuple(target_image_size) if target_image_size else None
        self.return_stereo = return_stereo
        self.return_fisheye = return_fisheye
        self.return_depth = return_depth
        self.return_lidar_depth = return_lidar_depth
        self.return_3d_bboxes = return_3d_bboxes
        self.return_segmentation = return_segmentation
        self.segmentation_mode = segmentation_mode
        self.frame_count = frame_count
        self.dilation = dilation
        self.fisheye_rotation = fisheye_rotation
        self.fisheye_offset = fisheye_offset
        self.keyframe_offset = keyframe_offset
        self.eigen_depth = eigen_depth
        self.color_aug = color_aug
        self.is_preprocessed = is_preprocessed
        self.load_kitti_360_segmentation_gt = load_kitti_360_segmentation_gt
        self.constrain_to_datapoints = constrain_to_datapoints
        
        # Online DA3 depth generation settings
        self.depth_mode = depth_mode.lower() if depth_mode else "offline"
        self.online_depth_generation = (self.depth_mode == "online") or online_depth_generation
        self.da3_checkpoint_path = da3_checkpoint_path
        self.da3_conf_percentile = da3_conf_percentile
        
        self._segmentation_mode_norm = (self.segmentation_mode or "").lower()

        if isinstance(self.fisheye_rotation, float) or isinstance(self.fisheye_rotation, int):
            self.fisheye_rotation = (0, self.fisheye_rotation)
        self.fisheye_rotation = tuple(self.fisheye_rotation)

        # Support random fisheye offset
        if type(self.fisheye_offset) == int:
            self.random_fisheye_offset = False
            self.fisheye_offset = (self.fisheye_offset, )
        elif type(self.fisheye_offset) in [tuple, list, omegaconf.listconfig.ListConfig]:
            self.random_fisheye_offset = True
            self.fisheye_offset = tuple(sorted(self.fisheye_offset))
        else:
            raise ValueError(f"Invalid datatype for fisheye offset: {type(self.fisheye_offset)}")

        if additional_random_front_offset and not self.random_fisheye_offset:
            raise ValueError("Random Fisheye Offset needs to be active for additional random front offset!")
        else:
            self.additional_random_front_offset = additional_random_front_offset

        self._sequences = self._get_sequences(self.data_path)

        self._calibs = self._load_calibs(self.data_path, self.fisheye_rotation)
        
        # For original resolution, get size from first image
        if self.target_image_size is None:
            self._original_perspective_size = self._calibs["im_size"]  # (H, W)
            self._original_fisheye_size = (1400, 1400)  # KITTI-360 fisheye is always 1400x1400
            print(f"[Dataset] Using original resolution: perspective={self._original_perspective_size}, fisheye={self._original_fisheye_size}")
        else:
            self._original_perspective_size = self.target_image_size
            self._original_fisheye_size = self.target_image_size
        
        # Resamplers need target size - use original fisheye size if no target specified
        resampler_target_size = self.target_image_size if self.target_image_size else self._original_perspective_size
        self._resampler_02, self._resampler_03 = self._get_resamplers(self._calibs, self._calibs["K_fisheye"], resampler_target_size)
        self._img_ids, self._poses = self._load_poses(self.pose_path, self._sequences)
        self._left_offset = ((self.frame_count - 1) // 2 + self.keyframe_offset) * self.dilation

        self._perspective_folder = "data_rect" if not self.is_preprocessed else f"data_{self.target_image_size[0]}x{self.target_image_size[1]}"
        # Segmentation folder: use 'data_192x640' as subfolder name 
        # Note: The actual files inside may be at original resolution (1408x376)
        # This is just the folder name convention used in the dataset
        self._segmentation_perspective_folder = "data_192x640"
        self._segmentation_fisheye_folder = "data_192x640_0x-15"
        self._fisheye_folder = "data_rgb" if not self.is_preprocessed else f"data_{self.target_image_size[0]}x{self.target_image_size[1]}_{self.fisheye_rotation[0]}x{self.fisheye_rotation[1]}"
        
        # Store DA3 generator reference (will be initialized lazily in trainer)
        self._da3_generator = None
        
        # Initialize online depth model if needed
        self._depth_model = None
        if self.online_depth_generation:
            self._init_online_depth_model()

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path, self._img_ids)
        elif self.return_segmentation:
            self._datapoints = self._semantics_split(self._sequences, self.data_path, self._img_ids)
        else:
            self._datapoints = self._full_split(self._sequences, self._img_ids, self.check_file_integrity)

        if self.return_3d_bboxes:
            self._3d_bboxes = self._load_3d_bboxes(Path(data_path) / "data_3d_bboxes" / "train_full", self._sequences)

        if self._segmentation_mode_norm in ("kitti-360", "kitti_360", "kitti360") or self.load_kitti_360_segmentation_gt:
            # Segmentations are only provided for the left camera
            self._datapoints = [dp for dp in self._datapoints if not dp[2]]

            # make sure we can load all segmentation masks
            self._datapoints = [dp for dp in self._datapoints if self.check_segmentation(dp)]

        if self.constrain_to_datapoints:
            print("Using maximum datapoint as last image of sequence.")
            seq_max_id = {seq: max([0] + [d[1] for d in self._datapoints if d[0] == seq]) for seq in self._sequences}
            for seq in self._sequences:
                self._poses[seq] = self._poses[seq][:seq_max_id[seq]+1]
                self._img_ids[seq] = self._img_ids[seq][:seq_max_id[seq]+1]

        self._skip = 0
        self.length = len(self._datapoints)

    def _init_online_depth_model(self):
        """Initialize Depth Anything V2 model for online depth generation."""
        try:
            # Try to import Depth Anything V2
            import sys
            depth_anything_path = "/home/lmh/Depth-Anything-V2"
            if depth_anything_path not in sys.path:
                sys.path.insert(0, depth_anything_path)
            
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # Model configuration for ViT-L
            model_configs = {
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
            }
            
            encoder = 'vitl'
            self._depth_model = DepthAnythingV2(**model_configs[encoder])
            
            # Load pretrained weights
            weights_path = os.path.join(depth_anything_path, "checkpoints", "depth_anything_v2_metric_hypersim_vitl.pth")
            if not os.path.exists(weights_path):
                weights_path = os.path.join(depth_anything_path, "checkpoints", "depth_anything_v2_vitl.pth")
            
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
                self._depth_model.load_state_dict(state_dict, strict=False)
                print(f"[Online Depth] Loaded weights from {weights_path}")
            else:
                print(f"[Online Depth] Warning: weights not found at {weights_path}")
            
            self._depth_model.eval()
            # Move to GPU if available
            if torch.cuda.is_available():
                self._depth_model = self._depth_model.cuda()
            
            print("[Online Depth] Depth Anything V2 model initialized successfully")
            
        except Exception as e:
            print(f"[Online Depth] Failed to initialize: {e}")
            print("[Online Depth] Falling back to offline depth loading")
            self._depth_model = None
            self.online_depth_generation = False

    def _generate_depth_online(self, img_path):
        """Generate depth map online using Depth Anything V2."""
        if self._depth_model is None:
            return None
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            
            # Prepare input
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            with torch.no_grad():
                depth = self._depth_model(img_tensor)
            
            # Convert to numpy
            depth = depth.squeeze().cpu().numpy()
            
            # Resize if needed
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return depth[None, :, :]  # Add channel dimension
            
        except Exception as e:
            print(f"[Online Depth] Generation failed: {e}")
            return None

    def check_segmentation(self, dp):
        """Checks for a datapoint dp if we can load all the segmentation masks for all image_ids."""
        sequence, id, is_right = dp
        seq_len = self._img_ids[sequence].shape[0]

        ids = [id] + [max(min(i, seq_len - 1), 0) for i in
                      range(id - self._left_offset, id - self._left_offset + self.frame_count * self.dilation,
                            self.dilation) if i != id]
        img_ids = [self.get_img_id_from_id(sequence, id) for id in ids]

        for img_id in img_ids:
            # /data/lmh_data/KITTI360/data_2d_semantics/2013_05_28_drive_0000_sync/image_00/data_192x640
            # _p = os.path.join(self.data_path, "data_2d_semantics", "train", sequence, "image_00", "semantic",
            _p = os.path.join(self.data_path, "data_2d_semantics", "train", sequence, "image_00", "data_192x640",
                              f"{img_id:010d}.png")

            if not os.path.isfile(_p):
                return False

        return True

    def check_file_integrity(self, seq, id):
        dp = Path(self.data_path)
        image_00 = dp / "data_2d_raw" / seq / "image_00" / self._perspective_folder
        image_01 = dp / "data_2d_raw" / seq / "image_01" / self._perspective_folder
        image_02 = dp / "data_2d_raw" / seq / "image_02" / self._fisheye_folder
        image_03 = dp / "data_2d_raw" / seq / "image_03" / self._fisheye_folder

        fisheye_offset = self.fisheye_offset[-1]

        seq_len = self._img_ids[seq].shape[0]

        ids = [id] + [max(min(i, seq_len - 1), 0) for i in range(id - self._left_offset, id - self._left_offset + self.frame_count * self.dilation, self.dilation) if i != id]
        ids_fish = [max(min(id + fisheye_offset, seq_len - 1), 0)] + [max(min(i, seq_len - 1), 0) for i in range(id + fisheye_offset - self._left_offset, id + fisheye_offset - self._left_offset + self.frame_count * self.dilation, self.dilation) if i != id + fisheye_offset]

        img_ids = [self.get_img_id_from_id(seq, id) for id in ids]
        img_ids_fish = [self.get_img_id_from_id(seq, id) for id in ids_fish]

        for img_id in img_ids:
            if not ((image_00 / f"{img_id:010d}.png").exists() and (image_01 / f"{img_id:010d}.png").exists()):
                return False
        if self.return_fisheye:
            for img_id in img_ids_fish:
                if not ((image_02 / f"{img_id:010d}.png").exists() and (image_03 / f"{img_id:010d}.png").exists()):
                    return False
        return True

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        seqs_path = Path(data_path) / "data_2d_raw"
        for seq in seqs_path.iterdir():
            if not seq.is_dir():
                continue
            all_sequences.append(seq.name)

        return all_sequences

    @staticmethod
    def _full_split(sequences, img_ids, check_integrity):
        datapoints = []
        for seq in sorted(sequences):
            ids = [id for id in range(len(img_ids[seq])) if check_integrity(seq, id)]
            datapoints_seq = [(seq, id, False) for id in ids] + [(seq, id, True) for id in ids]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _semantics_split(sequences, data_path, img_ids):
        datapoints = []
        for seq in sorted(sequences):
            datapoints_seq = [(seq, id, False) for id in range(len(img_ids[seq]))]
            datapoints_seq = [dp for dp in datapoints_seq if os.path.exists(os.path.join(data_path, "data_2d_semantics", "train", seq, "image_00", "semantic_rgb", f"{img_ids[seq][dp[1]]:010d}.png"))]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _load_split(split_path, img_ids):
        img_id2id = {seq: {id: i for i, id in enumerate(ids)} for seq, ids in img_ids.items()}

        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            seq = segments[0]
            id = img_id2id[seq][int(segments[1])]
            return seq, id, segments[2][0] == "r"

        return list(map(split_line, lines))

    @staticmethod
    def _load_calibs(data_path, fisheye_rotation=0):
        data_path = Path(data_path)

        calib_folder = data_path / "calibration"
        cam_to_pose_file = calib_folder / "calib_cam_to_pose.txt"
        cam_to_velo_file = calib_folder / "calib_cam_to_velo.txt"
        intrinsics_file = calib_folder / "perspective.txt"
        fisheye_02_file = calib_folder / "image_02.yaml"
        fisheye_03_file = calib_folder / "image_03.yaml"

        cam_to_pose_data = {}
        with open(cam_to_pose_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    cam_to_pose_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass

        cam_to_velo_data = None
        with open(cam_to_velo_file, 'r') as f:
            line = f.readline()
            try:
                cam_to_velo_data = np.array([float(x) for x in line.split()], dtype=np.float32)
            except ValueError:
                pass

        intrinsics_data = {}
        with open(intrinsics_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    intrinsics_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass

        with open(fisheye_02_file, 'r') as f:
            f.readline() # Skips first line that defines the YAML version
            fisheye_02_data = yaml.safe_load(f)

        with open(fisheye_03_file, 'r') as f:
            f.readline() # Skips first line that defines the YAML version
            fisheye_03_data = yaml.safe_load(f)

        im_size_rect = (int(intrinsics_data["S_rect_00"][1]), int(intrinsics_data["S_rect_00"][0]))
        im_size_fish = (fisheye_02_data["image_height"], fisheye_02_data["image_width"])

        # Projection matrices
        # We use these projection matrices also when resampling the fisheye cameras.
        # This makes downstream processing easier, but it could be done differently.
        P_rect_00 = np.reshape(intrinsics_data['P_rect_00'], (3, 4))
        P_rect_01 = np.reshape(intrinsics_data['P_rect_01'], (3, 4))

        # Rotation matrices from raw to rectified -> Needs to be inverted later
        R_rect_00 = np.eye(4, dtype=np.float32)
        R_rect_01 = np.eye(4, dtype=np.float32)
        R_rect_00[:3, :3] = np.reshape(intrinsics_data['R_rect_00'], (3, 3))
        R_rect_01[:3, :3] = np.reshape(intrinsics_data['R_rect_01'], (3, 3))

        # Rotation matrices from resampled fisheye to raw fisheye
        fisheye_rotation = np.array(fisheye_rotation).reshape((1, 2))
        R_02 = np.eye(4, dtype=np.float32)
        R_03 = np.eye(4, dtype=np.float32)
        R_02[:3, :3] = Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]], degrees=True).as_matrix().astype(np.float32)
        R_03[:3, :3] = Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]] * np.array([[1, -1]]), degrees=True).as_matrix().astype(np.float32)

        # Load cam to pose transforms
        T_00_to_pose = np.eye(4, dtype=np.float32)
        T_01_to_pose = np.eye(4, dtype=np.float32)
        T_02_to_pose = np.eye(4, dtype=np.float32)
        T_03_to_pose = np.eye(4, dtype=np.float32)
        T_00_to_velo = np.eye(4, dtype=np.float32)

        T_00_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_00"], (3, 4))
        T_01_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_01"], (3, 4))
        T_02_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_02"], (3, 4))
        T_03_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_03"], (3, 4))
        T_00_to_velo[:3, :] = np.reshape(cam_to_velo_data, (3, 4))

        # Compute cam to pose transforms for rectified perspective cameras
        T_rect_00_to_pose = T_00_to_pose @ np.linalg.inv(R_rect_00)
        T_rect_01_to_pose = T_01_to_pose @ np.linalg.inv(R_rect_01)

        # Compute cam to pose transform for fisheye cameras
        T_02_to_pose = T_02_to_pose @ R_02
        T_03_to_pose = T_03_to_pose @ R_03

        # Compute velo to cameras and velo to pose transforms
        T_velo_to_rect_00 = R_rect_00 @ np.linalg.inv(T_00_to_velo)
        T_velo_to_pose = T_rect_00_to_pose @ T_velo_to_rect_00
        T_velo_to_rect_01 = np.linalg.inv(T_rect_01_to_pose) @ T_velo_to_pose

        # Calibration matrix is the same for both perspective cameras
        K = P_rect_00[:3, :3]

        # Normalize calibration
        f_x = K[0, 0] / im_size_rect[1]
        f_y = K[1, 1] / im_size_rect[0]
        c_x = K[0, 2] / im_size_rect[1]
        c_y = K[1, 2] / im_size_rect[0]

        # Change to image coordinates [-1, 1]
        K[0, 0] = f_x * 2.
        K[1, 1] = f_y * 2.
        K[0, 2] = c_x * 2. - 1
        K[1, 2] = c_y * 2. - 1

        # Convert fisheye calibration to [-1, 1] image dimensions
        fisheye_02_data["projection_parameters"]["gamma1"] = (fisheye_02_data["projection_parameters"]["gamma1"] / im_size_fish[1]) * 2.
        fisheye_02_data["projection_parameters"]["gamma2"] = (fisheye_02_data["projection_parameters"]["gamma2"] / im_size_fish[0]) * 2.
        fisheye_02_data["projection_parameters"]["u0"] = (fisheye_02_data["projection_parameters"]["u0"] / im_size_fish[1]) * 2. - 1.
        fisheye_02_data["projection_parameters"]["v0"] = (fisheye_02_data["projection_parameters"]["v0"] / im_size_fish[0]) * 2. - 1.

        fisheye_03_data["projection_parameters"]["gamma1"] = (fisheye_03_data["projection_parameters"]["gamma1"] / im_size_fish[1]) * 2.
        fisheye_03_data["projection_parameters"]["gamma2"] = (fisheye_03_data["projection_parameters"]["gamma2"] / im_size_fish[0]) * 2.
        fisheye_03_data["projection_parameters"]["u0"] = (fisheye_03_data["projection_parameters"]["u0"] / im_size_fish[1]) * 2. - 1.
        fisheye_03_data["projection_parameters"]["v0"] = (fisheye_03_data["projection_parameters"]["v0"] / im_size_fish[0]) * 2. - 1.

        # Use same camera calibration as perspective cameras for resampling
        # K_fisheye = np.eye(3, dtype=np.float32)
        # K_fisheye[0, 0] = 2
        # K_fisheye[1, 1] = 2

        K_fisheye = K

        calibs = {
            "K_perspective": K,
            "K_fisheye": K_fisheye,
            "T_cam_to_pose": {
                "00": T_rect_00_to_pose,
                "01": T_rect_01_to_pose,
                "02": T_02_to_pose,
                "03": T_03_to_pose,
            },
            "T_velo_to_cam": {
                "00": T_velo_to_rect_00,
                "01": T_velo_to_rect_01,
            },
            "T_velo_to_pose": T_velo_to_pose,
            "fisheye": {
                "calib_02": fisheye_02_data,
                "calib_03": fisheye_03_data,
                "R_02": R_02[:3, :3],
                "R_03": R_03[:3, :3]
            },
            "im_size": im_size_rect
        }

        return calibs

    @staticmethod
    def _get_resamplers(calibs, K_target, target_image_size):
        resampler_02 = FisheyeToPinholeSampler(K_target, target_image_size, calibs["fisheye"]["calib_02"], calibs["fisheye"]["R_02"])
        resampler_03 = FisheyeToPinholeSampler(K_target, target_image_size, calibs["fisheye"]["calib_03"], calibs["fisheye"]["R_03"])

        return resampler_02, resampler_03

    @staticmethod
    def _load_poses(pose_path, sequences):
        ids = {}
        poses = {}

        for seq in sequences:
            pose_file = Path(pose_path) / seq / f"poses.txt"

            try:
                pose_data = np.loadtxt(pose_file)
            except FileNotFoundError:
                print(f'Ground truth poses are not avaialble for sequence {seq}.')

            ids_seq = pose_data[:, 0].astype(int)
            poses_seq = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
            poses_seq = np.concatenate((poses_seq, np.zeros_like(poses_seq[:, :1, :])), axis=1)
            poses_seq[:, 3, 3] = 1

            ids[seq] = ids_seq
            poses[seq] = poses_seq
        return ids, poses

    @staticmethod
    def _load_3d_bboxes(bbox_path, sequences):
        bboxes = {}

        for seq in sequences:
            with open(Path(bbox_path) / f"{seq}.xml", "rb") as f:
                tree = ET.parse(f)
            root = tree.getroot()

            objects = defaultdict(list)

            num_bbox = 0

            for child in root:
                if child.find('transform') is None:
                    continue
                obj = KITTI360Bbox3D()
                if child.find("semanticId") is not None:
                    obj.parseBbox(child)
                else:
                    obj.parseStuff(child)
                # globalId = local2global(obj.semanticId, obj.instanceId)
                # objects[globalId][obj.timestamp] = obj
                objects[obj.timestamp].append(obj)
                num_bbox +=1

            # globalIds = np.asarray(list(objects.keys()))
            # semanticIds, instanceIds = global2local(globalIds)
            # for label in labels:
            #     if label.hasInstances:
            #         print(f'{label.name:<30}:\t {(semanticIds==label.id).sum()}')
            # print(f'Loaded {len(globalIds)} instances')
            # print(f'Loaded {num_bbox} boxes')

            bboxes[seq] = objects

        return bboxes

    def get_img_id_from_id(self, sequence, id):
        return self._img_ids[sequence][id]

    def load_images(self, seq, img_ids, load_left, load_right, img_ids_fish=None):
        imgs_p_left = []
        imgs_f_left = []
        imgs_p_right = []
        imgs_f_right = []

        if img_ids_fish is None:
            img_ids_fish = img_ids

        for id in img_ids:
            if load_left:
                img_perspective = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "data_2d_raw", seq, "image_00", self._perspective_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_p_left += [img_perspective]

            if load_right:
                img_perspective = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "data_2d_raw", seq, "image_01", self._perspective_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_p_right += [img_perspective]

        for id in img_ids_fish:
            if load_left:
                img_fisheye = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "data_2d_raw", seq, "image_02", self._fisheye_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_f_left += [img_fisheye]
            if load_right:
                img_fisheye = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "data_2d_raw", seq, "image_03", self._fisheye_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                imgs_f_right += [img_fisheye]

        return imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right

    def load_segmentation_images(self, seq, img_ids, load_left, load_right, img_ids_fish=None):
        imgs_p_left = []
        imgs_f_left = []
        imgs_p_right = []
        imgs_f_right = []

        if img_ids_fish is None:
            img_ids_fish = img_ids

        for id in img_ids:
            if load_left:
                img_perspective = self.load_segmentation_from_path(os.path.join(self.data_segmentation_path, seq, "image_00", self._segmentation_perspective_folder, f"{id:010d}.png"))
                imgs_p_left += [img_perspective]

            if load_right:
                img_perspective = self.load_segmentation_from_path(os.path.join(self.data_segmentation_path, seq, "image_01", self._segmentation_perspective_folder, f"{id:010d}.png"))
                imgs_p_right += [img_perspective]

        for id in img_ids_fish:
            if load_left:
                img_fisheye = self.load_segmentation_from_path(os.path.join(self.data_segmentation_path, seq, "image_02", self._segmentation_fisheye_folder, f"{id:010d}.png"))
                imgs_f_left += [img_fisheye]
            if load_right:
                img_fisheye = self.load_segmentation_from_path(os.path.join(self.data_segmentation_path, seq, "image_03", self._segmentation_fisheye_folder, f"{id:010d}.png"))
                imgs_f_right += [img_fisheye]

        segs = imgs_p_left + imgs_p_right + imgs_f_left + imgs_f_right
        return segs

    def process_img(self, img: np.array, color_aug_fn=None, resampler:FisheyeToPinholeSampler=None):
        if resampler is not None and not self.is_preprocessed:
            img = torch.tensor(img).permute(2, 0, 1)
            img = resampler.resample(img)
        else:
            if self.target_image_size:
                img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img)

        if color_aug_fn is not None:
            img = color_aug_fn(img)

        img = img * 2 - 1
        return img

    def get_3d_bboxes(self, seq, img_id, pose, projs):
        seq_3d_bboxes = self._3d_bboxes[seq]
        pose_w2c = np.linalg.inv(pose)

        def filter_bbox(bbox):
            verts = bbox.vertices
            verts = (projs @ (pose_w2c[:3, :3] @ verts.T + pose_w2c[:3, 3, None])).T
            verts[:, :2] /= verts[:, 2:3]
            valid = ((verts[:, 0] >= -1) & (verts[:, 0] <= 1)) & ((verts[:, 1] >= -1) & (verts[:, 1] <= 1)) & ((verts[:, 2] > 0) & (verts[:, 2] <= 80))
            valid = np.any(valid, axis=-1)
            return valid

        bboxes = seq_3d_bboxes[-1] + seq_3d_bboxes[img_id]

        bboxes = list(filter(filter_bbox, bboxes))

        bboxes = [{
            "vertices": bbox.vertices,
            "faces": bbox.faces,
            "semanticId": bbox.semanticId,
            "instanceId": bbox.instanceId
        } for i, bbox in enumerate(bboxes)] #if valid[i]

        return bboxes

    def load_segmentation_from_path(self, path, target_size=None):
        seg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Only resize if target_image_size is specified and differs from original
        if target_size is not None:
            if seg.shape[:2] != (target_size[0], target_size[1]):
                seg = cv2.resize(seg, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        elif self.target_image_size is not None:
            if seg.shape[:2] != (self.target_image_size[0], self.target_image_size[1]):
                seg = cv2.resize(seg, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_NEAREST)
        return seg

    def load_segmentation(self, seq, img_id, target_size=None):
        seg = cv2.imread(os.path.join(self.data_path, "data_2d_semantics", "train", seq, "image_00", "semantic", f"{img_id:010d}.png"), cv2.IMREAD_UNCHANGED)
        # Only resize if target_image_size is specified and differs from original
        if target_size is not None:
            if seg.shape[:2] != (target_size[0], target_size[1]):
                seg = cv2.resize(seg, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        elif self.target_image_size is not None:
            if seg.shape[:2] != (self.target_image_size[0], self.target_image_size[1]):
                seg = cv2.resize(seg, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_NEAREST)
        return seg

    def replace_values(self, input, rep_dict):
        """
        Replaces the keys of the rep_dict with its values (in the input tensor).

            Parameters:
                input (tensor): tensor with the values to be replaced
                rep_dict (dict): the keys in input will be replaced by the values

            Returns:
                output (tensor): input after replacement
        """
        output = np.copy(input)
        for k, v in rep_dict.items():
            output[input == k] = v

        return output


    def load_depth(self, seq, img_id, is_right):
        """Load depth map from DA3 file or LiDAR based on depth_source config."""
        target_size = self.target_image_size
        
        # Try online depth generation first if enabled
        if self.online_depth_generation and self._depth_model is not None:
            cam_folder = "image_01" if is_right else "image_00"
            img_path = os.path.join(self.data_path, "data_2d_raw", seq, cam_folder, self._perspective_folder, f"{img_id:010d}.png")
            
            depth = self._generate_depth_online(img_path)
            if depth is not None:
                # Resize if needed
                if depth.shape[1:] != tuple(target_size):
                    depth = cv2.resize(depth[0], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                    depth = depth[None, :, :]
                return depth.astype(np.float32)
        
        if self.depth_source in ("da3", "depthanything3", "depth_anything_3", "depth-anything-3"):
            # DepthAnything3 output is saved as uint16 PNG with: depth_uint16 = depth_meters * depth_da3_scale.
            depth_root = self.depth_da3_path
            if depth_root is None:
                depth_root = os.path.join(self.data_path, "data_2d_depth_da3")

            cam_folder = "image_01" if is_right else "image_00"

            # process_kitti.py writes to either data_rect or data_rgb.
            # If this dataset uses a preprocessed folder name (e.g. data_192x640), fall back to data_rect.
            depth_subdir = self._perspective_folder
            if depth_subdir not in ("data_rect", "data_rgb"):
                depth_subdir = "data_rect"

            depth_path = os.path.join(depth_root, seq, cam_folder, depth_subdir, f"{img_id:010d}.png")
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if depth_raw is None:
                # If file not found, return zeros
                depth = np.zeros(target_size, dtype=np.float32)
                return depth[None, :, :]

            if depth_raw.ndim == 3:
                depth_raw = depth_raw[..., 0]

            depth = depth_raw.astype(np.float32)
            if self.depth_da3_scale > 0:
                depth = depth / self.depth_da3_scale

            # Resize if target size differs
            if depth.shape != tuple(target_size):
                depth = cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

            depth[~np.isfinite(depth)] = 0
            depth[depth < 0] = 0
            return depth[None, :, :]

        # Default: sparse LiDAR-projected depth map.
        points = np.fromfile(os.path.join(self.data_path, "data_3d_raw", seq, "velodyne_points", "data", f"{img_id:010d}.bin"), dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0

        T_velo_to_cam = self._calibs["T_velo_to_cam"]["00" if not is_right else "01"]
        K = self._calibs["K_perspective"]

        # project the points to the camera
        velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

        # the projection is normalized to [-1, 1] -> transform to [0, height-1] x [0, width-1]
        velo_pts_im[:, 0] = np.round((velo_pts_im[:, 0] * .5 + .5) * target_size[1])
        velo_pts_im[:, 1] = np.round((velo_pts_im[:, 1] * .5 + .5) * target_size[0])

        # check if in bounds
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < target_size[1]) & (velo_pts_im[:, 1] < target_size[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(target_size)
        depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = velo_pts_im[:, 1] * (target_size[1] - 1) + velo_pts_im[:, 0] - 1
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        return depth[None, :, :]

    def load_lidar_depth(self, seq, img_id, is_right):
        """Load LiDAR-projected sparse depth map (always from LiDAR, for evaluation)."""
        points = np.fromfile(os.path.join(self.data_path, "data_3d_raw", seq, "velodyne_points", "data", f"{img_id:010d}.bin"), dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0

        T_velo_to_cam = self._calibs["T_velo_to_cam"]["00" if not is_right else "01"]
        K = self._calibs["K_perspective"]

        # project the points to the camera
        velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
        
        # Filter out points with z <= 0 (behind camera) to avoid divide by zero
        valid_z = velo_pts_im[:, 2] > 1e-5
        velo_pts_im = velo_pts_im[valid_z]
        
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

        # the projection is normalized to [-1, 1] -> transform to [0, height-1] x [0, width-1]
        velo_pts_im[:, 0] = np.round((velo_pts_im[:, 0] * .5 + .5) * self.target_image_size[1])
        velo_pts_im[:, 1] = np.round((velo_pts_im[:, 1] * .5 + .5) * self.target_image_size[0])

        # check if in bounds
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < self.target_image_size[1]) & (velo_pts_im[:, 1] < self.target_image_size[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(self.target_image_size)
        depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = velo_pts_im[:, 1] * (self.target_image_size[1] - 1) + velo_pts_im[:, 0] - 1
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        return depth[None, :, :]

    def __getitem__(self, index: int):
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        sequence, id, is_right = self._datapoints[index]
        seq_len = self._img_ids[sequence].shape[0]

        load_left = (not is_right) or self.return_stereo
        load_right = is_right or self.return_stereo

        if self.random_fisheye_offset:
            fisheye_offset = self.fisheye_offset[torch.randint(0, len(self.fisheye_offset), (1,)).item()]
        else:
            fisheye_offset = self.fisheye_offset[-1]

        ids = [id] + [max(min(i, seq_len-1), 0) for i in range(id - self._left_offset, id - self._left_offset + self.frame_count * self.dilation, self.dilation) if i != id]
        if self.additional_random_front_offset:
            front_offset = max(fisheye_offset - 10, 0)
            ids = ids + [max(min(i + front_offset, seq_len-1), 0) for i in ids]
        ids_fish = [max(min(id + fisheye_offset, seq_len-1), 0)] + [max(min(i, seq_len-1), 0) for i in range(id + fisheye_offset - self._left_offset, id + fisheye_offset - self._left_offset + self.frame_count * self.dilation, self.dilation) if i != id + fisheye_offset]
        img_ids = [self.get_img_id_from_id(sequence, id) for id in ids]
        img_ids_fish = [self.get_img_id_from_id(sequence, id) for id in ids_fish]

        if not self.return_fisheye:
            ids_fish = []
            img_ids_fish = []

        if self.color_aug:
            color_aug_fn = get_color_aug_fn(ColorJitter.get_params(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)))
        else:
            color_aug_fn = None

        _start_time_loading = time.time()
        imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right = self.load_images(sequence, img_ids, load_left, load_right, img_ids_fish=img_ids_fish)
        _loading_time = np.array(time.time() - _start_time_loading)

        _start_time_processing = time.time()
        imgs_p_left = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_left]
        imgs_f_left = [self.process_img(img, color_aug_fn=color_aug_fn, resampler=self._resampler_02) for img in imgs_f_left]
        imgs_p_right = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_right]
        imgs_f_right = [self.process_img(img, color_aug_fn=color_aug_fn, resampler=self._resampler_03) for img in imgs_f_right]
        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are camera to world !!
        poses_p_left = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["00"] for i in ids] if load_left else []
        poses_f_left = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["02"] for i in ids_fish] if load_left else []
        poses_p_right = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["01"] for i in ids] if load_right else []
        poses_f_right = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["03"] for i in ids_fish] if load_right else []

        projs_p_left = [self._calibs["K_perspective"] for _ in ids] if load_left else []
        projs_f_left = [self._calibs["K_fisheye"] for _ in ids_fish] if load_left else []
        projs_p_right = [self._calibs["K_perspective"] for _ in ids] if load_right else []
        projs_f_right = [self._calibs["K_fisheye"] for _ in ids_fish] if load_right else []

        imgs = imgs_p_left + imgs_p_right + imgs_f_left + imgs_f_right if not is_right else imgs_p_right + imgs_p_left + imgs_f_right + imgs_f_left
        projs = projs_p_left + projs_p_right + projs_f_left + projs_f_right if not is_right else projs_p_right + projs_p_left + projs_f_right + projs_f_left
        poses = poses_p_left + poses_p_right + poses_f_left + poses_f_right if not is_right else poses_p_right + poses_p_left + poses_f_right + poses_f_left
        ids = np.array(ids + ids + ids_fish + ids_fish, dtype=np.int32)

        if self.return_depth:
            # Load depth for all perspective images (both left and right cameras, all frames)
            depths_p_left = [self.load_depth(sequence, img_id, is_right=False) for img_id in img_ids] if load_left else []
            depths_p_right = [self.load_depth(sequence, img_id, is_right=True) for img_id in img_ids] if load_right else []
            # Depth for fisheye is not directly supported; use zeros as placeholder
            fisheye_size = self.target_image_size
            depths_f_left = [np.zeros((1, *fisheye_size), dtype=np.float32) for _ in img_ids_fish] if load_left else []
            depths_f_right = [np.zeros((1, *fisheye_size), dtype=np.float32) for _ in img_ids_fish] if load_right else []

            depths = depths_p_left + depths_p_right + depths_f_left + depths_f_right if not is_right else depths_p_right + depths_p_left + depths_f_right + depths_f_left
        else:
            depths = []
        
        # Load LiDAR depth for evaluation (only first frame / frame 0)
        lidar_depths = []
        if self.return_lidar_depth:
            # Only load LiDAR depth for frame 0 (evaluation uses frame 0)
            lidar_depth_0 = self.load_lidar_depth(sequence, img_ids[0], is_right=is_right)
            lidar_depths = [lidar_depth_0]

        if self.return_3d_bboxes:
            bboxes_3d = [self.get_3d_bboxes(sequence, img_ids[0], poses[0], projs[0])]
        else:
            bboxes_3d = []

        # Logic for filtering datapoints can now be found in the init
        segs = []
        mode = self._segmentation_mode_norm
        if mode in ("kitti-360", "kitti_360", "kitti360"):
            segs_raw = [self.load_segmentation(sequence, id) for id in img_ids]
            segs = [self.replace_values(seg, id2TrainId) for seg in segs_raw]

        elif mode in ("panoptic_deeplab", "mask2former", "m2f"):
            if not self.data_segmentation_path:
                raise ValueError(
                    "segmentation_mode requires `data_segmentation_path` to be set."
                )
            segs = self.load_segmentation_images(sequence, img_ids, load_left, load_right, img_ids_fish=img_ids_fish)

        # Load the kitti-360 gt
        segs_kitti_gt = torch.zeros(0)
        if self.load_kitti_360_segmentation_gt:
            segs_raw = [self.load_segmentation(sequence, id) for id in img_ids]
            segs_kitti_gt = [self.replace_values(seg, id2TrainId) for seg in segs_raw]

        _proc_time = np.array(time.time() - _start_time)

        # print(_loading_time, _processing_time, _proc_time)

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "depths": depths,
            "lidar_depths": lidar_depths,  # LiDAR GT depth for evaluation
            "ts": ids,
            "3d_bboxes": bboxes_3d,
            "segs_gt": segs,
            "segs_kitti_gt": segs_kitti_gt,
            "t__get_item__": np.array([_proc_time]),
            "index": np.array([index])
        }

        return data

    def __len__(self) -> int:
        return self.length
