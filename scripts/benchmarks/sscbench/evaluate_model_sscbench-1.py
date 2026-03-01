import argparse
import sys
from datetime import datetime

from omegaconf import open_dict

import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.extend([".", "../../../"])

from scripts.benchmarks.sscbench.generate_ply_sequence import get_cam_k
from scripts.benchmarks.sscbench.point_utils import read_calib, generate_point_grid, get_fov_mask

from scripts.voxel.gen_voxelgrid_npy import save_as_voxel_ply

import logging

from pathlib import Path
import subprocess
import yaml

import cv2
import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from hydra import compose, initialize

import matplotlib.pyplot as plt

from models.bts.model import BTSNet, ImageRaySampler
from models.common.render import NeRFRenderer

from sscbench_dataset import SSCBenchDataset
from pathlib import Path

# DA3 深度生成器
from models.da3_depth_generator import DA3DepthGenerator

RELOAD_DATASET = True
DATASET_LENGTH = 50 # 2000
FULL_EVAL = False  # 是否对完整数据集进行评估
SAMPLE_EVERY = None
SAMPLE_OFFSET = 2
# SAMPLE_RANGE = list(range(1000, 1600))
SAMPLE_RANGE = None

SIZE = 51.2 # Can be: 51.2, 25.6, 12.8
SIZES = (12.8, 25.6, 51.2)
VOXEL_SIZE = 0.1 # Needs: 0.2 % VOXEL_SIZE == 0 (S4C 原版使用 0.1)

USE_ADDITIONAL_INVALIDS = True  # True = 排除遮挡区域，IoU 更高

TEST_ALPHA_CUTOFFS = False
SEARCH_VALUES = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6, 10e-7]

SIGMA_CUTOFF = 0.25  # default; can be overridden via CLI

USE_ALPHA_WEIGHTING = True
USE_GROW = True  # True = 膨胀预测，填补空洞，IoU 更高

# ============================================================
# DA3 深度过滤配置 (Evaluation-Time Cleaning Strategy)
# ============================================================
# NOTE: DA3 输出的是 affine-invariant 相对深度，使用 median scaling 对齐后再过滤
USE_DA3_FILTERING = True  # 启用 DA3 深度先验过滤 (使用 median scaling 对齐)
DA3_NEAR_THRESHOLD = 0.5   # 近场 (d < 10m) 阈值：允许体素在 DA3 表面前方最多 0.5m
DA3_FAR_THRESHOLD = 0.2    # 远场 (d >= 10m) 阈值：允许体素在 DA3 表面前方最多 0.2m
DA3_NEAR_FAR_BOUNDARY = 10.0  # 近场/远场分界线 (米)

CREATE_SIGMA_TRADEOFF_PLOT = True
SIGMA_VALUES = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001]

PLOT_ALL_IMAGES = False

GENERATE_PLY_FILES = False  # True 需要同时设置 USE_GROW=False, USE_ADDITIONAL_INVALIDS=False
# GENERATE_PLY_FILES = True
PLY_ONLY_FOV = True
PLY_IDS = None  # None 表示处理所有帧，或指定列表如 list(range(1000, 1600))
PLY_PATH = Path("/home/lmh/dino-da3-adaptive-sampling1/out/voxel_outputs")
PLY_SIZES = [12.8, 25.6, 51.2]  # 添加 12.8

GENERATE_STATISTICS = False

if GENERATE_PLY_FILES:
    assert (not USE_GROW) and (not USE_ADDITIONAL_INVALIDS) and VOXEL_SIZE == 0.1

    # make the necessary dirs
    for size in PLY_SIZES:
        if not os.path.exists(PLY_PATH / str(int(size))):
            os.makedirs(PLY_PATH / str(int(size)))


# Setup of CUDA device and logging

os.system("nvidia-smi")

device = f'cuda:0'

# DO NOT TOUCH OR YOU WILL BREAK RUNS (should be None)
gpu_id = None

if gpu_id is not None:
    print("GPU ID: " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser("SSCBenchmark Output generation")
    parser.add_argument("--sscbench_data_root", "-ssc", type=str)
    parser.add_argument("--voxel_gt_path", "-vgt", type=str)
    parser.add_argument("--resolution", "-r", default=(192, 640))
    parser.add_argument("--checkpoint", "-cp", type=str, required=True)
    parser.add_argument("--full", "-f", action="store_true")
    parser.add_argument(
        "--sigma_cutoff",
        type=float,
        default=None,
        help="Override SIGMA_CUTOFF used to threshold occupancy (default: 0.25).",
    )
    parser.add_argument(
        "--sweep_sigma",
        type=str,
        default=None,
        help="Comma-separated sigma cutoffs to sweep for occupancy PR/IoU (e.g. '0.1,0.2,0.25,0.3'). "
             "If omitted, no sweep is performed.",
    )
    
    # DA3 深度过滤参数
    parser.add_argument(
        "--da3_filter",
        action="store_true",
        default=None,
        help="Enable DA3 depth-based filtering to remove ghost fog in free space.",
    )
    parser.add_argument(
        "--no_da3_filter",
        action="store_true",
        default=None,
        help="Disable DA3 depth-based filtering.",
    )
    parser.add_argument(
        "--da3_near_threshold",
        type=float,
        default=None,
        help="DA3 filter threshold for near-field (d < 10m). Default: 0.5m",
    )
    parser.add_argument(
        "--da3_far_threshold",
        type=float,
        default=None,
        help="DA3 filter threshold for far-field (d >= 10m). Default: 0.2m",
    )

    args = parser.parse_args()

    sscbench_data_root = args.sscbench_data_root
    voxel_gt_path = args.voxel_gt_path
    resolution = args.resolution
    cp_path = args.checkpoint
    full_evaluation = args.full
    sigma_cutoff = args.sigma_cutoff
    sweep_sigma = args.sweep_sigma

    global SIGMA_CUTOFF
    if sigma_cutoff is not None:
        SIGMA_CUTOFF = float(sigma_cutoff)
    logging.info(f"Using a sigma cutoff of {SIGMA_CUTOFF}")
    
    # 处理 DA3 过滤参数
    global USE_DA3_FILTERING, DA3_NEAR_THRESHOLD, DA3_FAR_THRESHOLD
    if args.da3_filter:
        USE_DA3_FILTERING = True
    elif args.no_da3_filter:
        USE_DA3_FILTERING = False
    if args.da3_near_threshold is not None:
        DA3_NEAR_THRESHOLD = args.da3_near_threshold
    if args.da3_far_threshold is not None:
        DA3_FAR_THRESHOLD = args.da3_far_threshold
    
    logging.info(f"DA3 depth filtering: {'ENABLED' if USE_DA3_FILTERING else 'DISABLED'}")
    if USE_DA3_FILTERING:
        logging.info(f"  Near-field threshold (d < {DA3_NEAR_FAR_BOUNDARY}m): {DA3_NEAR_THRESHOLD}m")
        logging.info(f"  Far-field threshold (d >= {DA3_NEAR_FAR_BOUNDARY}m): {DA3_FAR_THRESHOLD}m")
    logging.info("Setting up dataset")

    global SIGMA_VALUES
    if sweep_sigma is not None:
        try:
            SIGMA_VALUES = [float(x.strip()) for x in sweep_sigma.split(",") if x.strip() != ""]
        except Exception as e:
            raise ValueError(f"Failed to parse --sweep_sigma='{sweep_sigma}'. Expected comma-separated floats.") from e
        if len(SIGMA_VALUES) == 0:
            raise ValueError("Parsed --sweep_sigma is empty. Provide at least one value, e.g. --sweep_sigma '0.25'.")

    with open("scripts/benchmarks/sscbench/label_maps.yaml", "r") as f:
        label_maps = yaml.safe_load(f)

    # pickle the dataset so we don't have to wait all the time
    if os.path.isfile("dataset.pkl") and not RELOAD_DATASET:
        logging.info("Loading dataset from dataset.pkl file.")
        with open("dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        logging.info("Generating the dataset and dumping it to dataset.pkl")
        dataset = SSCBenchDataset(
            data_path=sscbench_data_root,
            voxel_gt_path=voxel_gt_path,
            sequences=(9,),
            target_image_size=resolution,
            return_stereo=False,
            frame_count=1,
            color_aug=False,
        )
        if not full_evaluation:
            dataset.length = min(int(DATASET_LENGTH), dataset.length)

        with open("dataset.pkl", 'wb') as f:
            pickle.dump(dataset, f)

    logging.info("Setting up the model...")

    config_path = "exp_kitti_360"


    cp_path = Path(cp_path)
    # cp_path = next(cp_path.glob("training*.pt"))

    if cp_path.is_file():
        # 如果你直接指定了文件（如 .pt），就直接使用它
        final_cp_path = cp_path
    elif cp_path.is_dir():
        # 如果你指定的是文件夹，则保持原有的搜索逻辑
        try:
            final_cp_path = next(cp_path.glob("training*.pt"))
        except StopIteration:
            raise FileNotFoundError(f"在文件夹 {cp_path} 中找不到 training*.pt 文件")
    else:
        raise FileNotFoundError(f"路径不存在: {cp_path}")

    cp_path = final_cp_path

    initialize(version_base=None, config_path="../../../configs", job_name="gen_sscbench_outputs")
    config = compose(config_name=config_path, overrides=[])

    logging.info('Loading checkpoint')
    cp = torch.load(cp_path, map_location=device)

    with open_dict(config):
        config["renderer"]["hard_alpha_cap"] = True
        config["model_conf"]["code_mode"] = "z"
        # config["model_conf"]["z_near"] = 8
        config["model_conf"]["mlp_coarse"]["n_blocks"] = 0
        config["model_conf"]["mlp_coarse"]["d_hidden"] = 64
        config["model_conf"]["encoder"]["d_out"] = 64
        config["model_conf"]["encoder"]["type"] = "monodepth2"
        config["model_conf"]["grid_learn_empty"] = False
        config["model_conf"]["sample_color"] = True

        # stuff for segmentation
        config["model_conf"]["segmentation_mode"] = 'panoptic_deeplab'
        
        # 自动检测 checkpoint 是否使用 RBF encoding
        # 通过检查 mlp_coarse.lin_in.weight 的维度来判断
        state_dict = cp["model"] if "model" in cp else cp
        if "renderer.net.mlp_coarse.lin_in.weight" in state_dict:
            mlp_in_dim = state_dict["renderer.net.mlp_coarse.lin_in.weight"].shape[1]
            # d_in=103 表示没有 RBF (features=64 + xyz_code=39)
            # d_in=107 表示有 RBF (features=64 + xyz_code=39 + rbf_code=4)
            if mlp_in_dim == 103:
                logging.info(f"[Auto-detect] Checkpoint uses d_in={mlp_in_dim}, disabling RBF encoding")
                config["model_conf"]["use_rbf_encoding"] = False
            else:
                logging.info(f"[Auto-detect] Checkpoint uses d_in={mlp_in_dim}, keeping RBF encoding")
                config["model_conf"]["use_rbf_encoding"] = True

    net = BTSNet(config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    # 兼容两种权重格式：training_checkpoint (包含"model"键) 和 best_model (直接是state_dict)
    if "model" in cp:
        state_dict = cp["model"]
    else:
        state_dict = cp
    _wrapper.load_state_dict(state_dict, strict=False)
    renderer.to(device)
    renderer.eval()

    logging.info("Loading the Lidar to Camera matrices...")

    calib = read_calib()
    T_velo_2_cam = calib["Tr"]

    logging.info("Generating the point cloud...")

    pts, _, projected_pix, pix_z = generate_point_grid(vox_origin=np.array([0, -25.6, -2]),
                              scene_size=(51.2, 51.2, 6.4),
                              voxel_size=VOXEL_SIZE,
                              cam_E=T_velo_2_cam,
                              cam_k=get_cam_k())

    fov_mask = get_fov_mask()

    pts = torch.tensor(pts).to(device).reshape(1, -1, 3).float()
    fov_mask = fov_mask.reshape(256, 256, 32)
    
    # 为 DA3 过滤生成与 fov_mask 相同分辨率的投影坐标
    # 使用 voxel_size=0.2 来匹配 fov_mask 的分辨率 (256x256x32)
    _, _, projected_pix_for_filter, pix_z_for_filter = generate_point_grid(
        vox_origin=np.array([0, -25.6, -2]),
        scene_size=(51.2, 51.2, 6.4),
        voxel_size=0.2,  # 与 fov_mask 相同分辨率
        cam_E=T_velo_2_cam,
        cam_k=get_cam_k()
    )
    projected_pix_tensor = torch.tensor(projected_pix_for_filter).to(device).float()
    pix_z_tensor = torch.tensor(pix_z_for_filter).to(device).float()
    
    # 初始化 DA3 深度生成器
    da3_generator = None
    if USE_DA3_FILTERING:
        logging.info("Initializing DA3 depth generator for evaluation-time filtering...")
        da3_generator = DA3DepthGenerator(
            checkpoint_path="/home/lmh/dino-da3-adaptive-sampling1/DA3/ckpt/DA3NESTED-GIANT-LARGE",
            conf_percentile=20.0
        ).to(device)
        da3_generator.eval()

    logging.info("Setting up folders...")

    downsample_factor = int(0.2 // VOXEL_SIZE)

    results = {}
    for size in SIZES:
        results[size] = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tp_seg": np.zeros(15),
            "fp_seg": np.zeros(15),
            "tn_seg": np.zeros(15),
            "fn_seg": np.zeros(15),
        }

    # for the sigma tradeoff plots
    trade_off_values = np.zeros([len(SIGMA_VALUES), 4])

    cutoff_results = {i: {sv: {"tp":0, "fp": 0, "tn": 0, "fn": 0} for sv in SEARCH_VALUES} for i in range(1, 16)}

    pbar = tqdm(range(len(dataset)))

    images = {"ids": [], "images": []}

    ids = [125, 280, 960, 1000, 1150, 1325, 2300, 3175, 3750, 4300, 5155, 5475, 5750, 6475, 6525, 6670, 6775, 7500, 7860, 8000, 8350, 9000, 9350, 10975]

    ids = [60, 250, 455, 690, 835, 2235, 2385, 2495, 3385, 4235, 4360, 4550, 4875, 5550, 6035, 7010, 7110, 8575, 9010, 9410, 11260, 11460, 11885]

    # for our statistics
    tframeIds = []
    tinval = []
    ttp = []
    tfp = []
    ttn = []
    tfn = []

    # plot_image_at_frame_id(dataset, 952)
    for i in pbar:
        if SAMPLE_EVERY:
            if (i - SAMPLE_OFFSET) % SAMPLE_EVERY != 0:
                continue

        sequence, id, is_right = dataset._datapoints[i]

        if SAMPLE_RANGE:
            if id not in SAMPLE_RANGE:
                continue

        if GENERATE_PLY_FILES and PLY_IDS is not None and id not in PLY_IDS:
            continue

        if GENERATE_STATISTICS:
            tframeIds.append(id)

        data = dataset[i]

        # downsample the sigmas
        sigmas, segs = downsample_and_predict(data, net, pts, downsample_factor)
        
        # ============================================================
        # DA3 深度过滤 (Evaluation-Time Cleaning Strategy)
        # 利用 DA3 深度先验清除空域中的"幽灵雾"
        # ============================================================
        if USE_DA3_FILTERING and da3_generator is not None:
            # 获取原始图像 (需要恢复到 [0, 1] 范围)
            img_for_da3 = (data["imgs"][0] + 1) / 2  # (C, H, W)
            img_for_da3 = img_for_da3.unsqueeze(0).to(device)  # (1, C, H, W)
            
            # DA3 推理获取深度图
            with torch.no_grad():
                da3_output = da3_generator(img_for_da3)
                da3_depth = da3_output["depth"].squeeze()  # (H, W)
            
            # 应用 DA3 深度过滤
            sigmas_tensor = torch.tensor(sigmas).to(device)
            fov_mask_flat = torch.tensor(fov_mask).to(device).reshape(-1)
            
            sigmas_tensor, num_filtered, num_valid = apply_da3_depth_filtering(
                sigmas_tensor, 
                da3_depth, 
                projected_pix_tensor, 
                pix_z_tensor,
                fov_mask_flat,
                orig_img_H=376,  # KITTI-360 原始分辨率
                orig_img_W=1408
            )
            sigmas = sigmas_tensor.cpu().numpy()
            
            # 可选：打印过滤统计
            if i % 100 == 0:
                logging.info(f"[DA3 Filter] Frame {id}: filtered {num_filtered}/{num_valid} voxels "
                           f"({100*num_filtered/num_valid:.1f}%)")

        # convert both to the right format
        segs = convert_voxels(segs, label_maps["cityscapes_to_label"])
        target = convert_voxels(data["voxel_gt"][0].astype(int), label_maps["sscbench_to_label"])


        if PLOT_ALL_IMAGES:
            images["ids"].append(id)
            images["images"].append(((data["imgs"][0] + 1) / 2).permute(1, 2, 0))

            if len(images["ids"]) == 6:
                plot_images(images)
                images = {"images": [], "ids": []}

        # print(f"Image_Id: {id}")
        #
        # plt.imshow(((data["imgs"][0] + 1) / 2).permute(1, 2, 0))
        # plt.show()
        #
        # out_dict = {"sigmas": sigmas, "segs": segs.copy(), "gt": target, "fov_mask": fov_mask}
        #
        # with open(f'plots10_40/{id:06d}.pkl', 'wb') as f:
        #     pickle.dump(out_dict, f)

        if GENERATE_PLY_FILES:
            _segs = segs.copy()
            _target = target.copy()
            _segs[sigmas < SIGMA_CUTOFF] = 0
            mask = target != 255
            if PLY_ONLY_FOV:
                mask = mask & fov_mask
            _segs[~mask] = 0
            _target[~mask] = 0
            is_occupied_seg = torch.tensor(_segs > 0)
            is_occupied_gt = torch.tensor(_target > 0)

            for size in PLY_SIZES:
                num_voxels = int(size // 0.2)
                save_as_voxel_ply(PLY_PATH / str(int(size)) / f"{id:06d}.ply",
                                  is_occupied_seg[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:],
                                  classes=torch.tensor(_segs[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:]))
                save_as_voxel_ply(PLY_PATH / str(int(size)) / f"{id:06d}_gt.ply",
                                  is_occupied_gt[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:],
                                  classes=torch.tensor(_target[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2),:]))

        if USE_ADDITIONAL_INVALIDS:
            invalids = identify_additional_invalids(target)
            # logging.info(np.mean(invalids))
            target[invalids == 1] = 255

            if GENERATE_STATISTICS:
                tinval.append(np.mean(invalids))

        # test and summarize different alpha cutoffs
        if TEST_ALPHA_CUTOFFS:
            for i in range(1, 16):
                for search_value in SEARCH_VALUES:
                    _tmp = segs.copy()
                    _tmp[np.logical_and(segs == i, sigmas < search_value)] = 0
                    _tp_seg, _fp_seg, _tn_seg, _fn_seg = compute_occupancy_numbers_segmentation(
                        y_pred=_tmp, y_true=target, fov_mask=fov_mask, labels=label_maps["labels"])
                    cutoff_results[i][search_value]["tp"] += _tp_seg[i-1]
                    cutoff_results[i][search_value]["fp"] += _fp_seg[i-1]
                    cutoff_results[i][search_value]["tn"] += _tn_seg[i-1]
                    cutoff_results[i][search_value]["fn"] += _fn_seg[i-1]

        if CREATE_SIGMA_TRADEOFF_PLOT:
            for i, val in enumerate(SIGMA_VALUES):
                _tmp = segs.copy()
                _tmp[sigmas < val] = 0
                _tp, _fp, _tn, _fn = compute_occupancy_numbers(y_pred=_tmp, y_true=target, fov_mask=fov_mask)
                trade_off_values[i] += np.array([_tp, _fp, _tn, _fn])

        segs[sigmas < SIGMA_CUTOFF] = 0

        for size in SIZES:
            num_voxels = int(size // 0.2)

            # resize to right scene size
            _segs = segs[:num_voxels, (128 - num_voxels//2):(128 + num_voxels//2), :]
            _target = target[:num_voxels, (128 - num_voxels//2):(128 + num_voxels//2), :]
            _fov_mask = fov_mask[:num_voxels, (128 - num_voxels // 2):(128 + num_voxels // 2), :]

            _tp, _fp, _tn, _fn = compute_occupancy_numbers(y_pred=_segs, y_true=_target, fov_mask=_fov_mask)
            _tp_seg, _fp_seg, _tn_seg, _fn_seg = compute_occupancy_numbers_segmentation(
                y_pred=_segs, y_true=_target, fov_mask=_fov_mask, labels=label_maps["labels"])

            if size == 51.2 and GENERATE_STATISTICS:
                ttp += [_tp]
                tfp += [_fp]
                ttn += [_fn]
                tfn += [_fn]

            results[size]["tp"] += _tp
            results[size]["fp"] += _fp
            results[size]["tn"] += _tn
            results[size]["fn"] += _fn

            results[size]["tp_seg"] += _tp_seg
            results[size]["fp_seg"] += _fp_seg
            results[size]["tn_seg"] += _tn_seg
            results[size]["fn_seg"] += _fn_seg

            recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
            precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
            iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])

        pbar.set_postfix_str(f"IoU: {iou*100:.2f} Prec: {precision*100:.2f} Rec: {recall*100:.2f}")

    results_table = np.zeros((19, 3), dtype=np.float32)

    # Here we compute all the metrics
    for size_i, size in enumerate(SIZES):
        recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
        precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
        iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])

        results_table[0, size_i] = iou
        results_table[1, size_i] = precision
        results_table[2, size_i] = recall

        logging.info(f"#" * 50)
        logging.info(f"Results for size {size}. ")
        logging.info(f"#" * 50)

        logging.info("Occupancy metrics")
        logging.info(f"Recall: {recall*100:.2f}%")
        logging.info(f"Precision: {precision*100:.2f}%")
        logging.info(f"IoU: {iou*100:.2f}")

        recall_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fn_seg"])
        precision_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"])
        iou_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"] + results[size]["fn_seg"])

        weights = label_maps["weights"]
        weights_val = np.array(list(weights.values()))
        weighted_mean_iou = np.sum(weights_val * np.nan_to_num(iou_seg)) / np.sum(weights_val)

        mean_iou = np.mean(np.nan_to_num(iou_seg))
        # "Valid-class" mIoU: only classes that appear in GT for this evaluation set (avoid penalizing missing rare classes in small subsets)
        gt_present = (results[size]["tp_seg"] + results[size]["fn_seg"]) > 0
        valid_mean_iou = np.mean(np.nan_to_num(iou_seg)[gt_present]) if np.any(gt_present) else float("nan")

        results_table[3, size_i] = mean_iou
        results_table[4:, size_i] = iou_seg

        logging.info("Occupancy metrics segmentation")
        for i in range(15):
            logging.info(f"{label_maps['labels'][i+1]}; IoU: {iou_seg[i]*100:.2f}; Precision: {precision_seg[i]*100:.2f}%; Recall: {recall_seg[i]*100:.2f}%")

    logging.info(f"Results table for copying.")

    results_table_str = ""
    for i in range(19):
        results_table_str += f"{results_table[i, 0]*100:.2f}\t{results_table[i, 1]*100:.2f}\t{results_table[i, 2]*100:.2f}\n"
    print(results_table_str)

    logging.info(f"Mean IoU: {mean_iou*100:.2f}")
    logging.info(f"Valid-class Mean IoU: {valid_mean_iou*100:.2f}" if not np.isnan(valid_mean_iou) else "Valid-class Mean IoU: nan")
    logging.info(f"Weighted Mean IoU: {weighted_mean_iou*100:.2f}")

    if sweep_sigma is not None and CREATE_SIGMA_TRADEOFF_PLOT:
        # Print all sweep points (what you typically want to see in tmux logs)
        logging.info("[Sigma Sweep] Occupancy trade-off over provided SIGMA_VALUES:")
        logging.info("  sigma_cutoff\tIoU(%)\tPrec(%)\tRec(%)\tTP\tFP\tFN")
        for i, val in enumerate(SIGMA_VALUES):
            tp, fp, _, fn = trade_off_values[i]
            pres = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            _iou = tp / (tp + fp + fn + 1e-9)
            logging.info(
                f"  {val}\t{_iou*100:.2f}\t{pres*100:.2f}\t{rec*100:.2f}\t{int(tp)}\t{int(fp)}\t{int(fn)}"
            )

        best = None  # (iou, sigma, precision, recall)
        for i, val in enumerate(SIGMA_VALUES):
            tp, fp, _, fn = trade_off_values[i]
            pres = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            _iou = tp / (tp + fp + fn + 1e-9)
            cand = (_iou, val, pres, rec)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None:
            biou, bsigma, bpres, brec = best
            logging.info(
                f"[Sigma Sweep] Best occupancy IoU={biou*100:.2f} at SIGMA_CUTOFF={bsigma} "
                f"(Prec={bpres*100:.2f} Rec={brec*100:.2f})"
            )

    # 自动保存评估结果到与权重一致的目录
    save_evaluation_results(
        cp_path=cp_path,
        results=results,
        results_table=results_table,
        mean_iou=mean_iou,
        valid_mean_iou=valid_mean_iou,
        weighted_mean_iou=weighted_mean_iou,
        label_maps=label_maps,
        dataset_length=len(dataset),
        sizes=SIZES
    )

    if TEST_ALPHA_CUTOFFS:
        cutoff_metrics = \
            {i: {sv: {"precision": np.nan_to_num(100*cutoff_results[i][sv]["tp"] / (cutoff_results[i][sv]["tp"] + cutoff_results[i][sv]["fp"])),
                       "recall": np.nan_to_num(100*cutoff_results[i][sv]["tp"] / (cutoff_results[i][sv]["tp"] + cutoff_results[i][sv]["fn"])),
                       "IoU": np.nan_to_num(100*cutoff_results[i][sv]["tp"] / (cutoff_results[i][sv]["tp"] + cutoff_results[i][sv]["fn"] + cutoff_results[i][sv]["fp"]))}
                      for sv in SEARCH_VALUES} for i in range(1, 16)}

        best_values = {i: SEARCH_VALUES[torch.argmax(torch.tensor([cutoff_metrics[i][sv]["IoU"] for sv in SEARCH_VALUES]))] for i in range(1, 16)}

        print(best_values)

    if CREATE_SIGMA_TRADEOFF_PLOT:
        plt.figure(figsize=(10, 8))
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.xlim([10, 70])
        # plt.ylim([0, 100])

        for i, val in enumerate(SIGMA_VALUES):
            tp, fp, tn, fn = trade_off_values[i]
            pres = 100*tp / (tp + fp)
            recall = 100*tp/ (tp + fn)
            plt.scatter(pres, recall)
            plt.annotate(f"Sigma: {val}; IoU: {100*tp / (tp + fp + fn):.2f}", (pres, recall))

        identifier = os.path.basename(cp_path)
        if FULL_EVAL:
            path = f"figures/inv{str(USE_ADDITIONAL_INVALIDS)}_{VOXEL_SIZE:.1f}_mp{str(USE_GROW)}_{identifier}.png"
        else:
            path = f"figures/inv{str(USE_ADDITIONAL_INVALIDS)}_{DATASET_LENGTH}_{VOXEL_SIZE:.1f}_mp{str(USE_GROW)}_{identifier}.png"

        if os.path.isfile(path):
            os.remove(path)
        plt.savefig(path)

        plt.show()

    if GENERATE_STATISTICS:
        statistics_raw = {"frameId": tframeIds, "TP": ttp, "FP": tfp, "TN": ttn, "FN": tfn, "invalids": tinval}
        with open("stats.pkl", "wb") as f:
            pickle.dump(statistics_raw, f)
        logging.info("Saved the statistics for further analysis.")


def apply_da3_depth_filtering(sigmas, da3_depth, projected_pix, pix_z, fov_mask_flat, 
                               orig_img_H=376, orig_img_W=1408):
    """
    应用 DA3 深度先验过滤，清除"幽灵雾"（空域中的假阳性）。
    
    重要：DA3 输出的是 affine-invariant 相对深度，需要先进行 median scaling 对齐！
    
    原理：如果体素的深度 pix_z 显著小于对齐后的 DA3 深度（即体素在预测表面前方），
    则认为该体素在空域中，应该被清除。
    
    Args:
        sigmas: (256, 256, 32) 体素密度
        da3_depth: (H, W) DA3 预测的深度图 (可能与原始分辨率不同)
        projected_pix: (N, 2) 体素在图像上的投影坐标 (基于原始分辨率 1408x376)
        pix_z: (N,) 体素到相机的深度
        fov_mask_flat: (N,) FOV 掩码
        orig_img_H, orig_img_W: 原始图像尺寸 (投影坐标的基准)
    
    Returns:
        sigmas: 过滤后的体素密度
        num_filtered: 过滤的体素数量
        num_valid: FOV 内的有效体素数量
    """
    device = sigmas.device if hasattr(sigmas, 'device') else 'cpu'
    
    # 将 sigmas 展平
    sigmas_flat = sigmas.reshape(-1)
    N = sigmas_flat.shape[0]
    
    # 只处理 FOV 内的体素
    valid_mask = fov_mask_flat.reshape(-1)
    
    # DA3 深度图的实际尺寸
    da3_H, da3_W = da3_depth.shape
    
    # 缩放投影坐标到 DA3 深度图的分辨率
    # projected_pix 是基于原始分辨率 (1408, 376) 的坐标
    scale_x = da3_W / orig_img_W
    scale_y = da3_H / orig_img_H
    
    pix_x_scaled = (projected_pix[:, 0] * scale_x).clamp(0, da3_W - 1).long()
    pix_y_scaled = (projected_pix[:, 1] * scale_y).clamp(0, da3_H - 1).long()
    
    # 从 DA3 深度图采样 (最近邻)
    da3_depth_sampled = da3_depth[pix_y_scaled, pix_x_scaled]  # (N,)
    
    # ============================================================
    # Median Scaling 对齐 DA3 深度
    # DA3 是 affine-invariant 的，需要对齐到绝对深度
    # 使用 FOV 内的有效体素进行 median scaling
    # ============================================================
    valid_indices = valid_mask & (pix_z > 0) & (da3_depth_sampled > 0)
    if valid_indices.sum() > 100:  # 至少需要 100 个有效点
        pix_z_valid = pix_z[valid_indices]
        da3_valid = da3_depth_sampled[valid_indices]
        
        # Median scaling: scale = median(gt / pred)
        scale = torch.median(pix_z_valid / da3_valid)
        da3_depth_sampled = da3_depth_sampled * scale
        
        # 可选：打印对齐信息
        # print(f"[DA3 Align] scale={scale:.3f}, da3_mean_after={da3_depth_sampled[valid_indices].mean():.2f}")
    else:
        # 如果有效点不够，跳过过滤
        return sigmas_flat.reshape(256, 256, 32), 0, valid_mask.sum().item()
    
    # 计算残差：delta = voxel_depth - da3_depth
    # delta < 0 表示体素在 DA3 预测表面的前方（更靠近相机）
    delta = pix_z - da3_depth_sampled
    
    # 深度自适应阈值：近场宽松，远场严格
    threshold = torch.where(
        da3_depth_sampled < DA3_NEAR_FAR_BOUNDARY,
        torch.full_like(da3_depth_sampled, DA3_NEAR_THRESHOLD),
        torch.full_like(da3_depth_sampled, DA3_FAR_THRESHOLD)
    )
    
    # 过滤条件：体素在预测表面前方超过阈值
    # 只过滤 FOV 内的体素，避免影响 FOV 外的区域
    is_free_space = (delta < -threshold) & valid_mask
    
    # 应用过滤
    sigmas_flat[is_free_space] = 0.0
    
    # 统计过滤掉的体素数量
    num_filtered = is_free_space.sum().item()
    num_valid = valid_mask.sum().item()
    
    return sigmas_flat.reshape(256, 256, 32), num_filtered, num_valid


def downsample_and_predict(data, net, pts, factor):
    pts = pts.reshape(256*factor, 256*factor, 32*factor, 3)

    sigmas = torch.zeros(256, 256, 32).numpy()
    segs = torch.zeros(256, 256, 32).numpy()

    chunk_size_x = chunk_size_y = 256
    chunk_size_z = 32

    n_chunks_x = int(256*factor / chunk_size_x)
    n_chunks_y = int(256*factor / chunk_size_y)
    n_chunks_z = int(32*factor / chunk_size_z)


    b_x = chunk_size_x // factor # size of the mini blocks
    b_y = chunk_size_y // factor
    b_z = chunk_size_z // factor


    for i in range(n_chunks_x):
        for j in range(n_chunks_y):
            for k in range(n_chunks_z):
                pts_block = pts[i * chunk_size_x:(i + 1) * chunk_size_x, j * chunk_size_y:(j + 1) * chunk_size_y, k * chunk_size_z:(k + 1) * chunk_size_z]
                sigmas_block, segs_block = predict_grid(data, net, pts_block)
                sigmas_block = sigmas_block.reshape(chunk_size_x, chunk_size_y, chunk_size_z)
                segs_block = segs_block.reshape(chunk_size_x, chunk_size_y, chunk_size_z, 19)

                if USE_ALPHA_WEIGHTING:
                    alphas = 1 - torch.exp(- VOXEL_SIZE * sigmas_block)
                    segs_block = (alphas.unsqueeze(-1) * segs_block).unsqueeze(0)
                else:
                    segs_block = (sigmas_block.unsqueeze(-1) * segs_block).unsqueeze(0)

                segs_pool_list = [F.avg_pool3d(segs_block[..., i], kernel_size=factor, stride=factor, padding=0) for i in
                                  range(segs_block.shape[-1])]
                segs_pool = torch.stack(segs_pool_list, dim=-1).unsqueeze(0)
                segs_pool = torch.argmax(segs_pool, dim=-1).detach().cpu().numpy()

                # pool the observations
                sigmas_block = F.max_pool3d(sigmas_block.unsqueeze(0), kernel_size=factor, stride=factor, padding=0).squeeze(0).detach().cpu().numpy()

                sigmas[i * b_x:(i + 1) * b_x, j * b_y: (j + 1) * b_y, b_z * k:b_z * (k + 1)] = sigmas_block
                segs[i * b_x:(i + 1) * b_x, j * b_y: (j + 1) * b_y, b_z * k:b_z * (k + 1)] = segs_pool

                torch.cuda.empty_cache()

    if USE_GROW:
        sigmas = F.max_pool3d(torch.tensor(sigmas).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).numpy()

    return sigmas, segs

def use_custom_maxpool(_sigmas):
    sigmas = torch.zeros(258, 258, 34)
    sigmas[1:257, 1:257, 1:33] = torch.tensor(_sigmas)
    sigmas_pooled = torch.zeros(256, 256, 32)

    for i in range(256):
        for j in range(256):
            for k in range(32):
                sigmas_pooled[i, j, k] = max(sigmas[i+1, j+1, k+1],
                                             sigmas[i, j+1, k+1], sigmas[i+1, j, k+1],sigmas[i+1, j+1, k],
                                             sigmas[i+2, j+1, k+1], sigmas[i+1, j+2, k+1],sigmas[i+1, j+1, k+2])
    return sigmas_pooled

def plot_images(images_dict):
    """The images dict should include six images and six corresponding ids"""
    images = images_dict["images"]
    ids = images_dict["ids"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 6))

    axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(images[i])
        axes[i].axis("off")
        axes[i].set_title(f"FrameId: {ids[i]}")

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()

def plot_image_at_frame_id(dataset, frame_id):

    for i in range(len(dataset)):
        sequence, id, is_right = dataset._datapoints[i]
        if id == frame_id:
            data = dataset[i]
            plt.figure(figsize=(10, 4))
            plt.imshow(((data["imgs"][0] + 1) / 2).permute(1, 2, 0))
            plt.gca().set_axis_off()
            plt.show()
            return



def identify_additional_invalids(target):
    # Note: The Numpy implementation is a bit faster (about 0.1 seconds per iteration)

    _t = np.concatenate([np.zeros([256, 256, 1]), target], axis=2)
    invalids = np.cumsum(np.logical_and(_t != 255, _t != 0), axis=2)[:, :, :32] == 0
    # _t = torch.cat([torch.zeros([256, 256, 1], device=device, dtype=torch.int32), torch.tensor(target, dtype=torch.int32).to(device)], dim=2)
    # invalids = torch.cumsum((_t != 255) & (_t != 0), axis=2)[:,:, :32] == 0
    # height cut-off (z > 6 ==> no invalid)
    invalids[: , :, 7:] = 0
    # only empty voxels matter
    invalids[target != 0] = 0

    # return invalids.cpu().numpy()
    return invalids

def predict_grid(data_batch, net, points):
    images = torch.stack(data_batch["imgs"], dim=0).unsqueeze(0).to(device).float()
    poses = torch.tensor(np.stack(data_batch["poses"], 0)).unsqueeze(0).to(device).float()
    projs = torch.tensor(np.stack(data_batch["projs"], 0)).unsqueeze(0).to(device).float()

    poses = torch.inverse(poses[:, :1]) @ poses

    n, nv, c, h, w = images.shape

    net.compute_grid_transforms(projs, poses)
    # Pass empty data dict as first argument (required for DA3 depth extraction, not used in evaluation)
    net.encode({}, images, projs, poses, ids_encoder=[0], ids_render=[0])

    net.set_scale(0)

    # q_pts = get_pts(X_RANGE, Y_RANGE, Z_RANGE, p_res[1], p_res_y, p_res[0])
    # q_pts = q_pts.to(device).reshape(1, -1, 3)
    # # _, invalid, sigmas = net.forward(q_pts)
    #
    points = points.reshape(1, -1, 3)
    _, invalid, sigmas, segs = net.forward(points, predict_segmentation=True)

    return sigmas, segs


def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)


def compute_occupancy_numbers_segmentation(y_pred, y_true, fov_mask, labels):
    label_ids = list(labels.keys())[1:]
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()[mask]
    y_true = y_true.flatten()[mask]

    tp = np.zeros(len(label_ids))
    fp = np.zeros(len(label_ids))
    fn = np.zeros(len(label_ids))
    tn = np.zeros(len(label_ids))

    for label_id in label_ids:
        tp[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred == label_id))
        fp[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred == label_id))
        fn[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred != label_id))
        tn[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred != label_id))

    return tp, fp, tn, fn


def compute_occupancy_numbers(y_pred, y_true, fov_mask):
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    occ_true = y_true[mask] > 0
    occ_pred = y_pred[mask] > 0

    tp = np.sum(np.logical_and(occ_true == 1, occ_pred == 1))
    fp = np.sum(np.logical_and(occ_true == 0, occ_pred == 1))
    fn = np.sum(np.logical_and(occ_true == 1, occ_pred == 0))
    tn = np.sum(np.logical_and(occ_true == 0, occ_pred == 0))

    return tp, fp, tn, fn


def save_evaluation_results(cp_path, results, results_table, mean_iou, valid_mean_iou, weighted_mean_iou, label_maps, dataset_length, sizes):
    """
    保存评估结果到与权重一致的目录，文件名以权重名+时间命名
    """
    # 获取权重文件所在目录和文件名
    cp_path = Path(cp_path)
    weight_dir = cp_path.parent
    weight_name = cp_path.stem  # 不带扩展名的文件名
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成输出文件名
    output_filename = f"evaluation_{weight_name}_{timestamp}.txt"
    output_path = weight_dir / output_filename
    
    # 构建结果内容
    content_lines = []
    content_lines.append("# SSCBench Evaluation Results")
    content_lines.append(f"# Checkpoint: {cp_path.name}")
    content_lines.append(f"# Model Directory: {weight_dir}")
    content_lines.append(f"# Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content_lines.append("")
    content_lines.append("=" * 80)
    content_lines.append("                           SSCBench Evaluation Results")
    content_lines.append("=" * 80)
    content_lines.append("")
    content_lines.append(f"Checkpoint: {cp_path.name}")
    content_lines.append(f"Dataset: SSCBench KITTI ({dataset_length} samples)")
    content_lines.append("")
    
    labels = label_maps["labels"]
    
    for size in sizes:
        content_lines.append("-" * 80)
        content_lines.append(f"Results for size {size}")
        content_lines.append("-" * 80)
        
        recall = results[size]["tp"] / (results[size]["tp"] + results[size]["fn"])
        precision = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"])
        iou = results[size]["tp"] / (results[size]["tp"] + results[size]["fp"] + results[size]["fn"])
        
        content_lines.append("Occupancy Metrics:")
        content_lines.append(f"  - IoU:       {iou*100:.2f}")
        content_lines.append(f"  - Precision: {precision*100:.2f}%")
        content_lines.append(f"  - Recall:    {recall*100:.2f}%")
        content_lines.append("")
        
        recall_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fn_seg"])
        precision_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"])
        iou_seg = results[size]["tp_seg"] / (results[size]["tp_seg"] + results[size]["fp_seg"] + results[size]["fn_seg"])
        
        seg_miou = np.mean(np.nan_to_num(iou_seg))
        
        content_lines.append(f"Segmentation Metrics (mIoU: {seg_miou*100:.2f}):")
        content_lines.append("  | Class          | IoU    | Precision | Recall |")
        content_lines.append("  |----------------|--------|-----------|--------|")
        
        for i in range(15):
            class_name = labels[i+1]
            p = precision_seg[i]*100 if not np.isnan(precision_seg[i]) else float('nan')
            r = recall_seg[i]*100 if not np.isnan(recall_seg[i]) else float('nan')
            content_lines.append(f"  | {class_name:<14} | {iou_seg[i]*100:6.2f} | {p:8.2f}% | {r:6.2f}% |")
        
        content_lines.append("")
    
    # 添加汇总表
    content_lines.append("=" * 80)
    content_lines.append("                               Summary Table")
    content_lines.append("=" * 80)
    content_lines.append("")
    content_lines.append("Results table (for copying):")
    content_lines.append("             | 12.8   | 25.6   | 51.2   |")
    content_lines.append("-------------|--------|--------|--------|")
    
    row_names = ["IoU", "Precision", "Recall", "mIoU"] + [labels[i+1] for i in range(15)]
    for i, row_name in enumerate(row_names):
        content_lines.append(f"{row_name:<13}| {results_table[i, 0]*100:6.2f} | {results_table[i, 1]*100:6.2f} | {results_table[i, 2]*100:6.2f} |")
    
    content_lines.append("")
    content_lines.append(f"Mean IoU: {mean_iou*100:.2f}")
    if valid_mean_iou is not None and not np.isnan(valid_mean_iou):
        content_lines.append(f"Valid-class Mean IoU: {valid_mean_iou*100:.2f}")
    else:
        content_lines.append("Valid-class Mean IoU: nan")
    content_lines.append(f"Weighted Mean IoU: {weighted_mean_iou*100:.2f}")
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write('\n'.join(content_lines))
    
    logging.info(f"Evaluation results saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    with torch.no_grad():
        main()
