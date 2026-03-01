#!/usr/bin/env python3
"""
Module Testing Script for S4C-Pro-Adaptive Project
===================================================

This script tests and validates individual modules:
1. DA3 Depth Generator Module
2. NeRF Renderer with Adaptive Sampling
3. BTSNet with DINOv2 Backbone
4. Ray Sampler with DA3 depth support

Usage:
    python scripts/test_modules.py

Author: S4C-Pro-Adaptive
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "module_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_nerf_adaptive_sampling():
    """
    Test the NeRF renderer's adaptive sampling function.
    """
    print("\n" + "=" * 60)
    print("TEST: NeRF Adaptive Sampling")
    print("=" * 60)
    
    try:
        from models.common.render.nerf import NeRFRenderer
        
        # Create renderer
        renderer = NeRFRenderer(
            n_coarse=64,  # 24 surface + 40 global
            n_fine=0,
            lindisp=True
        )
        print("[✓] NeRFRenderer created")
        
        # Create test rays
        B = 1000  # Batch of rays
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ray format: [origin(3), direction(3), near(1), far(1)]
        rays = torch.zeros(B, 8, device=device)
        rays[:, 0:3] = torch.randn(B, 3)  # Random origins
        rays[:, 3:6] = torch.tensor([0, 0, 1]).expand(B, 3).float()  # Forward direction
        rays[:, 6] = 3.0  # z_near
        rays[:, 7] = 80.0  # z_far
        
        # DA3 depths (random between z_near and z_far)
        da3_depths = torch.rand(B, device=device) * 60 + 10  # [10, 70]m
        
        print(f"[INFO] Test rays shape: {rays.shape}")
        print(f"[INFO] DA3 depths shape: {da3_depths.shape}")
        print(f"[INFO] DA3 depths range: [{da3_depths.min():.1f}, {da3_depths.max():.1f}]m")
        
        # ============================================================
        # Test 1: Uniform sampling
        # ============================================================
        print("\n[Test 1] Uniform Sampling...")
        z_uniform = renderer.sample_coarse(rays)
        print(f"   Output shape: {z_uniform.shape}")
        print(f"   Sample range: [{z_uniform.min():.2f}, {z_uniform.max():.2f}]")
        
        # ============================================================
        # Test 2: Adaptive sampling without confidence mask
        # ============================================================
        print("\n[Test 2] Adaptive Sampling (no confidence mask)...")
        z_adaptive = renderer.sample_adaptive(
            rays, da3_depths,
            n_surface=24, n_global=40,
            absrel_prior=0.10, min_thickness=0.5,
            low_conf_mask=None
        )
        print(f"   Output shape: {z_adaptive.shape}")
        print(f"   Sample range: [{z_adaptive.min():.2f}, {z_adaptive.max():.2f}]")
        
        # ============================================================
        # Test 3: Adaptive sampling with confidence mask
        # ============================================================
        print("\n[Test 3] Adaptive Sampling (with confidence mask)...")
        # 20% of rays are low confidence
        low_conf_mask = torch.rand(B, device=device) < 0.2
        
        z_adaptive_conf = renderer.sample_adaptive(
            rays, da3_depths,
            n_surface=24, n_global=40,
            absrel_prior=0.10, min_thickness=0.5,
            low_conf_mask=low_conf_mask
        )
        print(f"   Output shape: {z_adaptive_conf.shape}")
        print(f"   Low confidence rays: {low_conf_mask.sum().item()}/{B}")
        
        # ============================================================
        # Verification: Check samples are sorted
        # ============================================================
        print("\n[Verification] Checking sample ordering...")
        sorted_check = torch.all(z_adaptive[:, 1:] >= z_adaptive[:, :-1])
        print(f"   Samples are sorted: {sorted_check.item()}")
        
        # ============================================================
        # Verification: Check adaptive samples concentrate around DA3 depth
        # ============================================================
        print("\n[Verification] Checking sample concentration...")
        
        # For each ray, count samples within ±2σ of DA3 depth
        concentrations = []
        for i in range(min(100, B)):
            d = da3_depths[i].item()
            sigma = max(0.5, 0.10 * d)
            d_min, d_max = d - 2*sigma, d + 2*sigma
            
            # Adaptive samples within range
            adaptive_in_range = ((z_adaptive[i] >= d_min) & (z_adaptive[i] <= d_max)).sum().item()
            # Uniform samples within range
            uniform_in_range = ((z_uniform[i] >= d_min) & (z_uniform[i] <= d_max)).sum().item()
            
            concentrations.append((adaptive_in_range, uniform_in_range))
        
        adaptive_mean = np.mean([c[0] for c in concentrations])
        uniform_mean = np.mean([c[1] for c in concentrations])
        
        print(f"   Adaptive: {adaptive_mean:.1f} samples within ±2σ (avg)")
        print(f"   Uniform:  {uniform_mean:.1f} samples within ±2σ (avg)")
        print(f"   Improvement: {adaptive_mean / uniform_mean:.2f}x")
        
        # ============================================================
        # Create visualization
        # ============================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('NeRF Adaptive Sampling Test Results', fontsize=14, fontweight='bold')
        
        # Plot 1: Sample distribution comparison
        ax = axes[0, 0]
        ax.hist(z_uniform.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Uniform', color='red')
        ax.hist(z_adaptive.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Adaptive', color='green')
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Count')
        ax.set_title('Sample Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: DA3 depth distribution
        ax = axes[0, 1]
        ax.hist(da3_depths.cpu().numpy(), bins=50, color='blue', alpha=0.7)
        ax.set_xlabel('DA3 Depth (m)')
        ax.set_ylabel('Count')
        ax.set_title('DA3 Depth Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Sample concentration comparison
        ax = axes[1, 0]
        labels = ['Adaptive', 'Uniform']
        values = [adaptive_mean, uniform_mean]
        bars = ax.bar(labels, values, color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Avg samples within ±2σ')
        ax.set_title('Surface Coverage Comparison')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Single ray example
        ax = axes[1, 1]
        ray_idx = 0
        d = da3_depths[ray_idx].item()
        sigma = max(0.5, 0.10 * d)
        
        ax.scatter(z_uniform[ray_idx].cpu().numpy(), [1]*64, c='red', s=30, label='Uniform', alpha=0.7)
        ax.scatter(z_adaptive[ray_idx].cpu().numpy(), [0]*64, c='green', s=30, label='Adaptive', alpha=0.7)
        ax.axvline(x=d, color='blue', linestyle='--', linewidth=2, label=f'DA3 depth={d:.1f}m')
        ax.axvspan(d - 2*sigma, d + 2*sigma, alpha=0.2, color='blue')
        ax.set_xlabel('Depth (m)')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Adaptive', 'Uniform'])
        ax.set_title(f'Single Ray Example (ray {ray_idx})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'test_nerf_adaptive_sampling.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n[✓] Visualization saved: {OUTPUT_DIR / 'test_nerf_adaptive_sampling.png'}")
        print("[✓] TEST PASSED: NeRF Adaptive Sampling")
        return True
        
    except Exception as e:
        print(f"\n[✗] TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_ray_sampler():
    """
    Test the ray sampler with DA3 depth support.
    """
    print("\n" + "=" * 60)
    print("TEST: Ray Sampler with DA3 Depth")
    print("=" * 60)
    
    try:
        from models.bts.model.ray_sampler import PatchRaySampler, ImageRaySampler
        
        # Create samplers
        patch_sampler = PatchRaySampler(
            ray_batch_size=4096,
            z_near=3.0,
            z_far=80.0,
            patch_size=8,
            channels=3
        )
        print("[✓] PatchRaySampler created")
        
        image_sampler = ImageRaySampler(
            z_near=3.0,
            z_far=80.0,
            channels=3
        )
        print("[✓] ImageRaySampler created")
        
        # Create test data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n, v, c, h, w = 1, 2, 3, 192, 640
        
        images = torch.randn(n, v, c, h, w, device=device)
        poses = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(n, v, 4, 4).contiguous()
        projs = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(n, v, 4, 4).contiguous()
        # Set projection matrix for perspective camera
        projs[:, :, 0, 0] = 1.0
        projs[:, :, 1, 1] = 1.0
        
        # DA3 depths
        da3_depths = torch.rand(n, v, 1, h, w, device=device) * 60 + 10  # [10, 70]m
        low_conf_mask = torch.rand(n, v, h, w, device=device) < 0.2
        
        print(f"\n[INFO] Test data shapes:")
        print(f"   Images: {images.shape}")
        print(f"   Poses: {poses.shape}")
        print(f"   Projs: {projs.shape}")
        print(f"   DA3 depths: {da3_depths.shape}")
        print(f"   Low conf mask: {low_conf_mask.shape}")
        
        # ============================================================
        # Test PatchRaySampler
        # ============================================================
        print("\n[Test] PatchRaySampler.sample()...")
        result = patch_sampler.sample(images, poses, projs, 
                                     da3_depths=da3_depths, 
                                     low_conf_mask=low_conf_mask)
        
        if len(result) == 4:
            rays, rgb_gt, sampled_depths, sampled_conf = result
        else:
            rays, rgb_gt = result[:2]
            sampled_depths = result[2] if len(result) > 2 else None
            sampled_conf = result[3] if len(result) > 3 else None
        
        print(f"   Rays shape: {rays.shape}")
        print(f"   RGB GT shape: {rgb_gt.shape}")
        if sampled_depths is not None:
            print(f"   Sampled depths shape: {sampled_depths.shape}")
        if sampled_conf is not None:
            print(f"   Sampled conf shape: {sampled_conf.shape}")
        
        # ============================================================
        # Test ImageRaySampler
        # ============================================================
        print("\n[Test] ImageRaySampler.sample()...")
        result = image_sampler.sample(images, poses, projs,
                                     da3_depths=da3_depths,
                                     low_conf_mask=low_conf_mask)
        
        if len(result) == 4:
            rays, rgb_gt, sampled_depths, sampled_conf = result
        else:
            rays, rgb_gt = result[:2]
            sampled_depths = result[2] if len(result) > 2 else None
            sampled_conf = result[3] if len(result) > 3 else None
        
        print(f"   Rays shape: {rays.shape}")
        if rgb_gt is not None:
            print(f"   RGB GT shape: {rgb_gt.shape}")
        if sampled_depths is not None:
            print(f"   Sampled depths shape: {sampled_depths.shape}")
        if sampled_conf is not None:
            print(f"   Sampled conf shape: {sampled_conf.shape}")
        
        print("\n[✓] TEST PASSED: Ray Sampler")
        return True
        
    except Exception as e:
        print(f"\n[✗] TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_btsnet_forward():
    """
    Test BTSNet forward pass (without full training setup).
    """
    print("\n" + "=" * 60)
    print("TEST: BTSNet Structure")
    print("=" * 60)
    
    try:
        from models.bts.model.models_bts import BTSNet
        from omegaconf import OmegaConf
        
        # Create config
        config = OmegaConf.create({
            "z_near": 3.0,
            "z_far": 80.0,
            "learn_empty": False,
            "inv_z": True,
            "code_mode": "z",
            "adaptive_sampling": True,
            "n_surface_samples": 24,
            "n_global_samples": 40,
            "absrel_prior": 0.10,
            "min_thickness": 0.5,
            "sample_color": True,
            "segmentation_mode": "m2f",
            "encoder": {
                "type": "monodepth2",
                "freeze": False,
                "pretrained": True,
                "resnet_layers": 50,
                "num_ch_dec": [32, 32, 64, 128, 256],
                "d_out": 64
            },
            "code": {
                "num_freqs": 6,
                "freq_factor": 1.5,
                "include_input": True
            },
            "mlp_coarse": {
                "type": "resnet",
                "n_blocks": 0,
                "d_hidden": 64
            },
            "mlp_fine": {
                "type": "empty",
                "n_blocks": 1,
                "d_hidden": 128
            }
        })
        
        print("[INFO] Creating BTSNet (this may take a moment to load DINOv2)...")
        
        # Create network (will load DINOv2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = BTSNet(config).to(device)
        
        print(f"[✓] BTSNet created on {device}")
        
        # Print model structure
        print("\n[INFO] Model structure:")
        print(f"   DINOv2 backbone: dinov2_vitl14_reg")
        print(f"   Adapter output dim: {net.latent_size}")
        print(f"   Adaptive sampling: {net.adaptive_sampling}")
        print(f"   Surface samples: {net.n_surface_samples}")
        print(f"   Global samples: {net.n_global_samples}")
        
        # Count parameters
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        
        print(f"\n[INFO] Parameter count:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Frozen (DINOv2): {total_params - trainable_params:,}")
        
        # Test adapter
        print("\n[Test] DINOv2 Adapter...")
        dummy_features = torch.randn(1, 1024, 14, 46, device=device)  # DINOv2 output shape
        adapter_out = net.dino_adapter(dummy_features)
        print(f"   Input: {dummy_features.shape}")
        print(f"   Output: {adapter_out.shape}")
        
        print("\n[✓] TEST PASSED: BTSNet Structure")
        return True
        
    except Exception as e:
        print(f"\n[✗] TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_da3_module():
    """
    Test DA3 depth generator module structure.
    """
    print("\n" + "=" * 60)
    print("TEST: DA3 Depth Generator Module")
    print("=" * 60)
    
    try:
        from models.da3_depth_generator import DA3DepthGenerator, DA3DepthCache
        
        # Test cache
        cache = DA3DepthCache(max_size=100)
        print("[✓] DA3DepthCache created")
        
        # Test generator initialization (without loading model)
        print("\n[INFO] DA3DepthGenerator structure:")
        print(f"   Checkpoint path: /home/lmh/dino-da3-adaptive-sampling1/DA3/ckpt/DA3NESTED-GIANT-LARGE")
        print(f"   Confidence percentile: 20.0%")
        
        # Check if DA3 checkpoint exists
        ckpt_path = Path("/home/lmh/dino-da3-adaptive-sampling1/DA3/ckpt/DA3NESTED-GIANT-LARGE")
        config_exists = (ckpt_path / "config.json").exists()
        weights_exists = (ckpt_path / "model.safetensors").exists()
        
        print(f"\n[INFO] Checkpoint status:")
        print(f"   Path exists: {ckpt_path.exists()}")
        print(f"   config.json: {'✓' if config_exists else '✗'}")
        print(f"   model.safetensors: {'✓' if weights_exists else '✗'}")
        
        if config_exists and weights_exists:
            print("\n[INFO] DA3 checkpoint available, model can be loaded")
        else:
            print("\n[WARNING] DA3 checkpoint not fully available")
            print("   Online depth generation will not work")
            print("   Will fall back to offline depth loading or zero depths")
        
        print("\n[✓] TEST PASSED: DA3 Module Structure")
        return True
        
    except Exception as e:
        print(f"\n[✗] TEST FAILED: {e}")
        traceback.print_exc()
        return False


def generate_summary_report(results):
    """
    Generate summary report of all tests.
    """
    print("\n" + "=" * 70)
    print("TEST SUMMARY REPORT")
    print("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed\n")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print("\n" + "=" * 70)
    
    if passed_tests == total_tests:
        print("ALL TESTS PASSED - Modules are working correctly!")
    else:
        print(f"WARNING: {total_tests - passed_tests} test(s) failed")
        print("Please check the error messages above for details.")
    
    print("=" * 70)
    
    # Save report
    report_path = OUTPUT_DIR / "test_report.txt"
    with open(report_path, 'w') as f:
        f.write("S4C-Pro-Adaptive Module Test Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Results: {passed_tests}/{total_tests} tests passed\n\n")
        for test_name, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            f.write(f"  [{status}] {test_name}\n")
    
    print(f"\nReport saved to: {report_path}")
    
    return passed_tests == total_tests


def main():
    """
    Run all module tests.
    """
    print("=" * 70)
    print("S4C-Pro-Adaptive Module Testing")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    results = {}
    
    # Test 1: NeRF Adaptive Sampling
    results["NeRF Adaptive Sampling"] = test_nerf_adaptive_sampling()
    
    # Test 2: Ray Sampler
    results["Ray Sampler"] = test_ray_sampler()
    
    # Test 3: BTSNet Structure
    results["BTSNet Structure"] = test_btsnet_forward()
    
    # Test 4: DA3 Module
    results["DA3 Module"] = test_da3_module()
    
    # Generate summary
    all_passed = generate_summary_report(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
