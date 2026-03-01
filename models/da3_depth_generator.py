"""
DA3 Online Depth Generator Module

This module provides online depth estimation using Depth Anything 3 (DA3NESTED-GIANT-LARGE).
It generates depth maps and confidence maps for adaptive sampling in S4C training.

Features:
- Online depth inference with DA3NESTED-GIANT-LARGE
- Confidence-based filtering (bottom 20% uses uniform sampling)
- Batch processing for multiple views
- Caching support to avoid redundant computation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# Add DA3 to path
DA3_PATH = Path(__file__).parent.parent / "DA3" / "src"
if str(DA3_PATH) not in sys.path:
    sys.path.insert(0, str(DA3_PATH))


class DA3DepthGenerator(nn.Module):
    """
    Online depth generator using Depth Anything 3 (DA3NESTED-GIANT-LARGE).
    
    This module handles:
    1. Model loading from local checkpoint
    2. Batch inference for multiple views
    3. Confidence map extraction
    4. Confidence-based filtering mask generation
    
    Args:
        checkpoint_path: Path to DA3NESTED-GIANT-LARGE checkpoint directory
        device: Device to run inference on
        conf_percentile: Percentile threshold for low-confidence filtering (default: 20)
    """
    
    def __init__(
        self,
        checkpoint_path: str = "/home/lmh/dino-da3-adaptive-sampling1/DA3/ckpt/DA3NESTED-GIANT-LARGE",
        device: Optional[torch.device] = None,
        conf_percentile: float = 20.0,
    ):
        super().__init__()
        
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_percentile = conf_percentile
        
        # Model will be lazily loaded
        self._model = None
        self._is_initialized = False
        
        print(f"[DA3DepthGenerator] Initialized with checkpoint: {checkpoint_path}")
        print(f"[DA3DepthGenerator] Confidence percentile threshold: {conf_percentile}%")
    
    def _load_model(self):
        """Lazily load DA3 model from checkpoint."""
        if self._is_initialized:
            return
        
        try:
            from depth_anything_3.api import DepthAnything3
            
            # Check if checkpoint exists
            config_path = self.checkpoint_path / "config.json"
            weights_path = self.checkpoint_path / "model.safetensors"
            
            if not config_path.exists() or not weights_path.exists():
                raise FileNotFoundError(
                    f"DA3 checkpoint not found at {self.checkpoint_path}. "
                    f"Expected config.json and model.safetensors"
                )
            
            print(f"[DA3DepthGenerator] Loading model from {self.checkpoint_path}...")
            
            # Load from local checkpoint
            self._model = DepthAnything3.from_pretrained(
                str(self.checkpoint_path),
                local_files_only=True
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Freeze all parameters
            for param in self._model.parameters():
                param.requires_grad = False
            
            self._is_initialized = True
            print(f"[DA3DepthGenerator] Model loaded successfully on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import DA3 modules. Make sure DA3 is properly installed. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DA3 model: {e}")
    
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_confidence_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate depth maps and confidence maps for input images.
        
        Args:
            images: Input images tensor (N, 3, H, W) in range [-1, 1] or [0, 1]
            return_confidence_mask: Whether to return confidence-based mask
        
        Returns:
            Dictionary containing:
            - depth: Depth maps (N, H, W) in meters
            - conf: Confidence maps (N, H, W) in range [0, 1]
            - low_conf_mask: Boolean mask for low-confidence pixels (N, H, W)
        """
        self._load_model()
        
        N, C, H, W = images.shape
        device = images.device
        
        # Convert from [-1, 1] to [0, 1] if needed
        if images.min() < 0:
            images = images * 0.5 + 0.5
        
        # Convert to numpy for DA3 inference
        # DA3 expects list of numpy arrays or PIL images
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)
        images_np = (images_np * 255).astype(np.uint8)
        images_list = [img for img in images_np]
        
        # Run DA3 inference
        prediction = self._model.inference(
            image=images_list,
            process_res=max(H, W),  # Use original resolution
            process_res_method="upper_bound_resize",
        )
        
        # Extract depth and confidence
        depth = torch.from_numpy(prediction.depth).to(device).float()  # (N, H, W)
        
        # Handle confidence - may be None for some models
        if prediction.conf is not None:
            conf = torch.from_numpy(prediction.conf).to(device).float()  # (N, H, W)
        else:
            # If no confidence available, use depth gradient as proxy
            conf = self._compute_proxy_confidence(depth)
        
        # Resize to original resolution if needed
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(
                depth.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1)
            conf = F.interpolate(
                conf.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1)
        
        result = {
            "depth": depth,
            "conf": conf,
        }
        
        # Compute low-confidence mask
        if return_confidence_mask:
            low_conf_mask = self._compute_low_confidence_mask(conf)
            result["low_conf_mask"] = low_conf_mask
        
        return result
    
    def _compute_proxy_confidence(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute proxy confidence from depth map when confidence is not available.
        Uses inverse of depth gradient magnitude as proxy.
        
        Args:
            depth: Depth tensor (N, H, W)
        
        Returns:
            Confidence tensor (N, H, W) in range [0, 1]
        """
        # Compute depth gradients
        depth_pad = F.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        grad_x = depth_pad[:, :, :, 2:] - depth_pad[:, :, :, :-2]
        grad_y = depth_pad[:, :, 2:, :] - depth_pad[:, :, :-2, :]
        
        grad_x = grad_x[:, :, 1:-1, :]
        grad_y = grad_y[:, :, :, 1:-1]
        
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8).squeeze(1)
        
        # Normalize and invert (lower gradient = higher confidence)
        grad_norm = grad_magnitude / (grad_magnitude.max() + 1e-8)
        conf = 1.0 - grad_norm.clamp(0, 1)
        
        return conf
    
    def _compute_low_confidence_mask(self, conf: torch.Tensor) -> torch.Tensor:
        """
        Compute mask for low-confidence pixels (bottom percentile).
        
        Args:
            conf: Confidence tensor (N, H, W)
        
        Returns:
            Boolean mask where True indicates low confidence (N, H, W)
        """
        N = conf.shape[0]
        low_conf_mask = torch.zeros_like(conf, dtype=torch.bool)
        
        for i in range(N):
            conf_flat = conf[i].flatten()
            threshold = torch.quantile(conf_flat, self.conf_percentile / 100.0)
            low_conf_mask[i] = conf[i] < threshold
        
        return low_conf_mask
    
    def generate_for_batch(
        self,
        images: torch.Tensor,
        is_perspective: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate depth maps for a batch of images, handling perspective vs fisheye cameras.
        
        Args:
            images: Input images (N, 3, H, W)
            is_perspective: Boolean tensor indicating perspective cameras (N,)
        
        Returns:
            Tuple of:
            - depth: Depth maps (N, H, W) - zeros for non-perspective cameras
            - conf: Confidence maps (N, H, W) - zeros for non-perspective cameras
            - low_conf_mask: Low confidence mask (N, H, W)
        """
        N, C, H, W = images.shape
        device = images.device
        
        # Initialize outputs with zeros
        depth = torch.zeros(N, H, W, device=device)
        conf = torch.zeros(N, H, W, device=device)
        low_conf_mask = torch.ones(N, H, W, device=device, dtype=torch.bool)  # All low conf by default
        
        # Find perspective camera indices
        perspective_indices = torch.where(is_perspective)[0]
        
        if len(perspective_indices) > 0:
            # Extract perspective images
            perspective_images = images[perspective_indices]
            
            # Run DA3 inference
            result = self.forward(perspective_images, return_confidence_mask=True)
            
            # Fill in results
            depth[perspective_indices] = result["depth"]
            conf[perspective_indices] = result["conf"]
            low_conf_mask[perspective_indices] = result["low_conf_mask"]
        
        return depth, conf, low_conf_mask


class DA3DepthCache:
    """
    Simple cache for DA3 depth results to avoid redundant computation.
    
    Uses image hash as key for caching.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._access_order: List[int] = []
    
    def _compute_hash(self, image: torch.Tensor) -> int:
        """Compute hash for an image tensor."""
        # Use a simple hash based on sampled pixels
        sampled = image[:, ::16, ::16].flatten()
        return hash(tuple(sampled[:100].tolist()))
    
    def get(self, image: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached result for an image."""
        key = self._compute_hash(image)
        if key in self._cache:
            # Move to end of access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, image: torch.Tensor, result: Dict[str, torch.Tensor]):
        """Cache result for an image."""
        key = self._compute_hash(image)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = result
        self._access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


# Global singleton instance
_da3_generator: Optional[DA3DepthGenerator] = None
_da3_cache: Optional[DA3DepthCache] = None


def get_da3_generator(
    checkpoint_path: str = "/home/lmh/dino-da3-adaptive-sampling1/DA3/ckpt/DA3NESTED-GIANT-LARGE",
    device: Optional[torch.device] = None,
    conf_percentile: float = 20.0,
) -> DA3DepthGenerator:
    """
    Get or create the global DA3 depth generator instance.
    
    Args:
        checkpoint_path: Path to DA3 checkpoint
        device: Device for inference
        conf_percentile: Confidence percentile threshold
    
    Returns:
        DA3DepthGenerator instance
    """
    global _da3_generator
    
    if _da3_generator is None:
        _da3_generator = DA3DepthGenerator(
            checkpoint_path=checkpoint_path,
            device=device,
            conf_percentile=conf_percentile,
        )
    
    return _da3_generator


def get_da3_cache() -> DA3DepthCache:
    """Get or create the global DA3 cache instance."""
    global _da3_cache
    
    if _da3_cache is None:
        _da3_cache = DA3DepthCache()
    
    return _da3_cache
