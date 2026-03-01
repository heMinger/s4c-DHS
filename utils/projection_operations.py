import torch


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    """Analytic inverse of a batch of SE(3) 4x4 pose matrices.

    Avoids torch.linalg.inv which triggers 'lazy wrapper should be called
    at most once' under DataParallel in PyTorch 1.13.x.

    Args:
        pose: (..., 4, 4) rigid-body transformation matrices.

    Returns:
        (..., 4, 4) inverse transformation matrices.
    """
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -R_inv @ t
    inv = torch.zeros_like(pose)
    inv[..., :3, :3] = R_inv
    inv[..., :3, 3:] = t_inv
    inv[..., 3, 3] = 1.0
    return inv


def distance_to_z(depths: torch.Tensor, projs: torch.Tensor):
    n, nv, h, w = depths.shape
    device = depths.device

    inv_K = torch.linalg.inv(projs)

    grid_x = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    grid_y = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    img_points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).expand(n, nv, -1, -1, -1)
    cam_points = (inv_K @ img_points.view(n, nv, 3, -1)).view(n, nv, 3, h, w)
    factors = cam_points[:, :, 2, :, :] / torch.norm(cam_points, dim=2)

    return depths * factors
