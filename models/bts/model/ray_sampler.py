import torch
from omegaconf import ListConfig

from models.common.util import util


class RaySampler:
    def sample(self, images, poses, projs):
        raise NotImplementedError

    def reconstruct(self, render_dict):
        raise NotImplementedError


class RandomRaySampler(RaySampler):
    def __init__(self, ray_batch_size, z_near, z_far, channels=3):
        self.ray_batch_size = ray_batch_size
        self.z_near = z_near
        self.z_far = z_far
        self.channels = channels

    def sample(self, images, poses, projs):
        n, v, c, h, w = images.shape

        all_rgb_gt = []
        all_rays = []

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays = util.gen_rays(poses[n_].view(-1, 4, 4), w, h, focal=focals, c=centers, z_near=self.z_near, z_far=self.z_far).view(-1, 8)

            rgb_gt = images[n_].view(-1, self.channels, h, w)
            rgb_gt = (rgb_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, self.channels))

            pix_inds = torch.randint(0, v * h * w, (self.ray_batch_size,))

            rgb_gt = rgb_gt[pix_inds]
            rays = rays[pix_inds]

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays = torch.stack(all_rays)

        return all_rays, all_rgb_gt

    def reconstruct(self, render_dict, channels=None):
        coarse = render_dict["coarse"]
        fine = render_dict["fine"]

        if channels is None:
            channels = self.channels

        c_rgb = coarse["rgb"]  # n, n_pts, v * 3
        c_weights = coarse["weights"]
        c_depth = coarse["depth"]
        c_invalid = coarse["invalid"]

        f_rgb = fine["rgb"]  # n, n_pts, v * 3
        f_weights = fine["weights"]
        f_depth = fine["depth"]
        f_invalid = fine["invalid"]

        rgb_gt = render_dict["rgb_gt"]

        n, n_pts, v_c = c_rgb.shape
        v = v_c // self.channels
        c_n_smps = c_weights.shape[-1]
        f_n_smps = f_weights.shape[-1]

        coarse["rgb"] = c_rgb.view(n, n_pts, v, channels)
        coarse["weights"] = c_weights.view(n, n_pts, c_n_smps)
        coarse["depth"] = c_depth.view(n, n_pts)
        coarse["invalid"] = c_invalid.view(n, n_pts, c_n_smps, v)

        fine["rgb"] = f_rgb.view(n, n_pts, v, channels)
        fine["weights"] = f_weights.view(n, n_pts, f_n_smps)
        fine["depth"] = f_depth.view(n, n_pts)
        fine["invalid"] = f_invalid.view(n, n_pts, f_n_smps, v)

        if "alphas" in coarse:
            c_alphas = coarse["alphas"]
            f_alphas = fine["alphas"]
            coarse["alphas"] = c_alphas.view(n, n_pts, c_n_smps)
            fine["alphas"] = f_alphas.view(n, n_pts, f_n_smps)

        if "z_samps" in coarse:
            c_z_samps = coarse["z_samps"]
            f_z_samps = fine["z_samps"]
            coarse["z_samps"] = c_z_samps.view(n, n_pts, c_n_smps)
            fine["z_samps"] = f_z_samps.view(n, n_pts, f_n_smps)

        if "rgb_samps" in coarse:
            c_rgb_samps = coarse["rgb_samps"]
            f_rgb_samps = fine["rgb_samps"]
            coarse["rgb_samps"] = c_rgb_samps.view(n, n_pts, c_n_smps, v, channels)
            fine["rgb_samps"] = f_rgb_samps.view(n, n_pts, f_n_smps, v, channels)

        render_dict["coarse"] = coarse
        render_dict["fine"] = fine
        render_dict["rgb_gt"] = rgb_gt.view(n, n_pts, channels)

        return render_dict


class PatchRaySampler(RaySampler):
    """Patch-based ray sampler with optional DA3-guided adaptive sampling."""
    
    def __init__(self, ray_batch_size, z_near, z_far, patch_size, channels=3,
                 adaptive_sampling=True, n_surface_samples=16, n_global_samples=8,
                 absrel_prior=0.12, min_thickness=0.5):
        self.ray_batch_size = ray_batch_size
        self.z_near = z_near
        self.z_far = z_far
        if isinstance(patch_size, int):
            self.patch_size_x, self.patch_size_y = patch_size, patch_size
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list) or isinstance(patch_size, ListConfig):
            self.patch_size_y = patch_size[0]
            self.patch_size_x = patch_size[1]
        else:
            raise ValueError(f"Invalid format for patch size")
        self.channels = channels
        assert (ray_batch_size % (self.patch_size_x * self.patch_size_y)) == 0
        self._patch_count = self.ray_batch_size // (self.patch_size_x * self.patch_size_y)
        
        # Adaptive sampling parameters (for DA3-guided sampling)
        self.adaptive_sampling = adaptive_sampling
        self.n_surface_samples = n_surface_samples
        self.n_global_samples = n_global_samples
        self.absrel_prior = absrel_prior
        self.min_thickness = min_thickness

    def sample(self, images, poses, projs, segs=None, sample_segs=False, da3_depths=None, low_conf_mask=None):
        n, v, c, h, w = images.shape
        device = images.device

        images = images.permute(0, 1, 3, 4, 2)          # n, v, h, w, c

        all_rgb_gt = []
        all_rays = []
        all_segs_gt = []
        all_da3_depths = []
        all_low_conf_mask = []

        patch_coords_v_list = []
        
        # Check if DA3 depths are provided
        has_da3_depths = da3_depths is not None and da3_depths.numel() > 0
        has_low_conf_mask = low_conf_mask is not None

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays = util.gen_rays(poses[n_].view(-1, 4, 4), w, h, focal=focals, c=centers, z_near=self.z_near, z_far=self.z_far)

            patch_coords_v = torch.randint(0, v, (self._patch_count, ))
            patch_coords_y = torch.randint(0, h-self.patch_size_y, (self._patch_count, ))
            patch_coords_x = torch.randint(0, w-self.patch_size_x, (self._patch_count, ))

            patch_coords_v_list.append(patch_coords_v)

            sample_rgb_gt = []
            sample_rays = []
            sample_segs_gt = []
            sample_da3_depths = []
            sample_low_conf_mask = []

            for v_, y, x in zip(patch_coords_v, patch_coords_y, patch_coords_x):
                rgb_gt_patch = images[n_][v_, y:y+self.patch_size_y, x:x+self.patch_size_x, :].reshape(-1, self.channels)
                rays_patch = rays[v_, y:y+self.patch_size_y, x:x+self.patch_size_x, :].reshape(-1, 8)

                if sample_segs:
                    segs_gt_patch = segs[n_][v_, y:y + self.patch_size_y, x:x + self.patch_size_x].reshape(-1)
                    sample_segs_gt.append(segs_gt_patch)
                    
                # Sample DA3 depths for the same patch
                if has_da3_depths:
                    # da3_depths shape: (n, v, 1, h, w) -> access (n_, v_, 0, y:y+h, x:x+w)
                    da3_patch = da3_depths[n_, v_, 0, y:y+self.patch_size_y, x:x+self.patch_size_x].reshape(-1)
                    sample_da3_depths.append(da3_patch)
                
                # Sample low_conf_mask for the same patch
                if has_low_conf_mask:
                    # low_conf_mask shape: (n, v, h, w) -> access (n_, v_, y:y+h, x:x+w)
                    conf_patch = low_conf_mask[n_, v_, y:y+self.patch_size_y, x:x+self.patch_size_x].reshape(-1)
                    sample_low_conf_mask.append(conf_patch)

                sample_rgb_gt.append(rgb_gt_patch)
                sample_rays.append(rays_patch)

            sample_rgb_gt = torch.cat(sample_rgb_gt, dim=0)
            sample_rays = torch.cat(sample_rays, dim=0)

            if sample_segs:
                sample_segs_gt = torch.cat(sample_segs_gt, dim=0)
                
            if has_da3_depths:
                sample_da3_depths = torch.cat(sample_da3_depths, dim=0)
            
            if has_low_conf_mask:
                sample_low_conf_mask = torch.cat(sample_low_conf_mask, dim=0)

            all_rgb_gt.append(sample_rgb_gt)
            all_rays.append(sample_rays)
            all_segs_gt.append(sample_segs_gt)
            if has_da3_depths:
                all_da3_depths.append(sample_da3_depths)
            if has_low_conf_mask:
                all_low_conf_mask.append(sample_low_conf_mask)

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays = torch.stack(all_rays)

        patch_coords_v = torch.cat(patch_coords_v_list)

        if sample_segs:
            all_segs_gt = torch.stack(all_segs_gt)
            
        if has_da3_depths:
            all_da3_depths = torch.stack(all_da3_depths)  # (n, ray_batch_size)
        else:
            all_da3_depths = None
        
        if has_low_conf_mask:
            all_low_conf_mask = torch.stack(all_low_conf_mask)  # (n, ray_batch_size)
        else:
            all_low_conf_mask = None

        if sample_segs:
            return all_rays, all_rgb_gt, all_segs_gt, patch_coords_v, all_da3_depths, all_low_conf_mask
        else:
            return all_rays, all_rgb_gt, all_da3_depths, all_low_conf_mask

    def reconstruct(self, render_dict, channels=None, reconstruct_segmentation=False):
        coarse = render_dict["coarse"]
        fine = render_dict["fine"]

        if channels is None:
            channels = self.channels

        c_rgb = coarse["rgb"]  # n, n_pts, v * 3
        c_weights = coarse["weights"]
        c_depth = coarse["depth"]
        c_invalid = coarse["invalid"]

        if reconstruct_segmentation:
            c_segmentation = coarse["segs"]
            c_segmentation_gt = render_dict["segmentation_gt"]
            # c_segmentation_raw = coarse["segs_raw"]
            n_classes = c_segmentation.shape[-1]
            # n_samples = c_segmentation_raw.shape[-2]

        f_rgb = fine["rgb"]  # n, n_pts, v * 3
        f_weights = fine["weights"]
        f_depth = fine["depth"]
        f_invalid = fine["invalid"]

        rgb_gt = render_dict["rgb_gt"]

        n, n_pts, v_c = c_rgb.shape
        v = v_c // channels
        c_n_smps = c_weights.shape[-1]
        f_n_smps = f_weights.shape[-1]
        # (This can be a different v from the sample method)

        coarse["rgb"] = c_rgb.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, v, channels)
        coarse["weights"] = c_weights.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps)
        coarse["depth"] = c_depth.view(n, self._patch_count, self.patch_size_y, self.patch_size_x)
        coarse["invalid"] = c_invalid.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps, v)

        if reconstruct_segmentation:
            coarse["segs"] = c_segmentation.view(n, self._patch_count,
                                                            self.patch_size_y, self.patch_size_x, n_classes)
            # coarse["segs_raw"] = c_segmentation_raw.view(n, self._patch_count, self.patch_size_y, self.patch_size_x,
            #                                              n_samples, n_classes)
            render_dict["segmentation_gt"] = c_segmentation_gt.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, 1)

        fine["rgb"] = f_rgb.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, v, channels)
        fine["weights"] = f_weights.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps)
        fine["depth"] = f_depth.view(n, self._patch_count, self.patch_size_y, self.patch_size_x)
        fine["invalid"] = f_invalid.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps, v)

        if "alphas" in coarse:
            c_alphas = coarse["alphas"]
            f_alphas = fine["alphas"]
            coarse["alphas"] = c_alphas.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps)
            fine["alphas"] = f_alphas.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps)

        if "z_samps" in coarse:
            c_z_samps = coarse["z_samps"]
            f_z_samps = fine["z_samps"]
            coarse["z_samps"] = c_z_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps)
            fine["z_samps"] = f_z_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps)

        if "rgb_samps" in coarse:
            c_rgb_samps = coarse["rgb_samps"]
            f_rgb_samps = fine["rgb_samps"]
            coarse["rgb_samps"] = c_rgb_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps, v, channels)
            fine["rgb_samps"] = f_rgb_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps, v, channels)

        render_dict["coarse"] = coarse
        render_dict["fine"] = fine
        render_dict["rgb_gt"] = rgb_gt.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, channels)

        return render_dict


class ImageRaySampler(RaySampler):
    def __init__(self, z_near, z_far, height=None, width=None, channels=3, norm_dir=True):
        self.z_near = z_near
        self.z_far = z_far
        self.height = height
        self.width = width
        self.channels = channels
        self.norm_dir = norm_dir

    def sample(self, images, poses, projs, segs=None, sample_segs=False, da3_depths=None, low_conf_mask=None):
        n, v, _, _ = poses.shape

        if self.height is None:
            self.height, self.width = images.shape[-2:]

        all_rgb_gt = []
        all_rays = []
        all_segs_gt = []
        all_da3_depths = []
        all_low_conf_mask = []
        
        has_da3_depths = da3_depths is not None and da3_depths.numel() > 0
        has_low_conf_mask = low_conf_mask is not None

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays = util.gen_rays(poses[n_].view(-1, 4, 4), self.width, self.height, focal=focals, c=centers, z_near=self.z_near, z_far=self.z_far, norm_dir=self.norm_dir).view(-1, 8)
            all_rays.append(rays)

            if images is not None:
                rgb_gt = images[n_].view(-1, self.channels, self.height, self.width)
                rgb_gt = (rgb_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, self.channels))
                all_rgb_gt.append(rgb_gt)

            if sample_segs:
                segs_gt = segs[n_].view(-1, 1, self.height, self.width)
                segs_gt = (segs_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, 1))
                all_segs_gt.append(segs_gt)
                
            if has_da3_depths:
                # da3_depths shape: (n, v, 1, h, w)
                da3_flat = da3_depths[n_].view(-1, 1, self.height, self.width)
                da3_flat = da3_flat.permute(0, 2, 3, 1).contiguous().reshape(-1)
                all_da3_depths.append(da3_flat)
            
            if has_low_conf_mask:
                # low_conf_mask shape: (n, v, h, w)
                conf_flat = low_conf_mask[n_].view(-1, 1, self.height, self.width)
                conf_flat = conf_flat.permute(0, 2, 3, 1).contiguous().reshape(-1)
                all_low_conf_mask.append(conf_flat)

        all_rays = torch.stack(all_rays)
        if images is not None:
            all_rgb_gt = torch.stack(all_rgb_gt)
        else:
            all_rgb_gt = None

        if sample_segs:
            all_segs_gt = torch.stack(all_segs_gt)
            
        if has_da3_depths:
            all_da3_depths = torch.stack(all_da3_depths)
        else:
            all_da3_depths = None
        
        if has_low_conf_mask:
            all_low_conf_mask = torch.stack(all_low_conf_mask)
        else:
            all_low_conf_mask = None

        if sample_segs:
            # the None accounts for the patch_to_image
            return all_rays, all_rgb_gt, all_segs_gt, None, all_da3_depths, all_low_conf_mask
        else:
            return all_rays, all_rgb_gt, all_da3_depths, all_low_conf_mask

    def reconstruct(self, render_dict, channels=None, reconstruct_segmentation=False):
        coarse = render_dict["coarse"]
        fine = render_dict["fine"]

        if channels is None:
            channels = self.channels

        if reconstruct_segmentation:
            c_segmentation = coarse["segs"]
            # c_segmentation_raw = coarse["segs_raw"]
            n_classes = c_segmentation.shape[-1]
            # n_samples = c_segmentation_raw.shape[-2]

        c_rgb = coarse["rgb"]  # n, n_pts, v * 3
        c_weights = coarse["weights"]
        c_depth = coarse["depth"]
        c_invalid = coarse["invalid"]

        f_rgb = fine["rgb"]  # n, n_pts, v * 3
        f_weights = fine["weights"]
        f_depth = fine["depth"]
        f_invalid = fine["invalid"]

        n, n_pts, v_c = c_rgb.shape
        v_in = n_pts // (self.height * self.width)
        v_render = v_c // channels
        c_n_smps = c_weights.shape[-1]
        f_n_smps = f_weights.shape[-1]
        # (This can be a different v from the sample method)

        if reconstruct_segmentation:
            coarse["segs"] = c_segmentation.view(n, v_in, self.height, self.width, n_classes)
            # coarse["segs_raw"] = c_segmentation_raw.view(n, v_in, self.height, self.width, n_samples, n_classes)

        coarse["rgb"] = c_rgb.view(n, v_in, self.height, self.width, v_render, channels)
        coarse["weights"] = c_weights.view(n, v_in, self.height, self.width, c_n_smps)
        coarse["depth"] = c_depth.view(n, v_in, self.height, self.width)
        coarse["invalid"] = c_invalid.view(n, v_in, self.height, self.width, c_n_smps, v_render)

        fine["rgb"] = f_rgb.view(n, v_in, self.height, self.width, v_render, channels)
        fine["weights"] = f_weights.view(n, v_in, self.height, self.width, f_n_smps)
        fine["depth"] = f_depth.view(n, v_in, self.height, self.width)
        fine["invalid"] = f_invalid.view(n, v_in, self.height, self.width, f_n_smps, v_render)

        if "alphas" in coarse:
            c_alphas = coarse["alphas"]
            f_alphas = fine["alphas"]
            coarse["alphas"] = c_alphas.view(n, v_in, self.height, self.width, c_n_smps)
            fine["alphas"] = f_alphas.view(n, v_in, self.height, self.width, f_n_smps)

        if "z_samps" in coarse:
            c_z_samps = coarse["z_samps"]
            f_z_samps = fine["z_samps"]
            coarse["z_samps"] = c_z_samps.view(n, v_in, self.height, self.width, c_n_smps)
            fine["z_samps"] = f_z_samps.view(n, v_in, self.height, self.width, f_n_smps)

        if "rgb_samps" in coarse:
            c_rgb_samps = coarse["rgb_samps"]
            f_rgb_samps = fine["rgb_samps"]
            coarse["rgb_samps"] = c_rgb_samps.view(n, v_in, self.height, self.width, c_n_smps, v_render, channels)
            fine["rgb_samps"] = f_rgb_samps.view(n, v_in, self.height, self.width, f_n_smps, v_render, channels)

        render_dict["coarse"] = coarse
        render_dict["fine"] = fine

        if "rgb_gt" in render_dict:
            rgb_gt = render_dict["rgb_gt"]
            render_dict["rgb_gt"] = rgb_gt.view(n, v_in, self.height, self.width, channels)

        return render_dict
