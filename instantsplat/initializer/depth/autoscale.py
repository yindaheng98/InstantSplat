from typing import List, Tuple

import torch

from instantsplat.initializer import InitializedPointCloud, InitializingCamera
from .abc import DepthInitializerWrapper
from .utils import fov2focal, count, get_min_depth


class AutoScaleDepthInitializerWrapper(DepthInitializerWrapper):
    def __init__(self, base_initializer_wrapper: DepthInitializerWrapper):
        super().__init__(base_initializer_wrapper.base_initializer)
        self.base_initializer_wrapper = base_initializer_wrapper

    def autoscale_depth(self, raw_invdepth: torch.Tensor, pointcloud: InitializedPointCloud, camera: InitializingCamera) -> torch.Tensor:
        xyz = pointcloud.points
        fx, fy = fov2focal(camera.FoVx, camera.image_width), fov2focal(camera.FoVy, camera.image_height)
        cx, cy = camera.image_width / 2, camera.image_height / 2
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=raw_invdepth.device)
        R = camera.R.to(dtype=xyz.dtype)
        T = camera.T.to(dtype=xyz.dtype)
        uvd = (K @ ((R @ xyz.T).T + T.unsqueeze(0)).T).T
        uv, d = (uvd[:, 0:2] / uvd[:, 2:3]).round().long(), uvd[:, 2]
        valid_mask = (0 <= uv[:, 0:2]).all(-1) & (uv[:, 0] < camera.image_width) & (uv[:, 1] < camera.image_height)
        uv, d = uv[valid_mask, :], d[valid_mask]
        d_count = count(uv, camera.image_height, camera.image_width)
        d_min = get_min_depth(uv, d, camera.image_height, camera.image_width)
        d_idx = d_count > 0 & (raw_invdepth > 1e-6) & (d_min > 1e-6)
        invd_raw = raw_invdepth[d_idx]
        invd_target = 1 / d_min[d_idx]
        center_raw = invd_raw.median()
        center_target = invd_target.median()
        extent_raw = (invd_raw.max() - center_raw).mean()
        extent_target = (invd_target.max() - center_target).mean()
        scale = extent_target / extent_raw
        shift = center_target - center_raw * scale
        return raw_invdepth * scale + shift

    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute depth and auto scale according to the point cloud and cameras."""
        raw_depths = self.base_initializer_wrapper.compute_depths(pointcloud, cameras)
        return [(self.autoscale_depth(raw_depth, pointcloud, camera), mask) for (raw_depth, mask), camera in zip(raw_depths, cameras)]
