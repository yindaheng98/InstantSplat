from typing import List, Tuple

import torch

from instantsplat.initializer import InitializedPointCloud, InitializingCamera
from .abc import DepthInitializerWrapper
from .utils import fov2focal, count, get_min_depth


class PointCloudCloudAsDepthInitializerWrapper(DepthInitializerWrapper):
    #!Duplicated:
    # Due to the sparsity of the point cloud,
    # this method may cause the depth of some pixels to correspond to occluded surfaces
    # rather than the correct foreground surfaces.

    def pcd2depth(self, pointcloud: InitializedPointCloud, camera: InitializingCamera) -> torch.Tensor:
        xyz = pointcloud.points
        fx, fy = fov2focal(camera.FoVx, camera.image_width), fov2focal(camera.FoVy, camera.image_height)
        cx, cy = camera.image_width / 2, camera.image_height / 2
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=xyz.device)
        R = camera.R.to(dtype=xyz.dtype)
        T = camera.T.to(dtype=xyz.dtype)
        uvd = (K @ ((R @ xyz.T).T + T.unsqueeze(0)).T).T
        uv, d = (uvd[:, 0:2] / uvd[:, 2:3]).round().long(), uvd[:, 2]
        valid_mask = (0 <= uv[:, 0:2]).all(-1) & (uv[:, 0] < camera.image_width) & (uv[:, 1] < camera.image_height)
        uv, d = uv[valid_mask, :], d[valid_mask]
        d_count = count(uv, camera.image_height, camera.image_width)
        d_min = get_min_depth(uv, d, camera.image_height, camera.image_width)
        d_idx = d_count > 0 & (d_min > 1e-6)
        invd = 1 / d_min[d_idx]
        invd = torch.zeros_like(d_count, dtype=invd.dtype)
        invd[d_idx] = 1 / d_min[d_idx]
        return invd, d_idx.float()

    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute depth and auto scale according to the point cloud and cameras."""
        return [self.pcd2depth(pointcloud, camera) for camera in cameras]
