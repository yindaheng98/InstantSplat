import abc
import math
from typing import List

import torch

from instantsplat.initializer import InitializedPointCloud, InitializingCamera
from .abc import DepthInitializerWrapper


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


class AutoScaleDepthInitializerWrapper(DepthInitializerWrapper):

    @abc.abstractmethod
    def compute_raw_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[torch.Tensor]:
        """Compute raw depth for the given point cloud and cameras."""
        raise NotImplementedError("Subclasses should implement this method.")

    def autoscale_depth(self, raw_depth: torch.Tensor, pointcloud: InitializedPointCloud, camera: InitializingCamera) -> torch.Tensor:
        xyz = pointcloud.points
        fx, fy = fov2focal(camera.FoVx, camera.image_width), fov2focal(camera.FoVy, camera.image_height)
        cx, cy = camera.image_width / 2, camera.image_height / 2
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=raw_depth.device)
        R = camera.R.to(dtype=xyz.dtype)
        T = camera.T.to(dtype=xyz.dtype)
        uvd = (K @ ((R @ xyz.T).T + T.unsqueeze(0)).T).T
        uv, d = uvd[:, 0:2] / uvd[:, 2:3], uvd[:, 2:3]
        valid_mask = (0 <= uv[:, 0:2]).all(-1) & (uv[:, 0] < camera.image_width) & (uv[:, 1] < camera.image_height)
        uv, d = uv[valid_mask, :], d[valid_mask, :]
        return raw_depth

    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[torch.Tensor]:
        """Compute depth and auto scale according to the point cloud and cameras."""
        raw_depths = self.compute_raw_depths(pointcloud, cameras)
        return [self.autoscale_depth(raw_depth, pointcloud, camera) for raw_depth, camera in zip(raw_depths, cameras)]
