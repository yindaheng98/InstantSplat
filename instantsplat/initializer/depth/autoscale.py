import abc
from typing import List

import torch

from instantsplat.initializer import InitializedPointCloud, InitializingCamera
from .abc import DepthInitializerWrapper


class AutoScaleDepthInitializerWrapper(DepthInitializerWrapper):

    @abc.abstractmethod
    def compute_raw_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[torch.Tensor]:
        """Compute raw depth for the given point cloud and cameras."""
        raise NotImplementedError("Subclasses should implement this method.")

    def autoscale_depth(self, raw_depth: torch.Tensor, pointcloud: InitializedPointCloud, camera: InitializingCamera) -> torch.Tensor:
        pass

    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[torch.Tensor]:
        """Compute depth and auto scale according to the point cloud and cameras."""
        raw_depths = self.compute_raw_depths(pointcloud, cameras)
        return [self.autoscale_depth(raw_depth, pointcloud, camera) for raw_depth, camera in zip(raw_depths, cameras)]
