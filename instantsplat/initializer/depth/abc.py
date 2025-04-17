import abc
from typing import List, Tuple

import torch

from instantsplat.initializer import AbstractInitializer, InitializedPointCloud, InitializingCamera


class DepthInitializerWrapper(AbstractInitializer):
    def __init__(self, base_initializer: AbstractInitializer):
        self.base_initializer = base_initializer

    def to(self, device: torch.device) -> 'AbstractInitializer':
        self.base_initializer = self.base_initializer.to(device)
        return self

    @abc.abstractmethod
    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[str]:
        """Compute depth for the given point cloud and cameras. Save them to disk and return the paths."""
        pass

    def __call__(self, image_path_list: List[str]) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        pointcloud, cameras = self.base_initializer.__call__(image_path_list)
        return pointcloud, [camera._replace(depth_path=depth_path) for camera, depth_path in zip(cameras, self.compute_depths(pointcloud, cameras))]
