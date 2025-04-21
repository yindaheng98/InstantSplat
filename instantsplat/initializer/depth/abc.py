import abc
import os
from typing import List, Tuple

import cv2
import numpy as np
import tifffile
import torch
import tqdm

from instantsplat.initializer import AbstractInitializer, InitializedPointCloud, InitializingCamera


class DepthInitializerWrapper(AbstractInitializer):
    def __init__(self, base_initializer: AbstractInitializer):
        self.base_initializer = base_initializer

    def to(self, device: torch.device) -> 'AbstractInitializer':
        self.base_initializer = self.base_initializer.to(device)
        return self

    @abc.abstractmethod
    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute depth and mask for the given point cloud and cameras."""
        raise NotImplementedError("Subclasses should implement this method.")

    def depth_path(self, image_path: str) -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(image_path)),
            "depths",
            os.path.splitext(os.path.basename(image_path))[0]+'.tiff'
        )

    def __call__(self, image_path_list: List[str]) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        pointcloud, cameras = self.base_initializer.__call__(image_path_list)
        depths = self.compute_depths(pointcloud, cameras)
        cameras_with_depth = []
        for i, camera in enumerate(tqdm.tqdm(cameras, desc="Saving Depths")):
            depth_path = self.depth_path(camera.image_path)
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            depth, mask = depths[i]
            if depth is None:
                continue
            if mask is None:
                mask = torch.ones_like(depth, dtype=depth.dtype)
            depth = depth.cpu().numpy()
            mask = mask.cpu().numpy()
            if depth_path.endswith('.tiff'):
                tifffile.imwrite(depth_path, depth)
                depth_scaled = np.repeat(((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)[..., np.newaxis], 3, axis=-1)
                cv2.imwrite(os.path.splitext(depth_path)[0] + '.png', depth_scaled)
                tifffile.imwrite(os.path.splitext(depth_path)[0] + '_mask.tiff', mask)
            else:
                cv2.imwrite(depth_path, depth)
                cv2.imwrite(os.path.splitext(depth_path)[0] + '_mask' + os.path.splitext(depth_path)[1], mask)
            cameras_with_depth.append(camera._replace(depth_path=depth_path))
        return pointcloud, cameras_with_depth
