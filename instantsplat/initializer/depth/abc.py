import abc
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tifffile
import torch
import tqdm

from instantsplat.initializer import AbstractInitializer, InitializedPointCloud, InitializingCamera


def save_depth(folder: str, image_name: Path, depth: torch.Tensor, mask: torch.Tensor = None) -> Tuple[str, str]:
    image_name = Path(image_name)
    name = image_name.with_name(image_name.name + ".tiff")
    depth_path = Path(folder) / "depths" / name
    mask_path = Path(folder) / "depth_masks" / name
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    if mask is None:
        mask = torch.ones_like(depth, dtype=depth.dtype)
    depth = depth.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    tifffile.imwrite(depth_path, depth)
    depth_scaled = np.repeat(((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)[..., np.newaxis], 3, axis=-1)
    cv2.imwrite(depth_path.with_suffix(".png"), depth_scaled)
    tifffile.imwrite(mask_path, mask)
    return str(depth_path), str(mask_path)


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

    def __call__(self, image_path_list: List[str], destination: str) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        pointcloud, cameras = self.base_initializer(image_path_list, destination)
        depths = self.compute_depths(pointcloud, cameras)
        cameras_with_depth = []
        for camera, (depth, mask) in zip(cameras, tqdm.tqdm(depths, desc="Saving Depths")):
            if depth is None:
                cameras_with_depth.append(camera)
                continue
            image_name = Path(camera.image_path).relative_to(Path(destination) / "images")
            depth_path, depth_mask_path = save_depth(destination, image_name, depth, mask)
            cameras_with_depth.append(camera._replace(depth_path=depth_path, depth_mask_path=depth_mask_path))
        return pointcloud, cameras_with_depth
