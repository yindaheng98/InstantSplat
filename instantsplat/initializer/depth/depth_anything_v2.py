
import os
from typing import List

import cv2
import numpy as np
import torch
import tqdm

from depth_anything_v2.dpt import DepthAnythingV2
from instantsplat.initializer import AbstractInitializer, InitializedPointCloud, InitializingCamera

from .abc import DepthInitializerWrapper

# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/run.py
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def default_image_path_to_depth_path(image_path: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(image_path)),
        "depths",
        os.path.splitext(os.path.basename(image_path))[0]+'.png'
    )


class DepthAnythingV2InitializerWrapper(DepthInitializerWrapper):
    def __init__(
            self,
            base_initializer: AbstractInitializer,
            input_size=518,
            encoder='vitl',
            checkpoints_folder='checkpoints',
            image_path_to_depth_path=None):
        super().__init__(base_initializer)
        self.input_size = input_size
        depth_anything = DepthAnythingV2(**model_configs[encoder])
        depth_anything.load_state_dict(torch.load(os.path.join(checkpoints_folder, f'depth_anything_v2_{encoder}.pth'), map_location='cpu'))
        self.depth_anything = depth_anything.eval()
        self.image_path_to_depth_path = default_image_path_to_depth_path if image_path_to_depth_path is None else image_path_to_depth_path

    def to(self, device):
        self.depth_anything = self.depth_anything.to(device).eval()
        return super().to(device)

    def compute_depth(self, image_path: str) -> str:
        depth_anything = self.depth_anything
        raw_image = cv2.imread(image_path)
        depth = depth_anything.infer_image(raw_image, self.input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_path = self.image_path_to_depth_path(image_path)
        depth = depth.astype(np.uint8)
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        cv2.imwrite(depth_path, depth)
        return depth_path

    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[str]:
        return [self.compute_depth(camera.image_path) for camera in tqdm.tqdm(cameras, desc="Computing depth")]
