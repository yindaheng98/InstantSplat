
import os
from typing import List, Tuple

import cv2
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


class DepthAnythingV2InitializerWrapper(DepthInitializerWrapper):
    def __init__(
            self,
            base_initializer: AbstractInitializer,
            input_size=518,
            encoder='vitl',
            checkpoints_folder='checkpoints',
            device="cuda"):
        super().__init__(base_initializer)
        self.input_size = input_size
        depth_anything = DepthAnythingV2(**model_configs[encoder])
        depth_anything.load_state_dict(torch.load(os.path.join(checkpoints_folder, f'depth_anything_v2_{encoder}.pth'), map_location='cpu'))
        self.depth_anything = depth_anything.eval()
        self.to(device)

    def to(self, device):
        self.device = device
        self.depth_anything = self.depth_anything.to(device).eval()
        return super().to(device)

    def compute_depth(self, image_path: str) -> torch.Tensor:
        depth_anything = self.depth_anything
        raw_image = cv2.imread(image_path)
        return torch.tensor(depth_anything.infer_image(raw_image, self.input_size), device=self.device), None

    def compute_depths(self, pointcloud: InitializedPointCloud, cameras: List[InitializingCamera]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.compute_depth(camera.image_path) for camera in tqdm.tqdm(cameras, desc="Computing Depths")]
