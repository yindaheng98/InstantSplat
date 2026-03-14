import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class VGGTInitializer(AbstractInitializer):
    def __init__(
        self,
        model_url: str = "checkpoints/vggt_1B_commercial.pt",
        vggt_fixed_resolution: int = 518,
        img_load_resolution: int = 1024,
        conf_threshold: float = 5.0,
        max_points: int = 100000,
        scene_scale: float = 1.0,
    ):
        self.vggt_fixed_resolution = vggt_fixed_resolution
        self.img_load_resolution = img_load_resolution
        self.conf_threshold = conf_threshold
        self.max_points = max_points
        self.scene_scale = scene_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L113-L118
        self.model = VGGT()
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
        self.model.eval()
        self.to(self.device)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self
