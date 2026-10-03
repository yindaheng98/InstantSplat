import math
from typing import List, Tuple

import numpy as np
import torch

from instantsplat.initializer.abc import InitializingCamera


# Copy from vggt.utils
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def centers_from_w2c(extrinsic: np.ndarray) -> torch.Tensor:
    """Camera centers from world-to-camera matrices of shape (N, 3, 4) or (N, 4, 4)."""
    rotation = torch.as_tensor(np.asarray(extrinsic)[:, :3, :3], dtype=torch.float64)
    translation = torch.as_tensor(np.asarray(extrinsic)[:, :3, 3], dtype=torch.float64).reshape(-1, 3)
    return torch.bmm(-rotation.transpose(1, 2), translation.unsqueeze(-1)).squeeze(-1)


def cameras_to_w2c_intrinsics(cameras: List[InitializingCamera]) -> Tuple[np.ndarray, np.ndarray]:
    """World-to-camera (N, 3, 4) and pinhole K (N, 3, 3) in each camera's own pixels.

    Principal point is the image center, the same assumption as getK from FoV.
    """
    extrinsic = np.zeros((len(cameras), 3, 4), dtype=np.float64)
    intrinsic = np.repeat(np.eye(3, dtype=np.float64)[None], len(cameras), axis=0)
    for i, camera in enumerate(cameras):
        extrinsic[i, :3, :3] = camera.R.detach().cpu().numpy()
        extrinsic[i, :3, 3] = camera.T.detach().reshape(3).cpu().numpy()
        intrinsic[i, 0, 0] = fov2focal(float(camera.FoVx), camera.image_width)
        intrinsic[i, 1, 1] = fov2focal(float(camera.FoVy), camera.image_height)
        intrinsic[i, 0, 2] = camera.image_width / 2.0
        intrinsic[i, 1, 2] = camera.image_height / 2.0
    return extrinsic, intrinsic
