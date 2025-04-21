import math

import torch


def fov2focal(fov: float, pixels: float):
    return pixels / (2 * math.tan(fov / 2))


def count(uv: torch.Tensor, height: int, width: int):
    """Count on each pixel on reference image: how many point projected to this pixel?"""
    index = (uv[..., 1] * width + uv[..., 0])
    src = torch.ones_like(index)
    counts = torch.zeros(height*width, dtype=src.dtype, device=src.device).scatter_add_(0, index, src)
    return counts.reshape(height, width)


def get_min_depth(uv: torch.Tensor, depth: torch.Tensor, height: int, width: int):
    """Count on each pixel: get min depth among all point projected to this pixel"""
    index = (uv[..., 1] * width + uv[..., 0])
    min_depth = torch.zeros(height*width, dtype=depth.dtype, device=depth.device).index_reduce_(0, index, depth, 'amin', include_self=False)
    return min_depth.reshape(height, width)
