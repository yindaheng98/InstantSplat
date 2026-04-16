import math
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from mapanything.utils.image import load_images

from instantsplat.initializer.depth import save_depth


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def recover_original_intrinsics(intrinsics, original_width, original_height, target_width, target_height):
    resize_scale = max(target_width / original_width, target_height / original_height) + 1e-8
    resized_width = math.floor(original_width * resize_scale)
    resized_height = math.floor(original_height * resize_scale)
    crop_left = (resized_width - target_width) // 2
    crop_top = (resized_height - target_height) // 2

    original_intrinsics = intrinsics.clone()
    original_intrinsics[0, 2] += crop_left
    original_intrinsics[1, 2] += crop_top
    original_intrinsics[0, :3] /= resize_scale
    original_intrinsics[1, :3] /= resize_scale
    return original_intrinsics


def interpolate_dense_output(tensor: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)[None]
    else:
        raise ValueError(f"Unsupported tensor shape for interpolation: {tuple(tensor.shape)}")

    if mode == "nearest":
        tensor = F.interpolate(tensor.float(), size=(height, width), mode=mode)
    else:
        tensor = F.interpolate(tensor.float(), size=(height, width), mode=mode, align_corners=False)

    tensor = tensor[0]
    if tensor.shape[0] == 1:
        return tensor[0]
    return tensor.permute(1, 2, 0)


def save_resized_depth(
    image_path: str,
    depth: torch.Tensor,
    mask: torch.Tensor,
    conf: torch.Tensor,
    original_height: int,
    original_width: int,
    save_conf_threshold: float,
) -> str:
    original_depth = interpolate_dense_output(depth, original_height, original_width, mode="bilinear")

    save_mask = torch.ones_like(original_depth, dtype=original_depth.dtype)
    if mask is not None:
        save_mask = interpolate_dense_output(mask.float(), original_height, original_width, mode="nearest") > 0.5
        save_mask = save_mask.float()

    if conf is not None:
        original_conf = interpolate_dense_output(conf, original_height, original_width, mode="bilinear")
        save_mask = save_mask * original_conf.clamp(min=0.0, max=save_conf_threshold) / max(float(save_conf_threshold), 1e-8)

    return save_depth(image_path=image_path, depth=original_depth, mask=save_mask)


def load_views(image_path_list: List[str], device: torch.device, *args, **kwargs):
    views = load_images(image_path_list, *args, **kwargs)
    original_sizes = []

    for view, image_path in zip(views, image_path_list):
        for key, value in list(view.items()):
            if torch.is_tensor(value):
                view[key] = value.to(device)

        with Image.open(image_path) as image:
            original_sizes.append(ImageOps.exif_transpose(image).size)

    return views, original_sizes
