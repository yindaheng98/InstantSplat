import numpy as np
import torch
import torch.nn.functional as F

from ..depth.abc import save_depth


def normalize_conf_for_save(conf: torch.Tensor, threshold: float) -> torch.Tensor:
    threshold = max(float(threshold), 1e-8)
    return conf.clamp(min=0.0, max=threshold) / threshold


def crop_and_resize_square_output(
    tensor: torch.Tensor,
    original_coord: torch.Tensor,
    src_resolution: int,
    dst_resolution: int,
    original_height: int,
    original_width: int,
    mode: str,
) -> torch.Tensor:
    scale = float(dst_resolution) / float(src_resolution)
    x1, y1, x2, y2 = original_coord[:4].tolist()
    x1 = max(0, min(dst_resolution, int(np.floor(x1 * scale))))
    y1 = max(0, min(dst_resolution, int(np.floor(y1 * scale))))
    x2 = max(0, min(dst_resolution, int(np.ceil(x2 * scale))))
    y2 = max(0, min(dst_resolution, int(np.ceil(y2 * scale))))

    tensor = tensor[y1:y2, x1:x2]
    if tensor.ndim != 2:
        raise ValueError(f"Unsupported tensor shape for resize: {tuple(tensor.shape)}")
    tensor = tensor[None, None].float()

    if mode == "nearest":
        tensor = F.interpolate(tensor, size=(original_height, original_width), mode=mode)
    else:
        tensor = F.interpolate(
            tensor,
            size=(original_height, original_width),
            mode=mode,
            align_corners=False,
        )
    return tensor[0, 0]


def save_vggt_depth(
    image_path: str,
    depth: torch.Tensor,
    conf: torch.Tensor,
    original_coord: torch.Tensor,
    src_resolution: int,
    dst_resolution: int,
    original_height: int,
    original_width: int,
    conf_threshold: float,
) -> str:
    original_depth = crop_and_resize_square_output(
        depth,
        original_coord,
        src_resolution,
        dst_resolution,
        original_height,
        original_width,
        mode="bilinear",
    )
    original_conf = crop_and_resize_square_output(
        conf,
        original_coord,
        src_resolution,
        dst_resolution,
        original_height,
        original_width,
        mode="bilinear",
    )
    save_mask = normalize_conf_for_save(original_conf, conf_threshold)
    return save_depth(image_path=image_path, depth=original_depth, mask=save_mask)
