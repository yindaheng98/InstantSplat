from copy import deepcopy

import cv2
import numpy as np
import PIL.Image
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose

from .bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import torchvision.transforms as tvf
from dust3r.utils.image import _resize_pil_image


def load_images(image_paths, size: int | None = None):
    imgs = []
    for path in image_paths:
        image = exif_transpose(PIL.Image.open(path)).convert("RGB")
        width_before, height_before = image.size
        image = _resize_pil_image(image, size)
        width_after, height_after = image.size
        width_after = width_after // 16 * 16
        height_after = height_after // 16 * 16
        image = np.array(image)
        image = cv2.resize(
            image,
            (width_after, height_after),
            interpolation=cv2.INTER_LINEAR,
        )
        image = PIL.Image.fromarray(image)

        print(
            f" - adding {path} with resolution {width_before}x{height_before} --> "
            f"{width_after}x{height_after}"
        )
        image_norm = tvf.Compose(
            [
                tvf.ToTensor(),
                tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        imgs.append(
            dict(
                img=image_norm(image)[None],
                true_shape=np.int32([image.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )
    return imgs, [None] * len(imgs)


def prepare_input(image_paths, size, reset_interval):
    images, _ = load_images(image_paths, size=size)

    views = []
    for index, image in enumerate(images):
        view = {
            "img": image["img"],
            "ray_map": torch.full(
                (
                    image["img"].shape[0],
                    6,
                    image["img"].shape[-2],
                    image["img"].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": index,
            "instance": str(index),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((index + 1) % reset_interval == 0).unsqueeze(0),
        }
        views.append(view)
        if (index + 1) % reset_interval == 0:
            overlap_view = deepcopy(view)
            overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap_view)
    return views


def accumulate_reset_poses(pr_poses, reset_mask, matrix_cumprod):
    if not reset_mask.any():
        return pr_poses
    concatenated = torch.cat(pr_poses, 0)
    identity = torch.eye(4, device=concatenated.device)
    reset_poses = torch.where(
        reset_mask.unsqueeze(-1).unsqueeze(-1), concatenated, identity
    )
    cumulative_bases = matrix_cumprod(reset_poses)
    shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
    composed = torch.einsum("bij,bjk->bik", shifted_bases, concatenated)
    return list(composed.unsqueeze(1).unbind(0))


def export_outputs(outputs, image_paths, min_conf_thr):
    from dust3r.post_process import estimate_focal_knowing_depth

    from .camera import pose_encoding_to_camera
    from .geometry import geotrf, matrix_cumprod

    outputs["pred"] = list(outputs["pred"])
    outputs["views"] = list(outputs["views"])

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat(
        [torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0
    )

    outputs["pred"] = [
        pred for pred, masked in zip(outputs["pred"], shifted_reset_mask) if not masked
    ]
    outputs["views"] = [
        view for view, masked in zip(outputs["views"], shifted_reset_mask) if not masked
    ]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self = torch.cat(
        [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]], 0
    )
    conf_self = torch.cat([output["conf_self"].cpu() for output in outputs["pred"]], 0)
    colors = torch.cat(
        [
            0.5 * (view["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for view in outputs["views"]
        ],
        0,
    )

    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    pr_poses = accumulate_reset_poses(pr_poses, reset_mask, matrix_cumprod)

    transformed_points = []
    filtered_colors = []
    for pose, self_points, self_conf, color in zip(pr_poses, pts3ds_self, conf_self, colors):
        mask = self_conf > min_conf_thr
        world_points = geotrf(pose, self_points.unsqueeze(0)).squeeze(0)
        transformed_points.append(world_points[mask])
        filtered_colors.append(color[mask])

    if not transformed_points or sum(points.shape[0] for points in transformed_points) == 0:
        raise RuntimeError(
            f"TTT3R produced no points above confidence threshold {min_conf_thr}."
        )

    principal_points = torch.stack(
        [
            torch.tensor(
                [int(view["true_shape"][0][1]) // 2, int(view["true_shape"][0][0]) // 2],
                device=pts3ds_self.device,
            ).float()
            for view in outputs["views"]
        ],
        0,
    )
    focals = estimate_focal_knowing_depth(
        pts3ds_self, principal_points, focal_mode="weiszfeld"
    ).cpu()
    cam2worlds = torch.cat(pr_poses, 0)
    world2cams = torch.linalg.inv(cam2worlds)

    original_sizes = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            original_sizes.append(image.size)

    return {
        "points": torch.cat(transformed_points, 0).numpy().astype(np.float32),
        "colors": torch.cat(filtered_colors, 0).numpy().astype(np.float32),
        "world2cams": world2cams.numpy().astype(np.float32),
        "focals": focals.numpy().astype(np.float32),
        "principal_points": principal_points.cpu().numpy().astype(np.float32),
        "image_paths": np.asarray(image_paths, dtype=object),
        "image_widths": np.asarray([size[0] for size in original_sizes], dtype=np.int32),
        "image_heights": np.asarray([size[1] for size in original_sizes], dtype=np.int32),
    }
