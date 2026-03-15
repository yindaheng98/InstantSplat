import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .load_fn import load_and_preprocess_images_square


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L65-L90
def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def build_valid_image_area_mask(original_coords, src_resolution, dst_resolution):
    """Build a mask that removes padded area after square-resize."""
    n = int(original_coords.shape[0])
    scale = float(dst_resolution) / float(src_resolution)
    valid_mask = np.zeros((n, dst_resolution, dst_resolution), dtype=bool)

    for i in range(n):
        x1, y1, x2, y2 = original_coords[i, :4]
        x1 = int(np.floor(x1 * scale))
        y1 = int(np.floor(y1 * scale))
        x2 = int(np.ceil(x2 * scale))
        y2 = int(np.ceil(y2 * scale))

        x1 = max(0, min(dst_resolution, x1))
        y1 = max(0, min(dst_resolution, y1))
        x2 = max(0, min(dst_resolution, x2))
        y2 = max(0, min(dst_resolution, y2))

        if x2 > x1 and y2 > y1:
            valid_mask[i, y1:y2, x1:x2] = True

    return valid_mask


class VGGTInitializer(AbstractInitializer):
    def __init__(
        self,
        model_url: str = "checkpoints/vggt_1B_commercial.pt",
        vggt_fixed_resolution: int = 518,
        img_load_resolution: int = 1024,
        conf_thres_value: float = 5.0,
        max_points: int = 100000,
        scene_scale: float = 1.0,
    ):
        self.vggt_fixed_resolution = vggt_fixed_resolution
        self.img_load_resolution = img_load_resolution
        self.conf_thres_value = conf_thres_value
        self.max_points = max_points
        self.scene_scale = scene_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L113-L118
        self.model = VGGT()
        self.model.load_state_dict(torch.load(model_url))
        self.model.eval()
        self.model.to(self.device)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(
        self, image_path_list: List[str]
    ) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        device = self.device
        vggt_fixed_resolution = self.vggt_fixed_resolution

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L107
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L132-L134
        images, original_coords = load_and_preprocess_images_square(image_path_list, self.img_load_resolution)
        images = images.to(device)
        original_coords = original_coords.to(device)  # (N, 6): [x1, y1, x2, y2, orig_width, orig_height]

        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(self.model, images, dtype, vggt_fixed_resolution)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (N, H, W, 3)
        torch.cuda.empty_cache()

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L203-L218
        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = points_rgb.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 3) [0, 1]

        conf_mask = depth_conf >= self.conf_thres_value
        valid_area_mask = build_valid_image_area_mask(
            original_coords.cpu().numpy(),
            src_resolution=self.img_load_resolution,
            dst_resolution=vggt_fixed_resolution,
        )
        conf_mask = np.logical_and(conf_mask, valid_area_mask)
        conf_mask = randomly_limit_trues(conf_mask, self.max_points)
        torch.cuda.empty_cache()

        cameras = []
        for i in range(len(image_path_list)):
            orig_w = original_coords[i, 4]
            orig_h = original_coords[i, 5]
            resize_ratio = max(orig_w, orig_h) / vggt_fixed_resolution

            fx_orig = intrinsic[i][0, 0] * resize_ratio
            fy_orig = intrinsic[i][1, 1] * resize_ratio

            cameras.append(
                InitializingCamera(
                    image_width=int(orig_w),
                    image_height=int(orig_h),
                    FoVx=focal2fov(fx_orig, orig_w),
                    FoVy=focal2fov(fy_orig, orig_h),
                    R=torch.from_numpy(extrinsic[i][:3, :3]).float().to(device),
                    T=torch.from_numpy(extrinsic[i][:3, 3]).float().to(device) * self.scene_scale,
                    image_path=image_path_list[i],
                )
            )

        point_cloud = InitializedPointCloud(
            points=torch.from_numpy(points_3d[conf_mask]).float().to(device) * self.scene_scale,
            colors=torch.from_numpy(points_rgb[conf_mask]).float().to(device),
        )

        return point_cloud, cameras
