import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .save_depth import save_vggt_depth
from .utils import focal2fov

RESOLUTION = 518


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
        img_load_resolution: int = 1024,
        conf_thres_value: float = 5.0,
        save_depth: bool = True,
        scene_scale: float = 1.0,
    ):
        self.img_load_resolution = img_load_resolution
        self.conf_thres_value = conf_thres_value
        self.save_depth = save_depth
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

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L132-L134
        images, original_coords = load_and_preprocess_images_square(image_path_list, self.img_load_resolution)
        images = images.to(device)
        original_coords = original_coords.to(device)  # (N, 6): [x1, y1, x2, y2, orig_width, orig_height]

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L65-L90
        batch = images
        if images.shape[-2:] != (RESOLUTION, RESOLUTION):
            batch = F.interpolate(images, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)
        batch = batch.unsqueeze(0)
        device = batch.device
        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L107
        dtype = (
            torch.bfloat16
            if device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, ps_idx = self.model.aggregator(batch)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, batch, ps_idx)

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (N, H, W, 3)
        torch.cuda.empty_cache()

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L203-L218
        points_rgb = batch.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 3) [0, 1]

        conf_mask = depth_conf >= self.conf_thres_value
        valid_area_mask = build_valid_image_area_mask(
            original_coords.cpu().numpy(),
            src_resolution=self.img_load_resolution,
            dst_resolution=RESOLUTION,
        )
        conf_mask = np.logical_and(conf_mask, valid_area_mask)
        torch.cuda.empty_cache()

        cameras = []
        for i in range(len(image_path_list)):
            orig_w = float(original_coords[i, 4].item())
            orig_h = float(original_coords[i, 5].item())
            resize_ratio = max(orig_w, orig_h) / RESOLUTION

            fx_orig = intrinsic[i][0, 0] * resize_ratio
            fy_orig = intrinsic[i][1, 1] * resize_ratio

            saved_depth_path = None
            if self.save_depth:
                saved_depth_path = save_vggt_depth(
                    image_path=image_path_list[i],
                    depth=torch.from_numpy(depth_map[i]).squeeze(-1),
                    conf=torch.from_numpy(depth_conf[i]),
                    original_coord=original_coords[i],
                    src_resolution=self.img_load_resolution,
                    dst_resolution=RESOLUTION,
                    original_height=int(orig_h),
                    original_width=int(orig_w),
                    conf_threshold=self.conf_thres_value,
                )

            cameras.append(
                InitializingCamera(
                    image_width=int(orig_w),
                    image_height=int(orig_h),
                    FoVx=focal2fov(fx_orig, orig_w),
                    FoVy=focal2fov(fy_orig, orig_h),
                    R=torch.from_numpy(extrinsic[i][:3, :3]).float().to(device),
                    T=torch.from_numpy(extrinsic[i][:3, 3]).float().to(device) * self.scene_scale,
                    image_path=image_path_list[i],
                    depth_path=saved_depth_path,
                )
            )

        point_cloud = InitializedPointCloud(
            points=torch.from_numpy(points_3d[conf_mask]).float().to(device) * self.scene_scale,
            colors=torch.from_numpy(points_rgb[conf_mask]).float().to(device),
        )

        return point_cloud, cameras
