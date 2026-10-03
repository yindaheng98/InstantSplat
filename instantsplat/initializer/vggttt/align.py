import numpy as np
import torch
from typing import List, Tuple

from vggttt.nets.vggt.utils.geometry import unproject_depth_map_to_point_map

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from ..align import camera_centers
from ..vggt.align import median_baseline_scale
from ..vggt.utils import cameras_to_w2c_intrinsics, centers_from_w2c

from .vggttt import PATCH_SIZE, RESOLUTION, VGGTTTInitializer


def intrinsics_on_vggttt_depth_grid(intrinsics: np.ndarray, cameras, images) -> np.ndarray:
    """Map original-pixel intrinsics onto the crop-and-pad depth grid.

    Depth values do not change with this resize. Only fx, fy, cx, cy do.
    """
    depth_intrinsics = np.array(intrinsics, dtype=np.float64, copy=True)
    batch_height, batch_width = int(images.shape[-2]), int(images.shape[-1])
    for i, camera in enumerate(cameras):
        width = int(camera.image_width)
        height = int(camera.image_height)
        resized_height = round(height * (RESOLUTION / width) / PATCH_SIZE) * PATCH_SIZE
        scale_x = RESOLUTION / width
        scale_y = resized_height / height
        crop_top = (resized_height - RESOLUTION) // 2 if resized_height > RESOLUTION else 0
        content_height = min(resized_height, RESOLUTION)
        batch_top = (batch_height - content_height) // 2
        batch_left = (batch_width - RESOLUTION) // 2
        depth_intrinsics[i, 0, 0] *= scale_x
        depth_intrinsics[i, 1, 1] *= scale_y
        depth_intrinsics[i, 0, 2] = intrinsics[i, 0, 2] * scale_x + batch_left
        depth_intrinsics[i, 1, 2] = intrinsics[i, 1, 2] * scale_y - crop_top + batch_top
    return depth_intrinsics


class VGGTTTAlign2Initializer(VGGTTTInitializer):
    def __init__(
        self,
        another_initializer: AbstractInitializer,
        *args,
        convert_image_path=lambda image_path_list, destination: image_path_list,
        update_camera=False,
        scene_scale=1.0,
        min_baseline_ratio=1e-3,
        **kwargs,
    ):
        super().__init__(*args, scene_scale=scene_scale, **kwargs)
        self.another_initializer = another_initializer
        self.convert_image_path = convert_image_path
        self.update_camera = update_camera
        self.min_baseline_ratio = min_baseline_ratio

    def to(self, device):
        super().to(device)
        self.another_initializer = self.another_initializer.to(device)
        return self

    def __call__(
        self, image_path_list: List[str], destination: str
    ) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        another_point_cloud, another_cameras = self.another_initializer(
            self.convert_image_path(image_path_list, destination), destination
        )
        image_paths = [camera.image_path for camera in another_cameras]
        images, extrinsic, intrinsic, depth_map, depth_conf, _points_3d = self.predict(image_paths)
        torch.cuda.empty_cache()
        extrinsic, intrinsic, depth_map, points_3d = self.align_predictions(
            another_cameras, images, extrinsic, depth_map
        )
        point_cloud, cameras = self.postprocess(
            image_paths,
            destination,
            images,
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            points_3d,
        )
        another_points = another_point_cloud.points.to(device=point_cloud.points.device, dtype=point_cloud.points.dtype)
        another_colors = another_point_cloud.colors.to(device=point_cloud.colors.device, dtype=point_cloud.colors.dtype)
        return InitializedPointCloud(
            points=torch.concatenate((point_cloud.points, another_points * self.scene_scale)),
            colors=torch.concatenate((point_cloud.colors, another_colors)),
        ), (cameras if self.update_camera else another_cameras)

    def align_predictions(self, cameras, images, extrinsic, depth_map):
        scale = median_baseline_scale(
            camera_centers(cameras),
            centers_from_w2c(extrinsic),
            min_baseline_ratio=self.min_baseline_ratio,
        )
        extrinsic, intrinsic = cameras_to_w2c_intrinsics(cameras)
        depth_map = np.asarray(depth_map) * scale
        points_3d = unproject_depth_map_to_point_map(
            depth_map, extrinsic, intrinsics_on_vggttt_depth_grid(intrinsic, cameras, images)
        )
        return extrinsic, intrinsic, depth_map, points_3d
