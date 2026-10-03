import numpy as np
import torch
from typing import List, Tuple

from vggt.utils.geometry import unproject_depth_map_to_point_map

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from ..align import camera_centers
from .utils import cameras_to_w2c_intrinsics, centers_from_w2c
from .vggt import RESOLUTION, VGGTInitializer


def median_baseline_scale(
    target_centers: torch.Tensor,
    source_centers: torch.Tensor,
    min_baseline_ratio: float = 1e-3,
) -> float:
    """One scale s with ||C_target_i - C_target_j|| ≈ s ||C_source_i - C_source_j||.

    Pairs whose source baseline is near zero are dropped. Their ratio is unstable.
    """
    target_centers = torch.as_tensor(target_centers, dtype=torch.float64).detach().cpu().reshape(-1, 3)
    source_centers = torch.as_tensor(source_centers, dtype=torch.float64).detach().cpu().reshape(-1, 3)
    source_distance = torch.cdist(source_centers, source_centers)
    target_distance = torch.cdist(target_centers, target_centers)
    row, col = torch.triu_indices(source_centers.shape[0], source_centers.shape[0], offset=1)
    source_baseline = source_distance[row, col]
    target_baseline = target_distance[row, col]
    keep = source_baseline > source_baseline.max() * min_baseline_ratio
    return float(torch.median(target_baseline[keep] / source_baseline[keep]))


def intrinsics_on_vggt_depth_grid(intrinsics: np.ndarray, original_coords: torch.Tensor) -> np.ndarray:
    """Map original-pixel intrinsics onto the padded square depth grid.

    Depth values do not change with this resize. Only fx, fy, cx, cy do.
    """
    coords = original_coords.detach().cpu().numpy()
    depth_intrinsics = np.array(intrinsics, dtype=np.float64, copy=True)
    for i in range(depth_intrinsics.shape[0]):
        width = int(coords[i, 4])
        height = int(coords[i, 5])
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        scale = RESOLUTION / max_dim
        depth_intrinsics[i, 0, 0] *= scale
        depth_intrinsics[i, 1, 1] *= scale
        depth_intrinsics[i, 0, 2] = (intrinsics[i, 0, 2] + left) * scale
        depth_intrinsics[i, 1, 2] = (intrinsics[i, 1, 2] + top) * scale
    return depth_intrinsics


class VGGTAlign2Initializer(VGGTInitializer):
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
        original_coords, batch, extrinsic, intrinsic, depth_map, depth_conf, _points_3d = self.predict(image_paths)
        extrinsic, intrinsic, depth_map, points_3d = self.align_predictions(
            another_cameras, original_coords, extrinsic, depth_map
        )
        point_cloud, cameras = self.postprocess(
            image_paths,
            destination,
            original_coords,
            batch,
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

    def align_predictions(self, cameras, original_coords: torch.Tensor, extrinsic, depth_map):
        scale = median_baseline_scale(
            camera_centers(cameras),
            centers_from_w2c(extrinsic),
            min_baseline_ratio=self.min_baseline_ratio,
        )
        extrinsic, intrinsic = cameras_to_w2c_intrinsics(cameras)
        depth_map = np.asarray(depth_map) * scale
        points_3d = unproject_depth_map_to_point_map(
            depth_map, extrinsic, intrinsics_on_vggt_depth_grid(intrinsic, original_coords)
        )
        return extrinsic, intrinsic, depth_map, points_3d
