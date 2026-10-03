import copy
import numpy as np
import open3d as o3d
import roma
from typing import List
import torch
from .abc import AbstractInitializer, InitializingCamera, InitializedPointCloud


def camera_centers(cameras: List[InitializingCamera]) -> torch.Tensor:
    """World-to-camera (R, T) gives center C = -R^T T."""
    rotation = torch.stack([camera.R for camera in cameras])
    translation = torch.stack([camera.T.reshape(3) for camera in cameras])
    return torch.bmm(-rotation.transpose(1, 2), translation.unsqueeze(-1)).squeeze(-1)


def global_registration_by_cameras(reference_points: torch.Tensor, reference_cameras: List[InitializingCamera], cameras: List[InitializingCamera]) -> torch.Tensor:
    reference_cameras = sorted(reference_cameras, key=lambda camera: camera.image_path)
    cameras = sorted(cameras, key=lambda camera: camera.image_path)
    source = camera_centers(cameras).to(device=reference_points.device, dtype=reference_points.dtype)
    target = camera_centers(reference_cameras).to(device=reference_points.device, dtype=reference_points.dtype)
    rotation, translation, scale = roma.rigid_points_registration(source, target, compute_scaling=True)
    return scale * reference_points @ rotation.T + translation


def registration_by_ICP(reference_points: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(reference_points.cpu().numpy())
    threshold = 0.02
    trans_init = torch.eye(4).cpu().numpy()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    source_transformed = copy.deepcopy(source).transform(reg_p2p.transformation)
    points_transformed = torch.from_numpy(np.asarray(source_transformed.points)).to(reference_points.device).to(reference_points.dtype)
    return points_transformed


class AlignInitializer(AbstractInitializer):
    def __init__(self, *initializers: AbstractInitializer):
        self.initializers = initializers

    def to(self, device):
        self.initializers = [initializer.to(device) for initializer in self.initializers]
        return self

    def __call__(self, image_path_list: List[str], destination: str):
        pointcloud, cameras = self.initializers[0](image_path_list, destination)
        for initializer in self.initializers[1:]:
            pcd, cams = initializer(image_path_list, destination)
            points = global_registration_by_cameras(pcd.points, cameras, cams)
            points = registration_by_ICP(pointcloud.points, points)
            pointcloud = pointcloud._replace(
                points=torch.cat((pointcloud.points, points)),
                colors=torch.cat((pointcloud.colors, pcd.colors)),
            )
        return pointcloud, cameras
