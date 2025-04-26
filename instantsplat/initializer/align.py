import copy
import numpy as np
import open3d as o3d
from typing import List
import torch
from .abc import AbstractInitializer, InitializingCamera, InitializedPointCloud


def global_registration_by_cameras(reference_points: torch.Tensor, reference_cameras: List[InitializingCamera], cameras: List[InitializingCamera]) -> torch.Tensor:
    reference_cameras = sorted(reference_cameras, key=lambda camera: camera.image_path)
    cameras = sorted(cameras, key=lambda camera: camera.image_path)
    R_ref = torch.stack([camera.R for camera in reference_cameras])
    T_ref = torch.stack([camera.T for camera in reference_cameras])
    R = torch.stack([camera.R for camera in cameras]).to(R_ref.dtype)
    T = torch.stack([camera.T for camera in cameras]).to(T_ref.dtype)
    R_tran = torch.bmm(R_ref.transpose(1, 2), R).median(0).values
    dist_T = (T.unsqueeze(0) - T.unsqueeze(1)).norm(dim=-1, p=2)
    dist_T_ref = (T_ref.unsqueeze(0) - T_ref.unsqueeze(1)).norm(dim=-1, p=2)
    scales = dist_T_ref / dist_T
    scale = scales[~scales.isnan()].median()
    T_tran = (T_ref - T @ R_tran.T * scale).median(0).values
    return (reference_points @ R_tran.T.to(reference_points.dtype)) * scale - T_tran


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

    def __call__(self, image_path_list: List[str]):
        pointcloud, cameras = self.initializers[0](image_path_list)
        for initializer in self.initializers[1:]:
            pcd, cams = initializer(image_path_list)
            points = global_registration_by_cameras(pcd.points, cameras, cams)
            points = registration_by_ICP(pointcloud.points, points)
            pointcloud = pointcloud._replace(
                points=torch.cat((pointcloud.points, points)),
                colors=torch.cat((pointcloud.colors, pcd.colors)),
            )
        return pointcloud, cameras
