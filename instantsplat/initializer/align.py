import os
from typing import List
import torch
from .abc import AbstractInitializer, InitializingCamera, InitializedPointCloud


def align_cameras(reference_points: torch.Tensor, reference_cameras: List[InitializingCamera], cameras: List[InitializingCamera]) -> torch.Tensor:
    R_ref = torch.stack([camera.R for camera in reference_cameras])
    T_ref = torch.stack([camera.T for camera in reference_cameras])
    R = torch.stack([camera.R for camera in cameras]).to(R_ref.dtype)
    T = torch.stack([camera.T for camera in cameras]).to(T_ref.dtype)
    R_tran = torch.bmm(R_ref.transpose(1, 2), R).mean(0)
    return reference_points @ R_tran.T.to(reference_points.dtype)


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
            points = align_cameras(pcd.points, cameras, cams)
            pointcloud = pointcloud._replace(
                points=torch.cat((pointcloud.points, points)),
                colors=torch.cat((pointcloud.colors, pcd.colors)),
            )
        return pointcloud, cameras
