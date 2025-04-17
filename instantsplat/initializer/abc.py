from abc import ABC, abstractmethod
from typing import NamedTuple, List, Tuple
import torch
import numpy as np
from plyfile import PlyData, PlyElement


class InitializingCamera(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor
    image_path: str
    depth_path: str = None


class InitializedPointCloud(NamedTuple):
    points: torch.Tensor
    colors: torch.Tensor


class InitializedPointCloud(InitializedPointCloud):
    def __new__(cls, points: torch.Tensor, colors: torch.Tensor):
        assert points.shape[0] == colors.shape[0]
        assert points.shape[1] == colors.shape[1] == 3
        return super().__new__(cls, points, colors)

    def save_ply(self, path):
        xyz = self.points.detach().cpu().numpy()
        rgb = (self.colors.detach() * 255.0).cpu().numpy()

        # Define the dtype for the structured array
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        normals = np.zeros_like(xyz)

        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(path)


class AbstractInitializer(ABC):

    @abstractmethod
    def to(self, device: torch.device) -> 'AbstractInitializer':
        return self

    @abstractmethod
    def __call__(self, image_path_list: List[str]) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        return InitializedPointCloud(points=torch.empty(0), colors=torch.empty(0)), []
