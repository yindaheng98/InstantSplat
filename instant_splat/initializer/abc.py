from abc import ABC, abstractmethod
from typing import NamedTuple, List, Tuple
import torch


class InitializingCamera(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor
    image_path: str


class InitializedPointCloud(NamedTuple):
    points: torch.Tensor
    colors: torch.Tensor


class AbstractInitializer(ABC):

    @abstractmethod
    def __call__(self, image_path_list: List[str]) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        return InitializedPointCloud(points=torch.empty(0), colors=torch.empty(0)), []
