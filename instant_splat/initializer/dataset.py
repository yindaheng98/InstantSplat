from typing import List
import torch
from gaussian_splatting.camera import build_camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset

from .abc import InitializingCamera


class InitializedCameraDataset(CameraDataset):
    def __init__(self, cameras: List[InitializingCamera]):
        self.initializing_cameras = cameras
        self.cameras = [build_camera(**camera._asdict()) for camera in self.initializing_cameras]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def to(self, device):
        self.cameras = [build_camera(**camera._asdict(), device=device) for camera in self.initializing_cameras]
        return self


def TrainableInitializedCameraDataset(cameras: List[InitializingCamera], exposures: List[torch.Tensor] = []):
    return TrainableCameraDataset(InitializedCameraDataset(cameras), exposures)
