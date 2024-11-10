import os
import math
from typing import List
import torch
from gaussian_splatting.camera import build_camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.utils import matrix_to_quaternion

from .abc import InitializingCamera


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


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

    def save_colmap_cameras(self, path):
        camera_file = os.path.join(path, "cameras.txt")
        images_file = os.path.join(path, "images.txt")
        with open(camera_file, 'w') as camera_f, open(images_file, 'w') as images_f:
            for i, camera in enumerate(self.initializing_cameras, 1):
                width, height = camera.image_width, camera.image_height
                fx, fy = fov2focal(camera.FoVx, width), fov2focal(camera.FoVy, height)
                camera_f.write(f"{i} PINHOLE {width} {height} {fx} {fy} {width/2} {height/2}\n")
                q = matrix_to_quaternion(camera.R)
                t = camera.T
                image_path = os.path.basename(camera.image_path)
                images_f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {image_path}\n")
                images_f.write(f"\n")


def TrainableInitializedCameraDataset(cameras: List[InitializingCamera], exposures: List[torch.Tensor] = []):
    return TrainableCameraDataset(InitializedCameraDataset(cameras), exposures)
