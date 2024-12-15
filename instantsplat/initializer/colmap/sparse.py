import os
import tempfile
import subprocess
import shutil
import torch

from gaussian_splatting.dataset.colmap import read_colmap_cameras, read_colmap_points3D
from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .load_cameras import load_colmap_cameras


def execute(cmd):
    proc = subprocess.Popen(cmd, shell=False)
    proc.communicate()
    return proc.returncode


class ColmapSparseInitializer(AbstractInitializer):
    def __init__(self,
                 destination: str,
                 colmap_executable: str = "colmap",
                 camera: str = "OPENCV",
                 single_camera_per_image: bool = True,
                 load_camera: str = None,
                 scene_scale: float = 1.0):
        self.colmap_executable = os.path.abspath(colmap_executable)
        self.camera = camera
        self.single_camera_per_image = "1" if single_camera_per_image else "0"
        self.load_camera = load_camera
        self.destination = destination
        self.scene_scale = scene_scale
        self.use_gpu = "1"

    def to(self, device):
        self.use_gpu = "0" if device == "cpu" else "1"
        return self

    def feature_extractor(args, folder):
        os.makedirs(os.path.join(folder, "distorted"))
        cmd = [
            args.colmap_executable, "feature_extractor",
            "--database_path", os.path.join(folder, "distorted", "database.db"),
            "--image_path", os.path.join(folder, "input"),
            "--ImageReader.camera_model", args.camera,
            "--SiftExtraction.use_gpu", args.use_gpu,
            "--ImageReader.single_camera_per_image", args.single_camera_per_image,
        ]
        return execute(cmd)

    def exhaustive_matcher(args, folder):
        cmd = [
            args.colmap_executable, "exhaustive_matcher",
            "--database_path", os.path.join(folder, "distorted", "database.db"),
            "--SiftMatching.use_gpu", args.use_gpu,
        ]
        return execute(cmd)

    def mapper(args, folder):
        cmd = [
            args.colmap_executable, "mapper",
            "--database_path", os.path.join(folder, "distorted", "database.db"),
            "--image_path", os.path.join(folder, "input"),
            "--Mapper.ba_global_function_tolerance=0.000001",
        ]
        if args.load_camera:
            os.makedirs(os.path.join(folder, "distorted", "sparse", "0"), exist_ok=True)
            cmd += [
                "--input_path", load_colmap_cameras(args.load_camera, folder, args.colmap_executable, args.use_gpu),
                "--output_path", os.path.join(folder, "distorted", "sparse", "0")
            ]
        else:
            os.makedirs(os.path.join(folder, "distorted", "sparse"), exist_ok=True)
            cmd += [
                "--output_path", os.path.join(folder, "distorted", "sparse")
            ]
        return execute(cmd)

    def image_undistorter(args, folder):
        cmd = [
            args.colmap_executable, "image_undistorter",
            "--image_path", os.path.join(folder, "input"),
            "--input_path", os.path.join(folder, "distorted", "sparse", "0"),
            "--output_path", folder,
            "--output_type=COLMAP",
        ]
        return execute(cmd)

    def sparse_reconstruct(self, folder):
        if self.feature_extractor(folder) != 0:
            raise RuntimeError("Feature extraction failed")
        if self.exhaustive_matcher(folder) != 0:
            raise RuntimeError("Feature matching failed")
        if self.mapper(folder) != 0:
            raise RuntimeError("Mapping failed")
        if self.image_undistorter(folder) != 0:
            raise RuntimeError("Undistortion failed")

    def save_distorted(self, folder, image_path_list):
        os.makedirs(os.path.join(self.destination, "images"), exist_ok=True)
        for image_path in image_path_list:
            src = os.path.join(folder, "images", os.path.basename(image_path))
            if not os.path.exists(src):
                raise RuntimeError("Undistortion incomplete")
            dst = os.path.join(self.destination, "images", os.path.basename(image_path))
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)

    def read_points3D(self, folder):
        os.makedirs(os.path.join(folder, "sparse/0"), exist_ok=True)
        for entry in os.scandir(os.path.join(folder, "sparse")):
            if entry.name.startswith("points3D"):
                shutil.copy2(entry.path, os.path.join(folder, "sparse/0", entry.name))
        xyz, rgb, _ = read_colmap_points3D(folder)
        return InitializedPointCloud(
            points=torch.from_numpy(xyz)*self.scene_scale, colors=torch.from_numpy(rgb)/255.0
        )

    def read_camera(self, folder):
        os.makedirs(os.path.join(folder, "sparse/0"), exist_ok=True)
        for entry in os.scandir(os.path.join(folder, "sparse")):
            if entry.name.startswith("images") or entry.name.startswith("cameras"):
                shutil.copy2(entry.path, os.path.join(folder, "sparse/0", entry.name))
        return [
            InitializingCamera(
                image_width=camera.image_width, image_height=camera.image_height,
                FoVx=camera.FoVx, FoVy=camera.FoVy,
                R=camera.R, T=camera.T*self.scene_scale,
                image_path=os.path.join(self.destination, "images", os.path.basename(camera.image_path))
            )
            for camera in read_colmap_cameras(folder)]

    def __call__(self, image_path_list):
        with tempfile.TemporaryDirectory() as tempdir:
            os.makedirs(os.path.join(tempdir, "input"))
            for image_path in image_path_list:
                shutil.copy2(image_path, os.path.join(tempdir, "input", os.path.basename(image_path)))
            self.sparse_reconstruct(tempdir)
            self.save_distorted(tempdir, image_path_list)
            return self.read_points3D(tempdir), self.read_camera(tempdir)
