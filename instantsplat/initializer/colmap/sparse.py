import os
import tempfile
import subprocess
import shutil
import numpy as np
import torch

from gaussian_splatting.dataset.colmap import read_colmap_cameras
from gaussian_splatting.dataset.colmap.dataset import parse_colmap_camera
from gaussian_splatting.dataset.colmap.read_write_model import read_points3D_binary, read_cameras_binary, read_images_binary
from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .load_cameras import load_colmap_cameras


def execute(cmd):
    proc = subprocess.Popen(cmd, shell=False)
    proc.communicate()
    return proc.returncode


class ColmapSparseInitializer(AbstractInitializer):
    def __init__(self,
                 destination: str,
                 run_at_destination: bool = True,
                 colmap_executable: str = "colmap",
                 camera: str = "OPENCV",
                 single_camera_per_image: bool = True,
                 load_camera: str = None,
                 scene_scale: float = 1.0,
                 allow_undistortion_missing: bool = False):
        self.destination = destination
        self.run_at_destination = run_at_destination
        self.colmap_executable = colmap_executable
        self.camera = camera
        self.single_camera_per_image = "1" if single_camera_per_image else "0"
        self.load_camera = load_camera
        self.scene_scale = scene_scale
        self.use_gpu = "1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allow_undistortion_missing = allow_undistortion_missing

    def to(self, device):
        self.use_gpu = "0" if device == "cpu" else "1"
        self.device = device
        return self

    def put_distorted(self, image_path_list, folder):
        os.makedirs(os.path.join(folder, "input"), exist_ok=True)
        for image_path in image_path_list:
            if not os.path.samefile(image_path, os.path.join(folder, "input", os.path.basename(image_path))):
                shutil.copy2(image_path, os.path.join(folder, "input", os.path.basename(image_path)))

    def feature_extractor(args, folder):
        os.makedirs(os.path.join(folder, "distorted"), exist_ok=True)
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

    def sparse_reconstruct(self, folder, image_path_list):
        mapper_ok = True
        for file in ["cameras.bin", "images.bin", "points3D.bin"]:
            if not os.path.exists(os.path.join(folder, "distorted", "sparse", "0", file)):
                mapper_ok = False
                break
        if self.load_camera is not None or not mapper_ok:
            if self.feature_extractor(folder) != 0:
                raise RuntimeError("Feature extraction failed")
            if self.exhaustive_matcher(folder) != 0:
                raise RuntimeError("Feature matching failed")
            if self.mapper(folder) != 0:
                raise RuntimeError("Mapping failed")
            if self.image_undistorter(folder) != 0:
                raise RuntimeError("Undistortion failed")
            return
        undistorter_ok = True
        for image_path in image_path_list:
            if not os.path.exists(os.path.join(folder, "images", os.path.basename(image_path))):
                undistorter_ok = False
                break
        for file in ["cameras.bin", "images.bin", "points3D.bin"]:
            if not os.path.exists(os.path.join(folder, "sparse", file)):
                undistorter_ok = False
                break
        if not undistorter_ok:
            if self.image_undistorter(folder) != 0:
                raise RuntimeError("Undistortion failed")
            return

    def save_distorted(self, folder, image_path_list):
        os.makedirs(os.path.join(self.destination, "images"), exist_ok=True)
        for image_path in image_path_list:
            src = os.path.join(folder, "images", os.path.basename(image_path))
            if not os.path.exists(src):
                if self.allow_undistortion_missing:
                    continue
                else:
                    raise RuntimeError("Undistortion incomplete")
            dst = os.path.join(self.destination, "images", os.path.basename(image_path))
            if os.path.exists(dst):
                if os.path.samefile(src, dst):
                    continue
                os.remove(dst)
            shutil.copy2(src, dst)

    def read_points3D(self, folder):
        points3D = read_points3D_binary(os.path.join(folder, "sparse", "points3D.bin"))
        xyz = torch.from_numpy(np.array([points3D[key].xyz for key in points3D])).to(device=self.device, dtype=torch.float)
        rgb = torch.from_numpy(np.array([points3D[key].rgb for key in points3D])).to(device=self.device, dtype=torch.float)
        return InitializedPointCloud(points=xyz*self.scene_scale, colors=rgb/255.0)

    def read_camera(self, folder):
        image_dir = os.path.join(folder, "images")
        cameras_extrinsic_file = os.path.join(folder, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(folder, "sparse", "cameras.bin")
        cam_extrinsics = read_images_binary(cameras_extrinsic_file)
        cam_intrinsics = read_cameras_binary(cameras_intrinsic_file)
        return [
            InitializingCamera(
                image_width=camera.image_width, image_height=camera.image_height,
                FoVx=camera.FoVx, FoVy=camera.FoVy,
                R=camera.R.to(device=self.device, dtype=torch.float),
                T=camera.T.to(device=self.device, dtype=torch.float)*self.scene_scale,
                image_path=os.path.join(self.destination, "images", os.path.basename(camera.image_path))
            )
            for camera in parse_colmap_camera(cam_extrinsics, cam_intrinsics, image_dir, os.path.join(folder, "depths"))]

    def run(self, image_path_list, tempdir):
        self.put_distorted(image_path_list, tempdir)
        self.sparse_reconstruct(tempdir, image_path_list)
        self.save_distorted(tempdir, image_path_list)

    def __call__(self, image_path_list):
        if self.run_at_destination:
            self.run(image_path_list, self.destination)
            return self.read_points3D(self.destination), self.read_camera(self.destination)
        else:
            with tempfile.TemporaryDirectory() as tempdir:
                self.run(image_path_list, tempdir)
                return self.read_points3D(tempdir), self.read_camera(tempdir)
