import os
import tempfile
import subprocess
import shutil
from pathlib import Path
import numpy as np
import torch

from gaussian_splatting.dataset.colmap import read_colmap_cameras
from gaussian_splatting.dataset.colmap.dataset import parse_colmap_camera
from gaussian_splatting.dataset.colmap.read_write_model import read_points3D_binary, read_cameras_binary, read_images_binary
from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .load_cameras import load_colmap_cameras


def output_image_paths(destination: str, image_name: Path):
    image_name = Path(image_name)
    root = Path(destination)
    image_path = root / "images" / image_name
    assert image_path.is_file(), f"Image does not exist: {image_path}"
    image_mask_path = root / "image_masks" / image_name.with_name(image_name.name + ".png")
    return str(image_path), str(image_mask_path) if image_mask_path.is_file() else None


def relative_image_names(image_path_list):
    """Paths relative to the common directory of every image in the list."""
    paths = [Path(path).resolve() for path in image_path_list]
    prefix = Path(os.path.commonpath(path.parent for path in paths))
    return prefix, [path.relative_to(prefix) for path in paths]


def copy2(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.samefile(src):
            return
        dst.unlink()
    shutil.copy2(src, dst)


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
                 allow_undistortion_missing: bool = False,
                 image_mask_dirname: str = "input_mask",
                 feature_mask_dirname: str = "feature_mask"):
        self.run_at_destination = run_at_destination
        self.colmap_executable = colmap_executable
        self.camera = camera
        self.single_camera_per_image = "1" if single_camera_per_image else "0"
        self.load_camera = load_camera
        self.scene_scale = scene_scale
        self.use_gpu = "1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allow_undistortion_missing = allow_undistortion_missing
        self.image_mask_dirname = image_mask_dirname
        self.feature_mask_dirname = feature_mask_dirname

    def to(self, device):
        self.use_gpu = "0" if device == "cpu" else "1"
        self.device = device
        return self

    def put_distorted(self, image_path_list, folder):
        prefix, image_names = relative_image_names(image_path_list)
        folder = Path(folder)
        image_mask_root = prefix.parent / self.image_mask_dirname
        feature_mask_root = prefix.parent / self.feature_mask_dirname
        for image_path, image_name in zip(image_path_list, image_names):
            copy2(image_path, folder / "input" / image_name)
            image_mask_path = image_mask_root / image_name.with_name(image_name.name + ".png")
            if image_mask_path.exists():
                copy2(image_mask_path, folder / "input_mask" / image_name.with_name(image_name.name + ".png"))
            feature_mask_path = feature_mask_root / image_name.with_name(image_name.name + ".png")
            if feature_mask_path.exists():
                copy2(feature_mask_path, folder / "feature_mask" / image_name.with_name(image_name.name + ".png"))
        return image_names

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
        feature_mask_dir = os.path.join(folder, "feature_mask")
        if os.path.isdir(feature_mask_dir):
            cmd += ["--ImageReader.mask_path", feature_mask_dir]
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
        shutil.rmtree(os.path.join(folder, "images"), ignore_errors=True)
        cmd = [
            args.colmap_executable, "image_undistorter",
            "--image_path", os.path.join(folder, "input"),
            "--input_path", os.path.join(folder, "distorted", "sparse", "0"),
            "--output_path", folder,
            "--output_type=COLMAP",
        ]
        return execute(cmd)

    def mask_undistorter(args, folder, image_names: list[Path]):
        folder = Path(folder)
        if not (folder / "input_mask").is_dir():
            return 0
        shutil.rmtree(folder / "tmp_mask", ignore_errors=True)
        exists = False
        for image_name in image_names:
            src = folder / "input_mask" / image_name.with_name(image_name.name + ".png")
            dst = folder / "tmp_mask" / image_name
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            exists = True
        if not exists:
            shutil.rmtree(folder / "tmp_mask", ignore_errors=True)
            return 0
        shutil.rmtree(folder / "tmp_mask_sparse", ignore_errors=True)
        ret = execute([
            args.colmap_executable, "image_undistorter",
            "--image_path", os.fspath(folder / "tmp_mask"),
            "--input_path", os.fspath(folder / "distorted" / "sparse" / "0"),
            "--output_path", os.fspath(folder / "tmp_mask_sparse"),
            "--output_type=COLMAP",
        ])
        shutil.rmtree(folder / "tmp_mask", ignore_errors=True)
        if ret != 0:
            shutil.rmtree(folder / "tmp_mask_sparse", ignore_errors=True)
            return ret
        for image_name in image_names:
            src = folder / "tmp_mask_sparse" / "images" / image_name
            dst = folder / "image_masks" / image_name.with_name(image_name.name + ".png")
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        shutil.rmtree(folder / "tmp_mask_sparse", ignore_errors=True)
        return 0

    def sparse_reconstruct(self, folder, image_names: list[Path]):
        mapper_ok = all(
            os.path.exists(os.path.join(folder, "distorted", "sparse", "0", file))
            for file in ("cameras.bin", "images.bin", "points3D.bin")
        )
        if self.load_camera is not None or not mapper_ok:
            if self.feature_extractor(folder) != 0:
                raise RuntimeError("Feature extraction failed")
            if self.exhaustive_matcher(folder) != 0:
                raise RuntimeError("Feature matching failed")
            if self.mapper(folder) != 0:
                raise RuntimeError("Mapping failed")
            if self.image_undistorter(folder) != 0:
                raise RuntimeError("Undistortion failed")
            if self.mask_undistorter(folder, image_names) != 0:
                raise RuntimeError("Mask undistortion failed")
            return
        undistorter_ok = all(
            (Path(folder) / "images" / image_name).exists()
            for image_name in image_names
        ) and all(
            os.path.exists(os.path.join(folder, "sparse", file))
            for file in ("cameras.bin", "images.bin", "points3D.bin")
        )
        if not undistorter_ok:
            if self.image_undistorter(folder) != 0:
                raise RuntimeError("Undistortion failed")
        if (Path(folder) / "input_mask").is_dir():
            mask_undistorter_ok = all(
                not (Path(folder) / "input_mask" / image_name.with_name(image_name.name + ".png")).exists()
                or (Path(folder) / "image_masks" / image_name.with_name(image_name.name + ".png")).exists()
                for image_name in image_names
            )
            if not mask_undistorter_ok:
                if self.mask_undistorter(folder, image_names) != 0:
                    raise RuntimeError("Mask undistortion failed")

    def save_distorted(self, folder, image_names: list[Path], destination: str):
        for image_name in image_names:
            src = Path(folder) / "images" / image_name
            if not src.exists():
                if self.allow_undistortion_missing:
                    continue
                else:
                    raise RuntimeError("Undistortion incomplete")
            copy2(src, Path(destination) / "images" / image_name)
            mask_src = Path(folder) / "image_masks" / image_name.with_name(image_name.name + ".png")
            if not mask_src.exists():
                continue
            copy2(mask_src, Path(destination) / "image_masks" / image_name.with_name(image_name.name + ".png"))

    def read_points3D(self, folder):
        points3D = read_points3D_binary(os.path.join(folder, "sparse", "points3D.bin"))
        xyz = torch.from_numpy(np.array([points3D[key].xyz for key in points3D])).to(device=self.device, dtype=torch.float)
        rgb = torch.from_numpy(np.array([points3D[key].rgb for key in points3D])).to(device=self.device, dtype=torch.float)
        return InitializedPointCloud(points=xyz*self.scene_scale, colors=rgb/255.0)

    def read_camera(self, folder, destination: str):
        image_dir = Path(folder) / "images"
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
                image_path=image_path, image_mask_path=image_mask_path,
                depth_path=depth_path, depth_mask_path=depth_mask_path,
            )
            for camera in parse_colmap_camera(cam_extrinsics, cam_intrinsics, image_dir, load_mask=False)]

    def run(self, image_path_list, folder, destination: str):
        image_names = self.put_distorted(image_path_list, folder)
        self.sparse_reconstruct(folder, image_names)
        self.save_distorted(folder, image_names, destination)
        return image_names

    def __call__(self, image_path_list, destination: str):
        if self.run_at_destination:
            self.run(image_path_list, destination, destination)
            return self.read_points3D(destination), self.read_camera(destination, destination)
        else:
            with tempfile.TemporaryDirectory() as tempdir:
                self.run(image_path_list, tempdir, destination)
                return self.read_points3D(tempdir), self.read_camera(tempdir, destination)
