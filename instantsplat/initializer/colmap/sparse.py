import os
import tempfile
import subprocess
import shutil

from instantsplat.initializer.abc import AbstractInitializer


def execute(cmd):
    proc = subprocess.Popen(cmd, shell=False)
    proc.communicate()
    return proc.returncode


class ColmapSparseInitializer(AbstractInitializer):
    def __init__(self,
                 colmap_executable: str = "colmap",
                 camera: str = "OPENCV",
                 single_camera_per_image: bool = True,
                 save_distorted_images: str = None):
        self.colmap_executable = os.path.abspath(colmap_executable)
        self.camera = camera
        self.single_camera_per_image = single_camera_per_image
        self.save_distorted_images = save_distorted_images
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
        ]
        if args.single_camera_per_image:
            cmd += ["--ImageReader.single_camera_per_image=1"]
        return execute(cmd)

    def exhaustive_matcher(args, folder):
        cmd = [
            args.colmap_executable, "exhaustive_matcher",
            "--database_path", os.path.join(folder, "distorted", "database.db"),
            "--SiftMatching.use_gpu", args.use_gpu,
        ]
        return execute(cmd)

    def mapper(args, folder):
        os.makedirs(os.path.join(folder, "distorted", "sparse"), exist_ok=True)
        cmd = [
            args.colmap_executable, "mapper",
            "--database_path", os.path.join(folder, "distorted", "database.db"),
            "--image_path", os.path.join(folder, "input"),
            "--output_path", os.path.join(folder, "distorted", "sparse"),
            "--Mapper.ba_global_function_tolerance=0.000001",
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
        if not self.save_distorted_images:
            return
        os.makedirs(self.save_distorted_images, exist_ok=True)
        for image_path in image_path_list:
            src = os.path.join(folder, "input", os.path.basename(image_path))
            if not os.path.exists(src):
                raise RuntimeError("Undistortion incomplete")
            dst = os.path.join(self.save_distorted_images, os.path.basename(image_path))
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(os.path.join(folder, "images", os.path.basename(image_path)), os.path.join(self.save_distorted_images, os.path.basename(image_path)))

    def __call__(self, image_path_list):
        with tempfile.TemporaryDirectory() as tempdir:
            os.makedirs(os.path.join(tempdir, "input"))
            for image_path in image_path_list:
                shutil.copy2(image_path, os.path.join(tempdir, "input", os.path.basename(image_path)))
            self.sparse_reconstruct(tempdir)
            self.save_distorted(tempdir, image_path_list)
            pass  # TODO: load the result and return it