import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from instantsplat.initializer.abc import InitializedPointCloud, InitializingCamera
from instantsplat.initializer.depth.abc import save_depth
from .sparse import ColmapSparseInitializer, execute
from .delaunay2ply import read_ply, delaunay2ply
from .poisson2ply import poisson2ply


def read_colmap_depth_map(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        delimiter_count = 0
        while delimiter_count < 3:
            if fid.read(1) == b"&":
                delimiter_count += 1
        depth = np.fromfile(fid, np.float32)
    return np.transpose(depth.reshape((width, height, channels), order="F"), (1, 0, 2)).squeeze()


class ColmapDenseInitializer(ColmapSparseInitializer):
    def __init__(
            self,
            PatchMatchStereo_max_image_size=2000,
            PatchMatchStereo_cache_size=32,
            PoissonMeshing_num_threads=16,
            delaunay2ply_batch=512,
            delaunay2ply_reference_batch=512*512,
            poisson2ply_thresh=0.2,
            use_fused=False,
            use_mask=False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PatchMatchStereo_max_image_size = PatchMatchStereo_max_image_size
        self.PatchMatchStereo_cache_size = PatchMatchStereo_cache_size
        self.PoissonMeshing_num_threads = PoissonMeshing_num_threads
        self.delaunay2ply_batch = delaunay2ply_batch
        self.delaunay2ply_reference_batch = delaunay2ply_reference_batch
        self.poisson2ply_thresh = poisson2ply_thresh
        self.use_fused = use_fused
        self.use_mask = use_mask

    def patch_match_stereo(args, folder):
        cmd = [
            args.colmap_executable, "patch_match_stereo",
            "--workspace_path", folder,
            "--workspace_format=COLMAP",
            "--PatchMatchStereo.max_image_size", str(args.PatchMatchStereo_max_image_size),
            "--PatchMatchStereo.cache_size", str(args.PatchMatchStereo_cache_size),
            "--PatchMatchStereo.geom_consistency=true",
        ]
        return execute(cmd)

    def stereo_fusion(args, folder):
        cmd = [
            args.colmap_executable, "stereo_fusion",
            "--workspace_path", folder,
            "--output_path", os.path.join(folder, "fused.ply"),
            "--workspace_format=COLMAP",
            "--input_type=photometric",
        ]
        if args.use_mask:
            cmd += ["--StereoFusion.mask_path", os.path.join(folder, "image_masks")]
        return execute(cmd)

    def poisson_mesher(args, folder):
        cmd = [
            args.colmap_executable, "poisson_mesher",
            "--input_path", os.path.join(folder, "fused.ply"),
            "--output_path", os.path.join(folder, "meshed-poisson.ply"),
            "--PoissonMeshing.num_threads", str(args.PoissonMeshing_num_threads),
        ]
        return execute(cmd)

    def delaunay_mesher(args, folder):
        cmd = [
            args.colmap_executable, "delaunay_mesher",
            "--input_path", folder,
            "--output_path", os.path.join(folder, "meshed-delaunay.ply"),
        ]
        return execute(cmd)

    def delaunay2ply(args, folder):
        return delaunay2ply(
            os.path.join(folder, "meshed-delaunay.ply"),
            os.path.join(folder, "meshed-poisson.ply"),
            batch=args.delaunay2ply_batch,
            reference_batch=args.delaunay2ply_reference_batch)

    def poisson2ply(args, folder):
        return poisson2ply(
            os.path.join(folder, "meshed-poisson.ply"),
            os.path.join(folder, "colorful-delaunay.ply"),
            threshold=args.poisson2ply_thresh)

    @staticmethod
    def verify_patch_match_stereo(folder, image_names):
        folder = Path(folder)
        for image_name in image_names:
            if not (folder / "stereo" / "depth_maps" / image_name.with_name(image_name.name + ".geometric.bin")).exists():
                return False
            if not (folder / "stereo" / "depth_maps" / image_name.with_name(image_name.name + ".photometric.bin")).exists():
                return False
            if not (folder / "stereo" / "normal_maps" / image_name.with_name(image_name.name + ".geometric.bin")).exists():
                return False
            if not (folder / "stereo" / "normal_maps" / image_name.with_name(image_name.name + ".photometric.bin")).exists():
                return False
        return True

    def dense_reconstruct(self, folder, image_names):
        ok = True

        ok = ok and self.verify_patch_match_stereo(folder, image_names)
        if not ok:
            if self.patch_match_stereo(folder) != 0:
                raise ValueError("Patch match stereo failed")

        ok = ok and os.path.exists(os.path.join(folder, "fused.ply"))
        if not ok:
            if self.stereo_fusion(folder) != 0:
                raise ValueError("Stereo fusion failed")

        if self.use_fused:
            return

        ok = ok and os.path.exists(os.path.join(folder, "meshed-poisson.ply"))
        if not ok:
            if self.poisson_mesher(folder) != 0:
                raise ValueError("Poisson mesher failed")

        ok = ok and os.path.exists(os.path.join(folder, "meshed-delaunay.ply"))
        if not ok:
            if self.delaunay_mesher(folder) != 0:
                raise ValueError("Delaunay mesher failed")

        ok = ok and os.path.exists(os.path.join(folder, "colorful-delaunay.ply"))
        if not ok:
            self.delaunay2ply(folder).write(os.path.join(folder, "colorful-delaunay.ply"))

        ok = ok and os.path.exists(os.path.join(folder, "filtered-poisson.ply"))
        if not ok:
            self.poisson2ply(folder).write(os.path.join(folder, "filtered-poisson.ply"))

    def export_depth(self, folder, camera: InitializingCamera):
        image_name = Path(camera.image_path).relative_to(Path(self.destination) / "images")
        bin_path = Path(folder) / "stereo" / "depth_maps" / image_name.with_name(image_name.name + ".geometric.bin")
        if not os.path.exists(bin_path):
            return camera
        depth = read_colmap_depth_map(bin_path)
        mask = depth > 0
        if not mask.any():
            return camera
        height, width = camera.image_height, camera.image_width
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        depth = np.where(mask, depth, 0).astype(np.float32) * self.scene_scale
        return camera._replace(depth_path=save_depth(
            camera.image_path,
            torch.from_numpy(depth),
            torch.from_numpy(mask.astype(np.float32)),
        ))

    def read_camera(self, folder):
        return [self.export_depth(folder, camera) for camera in super().read_camera(folder)]

    def run(self, image_path_list, tempdir):
        image_names = super().run(image_path_list, tempdir)
        self.dense_reconstruct(tempdir, image_names)
        return image_names

    def __call__(self, image_path_list):
        use_file = "fused.ply" if self.use_fused else "filtered-poisson.ply"
        if self.run_at_destination:
            self.run(image_path_list, self.destination)
            xyz, rgb = read_ply(os.path.join(self.destination, use_file))
            return InitializedPointCloud(points=xyz.to(self.device)*self.scene_scale, colors=rgb.to(self.device)/255.0), self.read_camera(self.destination)
        else:
            with tempfile.TemporaryDirectory() as tempdir:
                self.run(image_path_list, tempdir)
                xyz, rgb = read_ply(os.path.join(tempdir, use_file))
                return InitializedPointCloud(points=xyz.to(self.device)*self.scene_scale, colors=rgb.to(self.device)/255.0), self.read_camera(tempdir)
