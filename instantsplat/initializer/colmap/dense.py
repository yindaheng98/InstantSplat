import os
import shutil
import tempfile
from instantsplat.initializer.abc import InitializedPointCloud
from .sparse import ColmapSparseInitializer, execute
from .delaunay2ply import read_ply, delaunay2ply
from .poisson2ply import poisson2ply


class ColmapDenseInitializer(ColmapSparseInitializer):
    def __init__(
            self, *args,
            PatchMatchStereo_max_image_size=2000,
            PatchMatchStereo_cache_size=32,
            delaunay2ply_batch=512,
            poisson2ply_thresh=0.2,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.PatchMatchStereo_max_image_size = PatchMatchStereo_max_image_size
        self.PatchMatchStereo_cache_size = PatchMatchStereo_cache_size
        self.delaunay2ply_batch = delaunay2ply_batch
        self.poisson2ply_thresh = poisson2ply_thresh

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
        return execute(cmd)

    def poisson_mesher(args, folder):
        cmd = [
            args.colmap_executable, "poisson_mesher",
            "--input_path", os.path.join(folder, "fused.ply"),
            "--output_path", os.path.join(folder, "meshed-poisson.ply"),
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
            batch=args.delaunay2ply_batch)

    def poisson2ply(args, folder):
        return poisson2ply(
            os.path.join(folder, "meshed-poisson.ply"),
            os.path.join(folder, "colorful-delaunay.ply"),
            threshold=args.poisson2ply_thresh)

    def dense_reconstruct(self, folder):
        if self.patch_match_stereo(folder) != 0:
            raise ValueError("Patch match stereo failed")
        if self.stereo_fusion(folder) != 0:
            raise ValueError("Stereo fusion failed")
        if self.poisson_mesher(folder) != 0:
            raise ValueError("Poisson mesher failed")
        if self.delaunay_mesher(folder) != 0:
            raise ValueError("Delaunay mesher failed")
        self.delaunay2ply(folder).write(os.path.join(folder, "colorful-delaunay.ply"))
        self.poisson2ply(folder).write(os.path.join(folder, "filtered-poisson.ply"))

    def __call__(self, image_path_list):
        with tempfile.TemporaryDirectory() as tempdir:
            os.makedirs(os.path.join(tempdir, "input"))
            for image_path in image_path_list:
                shutil.copy2(image_path, os.path.join(tempdir, "input", os.path.basename(image_path)))
            self.sparse_reconstruct(tempdir)
            self.save_distorted(tempdir, image_path_list)
            self.dense_reconstruct(tempdir)
            xyz, rgb = read_ply(os.path.join(tempdir, "filtered-poisson.ply"))
            return InitializedPointCloud(points=xyz*self.scene_scale, colors=rgb/255.0), self.read_camera(tempdir)
