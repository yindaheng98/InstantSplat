import os
import numpy as np
import torch

from gaussian_splatting.dataset.colmap.read_write_model import (
    Camera as ColmapCamera,
    Image as ColmapImage,
    Point3D as ColmapPoint3D,
    write_model,
    rotmat2qvec,
)
from instantsplat.initializer.colmap.sparse import ColmapSparseInitializer, execute
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.projection import project_3D_points_np

from .vggt import run_VGGT


class VGGTColmapSparseInitializer(ColmapSparseInitializer):
    """Uses VGGT + VGGSfM tracker + COLMAP BA to replace COLMAP's SfM pipeline
    (feature_extractor + matcher + mapper).

    Inherits image_undistorter and all downstream steps from ColmapSparseInitializer.
    """

    def __init__(
        self,
        model_url="checkpoints/vggt_1B_commercial.pt",
        vggt_fixed_resolution=518,
        img_load_resolution=1024,
        max_query_pts=4096,
        query_frame_num=8,
        vis_thresh=0.2,
        max_reproj_error=8.0,
        keypoint_extractor="aliked+sp",
        fine_tracking=True,
        camera="PINHOLE",
        **kwargs,
    ):
        kwargs.pop("load_camera", None)
        super().__init__(camera=camera, load_camera=None, **kwargs)

        self.vggt_fixed_resolution = vggt_fixed_resolution
        self.img_load_resolution = img_load_resolution
        self.max_query_pts = max_query_pts
        self.query_frame_num = query_frame_num
        self.vis_thresh = vis_thresh
        self.max_reproj_error = max_reproj_error
        self.keypoint_extractor = keypoint_extractor
        self.fine_tracking = fine_tracking

        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L113-L118
        self.model = VGGT()
        self.model.load_state_dict(torch.load(model_url))
        self.model.eval()
        self.model.to(self.device)

    def to(self, device):
        super().to(device)
        self.model = self.model.to(device)
        return self

    def bundle_adjuster(self, folder):
        sparse_path = os.path.join(folder, "distorted", "sparse", "0")
        return execute([
            self.colmap_executable, "bundle_adjuster",
            "--input_path", sparse_path,
            "--output_path", sparse_path,
        ])

    def sparse_reconstruct(self, folder, image_path_list):
        mapper_ok = all(
            os.path.exists(os.path.join(folder, "distorted", "sparse", "0", f))
            for f in ("cameras.bin", "images.bin", "points3D.bin")
        )
        if not mapper_ok:
            # Override: replace feature_extractor + matcher + mapper
            self.vggt_mapper(folder, image_path_list)
            if self.bundle_adjuster(folder) != 0:
                raise RuntimeError("Bundle adjustment failed")
            if self.image_undistorter(folder) != 0:
                raise RuntimeError("Undistortion failed")
            if self.mask_undistorter(folder) != 0:
                raise RuntimeError("Mask undistortion failed")
            return
        undistorter_ok = all(
            os.path.exists(os.path.join(folder, "images", os.path.basename(p)))
            for p in image_path_list
        ) and all(
            os.path.exists(os.path.join(folder, "sparse", f))
            for f in ("cameras.bin", "images.bin", "points3D.bin")
        )
        if not undistorter_ok:
            if self.image_undistorter(folder) != 0:
                raise RuntimeError("Undistortion failed")
            if self.mask_undistorter(folder) != 0:
                raise RuntimeError("Mask undistortion failed")

    def vggt_mapper(self, folder, image_path_list):
        pass
