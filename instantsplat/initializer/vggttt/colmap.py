import os
import torch
import torch.nn.functional as F

from gaussian_splatting.dataset.colmap.read_write_model import write_model
from instantsplat.initializer.colmap.sparse import ColmapSparseInitializer
from instantsplat.initializer.colmap.dense import ColmapDenseInitializer
from instantsplat.initializer.vggt.colmap import VGGTColmapSparseInitializer
from instantsplat.initializer.vggt.np_to_colmap import batch_np_matrix_to_colmap
from instantsplat.initializer.vggt.utils import predict_tracks
from instantsplat.initializer.vggt.vggt import RESOLUTION, load_and_preprocess_images_square
from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.utils.geometry import closed_form_inverse_se3


class VGGTTTColmapSparseInitializer(VGGTColmapSparseInitializer):
    """Uses VGG-T³ + VGGSfM tracker + COLMAP BA to replace COLMAP's SfM pipeline
    (feature_extractor + matcher + mapper).

    Inherits image_undistorter and all downstream steps from ColmapSparseInitializer.
    """

    def __init__(
        self,
        model_url="nvidia/vgg-ttt",
        img_load_resolution=1024,
        max_query_pts=4096,
        query_frame_num=8,
        vis_thresh=0.2,
        max_reproj_error=8.0,
        keypoint_extractor="aliked+sp",
        fine_tracking=True,
        camera="PINHOLE",
        num_ttt_steps=2,
        memory_efficient_inference=True,
        use_global_pred=True,
        offload_to_cpu=False,
        **kwargs,
    ):
        kwargs.pop("load_camera", None)
        ColmapSparseInitializer.__init__(self, camera=camera, load_camera=None, **kwargs)

        self.img_load_resolution = img_load_resolution
        self.max_query_pts = max_query_pts
        self.query_frame_num = query_frame_num
        self.vis_thresh = vis_thresh
        self.max_reproj_error = max_reproj_error
        self.keypoint_extractor = keypoint_extractor
        self.fine_tracking = fine_tracking
        self.num_ttt_steps = num_ttt_steps
        self.memory_efficient_inference = memory_efficient_inference
        self.use_global_pred = use_global_pred
        self.offload_to_cpu = offload_to_cpu

        self.model = VGGT.from_pretrained(model_url)
        self.model.eval()
        self.model.to(self.device)

    def vggt_mapper(self, folder, image_path_list):
        device = self.device

        images, original_coords = load_and_preprocess_images_square(image_path_list, self.img_load_resolution)
        images = images.to(device)
        original_coords = original_coords.to(device)  # (N, 6): [x1, y1, x2, y2, orig_width, orig_height]

        batch = images
        if images.shape[-2:] != (RESOLUTION, RESOLUTION):
            batch = F.interpolate(images, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)
        batch = batch.unsqueeze(0)
        device = batch.device
        dtype = (
            torch.bfloat16
            if device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )

        with torch.no_grad():
            predictions = self.model.infer(
                batch.squeeze(0),
                num_ttt_steps=self.num_ttt_steps,
                dtype=dtype,
                memory_efficient_inference=self.memory_efficient_inference,
                use_global_pred=self.use_global_pred,
                offload_to_cpu=self.offload_to_cpu,
            )

        extrinsic = closed_form_inverse_se3(predictions["pose"])[:, :3, :].cpu().numpy()
        intrinsic = predictions["intrinsics"].cpu().numpy()
        depth_conf = predictions["conf"].cpu().numpy()
        points_3d = predictions["pts3d"].cpu().numpy()
        torch.cuda.empty_cache()

        original_coords = original_coords.cpu().numpy()
        img_load_resolution = self.img_load_resolution
        scale = img_load_resolution / RESOLUTION

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=self.max_query_pts,
                    query_frame_num=self.query_frame_num,
                    keypoint_extractor=self.keypoint_extractor,
                    fine_tracking=self.fine_tracking,
                )

                torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > self.vis_thresh

        cameras, colmap_images, colmap_points3D, valid_track_mask = batch_np_matrix_to_colmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            original_coords,
            img_load_resolution,
            [os.path.basename(p) for p in image_path_list],
            masks=track_mask,
            max_reproj_error=self.max_reproj_error,
            camera_type=self.camera,
            points_rgb=points_rgb,
        )

        if len(colmap_points3D) == 0:
            raise RuntimeError("No valid tracks for bundle adjustment")

        sparse_dir = os.path.join(folder, "distorted", "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)
        write_model(cameras, colmap_images, colmap_points3D, sparse_dir)


class VGGTTTColmapDenseInitializer(ColmapDenseInitializer, VGGTTTColmapSparseInitializer):
    """VGG-T³ sparse reconstruction + COLMAP dense reconstruction.

    Combines VGGTTTColmapSparseInitializer (VGG-T³ + BA for sparse) with
    ColmapDenseInitializer (PatchMatch + fusion + meshing for dense).
    """
    pass
