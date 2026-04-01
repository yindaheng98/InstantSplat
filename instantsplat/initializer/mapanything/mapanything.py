import math
from typing import List, Tuple

import torch
from PIL import Image, ImageOps
from mapanything.models import MapAnything
from mapanything.utils.colmap_export import closed_form_pose_inverse
from mapanything.utils.image import load_images

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def recover_original_intrinsics(intrinsics, original_width, original_height, target_width, target_height):
    resize_scale = max(target_width / original_width, target_height / original_height) + 1e-8
    resized_width = math.floor(original_width * resize_scale)
    resized_height = math.floor(original_height * resize_scale)
    crop_left = (resized_width - target_width) // 2
    crop_top = (resized_height - target_height) // 2

    original_intrinsics = intrinsics.clone()
    original_intrinsics[0, 2] += crop_left
    original_intrinsics[1, 2] += crop_top
    original_intrinsics[0, :3] /= resize_scale
    original_intrinsics[1, :3] /= resize_scale
    return original_intrinsics


class MapAnythingInitializer(AbstractInitializer):
    def __init__(
        self,
        apache: bool = False,
        ################################################################
        # MapAnything.infer parameters
        # https://github.com/facebookresearch/map-anything/blob/f7ebafb4d8349776705aaa686cf928988d1bd7f4/mapanything/models/mapanything/model.py#L2019-L2037
        memory_efficient_inference: bool = True,
        minibatch_size: int = None,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        apply_mask: bool = True,
        mask_edges: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10,
        ignore_calibration_inputs: bool = False,
        ignore_depth_inputs: bool = False,
        ignore_pose_inputs: bool = False,
        ignore_depth_scale_inputs: bool = False,
        ignore_pose_scale_inputs: bool = False,
        use_multiview_confidence: bool = False,
        multiview_conf_depth_abs_thresh: float = 0.02,
        multiview_conf_depth_rel_thresh: float = 0.02,
        ################################################################
        scene_scale: float = 1.0,
    ):
        self.scene_scale = scene_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MapAnything.from_pretrained(
            "facebook/map-anything-apache" if apache else "facebook/map-anything"
        ).to(self.device)
        self.model.eval()

        self.infer_parameters = dict(
            memory_efficient_inference=memory_efficient_inference,
            minibatch_size=minibatch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            ignore_calibration_inputs=ignore_calibration_inputs,
            ignore_depth_inputs=ignore_depth_inputs,
            ignore_pose_inputs=ignore_pose_inputs,
            ignore_depth_scale_inputs=ignore_depth_scale_inputs,
            ignore_pose_scale_inputs=ignore_pose_scale_inputs,
            use_multiview_confidence=use_multiview_confidence,
            multiview_conf_depth_abs_thresh=multiview_conf_depth_abs_thresh,
            multiview_conf_depth_rel_thresh=multiview_conf_depth_rel_thresh,
        )

    def to(self, device):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def __call__(self, image_path_list: List[str]) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        views = load_images(image_path_list, norm_type="dinov2")
        target_height, target_width = map(int, views[0]["true_shape"][0])
        original_sizes = []
        for image_path in image_path_list:
            with Image.open(image_path) as image:
                original_sizes.append(ImageOps.exif_transpose(image).size)

        with torch.no_grad():
            outputs = self.model.infer(views, **self.infer_parameters)

        all_points = []
        all_colors = []
        cameras = []

        for output, image_path, (original_width, original_height) in zip(outputs, image_path_list, original_sizes):
            depth_z = output["depth_z"][0].squeeze(-1).detach()
            mask = output["mask"][0].squeeze(-1).detach().type(torch.bool)
            valid_mask = mask & (depth_z > 0)

            pts3d = output["pts3d"][0].detach()
            all_points.append(pts3d[valid_mask])

            img_no_norm = output["img_no_norm"][0].detach()
            img_uint8 = (img_no_norm.clamp(0.0, 1.0) * 255).type(torch.uint8)
            all_colors.append(img_uint8[valid_mask])

            intrinsics = output["intrinsics"][0].detach()
            cam2world = output["camera_poses"][0].detach()
            world2cam = closed_form_pose_inverse(cam2world[None])[0]
            original_intrinsics = recover_original_intrinsics(
                intrinsics,
                original_width=original_width,
                original_height=original_height,
                target_width=target_width,
                target_height=target_height,
            )

            cameras.append(
                InitializingCamera(
                    image_width=original_width,
                    image_height=original_height,
                    FoVx=focal2fov(original_intrinsics[0, 0].item(), original_width),
                    FoVy=focal2fov(original_intrinsics[1, 1].item(), original_height),
                    R=world2cam[:3, :3].float(),
                    T=world2cam[:3, 3].float() * self.scene_scale,
                    image_path=image_path,
                )
            )

        if len(all_points) == 0 or sum(len(points) for points in all_points) == 0:
            raise RuntimeError("MapAnything produced no valid 3D points")

        points = torch.cat(all_points, dim=0)
        colors = torch.cat(all_colors, dim=0)

        return InitializedPointCloud(
            points=points * self.scene_scale,
            colors=colors / 255.0,
        ), cameras
