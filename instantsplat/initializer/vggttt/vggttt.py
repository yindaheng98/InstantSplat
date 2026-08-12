import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.utils.geometry import closed_form_inverse_se3

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud
from instantsplat.initializer.vggt.vggt import (
    RESOLUTION,
    build_valid_image_area_mask,
    load_and_preprocess_images_square,
)

from ..vggt.save_depth import save_vggt_depth
from ..vggt.utils import focal2fov


class VGGTTTInitializer(AbstractInitializer):
    def __init__(
        self,
        model_url: str = "nvidia/vgg-ttt",
        img_load_resolution: int = 1024,
        conf_thres_value: float = 5.0,
        save_depth: bool = True,
        scene_scale: float = 1.0,
        num_ttt_steps: int = 2,
        memory_efficient_inference: bool = True,
        use_global_pred: bool = True,
        offload_to_cpu: bool = False,
    ):
        self.img_load_resolution = img_load_resolution
        self.conf_thres_value = conf_thres_value
        self.save_depth = save_depth
        self.scene_scale = scene_scale
        self.num_ttt_steps = num_ttt_steps
        self.memory_efficient_inference = memory_efficient_inference
        self.use_global_pred = use_global_pred
        self.offload_to_cpu = offload_to_cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VGGT.from_pretrained(model_url)
        self.model.eval()
        self.model.to(self.device)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(
        self, image_path_list: List[str]
    ) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
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
        depth_map = predictions["depth"].cpu().numpy()
        depth_conf = predictions["conf"].cpu().numpy()
        points_3d = predictions["pts3d"].cpu().numpy()
        torch.cuda.empty_cache()

        points_rgb = batch.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 3) [0, 1]

        conf_mask = depth_conf >= self.conf_thres_value
        valid_area_mask = build_valid_image_area_mask(
            original_coords.cpu().numpy(),
            src_resolution=self.img_load_resolution,
            dst_resolution=RESOLUTION,
        )
        conf_mask = np.logical_and(conf_mask, valid_area_mask)
        torch.cuda.empty_cache()

        cameras = []
        for i in range(len(image_path_list)):
            orig_w = float(original_coords[i, 4].item())
            orig_h = float(original_coords[i, 5].item())
            resize_ratio = max(orig_w, orig_h) / RESOLUTION

            fx_orig = intrinsic[i][0, 0] * resize_ratio
            fy_orig = intrinsic[i][1, 1] * resize_ratio

            saved_depth_path = None
            if self.save_depth:
                saved_depth_path = save_vggt_depth(
                    image_path=image_path_list[i],
                    depth=torch.from_numpy(depth_map[i]).squeeze(-1),
                    conf=torch.from_numpy(depth_conf[i]),
                    original_coord=original_coords[i],
                    src_resolution=self.img_load_resolution,
                    dst_resolution=RESOLUTION,
                    original_height=int(orig_h),
                    original_width=int(orig_w),
                    conf_threshold=self.conf_thres_value,
                )

            cameras.append(
                InitializingCamera(
                    image_width=int(orig_w),
                    image_height=int(orig_h),
                    FoVx=focal2fov(fx_orig, orig_w),
                    FoVy=focal2fov(fy_orig, orig_h),
                    R=torch.from_numpy(extrinsic[i][:3, :3]).float().to(device),
                    T=torch.from_numpy(extrinsic[i][:3, 3]).float().to(device) * self.scene_scale,
                    image_path=image_path_list[i],
                    depth_path=saved_depth_path,
                )
            )

        point_cloud = InitializedPointCloud(
            points=torch.from_numpy(points_3d[conf_mask]).float().to(device) * self.scene_scale,
            colors=torch.from_numpy(points_rgb[conf_mask]).float().to(device),
        )

        return point_cloud, cameras
