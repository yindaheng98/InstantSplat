import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple

from vggttt.nets.vggt.models.vggt import VGGT
from vggttt.nets.vggt.img import load_and_preprocess_images
from vggttt.nets.vggt.utils.geometry import closed_form_inverse_se3

from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud
from instantsplat.initializer.depth.abc import save_depth

from ..vggt.save_depth import normalize_conf_for_save
from ..vggt.utils import focal2fov

# From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/nets/vggt/img.py#L613-L615
RESOLUTION = 518
PATCH_SIZE = 14


def save_vggttt_depth(
    image_path: str,
    depth: torch.Tensor,
    conf: torch.Tensor,
    original_width: int,
    original_height: int,
    resized_height: int,
    batch_top: int,
    content_height: int,
    conf_threshold: float,
) -> str:
    def restore(tensor):
        batch_bottom = batch_top + content_height
        tensor = tensor[batch_top:batch_bottom, :]
        resized_tensor = torch.zeros(
            (resized_height, RESOLUTION),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        crop_top = max((resized_height - RESOLUTION) // 2, 0)
        crop_bottom = crop_top + content_height
        resized_tensor[crop_top:crop_bottom, :] = tensor
        return F.interpolate(
            resized_tensor[None, None].float(),
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    original_depth = restore(depth)
    original_conf = restore(conf)
    save_mask = normalize_conf_for_save(original_conf, conf_threshold)
    return save_depth(image_path=image_path, depth=original_depth, mask=save_mask)


class VGGTTTInitializer(AbstractInitializer):
    # From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/demo.py#L513-L519
    # From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/nets/vggt/models/vggt.py#L255-L268
    def __init__(
        self,
        model_url: str = "nvidia/vgg-ttt",
        conf_thres_value: float = 5.0,
        save_depth: bool = True,
        scene_scale: float = 1.0,
        num_ttt_steps: int = 2,
        memory_efficient_inference: bool = True,
        use_global_pred: bool = True,
        offload_to_cpu: bool = False,
    ):
        self.conf_thres_value = conf_thres_value
        self.save_depth = save_depth
        self.scene_scale = scene_scale
        self.num_ttt_steps = num_ttt_steps
        self.memory_efficient_inference = memory_efficient_inference
        self.use_global_pred = use_global_pred
        self.offload_to_cpu = offload_to_cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/demo.py#L71-L79
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

        # From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/demo.py#L368-L376
        images = load_and_preprocess_images(image_path_list)
        images = images.to(device)

        # From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/demo.py#L383-L391
        predictions = self.model.infer(
            images,
            num_ttt_steps=self.num_ttt_steps,
            memory_efficient_inference=self.memory_efficient_inference,
            use_global_pred=self.use_global_pred,
            offload_to_cpu=self.offload_to_cpu,
        )

        # Adapted from: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/nets/vggt/models/vggt.py#L331-L363
        extrinsic = closed_form_inverse_se3(predictions["pose"])[:, :3, :].cpu().numpy()
        # From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/demo.py#L395-L399
        intrinsic = predictions["intrinsics"].cpu().numpy()
        depth_map = predictions["depth"].cpu().numpy()
        depth_conf = predictions["conf"].cpu().numpy()
        points_3d = predictions["pts3d"].cpu().numpy()
        torch.cuda.empty_cache()

        # From: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/demo.py#L401-L402
        points_rgb = images.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 3) [0, 1]

        conf_mask = depth_conf >= self.conf_thres_value
        torch.cuda.empty_cache()

        cameras = []
        for i in range(len(image_path_list)):
            with Image.open(image_path_list[i]) as image:
                orig_w, orig_h = image.size

            # Adapted from: https://github.com/nv-dvl/vgg-ttt/blob/5b2d02d0598da9544e1dfa3b13a24ba09257a07f/vggttt/nets/vggt/img.py#L672-L740
            resized_height = round(orig_h * (RESOLUTION / orig_w) / PATCH_SIZE) * PATCH_SIZE
            content_height = min(resized_height, RESOLUTION)
            batch_top = (images.shape[-2] - content_height) // 2
            batch_bottom = batch_top + content_height
            conf_mask[i, :batch_top, :] = False
            conf_mask[i, batch_bottom:, :] = False

            orig_w = float(orig_w)
            orig_h = float(orig_h)
            fx_orig = intrinsic[i][0, 0] * orig_w / RESOLUTION
            fy_orig = intrinsic[i][1, 1] * orig_h / resized_height

            saved_depth_path = None
            if self.save_depth:
                saved_depth_path = save_vggttt_depth(
                    image_path=image_path_list[i],
                    depth=torch.from_numpy(depth_map[i]).squeeze(-1),
                    conf=torch.from_numpy(depth_conf[i]),
                    original_width=int(orig_w),
                    original_height=int(orig_h),
                    resized_height=resized_height,
                    batch_top=batch_top,
                    content_height=content_height,
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
