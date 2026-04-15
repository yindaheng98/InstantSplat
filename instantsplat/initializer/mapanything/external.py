import os
from typing import List, Tuple

import hydra
import torch
from mapanything.models import init_model
from mapanything.utils.colmap_export import closed_form_pose_inverse
from mapanything.utils.inference import postprocess_model_outputs_for_inference

from instantsplat.initializer.abc import (
    AbstractInitializer,
    InitializedPointCloud,
    InitializingCamera,
)

from .mapanything import focal2fov, load_views, recover_original_intrinsics, save_resized_depth

# https://github.com/facebookresearch/map-anything/blob/main/scripts/profile_memory_runtime.py#L203-L219
MODEL_CONFIG = {
    "anycalib": {"norm_type": "identity", "resolution_set": 518},
    "da3": {"norm_type": "dinov2", "resolution_set": 504},
    "dust3r": {"norm_type": "dust3r", "resolution_set": 512},
    "mast3r": {"norm_type": "dust3r", "resolution_set": 512},
    "moge": {"norm_type": "identity", "resolution_set": 518},
    "must3r": {"norm_type": "dust3r", "resolution_set": 512},
    "pi3": {"norm_type": "identity", "resolution_set": 518},
    "pi3x": {"norm_type": "identity", "resolution_set": 518},
    "pow3r": {"norm_type": "dust3r", "resolution_set": 512},
    "pow3r_ba": {"norm_type": "dust3r", "resolution_set": 512},
    "vggt": {"norm_type": "identity", "resolution_set": 518},
}

LOCAL_MODEL_CONFIG_NAME = {
    "moge": "moge_2",
    "vggt": "vggt_commercial",
}


def init_model_from_config(
    model_name: str,
    device: str,
    machine: str = "default",
    configs_dir: str = "./configs",
    checkpoints_dir: str = "./checkpoints",
):
    configs_dir = os.path.abspath(configs_dir)
    checkpoints_dir = os.path.abspath(checkpoints_dir)
    config_name = LOCAL_MODEL_CONFIG_NAME.get(model_name, model_name)
    config_path = os.path.join(configs_dir, "model", f"{config_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(version_base=None, config_dir=configs_dir)
    config = hydra.compose(
        config_name="train",
        overrides=[f"model={config_name}", f"machine={machine}"],
    )
    config.machine.root_pretrained_checkpoints_dir = checkpoints_dir
    config.machine.root_uniception_pretrained_checkpoints_dir = checkpoints_dir

    model = init_model(
        model_str=config.model.model_str,
        model_config=config.model.model_config,
        torch_hub_force_reload=False,
    )
    return model.to(device)


class MapAnythingExternalInitializer(AbstractInitializer):
    def __init__(
        self,
        model_name: str = "vggt",
        machine: str = "default",
        configs_dir: str = "./configs",
        checkpoints_dir: str = "./checkpoints",
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        ################################################################
        # postprocess_model_outputs_for_inference parameters
        # https://github.com/facebookresearch/map-anything/blob/f7ebafb4d8349776705aaa686cf928988d1bd7f4/mapanything/utils/inference.py#L288-L296
        apply_mask: bool = True,
        mask_edges: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10,
        use_multiview_confidence: bool = False,
        multiview_conf_depth_abs_thresh: float = 0.02,
        multiview_conf_depth_rel_thresh: float = 0.02,
        ################################################################
        save_depths: bool = True,
        save_conf_threshold: float = 1.0,
        scene_scale: float = 1.0,
    ):
        if model_name not in MODEL_CONFIG:
            raise ValueError(
                f"Unsupported external model '{model_name}'. "
                f"Expected one of: {', '.join(sorted(MODEL_CONFIG))}"
            )

        defaults = MODEL_CONFIG[model_name]

        self.model_name = model_name
        self.norm_type = str(defaults["norm_type"])
        self.resolution_set = int(defaults["resolution_set"])
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.save_depths = save_depths
        self.save_conf_threshold = save_conf_threshold
        self.scene_scale = scene_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = init_model_from_config(
            model_name=self.model_name,
            device=str(self.device),
            machine=machine,
            configs_dir=configs_dir,
            checkpoints_dir=checkpoints_dir,
        )
        self.model.eval()

        self.postprocess_parameters = dict(
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            use_multiview_confidence=use_multiview_confidence,
            multiview_conf_depth_abs_thresh=multiview_conf_depth_abs_thresh,
            multiview_conf_depth_rel_thresh=multiview_conf_depth_rel_thresh,
        )

    def to(self, device):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def __call__(
        self, image_path_list: List[str]
    ) -> Tuple[InitializedPointCloud, List[InitializingCamera]]:
        views, original_sizes = load_views(image_path_list, self.device, norm_type=self.norm_type, resolution_set=self.resolution_set)
        target_height, target_width = map(int, views[0]["true_shape"][0])

        with torch.no_grad():
            if self.use_amp and self.device.type == "cuda" and self.amp_dtype != "fp32":
                with torch.autocast(
                    "cuda",
                    dtype={
                        "bf16": torch.bfloat16,
                        "fp16": torch.float16,
                    }[self.amp_dtype],
                ):
                    raw_outputs = self.model(views)
            else:
                raw_outputs = self.model(views)

        outputs = postprocess_model_outputs_for_inference(
            raw_outputs,
            views,
            **self.postprocess_parameters,
        )

        all_points = []
        all_colors = []
        cameras = []

        for output, image_path, (original_width, original_height) in zip(
            outputs, image_path_list, original_sizes
        ):
            if "pts3d" not in output or "intrinsics" not in output or "camera_poses" not in output:
                raise RuntimeError(
                    f"External model '{self.model_name}' did not return enough geometry to initialize cameras"
                )

            pts3d = output["pts3d"][0].detach()
            depth_z = (
                output["depth_z"][0].detach()
                if "depth_z" in output
                else pts3d[..., 2:].detach()
            ).squeeze(-1)
            finite_mask = torch.isfinite(pts3d).all(dim=-1) & torch.isfinite(depth_z)
            valid_mask = finite_mask & (depth_z > 0)

            if "mask" in output:
                valid_mask = valid_mask & output["mask"][0].squeeze(-1).detach().bool()

            if "conf" in output:
                valid_mask = valid_mask & torch.isfinite(output["conf"][0]).detach()

            all_points.append(pts3d[valid_mask])

            img_no_norm = output["img_no_norm"][0].detach()
            img_uint8 = (img_no_norm.clamp(0.0, 1.0) * 255).to(torch.uint8)
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

            saved_depth_path = None
            if self.save_depths:
                saved_depth_path = save_resized_depth(
                    image_path=image_path,
                    depth=depth_z,
                    mask=output["mask"][0].squeeze(-1).detach() if "mask" in output else None,
                    conf=output["conf"][0].detach() if "conf" in output else None,
                    original_height=original_height,
                    original_width=original_width,
                    save_conf_threshold=self.save_conf_threshold,
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
                    depth_path=saved_depth_path,
                )
            )

        if len(all_points) == 0 or sum(len(points) for points in all_points) == 0:
            raise RuntimeError(
                f"External model '{self.model_name}' produced no valid 3D points"
            )

        points = torch.cat(all_points, dim=0)
        colors = torch.cat(all_colors, dim=0)

        return InitializedPointCloud(
            points=points * self.scene_scale,
            colors=colors / 255.0,
        ), cameras
