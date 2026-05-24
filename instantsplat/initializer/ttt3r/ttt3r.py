from math import atan
from pathlib import Path
from typing import List

import torch

from .bootstrap import ensure_runtime_paths
from .export import export_outputs, prepare_input

from instantsplat.initializer.abc import (
    AbstractInitializer,
    InitializedPointCloud,
    InitializingCamera,
)


def focal2fov(focal: float, pixels: float) -> float:
    return 2 * atan(pixels / (2 * focal))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class Ttt3rInitializer(AbstractInitializer):
    def __init__(
        self,
        model_path: str = "checkpoints/cut3r_512_dpt_4_64.pth",
        model_update_type: str = "ttt3r",
        min_conf_thr: float = 1.5,
        scene_scale: float = 1.0,
        resize: int = 512,
        reset_interval: int = 1000000,
    ):
        self.model_path = model_path
        self.model_update_type = model_update_type
        self.min_conf_thr = min_conf_thr
        self.scene_scale = scene_scale
        self.resize = resize
        self.reset_interval = reset_interval
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_path(self, path: str, root: Path | None = None) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        base = root if root is not None else _project_root()
        return (base / candidate).resolve()

    def to(self, device):
        self.device = str(device)
        return self

    def __call__(self, image_path_list: List[str]):
        ensure_runtime_paths()

        model_path = self._resolve_path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"TTT3R checkpoint not found at '{model_path}'."
            )

        from .inference import inference_recurrent_lighter
        from .model import ARCroco3DStereo

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        model = ARCroco3DStereo.from_pretrained(str(model_path)).to(device)
        model.config.model_update_type = self.model_update_type
        model.eval()

        views = prepare_input(image_path_list, self.resize, self.reset_interval)
        outputs, _ = inference_recurrent_lighter(views, model, device, verbose=False)
        exported = export_outputs(outputs, image_path_list, self.min_conf_thr)

        points = torch.from_numpy(exported["points"]).float() * self.scene_scale
        colors = torch.from_numpy(exported["colors"]).float()
        world2cams = torch.from_numpy(exported["world2cams"]).float()
        focals = exported["focals"]
        principal_points = exported["principal_points"]
        image_widths = exported["image_widths"]
        image_heights = exported["image_heights"]
        exported_paths = exported["image_paths"].tolist()

        cameras = []
        for world2cam, focal, principal_point, image_path, image_width, image_height in zip(
            world2cams,
            focals,
            principal_points,
            exported_paths,
            image_widths,
            image_heights,
        ):
            cameras.append(
                InitializingCamera(
                    image_width=int(image_width),
                    image_height=int(image_height),
                    FoVx=focal2fov(float(focal), float(principal_point[0]) * 2.0),
                    FoVy=focal2fov(float(focal), float(principal_point[1]) * 2.0),
                    R=world2cam[:3, :3],
                    T=world2cam[:3, 3] * self.scene_scale,
                    image_path=str(image_path),
                )
            )

        return InitializedPointCloud(points=points, colors=colors), cameras
