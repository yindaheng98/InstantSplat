import os

from .colmap import ColmapSparseInitializer, ColmapDenseInitializer, relative_image_names
from .dust3r import Dust3rAlign2Initializer
from .vggt import VGGTAlign2Initializer
from .vggttt import VGGTTTAlign2Initializer


def convert_image_path(image_path_list, destination):
    _, image_names = relative_image_names(image_path_list)
    return [os.path.join(destination, "images", image_name) for image_name in image_names]


# Dust3r align to Colmap dense

def Dust3rAlign2ColmapSparseInitializer(
        convert_image_path=convert_image_path,
        model_path: str = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        batch_size: int = 1,
        niter: int = 300,
        schedule: str = 'linear',
        lr: float = 0.01,
        focal_avg: bool = True,
        scene_scale: float = 1.0,
        resize: int = 512,
        *args, **kwargs):
    return Dust3rAlign2Initializer(
        ColmapSparseInitializer(*args, **kwargs),
        convert_image_path=convert_image_path,
        model_path=model_path,
        batch_size=batch_size,
        niter=niter,
        schedule=schedule,
        lr=lr,
        focal_avg=focal_avg,
        scene_scale=scene_scale,
        resize=resize,
    )


# Dust3r align to Colmap dense

def Dust3rAlign2ColmapDenseInitializer(
        convert_image_path=convert_image_path,
        model_path: str = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        batch_size: int = 1,
        niter: int = 300,
        schedule: str = 'linear',
        lr: float = 0.01,
        focal_avg: bool = True,
        scene_scale: float = 1.0,
        resize: int = 512,
        *args, **kwargs):
    return Dust3rAlign2Initializer(
        ColmapDenseInitializer(*args, **kwargs),
        convert_image_path=convert_image_path,
        model_path=model_path,
        batch_size=batch_size,
        niter=niter,
        schedule=schedule,
        lr=lr,
        focal_avg=focal_avg,
        scene_scale=scene_scale,
        resize=resize,
    )


def VGGTAlign2ColmapSparseInitializer(
        convert_image_path=convert_image_path,
        model_url: str = "checkpoints/vggt_1B_commercial.pt",
        img_load_resolution: int = 1024,
        conf_thres_value: float = 5.0,
        scene_scale: float = 1.0,
        update_camera: bool = False,
        min_baseline_ratio: float = 1e-3,
        *args, **kwargs):
    return VGGTAlign2Initializer(
        ColmapSparseInitializer(*args, **kwargs),
        convert_image_path=convert_image_path,
        update_camera=update_camera,
        min_baseline_ratio=min_baseline_ratio,
        model_url=model_url,
        img_load_resolution=img_load_resolution,
        conf_thres_value=conf_thres_value,
        scene_scale=scene_scale,
    )


def VGGTAlign2ColmapDenseInitializer(
        convert_image_path=convert_image_path,
        model_url: str = "checkpoints/vggt_1B_commercial.pt",
        img_load_resolution: int = 1024,
        conf_thres_value: float = 5.0,
        scene_scale: float = 1.0,
        update_camera: bool = False,
        min_baseline_ratio: float = 1e-3,
        *args, **kwargs):
    return VGGTAlign2Initializer(
        ColmapDenseInitializer(*args, **kwargs),
        convert_image_path=convert_image_path,
        update_camera=update_camera,
        min_baseline_ratio=min_baseline_ratio,
        model_url=model_url,
        img_load_resolution=img_load_resolution,
        conf_thres_value=conf_thres_value,
        scene_scale=scene_scale,
    )


def VGGTTTAlign2ColmapSparseInitializer(
        convert_image_path=convert_image_path,
        model_url: str = "nvidia/vgg-ttt",
        conf_thres_value: float = 5.0,
        scene_scale: float = 1.0,
        num_ttt_steps: int | None = 2,
        memory_efficient_inference: bool = True,
        use_global_pred: bool = True,
        update_camera: bool = False,
        min_baseline_ratio: float = 1e-3,
        *args, **kwargs):
    return VGGTTTAlign2Initializer(
        ColmapSparseInitializer(*args, **kwargs),
        convert_image_path=convert_image_path,
        update_camera=update_camera,
        min_baseline_ratio=min_baseline_ratio,
        model_url=model_url,
        conf_thres_value=conf_thres_value,
        scene_scale=scene_scale,
        num_ttt_steps=num_ttt_steps,
        memory_efficient_inference=memory_efficient_inference,
        use_global_pred=use_global_pred,
    )


def VGGTTTAlign2ColmapDenseInitializer(
        convert_image_path=convert_image_path,
        model_url: str = "nvidia/vgg-ttt",
        conf_thres_value: float = 5.0,
        scene_scale: float = 1.0,
        num_ttt_steps: int | None = 2,
        memory_efficient_inference: bool = True,
        use_global_pred: bool = True,
        update_camera: bool = False,
        min_baseline_ratio: float = 1e-3,
        *args, **kwargs):
    return VGGTTTAlign2Initializer(
        ColmapDenseInitializer(*args, **kwargs),
        convert_image_path=convert_image_path,
        update_camera=update_camera,
        min_baseline_ratio=min_baseline_ratio,
        model_url=model_url,
        conf_thres_value=conf_thres_value,
        scene_scale=scene_scale,
        num_ttt_steps=num_ttt_steps,
        memory_efficient_inference=memory_efficient_inference,
        use_global_pred=use_global_pred,
    )
