from .colmap import ColmapSparseInitializer, ColmapDenseInitializer
from .dust3r import Dust3rAlign2Initializer


# Dust3r align to Colmap dense

def Dust3rAlign2ColmapSparseInitializer(
        convert_image_path=lambda image_path: image_path,
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
        convert_image_path=lambda image_path: image_path,
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
