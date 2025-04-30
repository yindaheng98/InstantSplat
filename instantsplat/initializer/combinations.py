from .colmap import ColmapSparseInitializer, ColmapDenseInitializer
from .dust3r import Dust3rInitializer, Dust3rAlign2Initializer, Mast3rInitializer
from .depth import AutoScaleDepthAnythingV2InitializerWrapper


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


def DepthAnythingV2ColmapSparseInitializer(
        input_size=518,
        encoder='vitl',
        checkpoints_folder='checkpoints',
        device="cuda",
        *args, **kwargs):
    return AutoScaleDepthAnythingV2InitializerWrapper(
        ColmapSparseInitializer(*args, **kwargs),
        input_size=input_size,
        encoder=encoder,
        checkpoints_folder=checkpoints_folder,
        device=device,
    )


def DepthAnythingV2ColmapDenseInitializer(
        input_size=518,
        encoder='vitl',
        checkpoints_folder='checkpoints',
        device="cuda",
        *args, **kwargs):
    return AutoScaleDepthAnythingV2InitializerWrapper(
        ColmapDenseInitializer(*args, **kwargs),
        input_size=input_size,
        encoder=encoder,
        checkpoints_folder=checkpoints_folder,
        device=device,
    )


def DepthAnythingV2Dust3rInitializer(
        input_size=518,
        encoder='vitl',
        checkpoints_folder='checkpoints',
        device="cuda",
        *args, **kwargs):
    return AutoScaleDepthAnythingV2InitializerWrapper(
        Dust3rInitializer(*args, **kwargs),
        input_size=input_size,
        encoder=encoder,
        checkpoints_folder=checkpoints_folder,
        device=device,
    )


def DepthAnythingV2Mast3rInitializer(
        input_size=518,
        encoder='vitl',
        checkpoints_folder='checkpoints',
        device="cuda",
        *args, **kwargs):
    return AutoScaleDepthAnythingV2InitializerWrapper(
        Mast3rInitializer(*args, **kwargs),
        input_size=input_size,
        encoder=encoder,
        checkpoints_folder=checkpoints_folder,
        device=device,
    )


def DepthAnythingV2Dust3rAlign2ColmapDenseInitializer(
        input_size=518,
        encoder='vitl',
        checkpoints_folder='checkpoints',
        device="cuda",
        *args, **kwargs):
    return AutoScaleDepthAnythingV2InitializerWrapper(
        Dust3rAlign2ColmapDenseInitializer(*args, **kwargs),
        input_size=input_size,
        encoder=encoder,
        checkpoints_folder=checkpoints_folder,
        device=device,
    )
