from .colmap import ColmapDenseInitializer
from .dust3r import Dust3rInitializer
from .dust3r import Mast3rInitializer
from .align import AlignInitializer


def Dust3rAlign2ColmapDenseInitializer(
        model_path: str = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        batch_size: int = 1,
        niter: int = 300,
        schedule: str = 'linear',
        lr: float = 0.01,
        focal_avg: bool = True,
        scene_scale: float = 10.0,
        resize: int = 512,
        *args, **kwargs):
    return AlignInitializer(
        ColmapDenseInitializer(*args, **kwargs),
        Dust3rInitializer(
            model_path=model_path,
            batch_size=batch_size,
            niter=niter,
            schedule=schedule,
            lr=lr,
            focal_avg=focal_avg,
            scene_scale=scene_scale,
            resize=resize,
        )
    )


def Mast3rAlign2ColmapDenseInitializer(
        model_path: str = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        coarse_lr: float = 0.07,
        coarse_niter: int = 500,
        fine_lr: float = 0.014,
        fine_niter: int = 200,
        min_conf_thr: float = 2.,
        matching_conf_thr: float = 5.,
        scene_scale: float = 10.0,
        shared_intrinsics: bool = False,
        resize: int = 512,
        *args, **kwargs):
    return AlignInitializer(
        ColmapDenseInitializer(*args, **kwargs),
        Mast3rInitializer(
            model_path=model_path,
            coarse_lr=coarse_lr,
            coarse_niter=coarse_niter,
            fine_lr=fine_lr,
            fine_niter=fine_niter,
            min_conf_thr=min_conf_thr,
            matching_conf_thr=matching_conf_thr,
            scene_scale=scene_scale,
            shared_intrinsics=shared_intrinsics,
            resize=resize,
        )
    )
