from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import BaseCameraTrainer


def Trainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        camera_position_lr_init=0.001,
        camera_position_lr_final=0.00001,
        camera_position_lr_delay_mult=0.01,
        camera_position_lr_max_steps=1000,
        camera_rotation_lr_init=0.0001,
        camera_rotation_lr_final=0.000001,
        camera_rotation_lr_delay_mult=0.01,
        camera_rotation_lr_max_steps=1000,
        opacity_lr=0.05,
        position_lr_max_steps=1000,
        *args, **kwargs):
    return BaseCameraTrainer(
        model=model,
        scene_extent=scene_extent,
        dataset=dataset,
        opacity_lr=opacity_lr,
        position_lr_max_steps=position_lr_max_steps,
        camera_position_lr_init=camera_position_lr_init,
        camera_position_lr_final=camera_position_lr_final,
        camera_position_lr_delay_mult=camera_position_lr_delay_mult,
        camera_position_lr_max_steps=camera_position_lr_max_steps,
        camera_rotation_lr_init=camera_rotation_lr_init,
        camera_rotation_lr_final=camera_rotation_lr_final,
        camera_rotation_lr_delay_mult=camera_rotation_lr_delay_mult,
        camera_rotation_lr_max_steps=camera_rotation_lr_max_steps,
        *args, **kwargs
    )
