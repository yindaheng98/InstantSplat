from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import BaseCameraTrainer, CameraTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper


def BaseTrainer(
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
        model, scene_extent, dataset,
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
        depth_from_iter=0,
        depth_l1_weight_max_steps=1000,
        *args, **kwargs):
    return CameraTrainer(
        model, scene_extent, dataset,
        opacity_lr=opacity_lr,
        position_lr_max_steps=position_lr_max_steps,
        depth_from_iter=depth_from_iter,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps,
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


def BaseScaleRegularizeTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        scale_reg_from_iter=100,
        *args, **kwargs):
    return ScaleRegularizeTrainerWrapper(
        BaseTrainer,
        model, scene_extent, dataset,
        *args,
        scale_reg_from_iter=scale_reg_from_iter,
        **kwargs)


def ScaleRegularizeTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        scale_reg_from_iter=100,
        *args, **kwargs):
    return ScaleRegularizeTrainerWrapper(
        Trainer,
        model, scene_extent, dataset,
        *args,
        scale_reg_from_iter=scale_reg_from_iter,
        **kwargs)
