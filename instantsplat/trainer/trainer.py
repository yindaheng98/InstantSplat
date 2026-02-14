from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import BaseCameraTrainer, CameraTrainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper


def BaseTrainer(
        model: CameraTrainableGaussianModel,
        dataset: TrainableCameraDataset,
        camera_position_lr_init=0.001,
        camera_position_lr_final=0.00001,
        camera_position_lr_delay_mult=0.01,
        camera_position_lr_max_steps=1000,
        camera_rotation_lr_init=0.0001,
        camera_rotation_lr_final=0.000001,
        camera_rotation_lr_delay_mult=0.01,
        camera_rotation_lr_max_steps=1000,
        camera_exposure_lr_max_steps=1000,
        opacity_lr=0.05,
        position_lr_max_steps=1000,
        **configs):
    return BaseCameraTrainer(
        model, dataset,
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
        camera_exposure_lr_max_steps=camera_exposure_lr_max_steps,
        **configs
    )


def Trainer(
        model: CameraTrainableGaussianModel,
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
        **configs):
    return CameraTrainer(
        model, dataset,
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
        **configs
    )


def BaseScaleRegularizeTrainer(
        model: CameraTrainableGaussianModel,
        dataset: TrainableCameraDataset,
        scale_reg_from_iter=100,
        **configs):
    return ScaleRegularizeTrainerWrapper(
        BaseTrainer,
        model, dataset,
        scale_reg_from_iter=scale_reg_from_iter,
        **configs)


def ScaleRegularizeTrainer(
        model: CameraTrainableGaussianModel,
        dataset: TrainableCameraDataset,
        scale_reg_from_iter=100,
        **configs):
    return ScaleRegularizeTrainerWrapper(
        Trainer,
        model, dataset,
        scale_reg_from_iter=scale_reg_from_iter,
        **configs)
