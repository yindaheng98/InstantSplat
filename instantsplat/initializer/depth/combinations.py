from typing import Callable

from .abc import AbstractInitializer
from .autoscale import AutoScaleDepthInitializerWrapper
from .depth_anything_v2 import DepthAnythingV2InitializerWrapper


def AutoScaleDepthAnythingV2InitializerWrapper(
        base_initializer_constructor: Callable[..., AbstractInitializer],
        *args,
        input_size=518,
        encoder='vitl',
        checkpoints_folder='checkpoints',
        device="cuda",
        **configs):
    return AutoScaleDepthInitializerWrapper(
        base_initializer_wrapper=DepthAnythingV2InitializerWrapper(
            base_initializer_constructor(*args, **configs),
            input_size=input_size,
            encoder=encoder,
            checkpoints_folder=checkpoints_folder,
            device=device,
        )
    )
