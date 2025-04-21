from .abc import AbstractInitializer
from .autoscale import AutoScaleDepthInitializerWrapper
from .depth_anything_v2 import DepthAnythingV2InitializerWrapper


def AutoScaleDepthAnythingV2InitializerWrapper(base_initializer: AbstractInitializer, *args, **kwargs):
    """
    Combines the AutoScaleDepthInitializerWrapper and DepthAnythingV2InitializerWrapper.
    """
    return AutoScaleDepthInitializerWrapper(
        base_initializer_wrapper=DepthAnythingV2InitializerWrapper(base_initializer, *args, **kwargs)
    )
