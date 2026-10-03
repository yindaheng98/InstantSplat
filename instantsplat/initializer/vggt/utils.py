import math


# Copy from vggt.utils
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
