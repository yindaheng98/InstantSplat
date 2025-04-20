import os
import math
from typing import List
import cv2
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
from dust3r.utils.image import _resize_pil_image


def load_images(img_path_list: List[str], size: int = None):
    """ open and convert all images in a list or folder to proper input format for DUSt3R"""
    imgs, sizes = [], []
    for path in img_path_list:
        img = exif_transpose(PIL.Image.open(os.path.join(path))).convert('RGB')
        sizes.append(img.size)
        W1, H1 = img.size
        img = _resize_pil_image(img, size)
        W, H = img.size
        W2 = W//16*16
        H2 = H//16*16
        img = np.array(img)
        img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_LINEAR)
        img = PIL.Image.fromarray(img)

        print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs, sizes


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
