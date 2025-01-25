import os
import math
from typing import List
import cv2
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn as nn
import torchvision.transforms as tvf
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import _resize_pil_image
from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .alignment import compute_global_alignment


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


class Xst3rInitializer(AbstractInitializer):
    def __init__(self,
                 model: nn.Module,
                 batch_size: int = 1,
                 niter: int = 300,
                 schedule: str = 'linear',
                 lr: float = 0.01,
                 focal_avg: bool = True,
                 scene_scale: float = 10.0,
                 resize: int = 512):
        self.batch_size = batch_size
        self.niter = niter
        self.schedule = schedule
        self.lr = lr
        self.focal_avg = focal_avg
        self.scene_scale = scene_scale
        self.resize = resize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(args, image_path_list):
        device = args.device
        images, ori_sizes = load_images(image_path_list, size=args.resize)
        model = args.model
        #######################################################################################################################################
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=args.batch_size)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = compute_global_alignment(scene=scene, init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr, focal_avg=args.focal_avg)
        scene = scene.clean_pointcloud()

        imgs = [torch.from_numpy(img).to(device) for img in scene.imgs]
        focals = scene.get_focals()
        poses = torch.linalg.inv(scene.get_im_poses().detach())
        pts3d = [p.detach() for p in scene.get_pts3d()]
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
        confidence_masks = scene.get_masks()
        intrinsics = scene.get_intrinsics()
        #######################################################################################################################################
        return InitializedPointCloud(
            points=torch.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])*args.scene_scale,
            colors=torch.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
        ), [
            InitializingCamera(
                image_width=ori_sizes[i][0], image_height=ori_sizes[i][1],
                FoVx=focal2fov(intrinsics[i][0, 0], intrinsics[i][0, 2]*2),
                FoVy=focal2fov(intrinsics[i][1, 1], intrinsics[i][1, 2]*2),
                R=poses[i][:3, :3], T=poses[i][:3, 3]*args.scene_scale,
                image_path=image_path_list[i]
            )
            for i in range(len(image_path_list))
        ]


class Dust3rInitializer(Xst3rInitializer):
    def __init__(self,
                 model_path: str = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", **kwargs):
        super().__init__(model=AsymmetricCroCo3DStereo.from_pretrained(model_path), **kwargs)


class Mast3rInitializer(Xst3rInitializer):
    def __init__(self, model_path: str = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512.pth", **kwargs):
        super().__init__(model=AsymmetricMASt3R.from_pretrained(model_path), **kwargs)
