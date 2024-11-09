from typing import NamedTuple
import torch
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from instant_splat.utils.dust3r_utils import compute_global_alignment
from .abc import AbstractInitializer, InitializingCamera, InitializedPointCloud
from .utils import load_images, focal2fov


class Dust3rInitializerParameter(NamedTuple):
    model_path: str
    batch_size: int = 1
    niter: int = 300
    schedule: str = 'linear'
    lr: float = 0.01
    focal_avg: bool = True
    device: torch.device = 'cuda'


class Dust3rInitializer(AbstractInitializer, Dust3rInitializerParameter):

    def __call__(args, image_path_list):
        device = args.device
        images, ori_sizes = load_images(image_path_list, size=512)
        model = AsymmetricCroCo3DStereo.from_pretrained(args.model_path).to(device)
        #######################################################################################################################################
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=args.batch_size)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = compute_global_alignment(scene=scene, init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr, focal_avg=args.focal_avg)
        scene = scene.clean_pointcloud()

        imgs = [torch.from_numpy(img).to(device) for img in scene.imgs]
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
        confidence_masks = scene.get_masks()
        intrinsics = scene.get_intrinsics()
        #######################################################################################################################################
        return InitializedPointCloud(
            points=torch.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)]),
            colors=torch.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
        ), [
            InitializingCamera(
                image_width=ori_sizes[i][0], image_height=ori_sizes[i][1],
                FoVx=focal2fov(intrinsics[i][0, 0], intrinsics[i][0, 2]*2),
                FoVy=focal2fov(intrinsics[i][1, 1], intrinsics[i][1, 2]*2),
                R=poses[i][:3, :3], T=poses[i][:3, 3],
                image_path=image_path_list[i]
            )
            for i in range(len(image_path_list))
        ]
