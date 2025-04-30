from typing import List
import torch
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .utils import load_images, focal2fov, fov2focal
from .alignment import compute_global_alignment


def preset_cameras(scene, known_cameras: List[InitializingCamera]):
    Rt = torch.zeros((len(known_cameras), 4, 4), device=known_cameras[0].R.device)
    focal = torch.zeros(len(known_cameras), device=known_cameras[0].R.device)
    for i, camera in enumerate(known_cameras):
        Rt[i, :3, :3] = camera.R
        Rt[i, :3, 3] = camera.T.to(camera.R.device)
        Rt[i, 3, 3] = 1.0
        height, width = scene.imshapes[i]
        fx = fov2focal(camera.FoVx, width)
        fy = fov2focal(camera.FoVy, height)
        focal[i] = (fx + fy) / 2
    C2W = torch.linalg.inv(Rt)
    scene.preset_pose(C2W)
    scene.preset_focal(known_focals=focal.tolist())
    return scene


class Dust3rInitializer(AbstractInitializer):
    def __init__(self,
                 model_path: str = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
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
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(self.device)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(args, image_path_list, known_cameras: List[InitializingCamera] = []):
        device = args.device
        images, ori_sizes = load_images(image_path_list, size=args.resize)
        model = args.model
        #######################################################################################################################################
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=args.batch_size)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        init = "mst"
        if len(known_cameras) > 0:
            scene = preset_cameras(scene, known_cameras)
            init = "known_poses"
        loss = compute_global_alignment(scene=scene, init=init, niter=args.niter, schedule=args.schedule, lr=args.lr, focal_avg=args.focal_avg)
        scene = scene.clean_pointcloud()

        imgs = [torch.from_numpy(img).to(device) for img in scene.imgs]
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


class Dust3rAlign2Initializer(Dust3rInitializer):
    def __init__(self, another_initializer: AbstractInitializer, *args, convert_image_path=lambda image_path: image_path, update_camera=False, scene_scale=1., **kwargs):
        super().__init__(*args, scene_scale=scene_scale, **kwargs)
        self.convert_image_path = convert_image_path
        self.update_camera = update_camera
        self.another_initializer = another_initializer

    def __call__(self, image_path_list):
        another_point_cloud, another_cameras = self.another_initializer([self.convert_image_path(image_path) for image_path in image_path_list])
        point_cloud, cameras = super().__call__([camera.image_path for camera in another_cameras], known_cameras=another_cameras)
        return InitializedPointCloud(
            points=torch.concatenate((point_cloud.points, another_point_cloud.points*self.scene_scale)),
            colors=torch.concatenate((point_cloud.colors, another_point_cloud.colors))
        ), (cameras if self.update_camera else another_cameras)
