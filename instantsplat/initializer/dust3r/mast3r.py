import tempfile
import torch
from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment, SparseGA
from instantsplat.initializer.abc import AbstractInitializer, InitializingCamera, InitializedPointCloud

from .utils import load_images, focal2fov


def get_intrinsics(self: SparseGA, device):
    focals = self.get_focals()
    pps = self.get_principal_points()
    K = torch.zeros((len(focals), 3, 3), device=device)
    for i in range(len(focals)):
        K[i, 0, 0] = K[i, 1, 1] = focals[i]
        K[i, :2, 2] = pps[i]
        K[i, 2, 2] = 1
    return K


class Mast3rInitializer(AbstractInitializer):
    def __init__(self,
                 model_path: str = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                 coarse_lr: float = 0.07,
                 coarse_niter: int = 500,
                 fine_lr: float = 0.014,
                 fine_niter: int = 200,
                 min_conf_thr: float = 2.,
                 matching_conf_thr: float = 5.,
                 scene_scale: float = 10.0,
                 shared_intrinsics: bool = False,
                 resize: int = 512):
        self.coarse_lr = coarse_lr
        self.coarse_niter = coarse_niter
        self.fine_lr = fine_lr
        self.fine_niter = fine_niter
        self.min_conf_thr = min_conf_thr
        self.matching_conf_thr = matching_conf_thr
        self.scene_scale = scene_scale
        self.resize = resize
        self.shared_intrinsics = shared_intrinsics
        self.cache = tempfile.TemporaryDirectory()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AsymmetricMASt3R.from_pretrained(model_path).to(self.device)

    def __del__(self):
        self.cache.cleanup()

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
        scene = sparse_global_alignment(
            image_path_list, pairs, args.cache.name,
            model, lr1=args.coarse_lr, niter1=args.coarse_niter, lr2=args.fine_lr, niter2=args.fine_niter, device=device,
            opt_depth=True, shared_intrinsics=args.shared_intrinsics,
            matching_conf_thr=args.matching_conf_thr)

        imgs = [torch.from_numpy(img).to(device) for img in scene.imgs]
        poses = torch.linalg.inv(scene.get_im_poses().detach())
        pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
        confidence_masks = [(c > args.min_conf_thr) for c in confs]
        intrinsics = get_intrinsics(scene, device=device)
        #######################################################################################################################################
        return InitializedPointCloud(
            points=torch.concatenate([p.view(*m.shape, 3)[m] for p, m in zip(pts3d, confidence_masks)])*args.scene_scale,
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
