import json
import os
import random
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset.colmap import ColmapTrainableCameraDataset, colmap_compute_scene_extent
from gaussian_splatting.trainer import CameraTrainer

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", default=30000, type=int)
parser.add_argument("-l", "--load_ply", default=None, type=str)
parser.add_argument("--load_camera", default=None, type=str)
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--config", default=None, type=str)


def init_from_ply(gaussian: GaussianModel, path):
    import numpy as np
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    gaussian.create_from_pcd(torch.from_numpy(xyz), torch.from_numpy(rgb / 255.0))


def init_gaussians(sh_degree: int, source: str, device: str, load_ply: str = None, load_camera: str = None, configs={}):
    gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
    gaussians.load_ply(load_ply) if load_ply else init_from_ply(gaussians, os.path.join(source, "sparse/0", "points3D.ply"))
    dataset = (TrainableCameraDataset.from_json(load_camera) if load_camera else ColmapTrainableCameraDataset(source)).to(device)
    trainer = CameraTrainer(
        gaussians,
        scene_extent=colmap_compute_scene_extent(dataset),
        dataset=dataset,
        **configs
    )
    if load_ply:
        gaussians.activate_all_sh_degree()
    return dataset, gaussians, trainer


def read_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


default_configs = dict(
    opacity_lr=0.05,
    camera_position_lr_init=0.001,
    camera_position_lr_final=0.00001,
    camera_position_lr_delay_mult=0.01,
    camera_position_lr_max_steps=1000,
    camera_rotation_lr_init=0.0001,
    camera_rotation_lr_final=0.000001,
    camera_rotation_lr_delay_mult=0.01,
    camera_rotation_lr_max_steps=1000,
)


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    os.makedirs(destination, exist_ok=True)
    configs = default_configs if args.config is None else read_config(args.config)
    dataset, gaussians, trainer = init_gaussians(
        sh_degree=sh_degree, source=source, device=device,
        load_ply=args.load_ply, load_camera=args.load_camera, configs=configs)
    dataset.save_cameras(os.path.join(destination, "cameras_orig.json"))

    pbar = tqdm(range(1, iteration+1))
    epoch = list(range(len(dataset)))
    epoch_psnr = torch.empty(3, 0, device=device)
    ema_loss_for_log = 0.0
    avg_psnr_for_log = 0.0
    for step in pbar:
        epoch_idx = step % len(dataset)
        if epoch_idx == 0:
            avg_psnr_for_log = epoch_psnr.mean().item()
            epoch_psnr = torch.empty(3, 0, device=device)
            random.shuffle(epoch)
        idx = epoch[epoch_idx]
        loss, out = trainer.step(dataset[idx])
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            epoch_psnr = torch.concat([epoch_psnr, psnr(out["render"], dataset[idx].ground_truth_image)], dim=1)
            if step % 10 == 0:
                pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log, 'n': gaussians._xyz.shape[0]})
        if step in args.save_iterations:
            save_path = os.path.join(destination, "point_cloud", "iteration_" + str(step))
            os.makedirs(save_path, exist_ok=True)
            gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
            dataset.save_cameras(os.path.join(destination, "cameras.json"))
    save_path = os.path.join(destination, "point_cloud", "iteration_" + str(iteration))
    os.makedirs(save_path, exist_ok=True)
    gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
    dataset.save_cameras(os.path.join(destination, "cameras.json"))


if __name__ == "__main__":
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(False)
    main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
