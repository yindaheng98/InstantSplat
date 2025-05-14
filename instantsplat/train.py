import os
import shutil
from typing import Tuple
import torch
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractTrainer
from gaussian_splatting.dataset.colmap import ColmapTrainableCameraDataset, colmap_init
from gaussian_splatting.train import save_cfg_args, training
from instantsplat.trainer import Trainer, BaseTrainer
from instantsplat.trainer import ScaleRegularizeTrainer, BaseScaleRegularizeTrainer
from instantsplat.initializer import TrainableInitializedCameraDataset
from instantsplat.initialize import initialize, default_image_folder

basemodes = {
    "base": Trainer,
    "nodepth": BaseTrainer,
}
scaleregmodes = {
    "base": ScaleRegularizeTrainer,
    "nodepth": BaseScaleRegularizeTrainer,
}


def prepare_training(sh_degree: int, source: str, destination: str, device: str, mode: str, load_ply: str = None, load_camera: str = None, load_depth=False, with_scale_reg=False, configs={}, init=None, init_configs={}) -> Tuple[CameraDataset, GaussianModel, AbstractTrainer]:
    gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
    if init:  # initialize
        initialized_cameras, initialized_point_cloud = initialize(initializer=init, directory=source, configs=init_configs, device=device)
        dataset = TrainableInitializedCameraDataset(initialized_cameras).to(device)
        gaussians.create_from_pcd(initialized_point_cloud.points, initialized_point_cloud.colors)
        if os.path.exists(os.path.join(destination, "input.ply")):
            os.remove(os.path.join(destination, "input.ply"))
        initialized_point_cloud.save_ply(os.path.join(destination, "input.ply"))
    else:  # create_from_pcd
        dataset = (TrainableCameraDataset.from_json(load_camera, load_depth=load_depth) if load_camera else ColmapTrainableCameraDataset(source, load_depth=load_depth)).to(device)
        colmap_init(gaussians, source) if not load_ply else gaussians.load_ply(load_ply)
        if os.path.exists(os.path.join(destination, "input.ply")):
            os.remove(os.path.join(destination, "input.ply"))
        shutil.copy2(os.path.join(source, "sparse/0", "points3D.ply"), os.path.join(destination, "input.ply"))

    modes = basemodes if not with_scale_reg else scaleregmodes
    trainer = modes[mode](
        gaussians,
        scene_extent=dataset.scene_extent(),
        dataset=dataset,
        **configs
    )
    return dataset, gaussians, trainer


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", default=1000, type=int)
    parser.add_argument("-l", "--load_ply", default=None, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--no_depth_data", action='store_true')
    parser.add_argument("--with_scale_reg", action="store_true")
    parser.add_argument("--mode", choices=sorted(list(set(list(basemodes.keys()) + list(scaleregmodes.keys())))), default="base")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("--init", choices=list(default_image_folder.keys()), default=None, type=str)
    parser.add_argument("--init_option", default=[], action='append', type=str)
    args = parser.parse_args()
    save_cfg_args(args.destination, args.sh_degree, args.source)
    torch.autograd.set_detect_anomaly(False)

    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    init_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.init_option}
    dataset, gaussians, trainer = prepare_training(
        sh_degree=args.sh_degree, source=args.source, destination=args.destination, device=args.device, mode=args.mode,
        load_ply=args.load_ply, load_camera=args.load_camera, load_depth=not args.no_depth_data, with_scale_reg=args.with_scale_reg, configs=configs,
        init=args.init, init_configs=init_configs)
    dataset.save_cameras(os.path.join(args.destination, "cameras.json"))
    training(
        dataset=dataset, gaussians=gaussians, trainer=trainer,
        destination=args.destination, iteration=args.iteration, save_iterations=args.save_iterations,
        device=args.device)
