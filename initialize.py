import json
import os
from argparse import ArgumentParser

from instantsplat.initializer import Dust3rInitializer, InitializedCameraDataset


parser = ArgumentParser()
parser.add_argument("-i", "--initializer", choices=["dust3r", "colmap"], default="dust3r", type=str)
parser.add_argument("-d", "--directory", required=True, type=str)
parser.add_argument("-f", "--image_folder", default=None, type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--config", default=None, type=str)


def read_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


if __name__ == '__main__':

    args = parser.parse_args()
    image_folder = args.image_folder or os.path.join(args.directory, "images")
    configs = {} if args.config is None else read_config(args.config)
    image_path_list = [os.path.join(image_folder, file) for file in sorted(os.listdir(image_folder))]
    if args.initializer == "dust3r":
        initializer = Dust3rInitializer(**configs).to(args.device)
        initialized_point_cloud, initialized_cameras = initializer(image_path_list=image_path_list)
    else:
        raise ValueError(f"Unknown initializer {args.initializer}")
    dataset = InitializedCameraDataset(initialized_cameras)

    os.makedirs(os.path.join(args.directory, "sparse/0"), exist_ok=True)
    initialized_point_cloud.save_ply(os.path.join(args.directory, "sparse/0/points3D.ply"))
    dataset.save_colmap_cameras(os.path.join(args.directory, "sparse/0"))
