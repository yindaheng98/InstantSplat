import os
import shutil

from instantsplat.initializer import *

default_image_folder = {
    "dust3r": "images",
    "mast3r": "images",
    "colmap-sparse": "input",
    "colmap-dense": "input",
    "colmap-dense-dust3r": "input",
    "colmap-dense-mast3r": "input",
}


def initialize(initializer, directory, configs, device):
    image_folder = os.path.join(directory, default_image_folder[initializer])
    image_path_list = [os.path.join(image_folder, file) for file in sorted(os.listdir(image_folder))]
    if initializer == "dust3r":
        initializer = Dust3rInitializer(**configs).to(device)
    elif initializer == "mast3r":
        initializer = Mast3rInitializer(**configs).to(device)
    elif initializer == "colmap-sparse":
        initializer = ColmapSparseInitializer(destination=directory, **configs).to(device)
    elif initializer == "colmap-dense":
        initializer = ColmapDenseInitializer(destination=directory, **configs).to(device)
    elif initializer == "colmap-dense-dust3r":
        initializer = Dust3rAlign2ColmapDenseInitializer(destination=directory, **configs).to(device)
    elif initializer == "colmap-dense-mast3r":
        initializer = Mast3rAlign2ColmapDenseInitializer(destination=directory, **configs).to(device)
    else:
        raise ValueError(f"Unknown initializer {initializer}")
    initialized_point_cloud, initialized_cameras = initializer(image_path_list=image_path_list)
    return initialized_cameras, initialized_point_cloud


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--initializer", choices=list(default_image_folder.keys()), default="dust3r", type=str)
    parser.add_argument("-d", "--directory", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)

    args = parser.parse_args()
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    initialized_cameras, initialized_point_cloud = initialize(args.initializer, args.directory, configs, args.device)
    dataset = InitializedCameraDataset(initialized_cameras)

    shutil.rmtree(os.path.join(args.directory, "sparse/0"), ignore_errors=True)
    os.makedirs(os.path.join(args.directory, "sparse/0"), exist_ok=True)
    initialized_point_cloud.save_ply(os.path.join(args.directory, "sparse/0/points3D.ply"))
    dataset.save_colmap_cameras(os.path.join(args.directory, "sparse/0"))
