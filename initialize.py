import os
import shutil
import re
from typing import List

from tqdm import tqdm

from instantsplat.initialize import *

default_image_folder = {
    "colmap-sparse": "input",
    "colmap-dense": "input",
}


def initialize(initializer, directory, configs, device):
    image_folder = os.path.join(directory, default_image_folder[initializer])
    image_path_list = [os.path.join(image_folder, file) for file in sorted(os.listdir(image_folder))]
    match initializer:
        case "colmap-sparse":
            initializer = ColmapSparseInitializer(destination=directory, **configs).to(device)
        case "colmap-dense":
            initializer = ColmapDenseInitializer(destination=directory, **configs).to(device)
        case _:
            raise ValueError(f"Unknown initializer {initializer}")
    initialized_point_cloud, initialized_cameras = initializer(image_path_list=image_path_list)
    return initialized_cameras, initialized_point_cloud


default_image_folder_reinit = {
    "dust3r": "images",
}


def reinitialize(initializer, directory, configs, device, known_cameras: List[InitializingCamera] = []):
    image_folder = os.path.join(directory, default_image_folder_reinit[initializer])
    image_path_list = [os.path.join(image_folder, file) for file in sorted(os.listdir(image_folder))]
    initializer = Dust3rInitializer(**configs).to(device)
    initialized_point_cloud, initialized_cameras = initializer(image_path_list=image_path_list, known_cameras=known_cameras)
    return initialized_cameras, initialized_point_cloud


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--initializer", choices=list(default_image_folder.keys()), default="colmap-dense", type=str)
    parser.add_argument("-d", "--directory", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)
    parser.add_argument("-r", "--option_reinit", default=[], action='append', type=str)
    parser.add_argument("--load_camera", default=None, type=str)

    args = parser.parse_args()
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    initialized_cameras, initialized_point_cloud = initialize(args.initializer, args.directory, configs, args.device)
    dataset = InitializedCameraDataset(initialized_cameras)

    shutil.rmtree(os.path.join(args.directory, "sparse/0"), ignore_errors=True)
    os.makedirs(os.path.join(args.directory, "sparse/0"), exist_ok=True)
    initialized_point_cloud.save_ply(os.path.join(args.directory, "sparse/0/points3D.ply"))
    dataset.save_colmap_cameras(os.path.join(args.directory, "sparse/0"))

    refs = TrainableCameraDataset.from_json(args.load_camera).to(args.device) if args.load_camera else None

    frames = {}
    for init_camera in initialized_cameras:
        print(os.path.basename(init_camera.image_path))
        filename = os.path.basename(init_camera.image_path)
        m = re.findall(r'^frame([0-9]+)_image[0-9]+\.[a-z]+$', filename)
        if len(m) != 1:
            raise ValueError(f"Cannot parse frame index from {filename}")
        frame_idx = int(m[0])
        if frame_idx not in frames:
            frames[frame_idx] = []
        frames[frame_idx].append(init_camera)

    configs_reinit = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_reinit}
    for frame_idx, initialized_cameras in tqdm(frames.items(), desc="Writing frames"):
        directory = os.path.join(args.directory, f"frame{frame_idx}")
        shutil.rmtree(directory, ignore_errors=True)
        for init_camera in initialized_cameras:
            path = os.path.join(directory, os.path.relpath(init_camera.image_path, args.directory))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copy2(init_camera.image_path, path)
        dataset = InitializedCameraDataset(initialized_cameras)
        os.makedirs(os.path.join(directory, "sparse/0"), exist_ok=True)
        initialized_point_cloud.save_ply(os.path.join(directory, "sparse/0/points3D.ply"))
        dataset.save_colmap_cameras(os.path.join(directory, "sparse/0"))
        if refs:
            for ref in refs:
                for i in range(len(initialized_cameras)):
                    if os.path.basename(ref.ground_truth_image_path) == os.path.basename(initialized_cameras[i].image_path):
                        initialized_cameras[i] = initialized_cameras[i]._replace(
                            image_height=ref.image_height,
                            image_width=ref.image_width,
                            FoVx=ref.FoVx,
                            FoVy=ref.FoVy,
                            R=ref.R,
                            T=ref.T,
                        )
        if len(initialized_cameras) < 2:
            continue
        reinitialize("dust3r", directory, configs_reinit, args.device, known_cameras=initialized_cameras)
