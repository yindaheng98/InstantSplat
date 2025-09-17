import os
import shutil
import re

from tqdm import tqdm

from instantsplat.initialize import *

default_image_folder = {
    "colmap-sparse": "input",
    "colmap-dense": "input",
    "dust3r-align-colmap": "input",
    "nodepth-colmap-sparse": "input",
    "nodepth-colmap-dense": "input",
    "nodepth-dust3r-align-colmap": "input",
}


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
