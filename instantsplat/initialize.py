import os
import shutil

from instantsplat.initializer import *
from instantsplat.initializer.depth import AutoScaleDepthAnythingV2InitializerWrapper

default_image_folder = {
    "dust3r": "images",
    "ttt3r": "images",
    "mast3r": "images",
    "mapanything": "images",
    "mapanything-external": "images",
    "vggt": "images",
    "vggt-colmap-sparse": "input",
    "vggt-colmap-dense": "input",
    "colmap-sparse": "input",
    "colmap-dense": "input",
    "dust3r-align-colmap-sparse": "input",
    "dust3r-align-colmap-dense": "input",
}


def initialize(initializer, directory, configs, device, scale=1.0, with_depth_anything=False):
    image_folder = os.path.join(directory, default_image_folder[initializer])
    image_path_list = [os.path.join(image_folder, file) for file in sorted(os.listdir(image_folder))]
    def convert_image_path(image_path): return os.path.join(os.path.dirname(os.path.dirname(image_path)), "images", os.path.basename(image_path))
    match initializer:
        case "dust3r":
            constructor = Dust3rInitializer
        case "ttt3r":
            constructor = Ttt3rInitializer
        case "mast3r":
            constructor = Mast3rInitializer
        case "vggt":
            constructor = VGGTInitializer
        case "mapanything":
            constructor = MapAnythingInitializer
        case "mapanything-external":
            constructor = MapAnythingExternalInitializer
        case "vggt-colmap-sparse":
            constructor = lambda **configs: VGGTColmapSparseInitializer(destination=directory, **configs)
        case "vggt-colmap-dense":
            constructor = lambda **configs: VGGTColmapDenseInitializer(destination=directory, **configs)
        case "colmap-sparse":
            constructor = lambda **configs: ColmapSparseInitializer(destination=directory, **configs)
        case "colmap-dense":
            constructor = lambda **configs: ColmapDenseInitializer(destination=directory, **configs)
        case "dust3r-align-colmap-sparse":
            constructor = lambda **configs: Dust3rAlign2ColmapSparseInitializer(destination=directory, convert_image_path=convert_image_path, **configs)
        case "dust3r-align-colmap-dense":
            constructor = lambda **configs: Dust3rAlign2ColmapDenseInitializer(destination=directory, convert_image_path=convert_image_path, **configs)
        case _:
            raise ValueError(f"Unknown initializer {initializer}")
    if with_depth_anything:
        base_constructor = constructor
        constructor = lambda *args, **configs: AutoScaleDepthAnythingV2InitializerWrapper(base_constructor, *args, **configs)
    initializer = constructor(**configs).to(device)
    initialized_point_cloud, initialized_cameras = initializer(image_path_list=image_path_list)
    initialized_point_cloud = initialized_point_cloud._replace(points=initialized_point_cloud.points*scale)
    initialized_cameras = [camera._replace(T=camera.T*scale) for camera in initialized_cameras]
    return initialized_cameras, initialized_point_cloud


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--initializer", choices=list(default_image_folder.keys()), default="dust3r", type=str)
    parser.add_argument("-d", "--directory", required=True, type=str)
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--with_depth_anything", action="store_true", default=False)
    parser.add_argument("-o", "--option", default=[], action='append', type=str)

    args = parser.parse_args()
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    initialized_cameras, initialized_point_cloud = initialize(args.initializer, args.directory, configs, args.device, scale=args.scale, with_depth_anything=args.with_depth_anything)
    dataset = InitializedCameraDataset(initialized_cameras)

    shutil.rmtree(os.path.join(args.directory, "sparse/0"), ignore_errors=True)
    os.makedirs(os.path.join(args.directory, "sparse/0"), exist_ok=True)
    initialized_point_cloud.save_ply(os.path.join(args.directory, "sparse/0/points3D.ply"))
    dataset.save_colmap_cameras(os.path.join(args.directory, "sparse/0"))
