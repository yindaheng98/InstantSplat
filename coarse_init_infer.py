import os
import torch
import numpy as np
import argparse
import time

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from instant_splat.utils.dust3r_utils import compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images
from instant_splat.initializer.dust3r import Dust3rInitializer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_base_path", type=str, default="data/sora/santorini/3_views")

    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--model_path", type=str, default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--focal_avg", action="store_true")

    return parser


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    device = args.device

    img_folder_path = os.path.join(args.img_base_path, "images")
    train_img_list = sorted(os.listdir(img_folder_path))
    images, ori_size = load_images(img_folder_path, size=512)
    print("ori_size", ori_size)
    initializer = Dust3rInitializer(args.model_path, args.batch_size, args.niter, args.schedule, args.lr, args.focal_avg, args.device)
    initialized_point_cloud, initialized_cameras = initializer(image_path_list=[os.path.join(img_folder_path, path) for path in train_img_list])

    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_path).to(device)
    start_time = time.time()
    ##########################################################################################################################################################################################
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=args.batch_size)
    output_colmap_path = img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene=scene, init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr, focal_avg=args.focal_avg)
    scene = scene.clean_pointcloud()

    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())
    ##########################################################################################################################################################################################
    end_time = time.time()
    print(f"Time consumption: {end_time-start_time} seconds")

    # save
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    color_4_3dgs_all = (np.array(imgs).reshape(-1, 3) * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D_all.ply"), pts_4_3dgs_all, color_4_3dgs_all)
    np.save(os.path.join(output_colmap_path, "focal.npy"), np.array(focals.cpu()))
    np.save(os.path.join(output_colmap_path, "confidence_masks.npy"), np.array(confidence_masks))
