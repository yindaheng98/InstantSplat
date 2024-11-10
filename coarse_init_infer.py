import os
import torch
import numpy as np
import argparse
import time
import cv2
from dust3r.utils.image import _resize_pil_image
import PIL.Image
from PIL.ImageOps import exif_transpose
from plyfile import PlyData, PlyElement
import torchvision.transforms as tvf

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from instant_splat.initializer.dust3r.alignment import compute_global_alignment
from instant_splat.initializer import Dust3rInitializer, InitializedCameraDataset
from gaussian_splatting.dataset.colmap import ColmapCameraDataset

######################## dust3r_utils.py ########################


def R_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.

    Parameters:
    - R: A 3x3 numpy array representing a rotation matrix.

    Returns:
    - A numpy array representing the quaternion [w, x, y, z].
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def save_colmap_cameras(ori_size, intrinsics, camera_file):
    with open(camera_file, 'w') as f:
        for i, K in enumerate(intrinsics, 1):  # Starting index at 1
            width, height = ori_size
            scale_factor_x = width/2 / K[0, 2]
            scale_factor_y = height/2 / K[1, 2]
            # assert scale_factor_x==scale_factor_y, "scale factor is not same for x and y"
            print(f'scale factor is not same for x{scale_factor_x} and y {scale_factor_y}')
            f.write(f"{i} PINHOLE {width} {height} {K[0, 0]*scale_factor_x} {K[1, 1]*scale_factor_x} {width/2} {height/2}\n")  # scale focal
            # f.write(f"{i} PINHOLE {width} {height} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")


def save_colmap_images(poses, images_file, train_img_list):
    with open(images_file, 'w') as f:
        for i, pose in enumerate(poses, 1):  # Starting index at 1
            # breakpoint()
            pose = np.linalg.inv(pose)
            R = pose[:3, :3]
            t = pose[:3, 3]
            q = R_to_quaternion(R)  # Convert rotation matrix to quaternion
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {train_img_list[i-1]}\n")
            f.write(f"\n")


def load_images(folder_or_list, size, square_ok=False):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    imgs = []
    for path in folder_content:
        if not path.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)1
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        W2 = W//16*16
        H2 = H//16*16
        img = np.array(img)
        img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_LINEAR)
        img = PIL.Image.fromarray(img)

        print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    print(f' (Found {len(imgs)} images)')
    return imgs, (W1, H1)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

######################## dust3r_utils.py ########################


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
    initialized_point_cloud.save_ply(os.path.join(output_colmap_path, "points3D_test.ply"))
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    color_4_3dgs_all = (np.array(imgs).reshape(-1, 3) * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D_all.ply"), pts_4_3dgs_all, color_4_3dgs_all)
    np.save(os.path.join(output_colmap_path, "focal.npy"), np.array(focals.cpu()))
    np.save(os.path.join(output_colmap_path, "confidence_masks.npy"), np.array(confidence_masks))

    dataset1 = InitializedCameraDataset(initialized_cameras)
    dataset2 = ColmapCameraDataset(args.img_base_path)
    for cam1, cam2 in zip(dataset1, dataset2):
        print(cam1.image_height - cam2.image_height)
        print(cam1.image_width - cam2.image_width)
        print(cam1.FoVx - cam2.FoVx)
        print(cam1.FoVy - cam2.FoVy)
        print(cam1.R - cam2.R)
        print(cam1.T - cam2.T)
    os.makedirs(os.path.join(args.img_base_path, "sparse/1"), exist_ok=True)
    dataset1.save_colmap_cameras(os.path.join(args.img_base_path, "sparse/1"))
