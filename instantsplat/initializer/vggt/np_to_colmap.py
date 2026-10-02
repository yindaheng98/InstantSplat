import copy

import numpy as np
from gaussian_splatting.dataset.colmap.read_write_model import (
    Camera as ColmapCamera,
    Image as ColmapImage,
    Point3D as ColmapPoint3D,
    read_model,
    rotmat2qvec,
    write_model,
)
from vggt.dependency.projection import project_3D_points_np


# From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L293-L320
def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri


# Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L12-L145
# pycolmap.Reconstruction is replaced by read_write_model namedtuples.
def batch_np_matrix_to_colmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    Convert Batched NumPy Arrays to COLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    if masks.sum(1).min() < min_inlier_per_frame:
        print(f"Not enough inliers per frame, skip BA.")
        # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L71-L73
        # Four Nones so the caller can unpack cameras, images, points3D, valid_mask.
        return None, None, None, None

    # From: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L78-L86
    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    points3D = {}
    for point3D_id, vidx in enumerate(valid_idx, start=1):
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3, dtype=np.uint8)
        points3D[point3D_id] = {
            "xyz": points3d[vidx],
            "rgb": rgb,
            "image_ids": [],
            "point2D_idxs": [],
        }

    num_points3D = len(valid_idx)
    cameras = {}
    images = {}
    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)

            # Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L96-L101
            camera = ColmapCamera(
                id=fidx + 1,
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=pycolmap_intri,
            )
            cameras[camera.id] = camera

        # Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L103-L109
        # rotmat2qvec replaces pycolmap.Rigid3d. The placeholder name is renamed after BA.
        qvec = rotmat2qvec(extrinsics[fidx][:3, :3])
        tvec = extrinsics[fidx][:3, 3]

        xys_list = []
        p3d_ids_list = []
        point2D_idx = 0

        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (points3D[point3D_id]["xyz"] < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    xys_list.append(point2D_xy)
                    p3d_ids_list.append(point3D_id)

                    # add element
                    points3D[point3D_id]["image_ids"].append(fidx + 1)
                    points3D[point3D_id]["point2D_idxs"].append(point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(xys_list)

        # Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L135-L143
        images[fidx + 1] = ColmapImage(
            id=fidx + 1,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera.id,
            name=f"image_{fidx + 1}",
            xys=np.array(xys_list) if xys_list else np.zeros((0, 2)),
            point3D_ids=np.array(p3d_ids_list, dtype=np.int64) if p3d_ids_list else np.array([], dtype=np.int64),
        )

    # Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L82-L86
    colmap_points3D = {}
    for point3D_id, point in points3D.items():
        colmap_points3D[point3D_id] = ColmapPoint3D(
            id=point3D_id,
            xyz=point["xyz"],
            rgb=point["rgb"],
            error=0.0,
            image_ids=np.array(point["image_ids"], dtype=np.int32),
            point2D_idxs=np.array(point["point2D_idxs"], dtype=np.int32),
        )

    return cameras, images, colmap_points3D, valid_mask


# Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L254-L292
# The original edits a pycolmap.Reconstruction in memory. Here BA has already
# written distorted/sparse/0, so the same steps run on the COLMAP files.
def rescale_colmap_to_original(
    sparse_dir,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    cameras, images, points3D = read_model(sparse_dir, ext=".bin")
    rescale_camera = True

    scaled_cameras = dict(cameras)
    scaled_images = {}
    for pyimageid in images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = images[pyimageid]
        pycamera = scaled_cameras[pyimage.camera_id]
        name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            # Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L276-L278
            scaled_cameras[pycamera.id] = pycamera._replace(
                width=int(real_image_size[0]),
                height=int(real_image_size[1]),
                params=pred_params,
            )

        xys = pyimage.xys
        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            # Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py#L280-L285
            top_left = original_coords[pyimageid - 1, :2]
            xys = np.asarray(pyimage.xys, dtype=np.float64)
            if xys.size:
                xys = (xys - top_left) * resize_ratio

        scaled_images[pyimageid] = pyimage._replace(name=name, xys=xys)

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    write_model(scaled_cameras, scaled_images, points3D, sparse_dir)
