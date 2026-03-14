# Adapted from: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/dependency/np_to_pycolmap.py#L12-L145

import os
import numpy as np
from gaussian_splatting.dataset.colmap.read_write_model import (
    Camera as ColmapCamera,
    Image as ColmapImage,
    Point3D as ColmapPoint3D,
    rotmat2qvec,
)
from vggt.dependency.projection import project_3D_points_np


def batch_np_matrix_to_colmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    original_coords,
    track_resolution,
    image_names,
    masks=None,
    max_reproj_error=None,
    camera_type="PINHOLE",
    points_rgb=None,
):
    """Convert batched NumPy arrays to COLMAP binary format dicts.

    Unlike batch_np_matrix_to_pycolmap which requires pycolmap, this uses
    read_write_model namedtuples and writes standard COLMAP binary files.

    Args:
        points3d: (P, 3) 3D world coordinates for each track.
        extrinsics: (N, 3, 4) camera extrinsic matrices.
        intrinsics: (N, 3, 3) camera intrinsic matrices, already scaled to track_resolution.
        tracks: (N, P, 2) 2D track coordinates at track_resolution.
        original_coords: (N, 6) [x1, y1, x2, y2, orig_w, orig_h] at track_resolution.
        track_resolution: Resolution of tracks/intrinsics (e.g. 1024).
        image_names: List of N image basenames.
        masks: (N, P) boolean mask (e.g. visibility). Default is None.
        max_reproj_error: Maximum reprojection error for filtering. Default is None.
        camera_type: COLMAP camera model ("PINHOLE" or "SIMPLE_PINHOLE").
        points_rgb: (P, 3) uint8 RGB colors. Default is None.

    Returns:
        (cameras, images, points3D) dicts suitable for write_model,
        and valid_mask (P,) boolean array indicating valid tracks.
    """
    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P

    # ── Reprojection error filtering ──

    reproj_mask = None
    if max_reproj_error is not None:
        projected_2d, projected_cam = project_3D_points_np(
            points3d, extrinsics, intrinsics)
        projected_2d[projected_cam[:, 2, :] <= 0] = 1e6

        reproj_error = np.linalg.norm(projected_2d - tracks, axis=-1)
        reproj_mask = reproj_error < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif reproj_mask is not None:
        masks = reproj_mask

    assert masks is not None

    valid_mask = masks.sum(axis=0) >= 2
    valid_idx = np.nonzero(valid_mask)[0]

    # ── Build cameras at original image resolution ──

    cameras = {}
    for i in range(N):
        orig_w, orig_h = original_coords[i, 4], original_coords[i, 5]
        rescale = max(orig_w, orig_h) / track_resolution
        fx = intrinsics[i][0, 0] * rescale
        fy = intrinsics[i][1, 1] * rescale
        cx, cy = orig_w / 2.0, orig_h / 2.0

        if camera_type == "PINHOLE":
            params = np.array([fx, fy, cx, cy])
        elif camera_type == "SIMPLE_PINHOLE":
            params = np.array([(fx + fy) / 2.0, cx, cy])
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

        cameras[i + 1] = ColmapCamera(
            id=i + 1, model=camera_type,
            width=int(orig_w), height=int(orig_h), params=params)

    # ── Build images (with 2D observations) and accumulate track info ──

    images = {}
    track_info = {
        rank + 1: {"image_ids": [], "point2d_idxs": []}
        for rank in range(len(valid_idx))
    }

    for fidx in range(N):
        orig_w, orig_h = original_coords[fidx, 4], original_coords[fidx, 5]
        x1, y1 = original_coords[fidx, 0], original_coords[fidx, 1]
        scale_to_orig = max(orig_w, orig_h) / track_resolution

        xys_list, p3d_ids_list = [], []
        p2d_idx = 0
        for rank, vidx in enumerate(valid_idx):
            p3d_id = rank + 1
            if masks[fidx, vidx]:
                xy_orig = (tracks[fidx, vidx] - [x1, y1]) * scale_to_orig
                xys_list.append(xy_orig)
                p3d_ids_list.append(p3d_id)
                track_info[p3d_id]["image_ids"].append(fidx + 1)
                track_info[p3d_id]["point2d_idxs"].append(p2d_idx)
                p2d_idx += 1

        images[fidx + 1] = ColmapImage(
            id=fidx + 1,
            qvec=rotmat2qvec(extrinsics[fidx][:3, :3]),
            tvec=extrinsics[fidx][:3, 3],
            camera_id=fidx + 1,
            name=image_names[fidx],
            xys=np.array(xys_list) if xys_list
                else np.zeros((0, 2)),
            point3D_ids=np.array(p3d_ids_list, dtype=np.int64)
                if p3d_ids_list else np.array([], dtype=np.int64))

    # ── Build Point3D entries ──

    points3D = {}
    for rank, vidx in enumerate(valid_idx):
        p3d_id = rank + 1
        info = track_info[p3d_id]
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3, dtype=np.uint8)
        points3D[p3d_id] = ColmapPoint3D(
            id=p3d_id,
            xyz=points3d[vidx],
            rgb=rgb,
            error=0.0,
            image_ids=np.array(info["image_ids"], dtype=np.int32),
            point2D_idxs=np.array(info["point2d_idxs"], dtype=np.int32))

    return cameras, images, points3D, valid_mask
