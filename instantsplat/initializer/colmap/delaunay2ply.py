import torch
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm


def read_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']])).T
    colors = torch.tensor(np.vstack([vertices['red'], vertices['green'], vertices['blue']])).T
    return positions, colors


def read_delaunay(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']])).T
    return positions, plydata['face']


def get_color(pos, pos_reference, color_reference, batch=1024, reference_batch=1024*1024):
    n_points = pos.shape[0]
    n_points_reference = pos_reference.shape[0]
    color = torch.zeros(size=(n_points, color_reference.shape[1]), dtype=color_reference.dtype)
    pbar = tqdm(desc="Processing point pairs", total=n_points*n_points_reference)
    for i in range(0, n_points, batch):
        step = batch if i+batch < n_points else (n_points-i)

        step_reference = min(batch+reference_batch, n_points_reference)
        pos_reference_step = pos_reference[0:step_reference, ...]
        color_reference_step = color_reference[0:step_reference, ...]
        dist = torch.norm(pos[i:i+step, ...].unsqueeze(1) - pos_reference_step.unsqueeze(0), p=2, dim=2)
        idx = dist.argmin(dim=1)
        color[i:i+step] = color_reference_step[idx, ...]
        pbar.update(step_reference)

        for j in range(batch+reference_batch, n_points_reference, batch):
            pos_reference_step[reference_batch:reference_batch+idx.shape[0], ...] = pos_reference_step[idx, ...]
            color_reference_step[reference_batch:reference_batch+idx.shape[0]:, ...] = color_reference_step[idx, ...]

            step_reference = reference_batch if j+reference_batch < n_points_reference else (n_points_reference-j)
            pos_reference_step[0:step_reference, ...] = pos_reference[j:j+step_reference, ...]
            color_reference_step[0:step_reference, ...] = color_reference[j:j+step_reference, ...]
            dist = torch.norm(pos[i:i+step, ...].unsqueeze(1) - pos_reference_step.unsqueeze(0), p=2, dim=2)
            idx = dist.argmin(dim=1)
            color[i:i+step] = color_reference_step[idx, ...]
            pbar.update(step_reference)
    return color


def save_ply(xyz, color, face=None):
    dtype_full = [(attr, 'float32') for attr in ['x', 'y', 'z']]
    dtype_full += [(attr, 'uint8') for attr in ['red', 'green', 'blue']]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, color), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    if face:
        return PlyData([el, face])
    return PlyData([el])


def delaunay2ply(delaunay, reference, batch=1024, reference_batch=1024*1024):
    with torch.device(device="cuda"):
        pos_delaunay, face_delaunay = read_delaunay(delaunay)
        pos_reference, color_reference = read_ply(reference)
        color_delaunay = get_color(pos_delaunay, pos_reference, color_reference, batch=batch, reference_batch=reference_batch)
        return save_ply(pos_delaunay.cpu().numpy(), color_delaunay.cpu().numpy(), face=face_delaunay)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delaunay", type=str, required=True, help="path to the delaunay point cloud")
    parser.add_argument("--reference", type=str, required=True, help="path to the reference point cloud")
    parser.add_argument("--save", type=str, required=True, help="path to the reference point cloud")
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()
    delaunay2ply(args.delaunay, args.reference, batch=args.batch).write(args.save)
