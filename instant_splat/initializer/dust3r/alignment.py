import torch
import numpy as np
import dust3r.cloud_opt.init_im_poses as init_fun
from dust3r.cloud_opt.base_opt import global_alignment_loop
from dust3r.utils.geometry import geotrf, inv
from dust3r.cloud_opt.commons import edge_str


def get_known_poses(scene):
    if scene.has_im_poses:
        known_poses_msk = torch.tensor([not (p.requires_grad) for p in scene.im_poses])
        known_poses = scene.get_im_poses()
        return known_poses_msk.sum(), known_poses_msk, known_poses
    else:
        return 0, None, None


def init_from_pts3d(scene, pts3d, im_focals, im_poses):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(scene)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = init_fun.align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = init_fun.sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)

    # set all pairwise poses
    for e, (i, j) in enumerate(scene.edges):
        i_j = edge_str(i, j)
        # compute transform that goes from cam to world
        s, R, T = init_fun.rigid_points_registration(scene.pred_i[i_j], pts3d[i], conf=scene.conf_i[i_j])
        scene._set_pose(scene.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = scene.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if scene.has_im_poses:
        for i in range(scene.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            scene._set_depthmap(i, depth)
            scene._set_pose(scene.im_poses, i, cam2world)
            if im_focals[i] is not None:
                scene._set_focal(i, im_focals[i])

    if scene.verbose:
        print(' init loss =', float(scene()))


@torch.no_grad()
def init_minimum_spanning_tree(scene, focal_avg=False, known_focal=None, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = scene.device
    pts3d, _, im_focals, im_poses = init_fun.minimum_spanning_tree(scene.imshapes, scene.edges,
                                                                   scene.pred_i, scene.pred_j, scene.conf_i, scene.conf_j, scene.im_conf, scene.min_conf_thr,
                                                                   device, has_im_poses=scene.has_im_poses, verbose=scene.verbose,
                                                                   **kw)

    if known_focal is not None:
        repeat_focal = np.repeat(known_focal, len(im_focals))
        for i in range(len(im_focals)):
            im_focals[i] = known_focal
        scene.preset_focal(known_focals=repeat_focal)
    elif focal_avg:
        im_focals_avg = np.array(im_focals).mean()
        for i in range(len(im_focals)):
            im_focals[i] = im_focals_avg
        repeat_focal = np.array(im_focals)  # .cpu().numpy()
        scene.preset_focal(known_focals=repeat_focal)

    return init_from_pts3d(scene, pts3d, im_focals, im_poses)


@torch.amp.autocast('cuda', enabled=False)
def compute_global_alignment(scene, init=None, niter_PnP=10, focal_avg=False, known_focal=None, **kw):
    if init is None:
        pass
    elif init == 'msp' or init == 'mst':
        init_minimum_spanning_tree(scene, niter_PnP=niter_PnP, focal_avg=focal_avg, known_focal=known_focal)
    elif init == 'known_poses':
        init_fun.init_from_known_poses(scene, min_conf_thr=scene.min_conf_thr,
                                       niter_PnP=niter_PnP)
    else:
        raise ValueError(f'bad value for {init=}')

    return global_alignment_loop(scene, **kw)
