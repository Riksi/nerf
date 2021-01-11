import tensorflow as tf
from helpers import ndc_rays, get_rays
import numpy as np
import imageio
import os
import time


def raw2outputs(raw, z_vals, rays_d):\

    def raw2alpha(raw, dists, act_fn=tf.nn.relu):
        return 1.0 - tf.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = tf.concat(
        [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
        axis=-1)

    dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

    # [N_rays, N_samples, 3]
    rgb = tf.math.sigmoid(raw[..., :3])

    noise = 0.
    if raw_noise_std > 0.:
        noise = tf.random.normal(
            raw[..., 3].shape
        ) * raw_noise_std

    # This is how you get alpha
    # sigma = tf.nn.relu(raw + noise)
    # alpha = 1 - exp(-sigma * dists)
    alpha = raw2alpha(raw[..., 3] + noise, dists)

    # T[0] = 1
    # T[i] = exp(-sum((sigma * dists)[1:i])
    # = prod(exp(-sigma * dists)[1:i])
    # = prod((1 - alpha)[1:i])
    # => T = cumprod(1 - alpha, exclusive=True)
    weights = alpha * tf.math.cumprod(1 - alpha + 1e-10, axis=-1, exclusive=True)

    # [N_rays, 3]
    rgb_map = tf.reduce_sum(rgb * weights[..., None], axis=-2)

    depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

    disp_map = 1. / tf.maximum(1e-10, depth_map / tf.reduce_sum(weights, axis=-1))

    acc_map = tf.reduce_sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False):
    N_rays = tf.shape(ray_batch)[0]

    # [N_rays, 3], [N_rays, 3]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]

    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # [N_rays, 1], [N_rays, 1]
    near, far = tf.split(ray_batch[..., 6:8], axis=-1, num_or_size_splits=2)

    # [N_samples]
    t_vals = tf.linspace(0., 1., N_samples)

    # Interpolate between near and far
    # either with equal spacing all the way along the depth
    # or inversely along the depth
    # [N_rays, N_samples]
    # TODO: they broadcast but I think it
    #  is already that shape but confirm
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1 - t_vals) + 1. / far * t_vals)

    # This is what get_sample_bounds does essentially I think
    if perturb:
        mid = (z_vals[..., 1:] + z_vals[..., :-1]) / 2.
        upper = tf.concat([mid, z_vals[..., 1:]], axis=-1)
        lower = tf.concat([z_vals[..., :-1], mid], axis=-1)
        t_rand = tf.random.uniform(tf.shape(z_vals))
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance,
            det=(perturb==0.)
        )
        z_samples = tf.stop_gradient(z_samples)

        z_vals = tf.sort(
            tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None, :]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map,
           'disp_map': disp_map,
           'acc_map': acc_map}

    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret.update(
         {'rgb0': rgb_map_0,
          'disp0': disp_map_0,
          'acc0': acc_map_0,
          'z_std': tf.math.reduce_std(z_samples, -1)}
        )

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret










def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    rays_dt = tf.data.Dataset.from_tensor_slices(
        rays_flat
    ).batch(chunk)

    all_ret = {}

    for rays_batch in rays_dt:
        ret = render_rays(rays_batch, **kwargs)
        for k, v in ret.items():
            all_ret.setdefault(k, []).append(v)

    all_ret = {k: tf.concat(v) for k, v in all_ret.items()}

    return  all_ret


def render(H, W, focal, chunk=1024 * 32, rays=None, c2w=None,
           ndc=True, near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d
        )
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)

    near = tf.empty(rays_d[..., :1], near)
    far = tf.empty(rays_d[..., :1], far)

    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)

    if use_viewdirs:
        rays = tf.concat([rays, viewdirs], axis=-1)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1] + list(all_ret[k].shape[1:]))
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: v for k, v in all_ret.items() if k not in k_extract}

    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs,
                gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4],
            **render_kwargs)
        rgbs.append(rgb.numpy())

        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps










