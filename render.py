import tensorflow as tf
from helpers import ndc_rays, get_rays


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
    N_rays = tf.shape(rays_batch)[0]

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



