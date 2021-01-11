import tensorflow as tf
import numpy as np


def img2mse(x, y):
    return tf.reduce_mean((x - y) ** 2)


def mse2psnr(x):
    return -10 * tf.math.log(x) / tf.math.log(10)


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def get_rays(H, W, focal, c2w):
    """
    For NDC you have
        x_ndc = - (x_c * focal / (W / 2)) / z_c
        y_ndc = - (y_c * focal / (W / 2)) / z_c
        z_ndc ~ 1 + 2 * near / z_c

    Here subtracting W/2 from 0 ... W-1,
    H/2 to 0 ... H-1 is equivalent to
    multiplying -1 ... (1 -  1 / (W / 2)) by W / 2
    i.e. a regular grid between -1 and 1
    the NDC bounds.

    (Note that y needs to increase from bottom to top hence -j)

    Then they divide by focal.

    So it looks like this transform is getting reversed
    in part at least. --But not sure why z is -1.--

    By their shifting of the origin (see page 21) too o + tn * d,
    the new origin has a oz = -1.

    By equation 26 d' = [ax * fx(d, o), ay * fy(d, o), - bz * fz(d, o)]
    which can be interpreted as a transformation of
    [fx(d, o), fy(d, o), fz(d, o)] via the matrix
    [[ax, 0, 0],
     [0, ay, 0],
     [0, 0,-bz]]

    Now they seem to be interpreting i and j as fx(d, o) and fy(d, o)
    and fz(d, o) = 1 / oz = -1 via the shifted origin.

    Applying the inverse transform thus yields the results
    shown

    """

    # [H, W]
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32),
                       indexing='xy')
    # [H, W, 3]
    dirs = tf.stack([(i - W * .5) / focal,
                     -(j - W * .5) / focal,
                     -tf.ones_like(i)], axis=-1)

    rays_d = tf.squeeze(c2w[:3, :3] @ dirs[..., None], axis=-1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    tn = -(near + rays_o[..., -1:]) / rays_d[..., -1:]

    rays_o = rays_o + tn * rays_d

    rays_ox, rays_oy, rays_oz = tf.split(rays_o, axis=-1, num_or_size_splits=3)
    rays_dx, rays_dy, rays_dz = tf.split(rays_d, axis=-1, num_or_size_splits=3)

    ax = -(focal / (W / 2))
    ay = -(focal / (H / 2))
    az = 1
    bz = 2 * near

    ox = ax * rays_ox / rays_oz
    oy = ay * rays_oy / rays_oz
    oz = az + bz / rays_oz

    dx = ox + ax * rays_dx / rays_dz
    dy = oy + ay * rays_dy / rays_dz
    dz = az - oz

    ndc_ray_o = tf.concat([ox, oy, oz], axis=-1)
    ndc_ray_d = tf.concat([dx, dy, dz], axis=-1)

    return ndc_ray_o, ndc_ray_d






