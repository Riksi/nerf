import tensorflow as tf
import numpy as np
from data import load_dv_data
import os
import time
from helpers import get_rays, ndc_rays, to8b, img2mse, mse2psnr
import imageio
from render import render, render_path


def render_only(images, render_poses, hwf,
                chunk_size, render_factor,
                render_kwargs, savedir):
    os.makedirs(savedir, exist_ok=True)
    print('test poses shape', render_poses.shape)

    rgbs, _ = render_path(render_poses,
                          hwf,
                          chunk_size,
                          render_kwargs,
                          gt_imgs=images,
                          savedir=savedir,
                          render_factor=render_factor)
    print('Rendering complete', savedir)
    imageio.mimwrite(os.path.join(savedir, 'video.mp4'),
                     to8b(rgbs), fps=30, quality=9)
    return


def train():

    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        # TODO: random seed for TF


    #TODO: add other datasets

    if args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape,
            basedir=args.datadir,
            testskip=args.testskip
        )

        print("Loaded deepvoxels", images.shape,
              render_poses.shape, hwf, args.datadir)

        i_train, i_val, i_test = i_split

        # length of z-direction of the pose matrix I think
        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    H, W, focal = hwf
    H, W = map(int, [H, W])
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)

    bds_dict = dict(
        near=f.cast(near, tf.float32),
        far=tf.cast(far, tf.float32)
    )

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.render_only:

        return render_only(
            images=images[i_test] if args.render_test else None,
            render_poses=render_poses,
            hwf = hwf,
            chunk_size=args.chunk,
            render_factor=args.render_factor,
            render_kwargs=render_kwargs_test,
            savedir=os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                  'test' if args.render_test else 'path',  start
                )
            )
        )


    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(
            lrate, decay_steps=args.lrate_decay * 1000,
            decay_rate=0.1
        )

    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.Variable(start, trainable=False)

    N_rand = args.N_rand

    use_batching = not args.no_batching

    if use_batching:

        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)

        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype('float32')

        np.random.shuffle(rays_rgb)
        i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        if use_batching:
            batch = rays_rgb[i_batch: i_batch + N_rand]
            batch = tf.transpose(batch, [1, 0, 2])

            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0


        else:
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, pose)
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = tf.stack(tf.meshgrid(
                    tf.range(H//2 - dH, H//2 + dH),
                    tf.range(W//2 - dW, W//2 + dW),
                    indexing='ij'
                ), -1)
                if i < 10:
                    print('precrop', dH, dW, coords[0, 0], coords[-1, -1])
            else:
                coords = tf.stack(tf.meshgrid(
                    tf.range(H), tf.range(W), indexing='ij'))
                coords = tf.reshape(coords, [-1, 2])

            coords = tf.reshape(coords, [-1, 2])
            select_inds = tf.random.shuffle(tf.range(tf.shape(coords)[0]))[:N_rand]
            rays_o = tf.gather_nd(rays_o, select_inds)
            rays_d = tf.gather_nd(rays_d, select_inds)
            batch_rays = tf.stack([rays_o, rays_d], 0)
            target_s = tf.gather_nd(target, select_inds)

    with tf.GradientTape() as tape:
        rgb, disp, acc, extras = render(
            H, W, focal, chunk=args.chunk, rays=batch_rays,
            verbose=i < 10, retraw=True, **render_kwargs_train)

        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss += img_loss0
            psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time() - time0

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, f'{prefix}_{i:06d}')
            np.save(path, np.get_weights())
            print('saved weights at', path)



        # TODO: add summaries, save weights, outputs

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:
            pass

        if i % args.i_testset == 0 and i > 0:
            pass

        if args.i_img == 0:
            pass












