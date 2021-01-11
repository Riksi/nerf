import tensorflow as tf

def init_nerf_model(D=8,
                    W=256,
                    input_ch=3,
                    input_ch_views=3,
                    output_ch=4,
                    skips=[4],
                    use_viewdirs=False):
    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu):
        return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(input_ch),
          type(input_ch_views), use_viewdirs)

    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views])
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts

    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], axis=-1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat([bottleneck, inputs_views], -1)
        outputs = inputs_viewdirs

        for i in range(1):
            outputs = dense(W // 2)(outputs)

        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_nerf(config):
    embed_fn, input_ch = get_embedder(config.multiires, config.i_embed)

    input_ch_views = 0
    embeddirs_fn = None

    if config.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(config.multires_views, config.i_embed)

    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=config.netdepth,
        W=config.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=config.use_viewdirs
    )
    grad_vars = model.trainable_variables
    models = {"model": model}

    model_fine = None

    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=config.netdepth_fine,
            W=config.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=config.use_viewdirs
        )

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(
            inputs,
            viewdirs,
            network_fn,
            embed_fn,
            embeddirs_fn,
            config.netchunk
        )

    render_kwargs_train = dict(
        network_query_fn=network_query_fn,
        perturb=config.perturb,
        N_importance=config.N_importance,
        network_fine=model_fine,
        N_samples=config.N_samples,
        network_fn=model,
        use_viewdirs=config.use_viewdirs,
        white_bkgd=config.white_bkgd,
        raw_noise_std=config.raw_noise_std
    )

    if config.dataset_type != 'llff' or lindisp.no_ndc:
        print('Not NDC')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = config.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k]
        for k in render_kwargs_train
    }
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = config.basedir
    expname = config.expname

    ## TODO: LOAD CHECKPOINT

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models










