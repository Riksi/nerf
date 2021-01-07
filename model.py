import tensorflow as tf
import math
from helpers import get_rays, ndc_rays


def get_sample_bounds(near, far, num_samples):
    diff = (far - near)
    idx = tf.range(num_samples)
    diff_term = diff / num_samples
    start = near + idx * diff_term
    end = start + diff_term
    return start, end


def get_embedding(data, num_dims):
    # [L]
    rng = tf.range(num_dims)
    # [P, Z, L]
    embed_term = 2 ** rng * math.pi * data[..., None]
    # [P, Z, 2 * L]
    embed = tf.dynamic_stitch(
        [tf.sin(embed_term), tf.cos(embed_term)],
        [(rng * 2), (rng * 2 + 1)]
    )
    return embed


class NeRF(tf.keras.models.Model):
    def __init__(self, units1=256, num_layers1=8, units2=128, num_layers2=2):
        super(NeRF, self).__init__()
        self.part1 = tf.keras.models.Sequential(
            [
                *(tf.keras.layers.Dense(
                    units=units1,
                    activation='relu'
                )
                    for _ in range(num_layers1 - 1)
                ),
                tf.keras.layers.Dense(
                    units=units1 + 1,
                )
            ]
        )
        self.part2 = tf.keras.models.Sequential(
            [
                *(tf.keras.layers.Dense(
                    units=units2,
                    activation='relu'
                )
                    for _ in range(num_layers2 - 1)
                ),
                tf.keras.layers.Dense(
                    units=256 * 3,
                )
            ]
        )

        self.loss_fn = tf.losses.MeanSquaredError()

    def estimate_color(self, samples, t_far, origin, direction, mask=None):
        # [P, N]
        delta = tf.concat([samples[1:], t_far[None]], axis=-1) - samples
        # [P, N, 3]
        coords = samples[..., None] * direction + origin
        # [P, N, 1 + Z]
        y1 = self.part1(coords)
        # [P, Nc, 1], [P, Nc, Z]
        sigma, features = tf.split(y1, [1, -1], axis=-1)
        # [P, N, 3 + Z]
        x2 = tf.concat([features, direction], axis=-1)
        # [P, N, 3]
        clr = self.part2(x2)

        # [P, N, 1]
        neg_sig_times_delta = - sigma * delta[..., None]

        # [P, N, 1]
        transmittance = tf.exp(-tf.cumsum(
            neg_sig_times_delta,
            axis=1,
            exclusive=True
        ))

        # [P, N, 1]
        weights = transmittance * (1 - tf.exp(neg_sig_times_delta))
        if mask is not None:
            weights = weights * mask

        # [P, 3]
        clr_est = tf.reduce_sum(
            clr * weights,
            axis=1
        )

        return clr_est, weights

    def call(self, inputs, num_samples=None, training=True):
        origin = get_embedding(inputs.origin, inputs.num_embed_dims)
        direction = get_embedding(inputs.direction, inputs.num_embed_dims)

        # [Nc], [Nc]
        starts, ends = get_sample_bounds(inputs.t_near, inputs.t_far, num_samples.coarse)

        t_coarse = tf.random.uniform(
            [tf.shape(inputs.direction)[0], num_samples.coarse],
            starts,
            ends
        )

        # [P, 3], [P, Nc]
        clr_coarse, coarse_weights = self.estimate_color(t_coarse, t_far, origin, direction)

        coarse_weights = tf.stop_gradient(coarse_weights)

        # [P, Nc]
        regions = tf.random.categorical(
            tf.squeeze(tf.log(coarse_weights) - tf.log(tf.reduce_sum(coarse_weights)), axis=-1),
            num_samples=num_samples.fine
        )

        # [P, Nf], [P, Nf]
        starts_for_regions = tf.gather(starts, regions)
        ends_for_regions = tf.gather(ends, regions)

        t_fine = tf.random.uniform(
            [tf.shape(direction)[0], num_samples.fine],
            starts_for_regions,
            ends_for_regions
        )

        t_union_ragged = tf.RaggedTensor.from_sparse(
            tf.sets.union(t_fine, t_coarse)
        )

        # [P, N']
        t_union = t_union_ragged.to_tensor()

        # [P, N']
        mask = tf.sequence_mask(
            t_union_ragged.row_lengths(),
            tf.shape(t_union)[-1]
        )

        clr_fine = self.estimate_color(t_union_ragged, inputs.t_far, origin, direction, mask)

        return clr_coarse, clr_fine

    def train_step(self, data):
        clr_coarse, clr_fine = self.call(data.inputs,  data.num_samples, training=True)
4
    def inference_step(self, data):
        _, clr_fine = self.call(data.inputs, data.num_samples, False)
        pred = tf.reshape(clr_fine, [data.grid_shape, 3])
        return pred























