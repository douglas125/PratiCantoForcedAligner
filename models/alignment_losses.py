import tensorflow as tf


binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)

possible_losses = [
    "all",
    "order_loss",
    "coverage_loss",
    "pred_max",
    "pred_min",
    "length_loss",
    "one_region_loss",
]


def create_mask(x):
    """Creates a mask that is 1 in unpadded shape and zero elsewhere
    e.g.
    1 1 1 1 1 1 0 0 0 0
    1 1 1 1 1 1 0 0 0 0
    1 1 1 1 1 1 0 0 0 0
    1 1 1 1 1 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    """
    unpadded_shape = x[0:2]
    padded_shape = x[2:4]
    vec = tf.ones(unpadded_shape, dtype=tf.float32)
    pad_shape = tf.stack(
        [tf.zeros((2,), dtype=tf.int32), padded_shape - unpadded_shape], axis=1
    )
    vec = tf.pad(vec, pad_shape)
    return vec


def alignment_loss(loss_to_return="all"):
    assert loss_to_return in possible_losses

    def alignment_loss_fn(y_true, y_pred):
        """Alignment loss

        This method is exclusively self supervised

        y-true contains char_length, melspec_length, ie, the shapes without padding
            shape is (batch_size, 2)

        y_pred - alignment predictions as a tensor of shape
            (batch_size, n_chars, n_mel_spectres)
            for each char, its correspondence vector should be 1
            where it corresponds to the mel spectrogram line and 0 otherwise

            y_pred should be softmax-ed in n_chars dimension to make sure
            that each mel_spec is predicted by only 1 char
        """
        # create masks to filter losses
        y_true = tf.cast(y_true, tf.int32)
        shapes = tf.concat(
            [
                y_true,
                tf.tile(
                    tf.expand_dims(tf.shape(y_pred)[1:3], axis=0),
                    [tf.shape(y_true)[0], 1],
                ),
            ],
            axis=1,
        )

        masks = tf.map_fn(
            create_mask,
            shapes,
            # tensorflow doesn't understand why the input is int
            # but the output is float
            fn_output_signature=tf.TensorSpec([None, None], dtype=tf.float32),
        )

        # apply mask to pred
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = y_pred * masks

        # force chars to be predicted in order
        pred_sum = tf.reduce_sum(y_pred, axis=2, keepdims=True)
        cum_pred = tf.cumsum(y_pred, axis=2) / (pred_sum + 1e-6)
        # apply masks - not needed since diff = 0 in the masked elements
        # because pred is masked
        # >cum_pred = cum_pred * masks< <- not needed

        # compute order loss
        v1 = tf.expand_dims(cum_pred, axis=2)
        v2 = tf.expand_dims(cum_pred, axis=1)
        v1v2diff = v1 - v2
        order_loss = tf.reduce_sum(tf.nn.relu(v1v2diff), axis=-1)

        # penalize only when next char comes before previous
        order_loss = tf.linalg.band_part(order_loss, -1, 0)
        order_loss = tf.reduce_mean(order_loss, axis=-1)
        order_loss = tf.reduce_mean(order_loss, axis=-1)

        """
        # someone has to predict 0 and 1 at some point
        pred_min = tf.reduce_min(y_pred, axis=-1)
        pred_min = tf.reduce_min(pred_min, axis=-1)
        pred_max = tf.reduce_max(y_pred, axis=-1)
        pred_max = tf.reduce_max(pred_max, axis=-1)
        pred_max = 1.0 - pred_max
        """

        # we don't want anyone predicting a very long length
        # each spectrogram step lasts ~10ms and smallest sound is ~70ms
        length_loss = tf.math.maximum(tf.abs(pred_sum) - 7.0, 0.0)
        length_loss = tf.reduce_mean(tf.squeeze(length_loss, axis=-1), axis=-1)

        # we don't want the same char spiking at
        # 2 different points in the spectrogram
        # rationale: if it goes from 0 to 1 back to 0, the derivative
        # ends up being 2. Anything bigger is not good
        one_region_loss = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        one_region_loss = tf.abs(one_region_loss)
        # with the sum, if the pred goes 0 -> 1 -> 0,
        # the shift sums to exactly 2
        one_region_loss = tf.reduce_sum(one_region_loss, axis=2)
        one_region_loss = tf.square(tf.math.maximum(one_region_loss - 2.0, 0.0))
        one_region_loss = tf.reduce_mean(one_region_loss, axis=-1)

        # though there's a softmax, we still want someone to actually predict 1
        coverage_pred = tf.reduce_max(
            y_pred, axis=1
        )  # shape is (batch_size, n_mel_specs)
        coverage_mask = tf.reduce_max(masks, axis=1)
        coverage_loss = binary_crossentropy(coverage_mask, coverage_pred)
        # print(coverage_mask, coverage_pred)

        # everyone has to predict 0 and 1 at some point
        # TODO: crossentropy loss? min is wrong
        pred_min = tf.reduce_min(y_pred + 1.0 - masks, axis=-1)
        pred_mask = tf.reduce_min(1.0 - masks, axis=-1)
        # pred_min = tf.reduce_sum(pred_min, axis=-1)
        pred_min = binary_crossentropy(pred_mask, pred_min)

        pred_max = tf.reduce_max(y_pred, axis=-1)
        pred_mask = tf.reduce_max(masks, axis=-1)
        # pred_max = tf.reduce_sum(pred_max * pred_mask, axis=-1)
        pred_max = binary_crossentropy(pred_mask, pred_max * pred_mask)

        order_loss = 1 * order_loss
        coverage_loss = 0.1 * coverage_loss
        pred_max = 1.0 * pred_max
        pred_min = 0.25 * pred_min
        length_loss = 0.1 * length_loss
        one_region_loss = 1.1 * one_region_loss
        if loss_to_return == "all":
            return (
                order_loss
                + coverage_loss
                + pred_max
                + pred_min
                + length_loss
                + one_region_loss
            )
        elif loss_to_return == "order_loss":
            return order_loss
        elif loss_to_return == "coverage_loss":
            return coverage_loss
        elif loss_to_return == "pred_max":
            return pred_max
        elif loss_to_return == "pred_min":
            return pred_min
        elif loss_to_return == "length_loss":
            return length_loss
        elif loss_to_return == "one_region_loss":
            return one_region_loss

    alignment_loss_fn.__name__ = f"AlignmentLoss_{loss_to_return}"
    return alignment_loss_fn
