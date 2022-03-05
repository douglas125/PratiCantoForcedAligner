import tensorflow as tf


def create_mask(x):
    """ Creates a mask that is 1 in unpadded shape and zero elsewhere
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
    pad_shape = tf.stack([
        tf.zeros((2,), dtype=tf.int32),
        padded_shape - unpadded_shape
    ], axis=1)
    vec = tf.pad(vec, pad_shape)
    return vec


def alignment_loss(y_true, y_pred):
    """ Alignment loss

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
    shapes = tf.concat([
        y_true,
        tf.tile(
            tf.expand_dims(tf.shape(y_pred)[1:3], axis=0),
            [tf.shape(y_true)[0], 1]
        )
    ], axis=1)

    masks = tf.map_fn(
        create_mask, shapes,
        # tensorflow doesn't understand why the input is int
        # but the output is float
        fn_output_signature=tf.TensorSpec([None, None], dtype=tf.float32)
    )

    # apply mask to pred
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = y_pred * masks

    # force chars to be predicted in order
    cum_pred = tf.cumsum(
        y_pred, axis=2
    ) / (tf.reduce_sum(y_pred, axis=2, keepdims=True) + 1e-6)
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
    order_loss = tf.reduce_sum(order_loss, axis=-1)
    order_loss = tf.reduce_sum(order_loss, axis=-1)

    # someone has to predict 0 and 1 at some point
    pred_min = tf.reduce_min(y_pred, axis=-1)
    pred_min = tf.reduce_min(pred_min, axis=-1)
    pred_max = tf.reduce_max(y_pred, axis=-1)
    pred_max = tf.reduce_max(pred_max, axis=-1)
    pred_max = 1.0 - pred_max
    return order_loss + 100 * (pred_max + pred_min)
