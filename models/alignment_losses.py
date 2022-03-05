import tensorflow as tf


def alignment_loss(y_true, y_pred):
    """ Alignment loss

    This method is exclusively self supervised

    y-true contains char_length, melspec_length, ie, the shapes without padding

    y_pred - alignment predictions as a tensor of shape
        (batch_size, n_chars, n_mel_spectres)
        for each char, its correspondence vector should be 1
        where it corresponds to the mel spectrogram line and 0 otherwise

        y_pred should be softmax-ed in n_chars dimension to make sure
        that each mel_spec is predicted by only 1 char
    """
    y_pred = tf.cast(y_pred, tf.float32)
    # force chars to be predicted in order
    cum_pred = tf.cumsum(y_pred, axis=2) / tf.reduce_sum(y_pred, axis=2,
                                                         keepdims=True)
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
