import tensorflow as tf


def alignment_loss(y_true, y_pred):
    """ Alignment loss

    This method is exclusively self supervised - y_true is not used

    y_pred - alignment predictions as a tensor of shape
        (batch_size, n_chars, n_mel_spectres)
        for each char, its correspondence vector should be 1
        where it corresponds to the mel spectrogram line and 0 otherwise

        y_pred should be softmax-ed in n_chars dimension to make sure
        that each mel_spec is predicted by only 1 char
    """
    # force chars to be predicted in order
    cum_pred = tf.cumsum(y_pred, axis=2) / tf.reduce_sum(y_pred, axis=2,
                                                         keepdims=True)
    v1 = tf.expand_dims(cum_pred, axis=2)
    v2 = tf.expand_dims(cum_pred, axis=1)
    order_loss = tf.reduce_sum(tf.nn.relu(v1 - v2), axis=-1)

    # penalize only when next char comes before previous
    order_loss = tf.linalg.band_part(order_loss, -1, 0)
    order_loss = tf.reduce_sum(order_loss, axis=-1)
    order_loss = tf.reduce_sum(order_loss, axis=-1)
    return order_loss
