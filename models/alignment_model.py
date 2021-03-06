import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as L


def get_spectrogram_model(n_samples=None, n_fft=1024):
    inp = L.Input((n_samples,))

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(inp, frame_length=1024, frame_step=256, fft_length=n_fft)
    spectrograms = tf.abs(stfts)
    return Model(inputs=inp, outputs=spectrograms, name="spectrogram")


def get_melspec_model(
    n_samples=None,
    n_fft=1024,
    sample_rate=22050,
    frame_length=1024,
    frame_step=256,
    lower_edge_hertz=60.0,
    upper_edge_hertz=7700.0,
    num_mel_bins=80,
):
    inp = L.Input((n_samples,))

    stfts = tf.signal.stft(
        inp, frame_length=frame_length, frame_step=frame_step, fft_length=n_fft
    )
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = n_fft // 2 + 1  # stfts.shape[-1].value

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return Model(inputs=inp, outputs=log_mel_spectrograms, name="log_mel_spectrogram")


class PraticantoForcedAligner:
    def __init__(
        self,
        sampling_rate,
        vocab,
        n_mels=80,
        emb_size=16 + 4,
        rnn_cells=128 + 32,
        proj_dim=256 + 32,
        frame_length=1024,
        frame_step=256,
        # if using CNN instead of RNN
        use_cnn=False,
        cnn_filters=96,
        cnn_num_convs=4,
        kernel_size_spec=9,
        kernel_size_char=5,
    ):
        # char encoder parameters
        self.vocab = vocab
        self.emb_size = emb_size
        self.rnn_cells = rnn_cells
        self.proj_dim = proj_dim

        # mel-spectrogram parameters
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step

        # CNN parameters
        self.use_cnn = use_cnn
        self.cnn_filters = cnn_filters
        self.kernel_size_char = kernel_size_char
        self.kernel_size_spec = kernel_size_spec
        self.cnn_num_convs = cnn_num_convs

    def build_models(self):
        self.MelSpectrogram = get_melspec_model(
            sample_rate=self.sampling_rate,
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
        )
        self.model_char_enc = self.char_seq_encoder()
        self.model_melspec_enc = self.melspec_encoder()

        inp1 = L.Input((None,), name="char_seq", dtype=tf.string)
        inp2 = L.Input((None,), name="waveform")
        char_enc = self.model_char_enc(inp1)
        spec = self.MelSpectrogram(inp2)
        spec_enc = self.model_melspec_enc(spec)

        # predict alignments
        char_align = L.Dot([2, 2])([char_enc, spec_enc])
        # char_align = L.Activation('sigmoid', name='output')(char_align)
        char_align = tf.keras.activations.softmax(char_align, axis=1)

        # postprocessing: figure out the region of each char in
        # (batch_size, n_chars, n_mels)
        # so that reducing n_mels gives the region that corresponds to the char

        """ constraints:
        - all audio must be predicted
        - no overlap? each char in sequence
        - force 1 prediction? NO
        """

        self.model_aligner = Model(inputs=[inp1, inp2], outputs=char_align)
        return self.model_aligner

    def char_seq_encoder(self):
        # note: vocab should probably include a [PAD] token
        self.char_table = tf.keras.layers.StringLookup(vocabulary=self.vocab)
        self.inv_char_table = tf.keras.layers.StringLookup(
            vocabulary=self.vocab, invert=True
        )
        inp = L.Input((None,), dtype=tf.string)
        x = self.char_table(inp)
        x = L.Embedding(self.char_table.vocabulary_size(), self.emb_size)(x)
        emb_x = x

        # encoding
        if self.use_cnn:
            # x shape is (batch_size, n_chars, emb_size)
            x = L.Conv1D(
                filters=self.cnn_filters,
                kernel_size=self.kernel_size_char,
                padding="same",
            )(x)
            for k in range(2 * self.cnn_num_convs):
                xbkp = x
                x = L.Conv1D(
                    filters=self.cnn_filters,
                    kernel_size=self.kernel_size_char,
                    padding="same",
                )(x)
                x = L.BatchNormalization()(x)
                x = L.Activation("relu")(x)
                x = L.Add()([x, xbkp])
        else:
            x = L.Bidirectional(L.LSTM(self.rnn_cells, return_sequences=True))(x)
            x = L.Bidirectional(L.LSTM(self.rnn_cells, return_sequences=True))(x)
        x = L.Concatenate()([emb_x, x])
        x = L.Dense(self.proj_dim)(x)
        return Model(inputs=inp, outputs=x, name="char_encoder")

    def melspec_encoder(self):
        # shape is (batch_size, seq_len, n_mels)
        inp = L.Input((None, self.n_mels))
        x = inp

        # TODO: check the need for preliminary dense layers here
        x = L.Dense(self.rnn_cells * 2, activation="relu")(x)
        xbkp = x
        x = L.Dense(self.rnn_cells * 2, activation="relu")(x)
        x = L.Dense(self.rnn_cells * 2, activation="relu")(x)
        x = L.Add()([xbkp, x])

        # encoding
        if self.use_cnn:
            x = L.Conv1D(
                filters=self.cnn_filters,
                kernel_size=self.kernel_size_spec,
                padding="same",
            )(x)
            for k in range(self.cnn_num_convs):
                xbkp = x
                x = L.Conv1D(
                    filters=self.cnn_filters,
                    kernel_size=self.kernel_size_spec,
                    padding="same",
                )(x)
                x = L.BatchNormalization()(x)
                x = L.Activation("relu")(x)
                x = L.Add()([x, xbkp])
            x = L.Concatenate()([xbkp, x])
        else:
            x = L.Bidirectional(L.LSTM(self.rnn_cells, return_sequences=True))(x)
            # x = L.Bidirectional(L.LSTM(self.rnn_cells, return_sequences=True))(x)
            x = L.Add()([xbkp, x])
        x = L.Dense(self.proj_dim)(x)
        return Model(inputs=inp, outputs=x, name="mel_encoder")
