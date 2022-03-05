import os

import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm


class LJSpeechReader:
    """ Helper class to retrieve LJSpeech audios and transcriptions
    """

    def __init__(self, ljspeech_folder):
        self.ljspeech_folder = ljspeech_folder
        self.df_audios = pd.read_csv(os.path.join(
            ljspeech_folder, 'metadata.csv'
        ), header=None, sep='|', names=[
            'ID', 'Original_text', 'Normalized_text'
        ])

        # build vocabulary
        self.tokens = list(set(' '.join(
            [str(x).lower() for x in self.df_audios.Normalized_text.tolist()]
        )))
        self.tokens = sorted(self.tokens) + ['[BOS]', '[EOS]', '[PAD]']

        # build lookup layers
        self.lookup = tf.keras.layers.StringLookup(
            max_tokens=None, num_oov_indices=1, mask_token=None,
            oov_token='[UNK]', vocabulary=self.tokens
        )
        self.lookup_inv = tf.keras.layers.StringLookup(
            max_tokens=None, num_oov_indices=1, mask_token=None,
            oov_token='[UNK]', vocabulary=self.tokens, invert=True
        )

    def generate_audios(self, max_audios=None):
        for idx, row in self.df_audios.iterrows():
            if max_audios is not None and idx > max_audios:
                break

            file_name = os.path.join(self.ljspeech_folder, 'wavs',
                                     row.ID + '.wav')
            file_contents = tf.io.read_file(file_name)
            audio, sr = tf.audio.decode_wav(file_contents)

            # reminder: parse tokens with
            # ljsr.lookup(tf.strings.bytes_split(cur_txt))
            if type(row.Normalized_text) == str:
                yield row.ID, row.Normalized_text.lower(), audio, sr

    def write_tfrecords_file(self, target_file):
        gen = self.generate_audios()
        with tf.io.TFRecordWriter(target_file) as file_writer:
            for item in tqdm(gen, total=len(self.df_audios)):
                serialized_tensors = LJSpeechReader.serialize(*item)
                file_writer.write(serialized_tensors.numpy())

    # Static methods
    def serialize(cur_id, text, audio, sr):
        s_id = tf.io.serialize_tensor(cur_id)
        s_text = tf.io.serialize_tensor(text)
        s_audio = tf.audio.encode_wav(audio, sr)
        s_sr = tf.io.serialize_tensor(sr)
        ans = tf.stack([s_id, s_text, s_audio, s_sr])
        ans = tf.io.serialize_tensor(ans)
        return ans

    def deserialize(v):
        v = tf.io.parse_tensor(v, tf.string)
        s_id = tf.io.parse_tensor(v[0], tf.string)
        text = tf.io.parse_tensor(v[1], tf.string)
        s_audio = v[2]
        cur_sr = tf.io.parse_tensor(v[3], tf.int32)
        cur_audio, sr = tf.audio.decode_wav(s_audio)
        return s_id, text, cur_audio, cur_sr
