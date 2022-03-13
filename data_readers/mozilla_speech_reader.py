import os
import tarfile
from io import StringIO

import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_io as tfio


class AudioTarReader:
    def __init__(self, filename, sr=48000):
        assert os.path.isfile(filename), f"{filename} does not exist"
        self.audio_tarfile = filename
        # we know that sampling rate is 48kHz
        self.sr = sr

        if self.audio_tarfile.endswith(".tar"):
            self.audios_tar = tarfile.open(self.audio_tarfile, "r")
        elif self.audio_tarfile.endswith(".tar.gz"):
            self.audios_tar = tarfile.open(self.audio_tarfile, "r|gz")
        else:
            raise ValueError("File must be .tar or .tar.gz")

        # get train.tsv, validated.tsv, test.tsv info
        self._retrieve_info()

        # build vocabulary
        self.tokens = list(
            set(
                " ".join(
                    [
                        str(x).lower()
                        for x in self.data_files["validated.tsv"].sentence.tolist()
                    ]
                )
            )
        )
        self.tokens = sorted(self.tokens) + ["[BOS]", "[EOS]", "[PAD]"]

        # build lookup layers
        self.lookup = tf.keras.layers.StringLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token=None,
            oov_token="[UNK]",
            vocabulary=self.tokens,
        )
        self.lookup_inv = tf.keras.layers.StringLookup(
            max_tokens=None,
            num_oov_indices=1,
            mask_token=None,
            oov_token="[UNK]",
            vocabulary=self.tokens,
            invert=True,
        )

    # aux functions
    def _retrieve_info(self):
        """Retrieves train/dev/test information from file
        Arguments:
        audios_tar - zip file with audios
        tar_file_list - list of files in zip file
        """
        self.data_files = {
            "train.tsv": None,
            "dev.tsv": None,
            "test.tsv": None,
            "validated.tsv": None,
        }

        remaining = 4
        for file in self.audios_tar:
            target_file = file.name.split("/")[-1]
            if target_file in list(self.data_files.keys()):
                with self.audios_tar.extractfile(file) as f:
                    bytes_data = f.read()
                    s = str(bytes_data, "utf-8")
                    data = StringIO(s)
                    df = pd.read_csv(data, sep="\t", low_memory=False)
                    self.data_files[target_file] = df
                remaining -= 1
            if remaining == 0:
                break

    def _retrieve_df(self, split):
        valid_splits = ["train", "test", "validated_only"]
        assert split in valid_splits, f"Valid splits are {valid_splits}"
        if split == "train":
            df = self.data_files["train.tsv"]
        elif split == "test":
            df = self.data_files["test.tsv"]
        else:
            # only files in validated but not train or test
            train_test_samples = list(
                set(self.data_files["train.tsv"].path).union(
                    set(self.data_files["test.tsv"].path)
                )
            )
            train_test_lookup = dict(
                zip(train_test_samples, [True] * len(train_test_samples))
            )
            validated_only = list(
                self.data_files["validated.tsv"].path.map(
                    lambda z: not train_test_lookup.get(z, False)
                )
            )
            df = self.data_files["validated.tsv"][validated_only]
            df = df.reset_index(drop=True)
        return df

    def gen_audios(self, split="validated_only"):
        df = self._retrieve_df(split)
        # print(f'Generating audios from {split}: {len(df)} files')
        # maps string names to where it is in the tarfile
        name_to_id_map = dict(zip(df.path.tolist(), df.index.tolist()))

        for file in self.audios_tar:
            target_file = file.name.split("/")[-1]
            row_id = name_to_id_map.get(target_file, -1)
            if row_id > 0:
                row = df.iloc[row_id]
                audio_data = self.audios_tar.extractfile(file).read()
                yield audio_data, row.sentence.lower(), str(row.age), str(row.gender)

    def write_tfrecords_file(self, target_file, split="validated_only"):
        df = self._retrieve_df(split)
        n_samples = len(df)

        gen = self.gen_audios(split)
        with tf.io.TFRecordWriter(target_file) as file_writer:
            for item in tqdm(gen, total=n_samples, desc=split):
                serialized_tensors = AudioTarReader.serialize(*item)
                file_writer.write(serialized_tensors.numpy())

    # Static methods
    def serialize(audio_data, sentence, age, gender):
        sentence = tf.io.serialize_tensor(sentence)
        age = tf.io.serialize_tensor(age)
        gender = tf.io.serialize_tensor(gender)
        ans = tf.stack([audio_data, sentence, age, gender])
        ans = tf.io.serialize_tensor(ans)
        return ans

    def deserialize(v):
        v = tf.io.parse_tensor(v, tf.string)
        audio_data = v[0]  # tf.io.parse_tensor(v[0], tf.string)
        audio_data = tfio.audio.decode_mp3(audio_data)
        sentence = tf.io.parse_tensor(v[1], tf.string)
        age = tf.io.parse_tensor(v[2], tf.string)
        gender = tf.io.parse_tensor(v[3], tf.string)

        # make sure that those are only really strings
        sentence = tf.ensure_shape(sentence, ())
        age = tf.ensure_shape(age, ())
        gender = tf.ensure_shape(gender, ())
        return audio_data, sentence, age, gender
