import os
import argparse
import datetime

import tensorflow as tf
from tqdm.keras import TqdmCallback

from data_readers.mozilla_speech_reader import AudioTarReader
from models.alignment_model import PraticantoForcedAligner
from models import alignment_losses


def prep_batch_inputs(cur_txt, cur_audio, seq_lengths):
    return {
        "char_seq": cur_txt,
        "waveform": cur_audio,
    }, seq_lengths


def scheduler(epoch, lr):
    if epoch <= 1:
        return 1e-4
    elif epoch <= 60:
        return 1e-3
    elif epoch <= 120:
        return 1e-4
    elif epoch <= 180:
        return 2e-5
    else:
        return 1e-5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Input file for training from Mozilla Speech Corpus",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--architecture",
        "-a",
        type=str,
        default="rnn",
        help="Architecture: `rnn` or `cnn`",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="",
        help="Checkpoint to load",
    )
    args = parser.parse_args()
    print(args)
    assert os.path.isfile(args.file), f"File not found: {args.file}"
    assert args.architecture.lower() in ["cnn", "rnn"], "Invalid architecture"

    atr = AudioTarReader(args.file)
    print("Preparing tfrecords files...")
    os.makedirs("data", exist_ok=True)
    data_file = "data/validated_not_traintest.tfrecords"
    if not os.path.isfile(data_file):
        atr.write_tfrecords_file(data_file)

    data_file = "data/train.tfrecords"
    if not os.path.isfile(data_file):
        atr.write_tfrecords_file(data_file, split="train")

    print("Creating model:")
    pfa = PraticantoForcedAligner(
        vocab=atr.tokens, sampling_rate=48000, use_cnn=args.architecture == "cnn"
    )
    alignment_model = pfa.build_models()
    alignment_model.summary()

    def prep_inputs(cur_audio, sentence, age, gender):
        cur_txt = tf.ensure_shape(sentence, ())
        cur_txt = tf.strings.unicode_split(cur_txt, "UTF-8")
        cur_txt = tf.concat([["[BOS]"], cur_txt, ["[EOS]"]], axis=0)

        shapes = tf.concat(
            [
                tf.shape(cur_txt),
                1 + (tf.shape(cur_audio[:, 0]) - pfa.frame_length) // pfa.frame_step,
            ],
            axis=0,
        )
        return cur_txt, cur_audio[:, 0], shapes

    dataset = tf.data.TFRecordDataset(
        ["data/validated_not_traintest.tfrecords", "data/train.tfrecords"],
        num_parallel_reads=2,
    )
    n_audio_samples = sum([1 for x in dataset])
    print(f"Training on {n_audio_samples} samples")

    batch_size = args.batch_size
    dataset = tf.data.TFRecordDataset(
        ["data/validated_not_traintest.tfrecords", "data/train.tfrecords"],
        num_parallel_reads=2,
    )
    dataset = (
        dataset.shuffle(5 * batch_size + 16)
        .repeat()
        .map(AudioTarReader.deserialize, num_parallel_calls=tf.data.AUTOTUNE)
        .map(prep_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size, padding_values=("[PAD]", 0.0, 0), drop_remainder=True)
        .map(prep_batch_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model_losses = [
        alignment_losses.alignment_loss(x) for x in alignment_losses.possible_losses
    ]
    print(f"Losses: {model_losses}")

    alignment_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3, clipnorm=0.1, beta_1=0.8, beta_2=0.99, epsilon=0.1
        ),
        loss=model_losses[0],
        metrics=model_losses[1:],
    )
    if os.path.isfile(args.checkpoint + ".index"):
        print(f"Loading weights from {args.checkpoint}")
        alignment_model.load_weights(args.checkpoint)
    else:
        print(f"Start training from scratch")

    # callbacks
    os.makedirs("checkpoints", exist_ok=True)
    filepath = "checkpoints/m_{epoch}_{loss:.3f}.chkpt"
    chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    """
    reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=10, verbose=1,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7
    )
    """

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    alignment_model.fit(
        dataset,
        epochs=300,
        steps_per_epoch=n_audio_samples // batch_size,
        callbacks=[
            lr_callback,
            chkpt_callback,
            tensorboard_callback,
            TqdmCallback(verbose=2),
        ],
        verbose=0,
    )


if __name__ == "__main__":
    main()
