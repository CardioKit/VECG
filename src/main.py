import argparse
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import logging
from model import Encoder, Decoder, TCVAE
from callbacks import LatentVectorSpaceSnapshot, ReconstructionPlot


def data_generator(dataset):
    iterator = iter(dataset)
    while True:
        try:
            batch = next(iterator)
            yield batch['ecg']['I']
        except StopIteration:
            iterator = iter(dataset)


def get_samples(dataset, n, label=None):
    k = None
    for example in dataset.take(1):
        k = example
    return (k['ecg']['I'][0:n], k[label][0:n]) if label else k['ecg']['I'][0:n]


def main(arguments):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(arguments.seed)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = arguments.path_results + '/model/best_vae_' + start_time
    wandb.init(project='vecg', dir=arguments.path_results, mode=arguments.wandb_mode, config=arguments)
    wandb_logger = logging.getLogger("wandb")
    wandb_logger.setLevel(logging.ERROR)

    ######################################################
    # DATA LOADING
    ######################################################
    data = tfds.load(
        arguments.dataset,
        split=['train[:90%]', 'train[10%:]'],
        shuffle_files=True,
    )

    train = data[0].shuffle(1024).batch(arguments.batch_size).prefetch(tf.data.AUTOTUNE)
    val = data[1].shuffle(1024).batch(arguments.batch_size).prefetch(tf.data.AUTOTUNE)

    ######################################################
    # MACHINE LEARNING
    ######################################################
    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    encoder = Encoder(arguments.latent_dim)
    decoder = Decoder(arguments.latent_dim)

    tc_vae = TCVAE(encoder, decoder, len(data[0]), mss=True, coefficients=tuple(arguments.coefficients))
    tc_vae.compile(optimizer=tf.keras.optimizers.legacy.RMSprop())
    data_sample, label_sample = get_samples(train, n=100, label=arguments.label)

    callbacks = [
        WandbMetricsLogger(),
        WandbModelCheckpoint(model_path, monitor='loss', save_best_only=True),
        ReconstructionPlot(get_samples(train, n=4)),
        LatentVectorSpaceSnapshot(data_sample, label_sample)
    ]

    tc_vae.fit(
        data_generator(train),
        steps_per_epoch=len(train),
        epochs=arguments.epochs,
        validation_data=data_generator(val),
        validation_steps=len(val),
        callbacks=callbacks,
        verbose=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='VECG', description='Representational Learning of ECG using TC-VAE')
    parser.add_argument('-r', '--path_results', type=str, default='../results',
                        help='location to save results (default: ../results)')
    parser.add_argument('-d', '--dataset', type=str, default='zheng', help='choose tensorflow dataset (default: zheng)')
    parser.add_argument('-l', '--label', type=str, default='quality', help='choose a labelling (default: quality)')
    parser.add_argument('-c', '--coefficients', nargs=3, type=float, default=(1.0, 4.0, 1.0),
                        help='coefficients of KL decomposition (a, b, c)')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs of train (default: 50)')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size for model training (default: 256)')
    parser.add_argument('-s', '--seed', type=int, default=42, help='seed for reproducibility (default: 42)')
    parser.add_argument('-ld', '--latent_dim', type=int, default=16, help='dimension of the latent vector space (default: 16)')
    parser.add_argument('-w', '--wandb_mode', type=str, default='online', help='Disable wandb tracking (default: online)')

    args = parser.parse_args()
    main(args)
