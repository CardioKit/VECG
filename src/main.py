import argparse
import datetime
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from model import Encoder, Decoder, TCVAE
from callbacks import LatentVectorSpaceSnapshot, ReconstructionPlot, KLCoefficientScheduler, CollapseCallback
from evaluate import Evaluate

# import logging
# import wandb
# from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TFDS_DATA_DIR'] = '/mnt/sdb/home/ml/tensorflow_datasets/'


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


def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.5)


def main(arguments):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(arguments.seed)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path_model = arguments.path_results + '/model/best_vae_' + start_time
    path_epoch_log = arguments.path_results + '/logs/epochs_' + start_time + '.log'
    # wandb.init(project=arguments.wandb_project, dir=arguments.path_results, mode=arguments.wandb_mode, config=arguments)
    # wandb_logger = logging.getLogger("wandb")
    # wandb_logger.setLevel(logging.ERROR)

    ######################################################
    # DATA LOADING
    ######################################################
    data_train = tfds.load(
        arguments.dataset,
        split=['train'],
        shuffle_files=True,
    )

    data_val = tfds.load(
        'synth',
        split=['train'],
        shuffle_files=True,
    )

    train = data_train[0].shuffle(1024).batch(arguments.batch_size).prefetch(tf.data.AUTOTUNE)
    val = data_val[0].shuffle(4096).batch(arguments.batch_size).prefetch(tf.data.AUTOTUNE)

    ######################################################
    # MACHINE LEARNING
    ######################################################
    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    encoder = Encoder(arguments.latent_dim)
    decoder = Decoder(arguments.latent_dim)

    tc_vae = TCVAE(encoder, decoder, len(data_train[0]), mss=True, coefficients=tuple(arguments.coefficients))
    tc_vae.compile(optimizer=tf.keras.optimizers.legacy.RMSprop())
    # data_sample, label_sample = get_samples(val, n=4096, label=arguments.label)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='recon', factor=0.05, patience=10, min_lr=0.000001),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(path_model, monitor='val_loss', save_best_only=True),
        # CollapseCallback(data_sample),
        # KLCoefficientScheduler(alpha, beta, gamma),
        # tf.keras.callbacks.CSVLogger(path_epoch_log),
        # WandbMetricsLogger(),
        # WandbModelCheckpoint(path_model, monitor='val_loss', save_best_only=True),
        # tf.keras.callbacks.LearningRateScheduler(scheduler),
        # ReconstructionPlot(get_samples(train, n=4)),
        # LatentVectorSpaceSnapshot(get_samples(val, n=1000, label=arguments.label))
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

    ev = Evaluate(start_time, tc_vae)
    ev.evaluate('zheng', 'train', [50, 100, 150, 200])
    ev.evaluate('medalcare', 'train', [50, 100, 150, 200])
    ev.evaluate('medalcare', 'test', [50, 100, 150, 200])
    ev.evaluate('synth', 'train', [50, 100, 150, 200])
    #ev.evaluate('ptb', 'train', [50, 100, 150, 200])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='VECG', description='Representational Learning of ECG using TC-VAE')
    parser.add_argument('-r', '--path_results', type=str, default='../results',
                        help='location to save results (default: ../results)')
    parser.add_argument('-d', '--dataset', type=str, default='ptb', help='choose tensorflow dataset (default: ptb)')
    parser.add_argument('-l', '--label', type=str, default='quality', help='choose a labelling (default: quality)')
    parser.add_argument('-c', '--coefficients', nargs=3, type=float, default=(0.01, 0.04, 0.01),
                        help='coefficients of KL decomposition (a, b, c)')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='logs of train (default: 100)')
    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help='batch size for model training (default: 256)')
    parser.add_argument('-s', '--seed', type=int, default=42, help='seed for reproducibility (default: 42)')
    parser.add_argument('-ld', '--latent_dim', type=int, default=16,
                        help='dimension of the latent vector space (default: 16)')
    parser.add_argument('-w', '--wandb_mode', type=str, default='online',
                        help='Disable wandb tracking (default: online)')
    parser.add_argument('-p', '--wandb_project', type=str, default='vecg', help='Wandb project name (default: vecg)')

    args = parser.parse_args()
    main(args)
