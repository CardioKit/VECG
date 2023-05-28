import argparse
import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
import wandb.keras as cb
import logging
from model import Encoder, Decoder, TC_VAE
from callbacks import KLCoefficientScheduler, LatentVectorSpaceSnapshot, ReconstructionPlot


def data_generator(dataset):
    iterator = iter(dataset)
    while True:
        try:
            batch = next(iterator)
            yield batch['ecg']['I']
        except StopIteration:
            iterator = iter(dataset)


def get_samples(dataset, n=4):
    ds = dataset.shuffle(1024).batch(n).prefetch(tf.data.AUTOTUNE)
    for example in ds.take(1):
        k = example
    return k['ecg']['I']


def main(args):
    ######################################################
    # INITIALIZATION
    ######################################################
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = args.path_results + '/model/best_vae_' + start_time
    wandb.init(project='vecg', dir=args.path_results, config=args)
    wandb_logger = logging.getLogger("wandb")
    wandb_logger.setLevel(logging.ERROR)

    ######################################################
    # DATA LOADING
    ######################################################
    data = tfds.load(
        args.dataset,
        split=['train[:10%]', 'train[10%:11%]', 'train[11%:100%]'],
        shuffle_files=True,
    )

    train = data[0].shuffle(1024).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val = data[1].shuffle(1024).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test = data[2].shuffle(1024).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    ######################################################
    # MACHINE LEARNING
    ######################################################
    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    encoder = Encoder(args.latent_dim)
    decoder = Decoder(args.latent_dim)

    tc_vae = TC_VAE(encoder, decoder, tuple(args.coefficients))
    tc_vae.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(), loss=tf.keras.losses.MeanAbsoluteError())

    callbacks = [
        cb.WandbMetricsLogger(),
        cb.WandbModelCheckpoint(model_path),
        ReconstructionPlot(get_samples(train)),
    ]

    history = tc_vae.fit(
        data_generator(train),
        steps_per_epoch=len(train),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='VECG', description='Representational Learning of ECG using TC-VAE')
    parser.add_argument('--path_results', type=str, default='../results',
                        help='location to save results (default: ../results)')
    parser.add_argument('--dataset', type=str, default='zheng', help='choose tensorflow dataset (default: zheng)')
    parser.add_argument('--coefficients', nargs=3, type=float, default=(1.0, 1.0, 1.0),
                        help='coefficients of KL decomposition (a, b, c)')
    parser.add_argument('--epochs', type=int, default=1, help='epochs of train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for model training (default: 256)')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility (default: 42)')
    parser.add_argument('--latent_dim', type=int, default=16, help='dimension of the latent vector space (default: 16)')

    args = parser.parse_args()
    main(args)
