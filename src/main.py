import datetime
import argparse
import tensorflow as tf
import numpy as np

from utils import create_directory
from dataset import DatasetZheng
from preprocessing import Preprocess
import model

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

def train_model(x_train, x_val, model, epochs=10, batch_size=512, model_path=None):
    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer, loss=tf.keras.losses.MeanAbsoluteError())
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = '../results/logs/' + time + '/'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "../results/model/best_vae_" + time + ".h5",
            save_best_only=True,
            save_weights_only=True,
            monitor="loss"),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=2000,
            verbose=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_path,
            histogram_freq=50,
            write_images=True,
            embeddings_freq=50),
        WandbMetricsLogger(),
        WandbModelCheckpoint("../results/model"),
        # lvss(x_test, y_test[:, 1], log_path, period=200),
        # rp(x_val[119:120], log_path, period=50),
    ]

    if model_path != None:
        model.fit(x_train, epochs=1, batch_size=512)
        model.built = True
        model.load_weights(model_path)

    model.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(x_val, x_val),
        initial_epoch=0,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECG Clustering')

    parser.add_argument('--preloaded', type=bool, default=True, metavar='N',
                        help='use preloaded data in ../temp folder')
    parser.add_argument('--log_path', type=str, default='../results/log/', metavar='N',
                        help='use log_path in ../results/logs/ folder')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='number of epochs to train (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--latent_dim', type=int, default=8, metavar='N',
                        help='dimension of the latent vector space (default: 32)')
    parser.add_argument('--intermediate_dim', type=int, default=128, metavar='N',
                        help='dimension of the intermediate layer (default: 128)')
    parser.add_argument('--sampling_rate_target', type=int, default=500, metavar='N',
                        help='length of preprocessed signal (default: 500)')
    parser.add_argument('--miliseconds', type=int, default=10000, metavar='N',
                        help='miliseconds the preprocessed signal represents (default: 10000)')
    parser.add_argument('--model', type=str, default=None, metavar='N',
                        help='load stored model')
    parser.add_argument('--beta', type=float, default=1.0, metavar='N',
                        help='beta parameter of beta-VAE (default: 1.0)')

    args = parser.parse_args()

    create_directory('../data')
    create_directory('../temp')
    create_directory('../results/model')
    create_directory('../results/logs')
    create_directory('../results/media')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    seed = args.seed

    wandb.init(
        entity="max_kapsecker",
        project="vecg",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "signal_length": args.sampling_rate_target,
        }
    )

    ######################################################
    # DATA LOADING AND PREPROCESSING
    ######################################################
    train = DatasetZheng('../data/zheng/', args.preloaded)
    train.X = train.X[0:100]
    train.Y = train.Y[0:100]
    preprocessor = Preprocess(250, 250, peak='R', final_length=args.sampling_rate_target)
    X, y, u, q = preprocessor.preprocess(train)

    n = len(X)
    index_val = np.random.choice(np.array(range(0, n)), size=int(0.2 * n), replace=False)
    index_val_select = [x in index_val for x in u]

    x_train = X[np.invert(index_val_select)]
    y_train = y[np.invert(index_val_select)]
    x_val = X[index_val_select]
    y_val = y[index_val_select]

    ######################################################
    # MACHINE LEARNING
    ######################################################
    vae = model.Autoencoder(args.latent_dim)
    train_model(x_train, x_val, vae, epochs=args.epochs, batch_size=args.batch_size, model_path=args.model)

    ######################################################
    # TESTING
    ######################################################
    # encoded_train = model.encoder(X)[0]

    ######################################################
    # RESULTS
    ######################################################
    # encoded_train = model.encoder(X)[0]
