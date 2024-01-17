import argparse
import datetime
import os

import tensorflow_datasets as tfds
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.src.optimizers import RMSprop

from evaluate.embedding import Embedding
from evaluate.personalization import Personalization
from utils.callbacks import ReconstructionPlot, CoefficientSchedulerTCVAE
from utils.helper import Helper

from model.encoder import Encoder
from model.decoder import Decoder
from model.dvae import DVAE
from metrics.loss import VAELoss, TCVAELoss, HFVAELoss

os.environ['TFDS_DATA_DIR'] = '/mnt/sdb/home/ml/tensorflow_datasets/'


def main(parameters):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(parameters['seed'])
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = parameters['save_results_path'] + start_time + '/'
    Helper.generate_paths(
        [base_path, base_path + 'evaluation/', base_path + 'model/', base_path + 'training/reconstruction/']
    )
    Helper.write_json_file(parameters, base_path + 'params.json')
    Helper.print_available_gpu()

    ######################################################
    # DATA LOADING
    ######################################################
    data_train = tfds.load(parameters['train_dataset']['name'], split=[parameters['train_dataset']['split']],
                           shuffle_files=True)
    train = data_train[0].shuffle(parameters['train_dataset']['shuffle_size']).batch(
        parameters['train_dataset']['batch_size']).prefetch(tf.data.AUTOTUNE)
    data_val = tfds.load(parameters['val_dataset']['name'], split=[parameters['val_dataset']['split']],
                         shuffle_files=True)
    val = data_val[0].shuffle(parameters['val_dataset']['shuffle_size']).batch(
        parameters['val_dataset']['batch_size']).prefetch(tf.data.AUTOTUNE)

    ######################################################
    # MACHINE LEARNING
    ######################################################
    loss = VAELoss(len(data_train[0]), parameters['coefficients'])
    callbacks = [
        ReduceLROnPlateau(monitor='recon', factor=0.05, patience=50, min_lr=0.000001),
        # TerminateOnNaN(),
        CSVLogger(base_path + 'training/training_progress.csv'),
        CoefficientSchedulerTCVAE(loss, parameters['epochs'], parameters['coefficients_raise'],
                                  parameters['coefficients']),
        # ModelCheckpoint(filepath=base_path + 'model/', monitor='loss', save_best_only=True, verbose=0),
        ReconstructionPlot(train, parameters['index_tracked_sample'], base_path + 'training/reconstruction/',
                           period=parameters['period_reconstruction_plot']),
        # CollapseCallback(train),
        EarlyStopping(monitor="val_loss", patience=parameters['early_stopping'])
    ]
    encoder = Encoder(parameters['latent_dimension'])
    decoder = Decoder(parameters['latent_dimension'])
    dvae = DVAE(encoder, decoder, loss)
    dvae.compile(optimizer=RMSprop(learning_rate=parameters['learning_rate']))
    dvae.fit(
        Helper.data_generator(train), steps_per_epoch=len(train), epochs=parameters['epochs'],
        validation_data=Helper.data_generator(val), validation_steps=len(val),
        callbacks=callbacks, verbose=1,
    )

    ######################################################
    # EVALUATION
    ######################################################
    embedding = Embedding(dvae, base_path)
    personalization = Personalization(dvae, base_path)

    for dataset in parameters['encode_data']:
        if parameters['encode_data'][dataset]['fine_tune']:
            personalization.fine_tune_evaluate(parameters['encode_data'][dataset])
        else:
            embedding.evaluate_dataset(parameters['encode_data'][dataset])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='VECG', description='Representational Learning of ECG using TC-VAE',
    )
    parser.add_argument(
        '-p', '--path_config', type=str, default='./params.yml',
        help='location of the params file (default: ./params.yml)',
    )

    args = parser.parse_args()
    parameters = Helper.load_yaml_file(args.path_config)
    main(parameters)
