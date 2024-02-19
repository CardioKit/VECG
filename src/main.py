import argparse
import datetime
import os

import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.src.optimizers import RMSprop

from utils.callbacks import ReconstructionPlot, CoefficientScheduler, CollapseCallback
from utils.helper import Helper

from model.encoder import Encoder
from model.decoder import Decoder
from model.tcvae import TCVAE

# TODO: Set path to the location of the tensorflow datasets
os.environ['TFDS_DATA_DIR'] = '/mnt/sdb/home/ml/tensorflow_datasets/'


def main(parameters):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(parameters['seed'])
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = parameters['save_results_path'] + start_time + '/'
    Helper.generate_paths(
        [base_path, base_path + 'evaluation/', base_path + 'model_best/', base_path + 'model_final/',
         base_path + 'training/reconstruction/', base_path + 'training/collapse/']
    )
    Helper.write_json_file(parameters, base_path + 'params.json')
    Helper.print_available_gpu()

    ######################################################
    # DATA LOADING
    ######################################################
    train, size_train = Helper.load_multiple_datasets(parameters['train_dataset'])
    val, size_val = Helper.load_multiple_datasets(parameters['val_dataset'])

    ######################################################
    # MACHINE LEARNING
    ######################################################
    callbacks = [
        TerminateOnNaN(),
        CollapseCallback(val, base_path + 'training/collapse/'),
        CSVLogger(base_path + 'training/training_progress.csv'),
        EarlyStopping(monitor="val_loss", patience=parameters['early_stopping']),
        CoefficientScheduler(parameters['epochs'], parameters['coefficients'], parameters['coefficients_raise']),
        ReduceLROnPlateau(monitor='recon', factor=0.05, patience=20, min_lr=0.000001),
        ModelCheckpoint(filepath=base_path + 'model_best/', monitor='loss', save_best_only=True, verbose=0),
        ReconstructionPlot(train[0], base_path + 'training/reconstruction/', parameters['reconstruction']),
    ]

    encoder = Encoder(parameters['latent_dimension'])
    decoder = Decoder(parameters['latent_dimension'])
    vae = TCVAE(encoder, decoder, parameters['coefficients'], size_train)
    vae.compile(optimizer=RMSprop(learning_rate=parameters['learning_rate']))
    vae.fit(
        Helper.data_generator(train), steps_per_epoch=size_train,
        validation_data=Helper.data_generator(val), validation_steps=size_val,
        epochs=parameters['epochs'], callbacks=callbacks, verbose=1,
    )

    vae.save(base_path + 'model_final/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='VECG', description='Representational Learning of ECG using disentangling VAE',
    )
    parser.add_argument(
        '-p', '--path_config', type=str, default='./params.yml',
        help='location of the params file (default: ./params.yml)',
    )

    args = parser.parse_args()
    parameters = Helper.load_yaml_file(args.path_config)

    combinations = [
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.01, 'beta': 0.04, 'gamma': 0.01}},
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.05, 'beta': 0.2, 'gamma': 0.05}},
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.1, 'beta': 0.4, 'gamma': 0.1}},
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.2, 'beta': 0.8, 'gamma': 0.2}},
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.3, 'beta': 1.2, 'gamma': 0.3}},
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.4, 'beta': 1.6, 'gamma': 0.4}},
        {'latent_dimension': 8,  'coefficients': {'alpha': 0.5, 'beta': 2.0, 'gamma': 0.5}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.01, 'beta': 0.04, 'gamma': 0.01}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.05, 'beta': 0.2, 'gamma': 0.05}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.1, 'beta': 0.4, 'gamma': 0.1}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.2, 'beta': 0.8, 'gamma': 0.2}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.3, 'beta': 1.2, 'gamma': 0.3}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.4, 'beta': 1.6, 'gamma': 0.4}},
        {'latent_dimension': 12, 'coefficients': {'alpha': 0.5, 'beta': 2.0, 'gamma': 0.5}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.01, 'beta': 0.04, 'gamma': 0.01}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.05, 'beta': 0.2, 'gamma': 0.05}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.1, 'beta': 0.4, 'gamma': 0.1}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.2, 'beta': 0.8, 'gamma': 0.2}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.3, 'beta': 1.2, 'gamma': 0.3}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.4, 'beta': 1.6, 'gamma': 0.4}},
        {'latent_dimension': 16, 'coefficients': {'alpha': 0.5, 'beta': 2.0, 'gamma': 0.5}},
        {'latent_dimension': 20, 'coefficients': {'alpha': 0.01, 'beta': 0.04, 'gamma': 0.01}},
        {'latent_dimension': 20, 'coefficients': {'alpha': 0.05, 'beta': 0.2, 'gamma': 0.05}},
        {'latent_dimension': 20, 'coefficients': {'alpha': 0.1, 'beta': 0.4, 'gamma': 0.1}},
        {'latent_dimension': 20, 'coefficients': {'alpha': 0.2, 'beta': 0.8, 'gamma': 0.2}},
        {'latent_dimension': 20, 'coefficients': {'alpha': 0.3, 'beta': 1.2, 'gamma': 0.3}},
        {'latent_dimension': 20, 'coefficients': {'alpha': 0.4, 'beta': 1.6, 'gamma': 0.4}},
        {'latent_dimension': 20, 'coefficients': {'alpha': 0.5, 'beta': 2.0, 'gamma': 0.5}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 0.01, 'beta': 0.04, 'gamma': 0.01}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 0.05, 'beta': 0.2, 'gamma': 0.05}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 0.1, 'beta': 0.4, 'gamma': 0.1}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 0.2, 'beta': 0.8, 'gamma': 0.2}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 0.3, 'beta': 1.2, 'gamma': 0.3}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 0.4, 'beta': 1.6, 'gamma': 0.4}},
        {'latent_dimension': 24, 'coefficients': {'alpha': 0.5, 'beta': 2.0, 'gamma': 0.5}},
    ]

    for k in combinations:
        parameters.update(k)
        main(parameters)
