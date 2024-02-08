import argparse
import datetime
import os

import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.src.optimizers import RMSprop

from evaluate.embedding import Embedding
from evaluate.personalization import Personalization
from utils.callbacks import ReconstructionPlot, CoefficientScheduler, CollapseCallback
from utils.helper import Helper

from model.encoder import Encoder
from model.decoder import Decoder
from model.tcvae import TCVAE

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
    train, size_train = Helper.load_multiple_datasets(parameters['train_dataset'])
    val, size_val = Helper.load_multiple_datasets(parameters['val_dataset'])

    ######################################################
    # MACHINE LEARNING
    ######################################################

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='../logs/'),
        ReduceLROnPlateau(monitor='recon', factor=0.05, patience=20, min_lr=0.000001),
        TerminateOnNaN(),
        CSVLogger(base_path + 'training/training_progress.csv'),
        CoefficientScheduler(parameters['epochs'], parameters['coefficients']),
        ModelCheckpoint(filepath=base_path + 'model/', monitor='loss', save_best_only=True, verbose=0),
        ReconstructionPlot(train[0], parameters['index_tracked_sample'], base_path + 'training/reconstruction/',
                           period=parameters['period_reconstruction_plot']),
        #CollapseCallback(val),
        EarlyStopping(monitor="val_loss", patience=parameters['early_stopping'])
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

    ######################################################
    # EVALUATION
    ######################################################
    embedding = Embedding(vae, base_path)
    personalization = Personalization(vae, base_path)

    for dataset in parameters['encode_data']:
        if parameters['encode_data'][dataset]['fine_tune']:
            personalization.fine_tune_evaluate(parameters['encode_data'][dataset])
        else:
            embedding.evaluate_dataset(parameters['encode_data'][dataset])


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

    main(parameters)

    '''
    for latent_dim in [8, 16]:
        for alpha in [0.5]:
            for beta in [1.0, 4.0, 8.0]:
                for gamma in [0.1, 1.0, 2.0]:
                    parameters['latent_dimension'] = latent_dim
                    parameters['coefficients']['alpha'] = float(alpha)
                    parameters['coefficients']['beta'] = float(beta)
                    parameters['coefficients']['gamma'] = float(gamma)
                    main(parameters)
    '''
