import argparse
import datetime
import os

import tensorflow_datasets as tfds
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger, ModelCheckpoint
from keras.src.optimizers import RMSprop

from utils.callbacks import ReconstructionPlot, CoefficientSchedulerTCVAE
from utils.utils import Utils
from metrics.loss import TCVAELoss
from model.encoder import Encoder
from model.decoder import Decoder
from model.dvae import DVAE
from evaluate.evaluate import Evaluate

os.environ['TFDS_DATA_DIR'] = '/mnt/sdb/home/ml/tensorflow_datasets/'

def main(parameters):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(parameters['seed'])
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = '../results/' + start_time + '/'
    Utils.generate_paths([base_path + 'media/', base_path + 'model/', base_path + 'reconstruction/'])
    Utils.write_json_file(parameters, base_path + 'params.json')

    ######################################################
    # DATA LOADING
    ######################################################
    data_train = tfds.load(parameters['train_dataset'], split=['train'], shuffle_files=True)
    train = data_train[0].shuffle(parameters['shuffle_size']).batch(parameters['batch_size']).prefetch(tf.data.AUTOTUNE)
    data_val = tfds.load(parameters['val_dataset'], split=['train'], shuffle_files=True)
    val = data_val[0].shuffle(parameters['shuffle_size']).batch(parameters['batch_size']).prefetch(tf.data.AUTOTUNE)

    ######################################################
    # MACHINE LEARNING
    ######################################################
    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    if parameters['load_model'] == 'None':
        loss = TCVAELoss(len(data_train[0]), parameters['coefficients'])
        callbacks = [
            ReduceLROnPlateau(monitor='recon', factor=0.05, patience=10, min_lr=0.000001),
            TerminateOnNaN(),
            CSVLogger(base_path + 'training_progress.csv'),
            CoefficientSchedulerTCVAE(loss, parameters['epochs'], parameters['coefficients_raise'],
                                      parameters['coefficients']),
            #ModelCheckpoint(filepath=base_path + 'model/', monitor='loss', save_best_only=True, verbose=0),
            ReconstructionPlot(train, parameters['index_tracked_sample'], base_path + 'reconstruction/',
                               period=parameters['period_reconstruction_plot']),
            # CollapseCallback(train),
        ]
        encoder = Encoder(parameters['latent_dimension'])
        decoder = Decoder(parameters['latent_dimension'])
        dvae = DVAE(encoder, decoder, loss)
        dvae.compile(optimizer=RMSprop(learning_rate=parameters['learning_rate']))
        dvae.fit(
            Utils.data_generator(train),
            steps_per_epoch=len(train),
            epochs=parameters['epochs'],
            validation_data=Utils.data_generator(val),
            validation_steps=len(val),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model_path = '../results/' + parameters['load_model'] + '/model/'
        print('Load model from:', model_path)
        dvae = tf.keras.models.load_model(model_path)

    ######################################################
    # EVALUATION
    ######################################################
    ev = Evaluate(base_path, dvae)
    for dataset in parameters['eval_data']:
        ev.evaluate(dataset, 'train')
    ev.evaluate('polar', 'Subject1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='VECG', description='Representational Learning of ECG using TC-VAE',
    )
    parser.add_argument(
        '-p', '--path_config', type=str, default='./params.yml',
        help='location of the params file (default: ./params.yml)',
    )

    args = parser.parse_args()
    parameters = Utils.load_yaml_file(args.path_config)
    main(parameters)
