import argparse
import datetime
import os

import tensorflow_datasets as tfds
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras.src.optimizers import RMSprop

from utils.callbacks import ReconstructionPlot, CoefficientSchedulerTCVAE
from utils.helper import Helper

from model.encoder import Encoder
from model.decoder import Decoder
from model.dvae import DVAE
from metrics.loss import TCVAELoss

from evaluate.disentanglement import EvaluateDisentanglement
from evaluate.anomaly import EvaluateAnomalyDetection
from evaluate.personalization import EvaluatePersonalization

os.environ['TFDS_DATA_DIR'] = '/mnt/sdb/home/ml/tensorflow_datasets/'

def main(parameters):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(parameters['seed'])
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = parameters['save_results_path'] + start_time + '/'
    Helper.generate_paths([base_path, base_path + 'evaluation/', base_path + 'model/', base_path + 'reconstruction/'])
    Helper.write_json_file(parameters, base_path + 'params.json')
    Helper.print_available_gpu()

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
    if parameters['load_model'] == 'None':
        loss = TCVAELoss(len(data_train[0]), parameters['coefficients'])
        callbacks = [
            ReduceLROnPlateau(monitor='recon', factor=0.05, patience=10, min_lr=0.000001),
            TerminateOnNaN(),
            CSVLogger(base_path + '/training_progress.csv'),
            CoefficientSchedulerTCVAE(loss, parameters['epochs'], parameters['coefficients_raise'],
                                      parameters['coefficients']),
            #ModelCheckpoint(filepath=base_path + '/model/', monitor='loss', save_best_only=True, verbose=0),
            ReconstructionPlot(train, parameters['index_tracked_sample'], base_path + '/reconstruction/',
                               period=parameters['period_reconstruction_plot']),
            # CollapseCallback(train),
        ]
        encoder = Encoder(parameters['latent_dimension'])
        decoder = Decoder(parameters['latent_dimension'])
        dvae = DVAE(encoder, decoder, loss)
        dvae.compile(optimizer=RMSprop(learning_rate=parameters['learning_rate']))
        dvae.fit(
            Helper.data_generator(train),
            steps_per_epoch=len(train),
            epochs=parameters['epochs'],
            validation_data=Helper.data_generator(val),
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

    evaluateDisentanglement = EvaluateDisentanglement(dvae, base_path)
    evaluateDisentanglement.evaluate(parameters['eval_data_disentanglement'])

    evaluateAnomalyDetection = EvaluateAnomalyDetection(dvae, base_path)
    evaluateAnomalyDetection.evaluate(parameters['eval_data_disentanglement'])


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
