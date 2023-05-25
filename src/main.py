import argparse
import datetime
import wandb
import tensorflow as tf
import tensorflow_datasets as tfds
from model import Encoder, Decoder, TC_VAE


def create_directory(dir_name):
    import os
    try:
        os.mkdir(dir_name)
        print('Directory ', dir_name, ' Created ')
    except FileExistsError:
        print('Directory ', dir_name, ' already exists')

def data_generator(dataset):
    iterator = iter(dataset)
    while True:
        try:
            batch = next(iterator)
            yield batch['ecg']['I']
        except StopIteration:
            iterator = iter(dataset)

def main(args):

    ######################################################
    # INITIALIZATION
    ######################################################
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    create_directory(args.path_results)
    # TODO: Check tracking possibilities with wandb
    wandb.init(entity='max_kapsecker', project='vecg', dir=args.path_results, config=args)
    #create_directory(args.path_results + '/models')
    #create_directory(args.path_results + '/logs')
    #create_directory(args.path_results + '/media')

    ######################################################
    # DATA LOADING
    ######################################################
    data = tfds.load(
        'zheng',
        split=['train[:10%]', 'train[10%:20%]', 'train[20%:100%]'],
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

    tc_vae = TC_VAE(encoder, decoder, alpha=0.1, beta=1.5, gamma=0.1)
    tc_vae.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(), loss=tf.keras.losses.MeanAbsoluteError())

    #log_path = args.path_results + '/logs/' + start_time + '/'
    #model_path = args.path_results + '/model/best_vae_' + start_time + '.h5',
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=10),
        # TODO: Check WandDB callbacks
        #tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=50, write_images=True, embeddings_freq=50),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True, monitor='loss'),
    ]

    history = tc_vae.fit(
        data_generator(train),
        steps_per_epoch=len(train),
        epochs=20,
        # validation_data=data_generator(val),
        callbacks=callbacks,
        verbose=1,
    )

    ######################################################
    # TESTING
    ######################################################

    ######################################################
    # RESULTS
    ######################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Representational Learning of ECG using TC-VAE')
    parser.add_argument('--path_results', type=str, default='../results', metavar='N',
                        help='location to save results (default: ../results)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='epochs of train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size for model training (default: 256)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='seed for reproducibility (default: 42)')
    parser.add_argument('--latent_dim', type=int, default=16, metavar='N',
                        help='dimension of the latent vector space (default: 16)')

    args = parser.parse_args()
    main(args)
