import os
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.tcvae import TCVAE
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.src.optimizers import RMSprop
from utils.helper import Helper
import pandas as pd

os.environ['TFDS_DATA_DIR'] = '/mnt/sdb/home/ml/tensorflow_datasets/'


def get_embeddings(model, datasets):
    split = datasets['split']
    batch_size = datasets['batch_size']
    result = []
    for dataset in datasets['name']:
        data_train = tfds.load(dataset, split=[split])
        train = data_train[0].batch(batch_size).prefetch(tf.data.AUTOTUNE)
        labels = Helper.get_labels(train)

        df = model._encoder.predict(Helper.data_generator([train], method='stop'))
        df = df[0]  # reparameterize(df[0], df[1]).numpy()
        ld = df.shape[1]

        labels.index = range(0, len(labels))
        df = pd.concat([pd.DataFrame(df), labels], axis=1)
        result.append(df)

    return result, ld


def fine_tune(path_model, datasets, splits, ld, path_save):
    model = tf.keras.models.load_model(path_model)

    for i, split in enumerate(splits):
        print(i, '\t', 'Split:', split)
        datasets.update({'split': split})

        data_train = tfds.load(datasets['name'][0], split=[split])
        train = data_train[0].batch(datasets['batch_size']).prefetch(tf.data.AUTOTUNE)

        encoder = Encoder(ld)
        decoder = Decoder(ld)

        model_personalization = TCVAE(encoder, decoder, {'alpha': 0.0, 'beta': 0.0, 'gamma': 0.0}, len(train))
        model_personalization.compile(optimizer=RMSprop(learning_rate=0.001))
        model_personalization.set_weights(model.get_weights())
        model_personalization.fit(
            Helper.data_generator([train]),
            steps_per_epoch=len(train),
            epochs=20, verbose=1,
        )
        embeddings, ld = get_embeddings(model_personalization, datasets)
        embedding = embeddings[0]
        embedding['split'] = split
        embedding.to_csv(path_save + split + '.csv')
