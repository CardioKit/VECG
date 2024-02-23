import os
from model.encoder import Encoder
from model.decoder import Decoder
from model.tcvae import TCVAE
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.src.optimizers import RMSprop
from src.utils.helper import Helper
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

model = tf.keras.models.load_model('../results_round_3/2024-02-16_15-25-04/model_best/')

datasets = {
    'name': ['icentia11k'],
    'shuffle_size': 1024,
    'batch_size': 1024,
}

splits = [
    '107', '5484', '6998', '3984', '3111', '4040', '3013', '6607', '4219', '8750', '5665', '9225',
    '8030', '9886', '1851', '1123', '3043', '3369', '6829', '10969', '3088', '9405', '9535', '4993',
    '4209', '10937', '6167', '4688', '6877', '10733', '8412', '10146', '10973', '9345', '2514', '2908',
    '5938', '5015', '9595', '8769', '4786', '2602', '7779', '2826', '1118', '3485', '2980', '10503',
    '7719', '6575', '1722', '7234', '8366', '3948', '5493', '10731', '8111', '2820', '5337', '5369',
    '4184', '9403', '9625', '303', '33', '3274', '1941', '9116', '9283', '3522', '4836', '7107', '251',
    '9071', '6899', '9733', '9440', '457', '2954', '1839', '5865', '8500', '9559', '1277', '1145', '10107',
    '9287', '8443', '9783', '9956', '10090', '3204', '6814', '4553', '6377', '5572', '1178', '5032', '1793', '4453',
]

ld = 12

for i, split in enumerate(splits):
    print(i, '\t', 'Split:', split)
    datasets.update({'split': split})

    data_train = tfds.load('icentia11k', split=[split])
    train = data_train[0].batch(datasets['batch_size']).prefetch(tf.data.AUTOTUNE)

    encoder = Encoder(ld)
    decoder = Decoder(ld)

    model_personalization = TCVAE(encoder, decoder, {'alpha': 0.2, 'beta': 0.4, 'gamma': 0.2}, len(train))
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
    embedding.to_csv('./embedding_personalization/' + split + '.csv')