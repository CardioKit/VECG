import tensorflow as tf
import tensorflow_datasets as tfds
from keras.src.callbacks import CSVLogger
from keras.src.optimizers import RMSprop

from utils.helper import Helper


class Personalization:
    def __init__(self, model, path_save):
        self._model = model
        self._path_save = path_save + 'evaluation/personalization/'

    def fine_tune_evaluate(self, dataset, batch_size=256):
        name = dataset['name']
        epochs = dataset['epochs']
        for split in dataset['splits']:
            path = self._path_save + name + '/' + split + '/'
            data_train = tfds.load(name, split=[split])
            train = data_train[0].batch(batch_size).prefetch(tf.data.AUTOTUNE)
            Helper.generate_paths([path])

            #model_fine_tune = tf.keras.models.clone_model(self._model)
            #model_fine_tune.compile(optimizer=RMSprop(learning_rate=0.001))
            self._model.fit(
                Helper.data_generator([train]),
                steps_per_epoch=len(train),
                epochs=epochs,
                callbacks=CSVLogger(path + '/training_progress.csv'),
            )

            _, _ = Helper.get_embedding(self._model, name, split, save_path=path)
