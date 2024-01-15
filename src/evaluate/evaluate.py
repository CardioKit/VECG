import tensorflow as tf
import tensorflow_datasets as tfds
from keras.src.callbacks import CSVLogger
from keras.src.optimizers import RMSprop

from utils.helper import Helper


class Evaluation():

    def __init__(self, model, path_save, seed=42):
        self._model = model
        self._path_save = path_save

    def evaluate_dataset(self, dataset):
        name = dataset['name']
        for split in dataset['splits']:
            path = self._path_save + name + '/' + split
            Helper.generate_paths([path])

            if dataset['fine_tune']:
                epochs = dataset['epochs']
                batch_size = dataset['batch_size']
                data_train = tfds.load(name, split=[split])
                train = data_train[0].batch(batch_size).prefetch(tf.data.AUTOTUNE)

                model_fine_tune = tf.keras.models.clone_model(self._model)
                model_fine_tune.compile(optimizer=RMSprop(learning_rate=0.001))
                model_fine_tune.fit(
                    Helper.data_generator(train),
                    steps_per_epoch=len(train),
                    epochs=epochs,
                    callbacks=CSVLogger(path + '/training_progress.csv'),
                )
                _, _ = Helper.get_embedding(model_fine_tune, name, split, save_path=path)
            else:
                _, _ = Helper.get_embedding(self._model, name, split, save_path=path)
