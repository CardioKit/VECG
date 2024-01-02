import tensorflow as tf
import tensorflow_datasets as tfds
import umap
from keras.src.callbacks import CSVLogger
from keras.src.optimizers import RMSprop
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.helper import Helper
from utils.visualizations import Visualizations


class EvaluatePersonalization():
    def __init__(self, model, path_save):
        self._model = model
        self._path_save = path_save + 'evaluation/personlization/'

    def fine_tune_evaluate(self, datasets, batch_size=256):
        for d in datasets.keys():
            name = datasets[d]['name']
            epochs = datasets[d]['epochs']
            for split in datasets[d]['splits']:
                path = self._path_save + name + '_' + split + '/'
                data_train = tfds.load(name, split=[split])
                train = data_train[0].batch(batch_size).prefetch(tf.data.AUTOTUNE)
                Helper.generate_paths([path])

                model_fine_tune = tf.keras.models.clone_model(self._model)
                model_fine_tune.compile(optimizer=RMSprop(learning_rate=0.001))
                model_fine_tune.fit(
                    Helper.data_generator(train),
                    steps_per_epoch=len(train),
                    epochs=epochs,
                    callbacks=CSVLogger(path + '/training_progress.csv'),
                )

                z, labels = Helper.get_embedding(model_fine_tune, name, split, save_path=path)

                embedding_pca = PCA(n_components=2).fit_transform(z[:, :, 0])
                Visualizations.plot_embedding(embedding_pca, labels=labels, path_eval=path + '/pca')
                embedding_umap = umap.UMAP().fit_transform(z[:, :, 0])
                Visualizations.plot_embedding(embedding_umap, labels=labels, path_eval=path + '/umap')
                embedding_tsne = TSNE().fit_transform(z[:, :, 0])
                Visualizations.plot_embedding(embedding_tsne, labels=labels, path_eval=path + '/tsne')
                # pacmap
