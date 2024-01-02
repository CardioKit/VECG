import os, yaml, codecs, json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import umap
from pacmap import pacmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Helper:

    @staticmethod
    def generate_paths(paths):
        for path in paths:
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
            except FileExistsError:
                print(f"Directory already exists: {path}")

    @staticmethod
    def data_generator(dataset, method='continue'):
        iterator = iter(dataset)
        while True:
            try:
                batch = next(iterator)
                yield batch['ecg']['I']
            except StopIteration:
                if method == 'continue':
                    iterator = iter(dataset)
                elif method == 'stop':
                    return

    @staticmethod
    def get_sample(dataset, n, label=None):
        k = None
        for example in dataset.take(1):
            k = example
        return (k['ecg']['I'][n:(n + 1)], k[label][n:(n + 1)]) if label else k['ecg']['I'][n:(n + 1)]

    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 20:
            return lr
        else:
            return lr * tf.math.exp(-0.5)

    @staticmethod
    def load_yaml_file(path):
        with open(path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @staticmethod
    def write_json_file(file, filepath):
        with codecs.open(filepath, 'w', 'utf8') as f:
            f.write(json.dumps(file, sort_keys=True, ensure_ascii=False))

    @staticmethod
    def print_available_gpu():
        print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    @staticmethod
    def embedding(X, metadata, method, method_name, base_path):
        embedding = method.fit_transform(X)
        for label in metadata.keys():
            if (label != 'ecg') & (label != 'quality'):
                try:
                    path_save = base_path + 'embedding_' + method_name + '_' + label
                    #Visualizations.plot_embedding(embedding, metadata[label], path_save)
                except Exception as e:
                    pass


    @staticmethod
    def get_labels(dataset):
        df = pd.DataFrame()
        for data in dataset:
            keys = set(data.keys()) - {'ecg', 'quality'}
            dict = {key: data[key] for key in keys}
            df = pd.concat([df, pd.DataFrame.from_dict(dict, dtype=np.float32)])
        return df

    @staticmethod
    def get_embedding(model, dataset, split='train', save_path=None, batch_size=512):

        data_train = tfds.load(dataset, split=[split])
        train = data_train[0].batch(batch_size).prefetch(tf.data.AUTOTUNE)
        labels = Helper.get_labels(train)
        z_mean, z_log_var = model._encoder.predict(Helper.data_generator(train, method='stop'))
        z = model.reparameterize(z_mean, z_log_var)

        z_mean = np.expand_dims(z_mean, axis=2)
        z_log_var = np.expand_dims(z_log_var, axis=2)
        z = np.expand_dims(z, axis=2)
        z = np.concatenate((z_mean, z_log_var, z), axis=2)

        if save_path != None:
            file_path = save_path + '/' + dataset + '_' + str(split) + '.npy'
            Helper.generate_paths([save_path])
            with open(file_path, 'wb') as f:
                np.save(f, z)
        return z, labels

    @staticmethod
    def embedd(X):
        embedding_tsne = TSNE().fit_transform(X)
        embedding_pca = PCA(n_components=2).fit_transform(X)
        embedding_umap = umap.UMAP().fit_transform(X)
        embedding_pacmap = None #pacmap.PaCMAP().fit_transform(X)
        return embedding_tsne, embedding_pca, embedding_umap, embedding_pacmap
