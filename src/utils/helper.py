import os, yaml, codecs, json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA


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
            file_path = save_path + '/' + dataset + '_' + str(split) + '_data.npy'
            Helper.generate_paths([save_path])
            labels.to_csv(save_path + '/' + dataset + '_' + str(split) + '_labels.csv', index=False)
            with open(file_path, 'wb') as f:
                np.save(f, z)
        return z, labels

    @staticmethod
    def embedding(df, labels, method=PCA(n_components=2)):
        x = method.fit_transform(df)
        x = pd.DataFrame(x)
        x = pd.concat([x, labels], axis=1)
        return x

    @staticmethod
    def load_embedding(path, dataset, split):
        X = np.load(path + '/' + dataset + '/' + split + '/' + dataset + '_' + split + '_data.npy')
        latent_dim = X.shape[1]
        y = pd.read_csv(path + '/' + dataset + '/' + split + '/' + dataset + '_' + split + '_labels.csv')
        df = pd.DataFrame(X[:, :, 0])
        df = pd.concat([df, y], axis=1)
        return df, latent_dim
