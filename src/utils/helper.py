import os, yaml, codecs, json
import numpy as np
import pandas as pd
import glob
import ipywidgets as widgets
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from src.metrics.disentanglement import Disentanglement
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

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
        k = 0
        n = len(dataset)
        iterator = iter(dataset[k])
        while True:
            try:
                batch = next(iterator)
                yield batch['ecg']['I']
            except StopIteration:
                if method == 'continue':
                    if k == n - 1:
                        k = 0
                    else:
                        k = k + 1
                    iterator = iter(dataset[k])
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
        z_mean, z_log_var = model._encoder.predict(Helper.data_generator([train], method='stop'))
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
        X = np.load(path + dataset + '/' + split + '/' + dataset + '_' + split + '_data.npy')
        latent_dim = X.shape[1]
        y = pd.read_csv(path + '/' + dataset + '/' + split + '/' + dataset + '_' + split + '_labels.csv')
        df = pd.DataFrame(X[:, :, 0])
        df = pd.concat([df, y], axis=1)
        return df, latent_dim

    @staticmethod
    def load_multiple_datasets(datasets):
        size = 0
        data_list = []
        for i, k in enumerate(datasets['name']):
            temp = tfds.load(k, split=[datasets['split']], shuffle_files=True)
            data = temp[0].shuffle(datasets['shuffle_size']).batch(datasets['batch_size']).prefetch(tf.data.AUTOTUNE)
            size = size + len(data)
            data_list.append(data)
        return data_list, size

    @staticmethod
    def load_dataset(dataset):
        temp = tfds.load(dataset['name'],split=[dataset['split']], shuffle_files=True)
        data = temp[0].shuffle(dataset['shuffle_size']).batch(dataset['batch_size']).prefetch(tf.data.AUTOTUNE)
        return data, len(temp[0])

    @staticmethod
    def feature_axis_mapping(embeddings, ld):
        n = len(embeddings)
        struct = []
        for ind in range(0, n):
            df = embeddings[ind].fillna(0.0)
            cols = df.columns
            X = np.array(df.iloc[:, 0:ld]).reshape(-1, ld)
            for j in cols[ld:]:
                y = df.loc[:, j]
                if len(np.unique(np.array(y))) > 1:
                    for k in range(0, ld):
                        reg = LinearRegression().fit(X[:, k].reshape(-1, 1), y)
                        score = reg.score(X[:, k].reshape(-1, 1), y)
                        struct.append({'Dim': k, 'Feature': j, 'Score': score})
        return struct

    @staticmethod
    def readable_axis_mapping(struct):
        merged = {}

        for item in struct:
            dim_key = f'Dim {item["Dim"]}'
            if dim_key not in merged:
                merged[dim_key] = {'Features': [], 'Scores': []}
            index = 0
            for i, score in enumerate(merged[dim_key]['Scores']):
                if item['Score'] > score:
                    index = i
                    break
                elif i == len(merged[dim_key]['Scores']) - 1:
                    index = i + 1
            merged[dim_key]['Features'].insert(index, item['Feature'])
            merged[dim_key]['Scores'].insert(index, item['Score'])
        return merged

    @staticmethod
    def axis_feature_mapping(df, ld):
        cols = df.columns
        for j in cols[ld:]:
            max_score = 0.0
            dimension = None
            for k in cols[:ld]:
                X = np.array(df.loc[:, k]).reshape(-1, 1)
                y = df.loc[:, j]
                try:
                    reg = LinearRegression().fit(X, y)
                    score = reg.score(X, y)
                    if score > max_score:
                        dimension = k
                        max_score = score
                except:
                    continue

            print('%20s' % j, '-', dimension, ':', '\t', np.round(max_score, 10))

    @staticmethod
    def experiments(datasets, path, filter='2024-01-01 00:00:00'):
        df = pd.DataFrame()
        for k in glob.glob(path + '*/'):
            params = Helper.load_yaml_file(k + 'params.json')
            try:
                train_progress = pd.read_csv(k + 'training/training_progress.csv')
                n = len(train_progress) - 1
                train_progress = train_progress.loc[n:n,
                                 ['alpha', 'beta', 'gamma', 'loss', 'recon', 'mi', 'tc', 'dw_kl']]
                train_progress['time'] = pd.to_datetime(k[len(path):-1], format='%Y-%m-%d_%H-%M-%S')
                train_progress['latent_dim'] = params['latent_dimension']
                train_progress['epoch'] = n + 1
                df = pd.concat([df, train_progress])
            except Exception as e:
                print('failed on:', k, e)
                continue
        df.reset_index(inplace=True, drop=True)
        df = df[(df.time > filter)].sort_values('time').reset_index(drop=True)
        df['total'] = df.recon + df.mi + df.tc + df.dw_kl

        for i, val in enumerate(df.time):
            model = tf.keras.models.load_model(
                path + str(val).replace(' ', '_').replace(':', '-') + '/model_best/')
            emb, ld = Helper.get_embeddings(model, datasets)
            mus_train = np.array(emb[0].iloc[:, :ld])
            ys_train = np.array(emb[0].loc[:, ['p_height', 't_height']])
            df.loc[i, 'MIG'] = Disentanglement.compute_mig(mus_train, ys_train)['discrete_mig']

        return df

    @staticmethod
    def calculate_distances(q, ld):
        mean = np.mean(q.iloc[:, 0:ld], axis=0)
        return np.mean(np.sqrt(np.sum(np.power(q.iloc[:, 0:ld] - mean, 2), axis=1)))

    @staticmethod
    def select_path(path):
        return widgets.Dropdown(
            options=sorted(glob.glob(path + '*/')),
            description='Base path:',
            disabled=False,
        )

    @staticmethod
    def number_to_category(df):
        df.diagnosis = df.diagnosis.replace(
            0.0, 'avblock'
        ).replace(
            1.0, 'fam'
        ).replace(
            2.0, 'iab'
        ).replace(
            3.0, 'lae'
        ).replace(
            4.0, 'lbbb'
        ).replace(
            5.0, 'mi'
        ).replace(
            6.0, 'rbbb'
        ).replace(
            7.0, 'sinus'
        )
        return df

    @staticmethod
    def reparameterize(mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return tf.add(mean, tf.multiply(eps, tf.exp(log_var * 0.5)), name="sampled_latent_variable")

    @staticmethod
    def get_embeddings(model, datasets):
        split = datasets['split']
        batch_size = datasets['batch_size']
        result = []
        for dataset in datasets['name']:
            data_train = tfds.load(dataset, split=[split])
            train = data_train[0].batch(batch_size).prefetch(tf.data.AUTOTUNE)
            labels = Helper.get_labels(train)

            df = model._encoder.predict(Helper.data_generator([train], method='stop'))
            df = df[0]
            ld = df.shape[1]

            labels.index = range(0, len(labels))
            df = pd.concat([pd.DataFrame(df), labels], axis=1)
            result.append(df)

        return result, ld

    @staticmethod
    def get_icentia_embedding(splits, model):
        datasets = {
            'name': ['icentia11k'],
            'shuffle_size': 1024,
            'batch_size': 1024,
        }
        df = pd.DataFrame()
        for i, k in enumerate(splits):
            datasets.update({'split': k})
            df_pers = Helper.get_embeddings(model, datasets)
            df_pers = df_pers[0][0]
            df_pers['subject'] = k
            df = pd.concat([df, df_pers])
        df.beat = df.beat.replace(0.0, 'Normal').replace(1.0, 'Unclassified').replace(2.0, 'PAC').replace(3.0, 'PVC')
        df = df[df.beat != 'Unclassified']
        return df


    @staticmethod
    def cross_validation_knn(X_train, X_val, y_train, y_val):
        X_combined = np.vstack((X_train, X_val))
        y_combined = np.concatenate((y_train, y_val))

        test_fold = np.concatenate([
            -np.ones(X_train.shape[0]),
            np.zeros(X_val.shape[0])
        ])

        ps = PredefinedSplit(test_fold)

        imputer = SimpleImputer(strategy='constant', fill_value=0)

        knn = KNeighborsClassifier()

        pipeline = Pipeline(steps=[('imputer', imputer), ('knn', knn)])

        param_grid = {
            'knn__n_neighbors': range(3, 50)
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=ps, scoring='accuracy')

        grid_search.fit(X_combined, y_combined)

        best_k = grid_search.best_params_['knn__n_neighbors']
        best_score = grid_search.best_score_

        print(f'Best k: {best_k} with accuracy: {best_score}')
        return best_k

