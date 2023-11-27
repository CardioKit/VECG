import os

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression


class Evaluate:

    def __init__(self, timestamp, model, config=None):
        self._model = model
        self._timestamp = timestamp
        self._path = '../results/media/' + self._timestamp + '/'
        self.generate_paths([self._path])
        with open(self._path + "config.txt", "w") as text_file:
            mod_str = str(model.get_config())
            text_file.write(mod_str)  # + '\n' + enc_str + '\n' + dec_str)

    def generate_paths(self, paths):
        for path in paths:
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
            except FileExistsError:
                print(f"Directory already exists: {path}")

    def eval_reconstruction(self, X, reconstruction, indices, path_eval, titles=None, xlabel=None, ylabel=None):
        """
        Plots original and reconstructed data for given indices.

        Parameters:
            X (numpy.ndarray): Original data.
            reconstruction (numpy.ndarray): Reconstructed data.
            indices (list): List of indices to plot.
            titles (list, optional): Titles for each subplot.
            xlabel (str, optional): Label for the x-axis.
            ylabel (str, optional): Label for the y-axis.

        Returns:
            None
        """
        assert len(indices) == 4
        num_rows = 2
        num_cols = 2

        fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(15, 5))

        for i, idx in enumerate(indices):
            row = i // num_cols
            col = i % num_cols
            ax[row, col].plot(X[idx], label='Original')
            ax[row, col].plot(reconstruction[idx], label='Reconstruction')

            if titles:
                ax[row, col].set_title(titles[i])

            ax[row, col].legend(loc='upper left')

        if xlabel:
            for a in ax[-1, :]:
                a.set_xlabel(xlabel)
        if ylabel:
            for a in ax[:, 0]:
                a.set_ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(path_eval + 'reconstruction.png')
        plt.close()

    def eval_embedding(self, embedding, label, path_eval, title=None, xlabel=None, ylabel=None, cmap='viridis',
                       marker_size=10, alpha=0.7):
        """
        Plots an embedding scatter plot.

        Parameters:
            X (numpy.ndarray): 2D array of data points (samples x features).
            label (numpy.ndarray): Labels or categories for each data point.
            path (str): File path to save the plot.
            title (str, optional): Title for the plot.
            xlabel (str, optional): Label for the x-axis.
            ylabel (str, optional): Label for the y-axis.
            cmap (str, optional): Colormap for coloring data points.
            marker_size (int, optional): Size of markers in the scatter plot.
            alpha (float, optional): Alpha value for marker transparency.

        Returns:
            None
        """

        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=marker_size, c=label, cmap=cmap, alpha=alpha)

        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        plt.colorbar(label='Label')
        plt.tight_layout()
        plt.savefig(path_eval + '.png')
        plt.close()

    def eval_dimensions(self, encoded, dimensions, path_eval, N=100, bound_factor=1.5):
        # ld = self._model.get_config()['encoder']['config']['latent_dim']
        # M = np.zeros((N, ld)).astype(np.float32)
        mean_enc = np.mean(encoded, axis=0)
        max_enc = np.max(encoded, axis=0)
        min_enc = np.min(encoded, axis=0)
        M = np.tile(mean_enc, (N, 1))
        for dimension in dimensions:
            bound_up = bound_factor * max_enc[dimension]
            bound_low = bound_factor * min_enc[dimension]
            M[:, dimension] = np.linspace(bound_low, bound_up, N)
        res = self._model.decoder.predict(M)
        mean = np.mean(res, axis=0)
        std = np.std(res, axis=0)
        plt.figure(figsize=(15, 5))
        plt.plot(range(0, len(mean)), mean, 'k-')
        plt.fill_between(range(0, len(mean)), mean - std, mean + std)
        plt.savefig(path_eval + '.png')
        plt.close()

    def eval_dimension_interpretation(self, encoded, labels, path):
        df = pd.DataFrame()
        # text = 'label,dim,method,score'
        for label in labels.keys():
            for dim in range(0, encoded.shape[-1]):
                try:
                    reg = LinearRegression().fit(np.array(encoded[:, dim]).reshape(-1, 1),
                                                 np.array(labels[label]))  # .reshape(-1, 1))
                    score = reg.score(np.array(encoded[:, dim]).reshape(-1, 1),
                                      np.array(labels[label]))  # .reshape(-1, 1))
                    df = pd.concat([df, pd.DataFrame(
                        {'label': [str(label)], 'dim': [str(dim)], 'method': ['LR'], 'score': [str(score)]})])
                    # text += str(label) + ',' + str(dim) + ',LR,' + str(score) + '\n'
                except Exception as e:
                    df = pd.concat(
                        [df, pd.DataFrame(
                            {'label': [str(label)], 'dim': [str(dim)], 'method': ['LR'], 'score': ['failed']})])
                    # text += str(label) + ',' + str(dim) + 'LR,failed\n'
                    pass
        df.to_csv(path + '.txt')
        # with open(path + ".txt", "w") as text_file:
        #    text_file.write(text)

    def evaluate(self, dataset, split, indices, batch_size=None):

        path_eval = self._path + dataset + '/' + split + '/'

        self.generate_paths([path_eval])

        data = tfds.load(dataset)
        if batch_size == None:
            ds = data[split].batch(len(data[split]))
        else:
            ds = data[split].batch(batch_size)

        for k in ds.take(1):
            k = k

        X = k['ecg']['I']
        z_mean, z_log_var, z = self._model.encode(X)
        reconstruction = self._model.decode(z)
        self.eval_reconstruction(X, reconstruction, indices, path_eval)

        embedding_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z)
        embedding_pca = PCA(n_components=2).fit_transform(z)

        for label in k.keys():
            try:
                self.eval_embedding(embedding_pca, k[label], path_eval + 'embedding_pca_' + label)
                self.eval_embedding(embedding_tsne, k[label], path_eval + 'embedding_tsne_' + label)
            except Exception as e:
                print(e)
                pass

        for dim in range(0, z.shape[-1]):
            try:
                self.eval_dimensions(z, [dim], path_eval + 'dimension_' + str(dim))
            except Exception as e:
                print(e)
                pass
        self.eval_dimension_interpretation(z, k, path_eval + 'dimension_interpretation')
