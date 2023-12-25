import numpy as np
import pandas as pd
import umap
import pacmap
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression

from utils.utils import Utils


class Evaluate:

    def __init__(self, base_path, model, config=None):
        self._model = model
        self._path = base_path
        Utils.generate_paths([self._path])
        with open(self._path + "config.txt", "w") as text_file:
            mod_str = str(model.get_config())
            text_file.write(mod_str)

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

    def eval_embedding(self, embedding, label, path_eval, loc_scaling=False,
                    title=None, xlabel=None, ylabel=None, cmap='viridis', marker_size=10, alpha=0.7):
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
        if loc_scaling:
            embedding[:, 0] = np.log((embedding[:, 0] - np.min(embedding[:, 0])) + 1.0)
            embedding[:, 1] = np.log((embedding[:, 1] - np.min(embedding[:, 1])) + 1.0)

        df = pd.DataFrame(embedding, columns=['First Axis', 'Second Axis'])
        df['label'] = label
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x="First Axis", y="Second Axis", hue="label", s=8)
        plt.savefig(path_eval + '.png')
        plt.close()

    def eval_dimensions(self, encoded, dimensions, path_eval, N=100, bound_factor=1.5):

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
        for label in labels.keys():
            for dim in range(0, encoded.shape[-1]):
                try:
                    reg = LinearRegression().fit(np.array(encoded[:, dim]).reshape(-1, 1),
                                                 np.array(labels[label]))
                    score = reg.score(np.array(encoded[:, dim]).reshape(-1, 1),
                                      np.array(labels[label]))
                    df = pd.concat([df, pd.DataFrame(
                        {'label': [str(label)], 'dim': [str(dim)], 'method': ['LR'], 'score': [str(score)]})])
                except Exception as e:
                    df = pd.concat(
                        [df, pd.DataFrame(
                            {'label': [str(label)], 'dim': [str(dim)], 'method': ['LR'], 'score': ['failed']})])
                    pass
        df.to_csv(path + '.txt')

    def evaluate(self, dataset, split, batch_size=50000):

        path_eval = self._path + 'media/' + dataset + '/' + split + '/'
        Utils.generate_paths([path_eval])
        data = tfds.load(dataset)
        ds = data[split].shuffle(128).batch(batch_size)

        iterator = iter(ds)
        batch = next(iterator)
        X = batch['ecg']['I']
        z_mean, z_log_var, z = self._model.encode(X)

        embedding_tsne = TSNE(
            n_components=2,
            learning_rate='auto',
            init='random',
            perplexity=3,
        ).fit_transform(z_mean)

        embedding_pca = PCA(
            n_components=2
        ).fit_transform(z_mean)

        embedding_umap = umap.UMAP().fit_transform(z_mean)

        embedding_pacmap = pacmap.PaCMAP(
            n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0
        ).fit_transform(z_mean, init="pca")


        for label in batch.keys():
            if (label != 'ecg') & (label != 'quality'):
                try:
                    self.eval_embedding(embedding_pca, batch[label], path_eval + 'embedding_pca_' + label)
                    self.eval_embedding(embedding_tsne, batch[label], path_eval + 'embedding_tsne_' + label)
                    self.eval_embedding(embedding_umap, batch[label], path_eval + 'embedding_umap_' + label)
                    self.eval_embedding(embedding_pacmap, batch[label], path_eval + 'embedding_pacmap_' + label)
                except Exception as e:
                    pass

        for dim in range(0, z.shape[-1]):
            try:
                self.eval_dimensions(z, [dim], path_eval + 'dimension_' + str(dim))
            except Exception as e:
                pass
        self.eval_dimension_interpretation(z, batch, path_eval + 'dimension_interpretation')
