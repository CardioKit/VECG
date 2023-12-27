import numpy as np
import pandas as pd
import umap
import pacmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression

from utils.helper import Helper
from utils.visualizations import Visualizations


class EvaluateDisentanglement():
    def __init__(self, model, path_save):
        self._model = model
        self._path_save = path_save + 'evaluation/disentanglement/'
        Helper.generate_paths([self._path_save])

    def _embedd(self, X, labels, path):
        embedding_tsne = TSNE().fit_transform(X)
        Visualizations.plot_embedding(embedding_tsne, labels=labels, path_eval=path + '/tsne')

        embedding_pca = PCA(n_components=2).fit_transform(X)
        Visualizations.plot_embedding(embedding_pca, labels=labels, path_eval=path + '/pca')

        embedding_umap = umap.UMAP().fit_transform(X)
        Visualizations.plot_embedding(embedding_umap, labels=labels, path_eval=path + '/umap')

        embedding_pacmap = pacmap.PaCMAP().fit_transform(X)
        Visualizations.plot_embedding(embedding_pacmap, labels=labels, path_eval=path + '/pacmap')

    def _dim_toggle(self, z, path, bound_factor=1.2, n=1000):
        mean_z = np.mean(z, axis=0)
        max_z = np.max(z, axis=0)
        min_z = np.min(z, axis=0)
        M = np.tile(mean_z, (n, 1))
        for dimension in range(z.shape[1]):
            bound_up = bound_factor * max_z[dimension]
            bound_low = bound_factor * min_z[dimension]
            M[:, dimension] = np.linspace(bound_low, bound_up, n)
            res = self._model._decoder.predict(M)
            save_path = path + '/reconstruction/'
            Helper.generate_paths([save_path])
            Visualizations.eval_dimensions(res, save_path + 'reconstruction_dim_' + "{:0>{}}".format(dimension, 2))

    def _disentanglement_metrics(self, encoded, labels, path):
        df = pd.DataFrame()
        for label in labels.columns:
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
                            {'label': [str(label)], 'dim': [str(dim)], 'method': ['LR'], 'score': ['failed']}
                            )
                        ]
                    )
                    pass
        df.to_csv(path + '/disentanglement_metrics.csv', index=False)

    def evaluate(self, datasets):
        for d in datasets:
            Helper.generate_paths([self._path_save + d])
            z, labels = Helper.get_embedding(self._model, d, save_path=self._path_save + d)
            self._embedd(z[:, :, 0], labels, self._path_save + d)
            self._dim_toggle(z[:, :, 2], self._path_save + d)
            self._disentanglement_metrics(z[:, :, 0], labels, self._path_save + d)
