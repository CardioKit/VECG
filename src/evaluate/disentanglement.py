import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE

from metrics.disentanglement import mutual_information_gap, DisentanglementMetricsInfo
from metrics.similarity import SimilarityMeasure, linear_regressor, mutual_information_regression, kl_divergence_bins
from utils.helper import Helper, dotdict
from utils.visualizations import Visualizations


class EvaluateDisentanglement():
    def __init__(self, model, path_save):
        self._model = model
        self._path_save = path_save + 'evaluation/disentanglement/'
        Helper.generate_paths([self._path_save])

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
        for d in datasets.keys():
            name = datasets[d]['name']
            for split in datasets[d]['splits']:
                path = self._path_save + name + '_' + split
                Helper.generate_paths([path])
                z, labels = Helper.get_embedding(self._model, name, split=split, save_path=path)
                '''
                    'meta_data': labels,
                    'z': z,
                    'discrete_features': [False] * len(labels.columns),
                }
                simMeasure = SimilarityMeasure(dotdict(data), kl_divergence_bins)
                disMetInfo = DisentanglementMetricsInfo(simMeasure, mutual_information_gap)
                print(disMetInfo.compute_score())
                '''

                self._dim_toggle(z[:, :, 2], path)
                self._disentanglement_metrics(z[:, :, 0], labels, path)

                embedding_tsne = TSNE().fit_transform(z[:, :, 0])
                embedding_pca = PCA(n_components=2).fit_transform(z[:, :, 0])
                embedding_umap = umap.UMAP().fit_transform(z[:, :, 0])

                Visualizations.plot_embedding(embedding_tsne, labels=labels, path_eval=path + '/tsne')
                Visualizations.plot_embedding(embedding_pca, labels=labels, path_eval=path + '/pca')
                Visualizations.plot_embedding(embedding_umap, labels=labels, path_eval=path + '/umap')
