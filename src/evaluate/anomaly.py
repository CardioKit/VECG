from utils.visualizations import Visualizations


class EvaluateAnomalyDetection():
    def __init__(self, model, path_save):
        self._model = model
        self._path_save = path_save + 'evaluation/anomaly/'

    def evaluate(self, datasets, labels):
        for data in datasets:
            z_mean, _, _ = self._model.encode(data)
            Visualizations.plot_embedding(z_mean, labels, self._path_save)
