import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from utils.helper import Helper


class Visualizations:

    @staticmethod
    def plot_embedding(embedding, labels):
        NotImplementedError

    @staticmethod
    def pair_plot(embedding, labels):
        NotImplementedError

    @staticmethod
    def eval_reconstruction(X, reconstruction, indices, path_eval, titles=None, xlabel=None, ylabel=None):
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

    @staticmethod
    def eval_dimensions(res, path_eval):
        mean = np.mean(res, axis=0)
        std = np.std(res, axis=0)
        plt.figure(figsize=(15, 5))
        plt.plot(range(0, len(mean)), mean, 'k-')
        plt.fill_between(range(0, len(mean)), mean - std, mean + std)
        plt.savefig(path_eval + '.png')
        plt.close()

    @staticmethod
    def plot_trainings_process(train_progress, metrics):
        plt.figure(figsize=(10, 5))
        for k in metrics:
            sns.lineplot(train_progress, x='epoch', y=k)
        #ax.set_yscale("log")
