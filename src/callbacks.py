import tensorflow as tf
from sklearn.manifold import TSNE
import wandb
from matplotlib import pyplot as plt
import numpy as np


class KLCoefficientScheduler(tf.keras.callbacks.Callback):

    def __init__(self, alpha_cycle, beta_cycle, gamma_cycle):
        super(KLCoefficientScheduler, self).__init__()
        self.alpha_cycle = alpha_cycle
        self.beta_cycle = beta_cycle
        self.gamma_cycle = gamma_cycle

    def on_epoch_begin(self, epoch, logs=None):
        self.model.alpha = self.alpha_cycle[epoch]
        self.model.beta = self.beta_cycle[epoch]
        self.model.gamma = self.gamma_cycle[epoch]


class LatentVectorSpaceSnapshot(tf.keras.callbacks.Callback):

    def __init__(self, data, labelling, period=1):
        super(LatentVectorSpaceSnapshot, self).__init__()
        self.data = data
        self.labelling = labelling
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            encoded = np.nan_to_num(self.model.encode(self.data)[2])
            x_tsne = TSNE(n_components=2).fit_transform(encoded)
            plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=self.labelling, s=2)
            wandb.log({"lv_space": wandb.Image(plt)})
            plt.close()


class NearestNeighbourPerformance(tf.keras.callbacks.Callback):

    def __init__(self, data, labelling, period=10):
        super(NearestNeighbourPerformance, self).__init__()
        self.data = data
        self.labelling = labelling
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            wandb.log({"nn_performance": None})


class ReconstructionPlot(tf.keras.callbacks.Callback):
    def __init__(self, data, period=1):
        super(ReconstructionPlot, self).__init__()
        self.data = data
        self.n = len(data)
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            reconstructed = self.model.predict(self.data)
            fig, axs = plt.subplots(self.n, 1, sharex=True)

            for k in range(self.n):
                axs[k].plot(self.data[k], color='tab:blue', label='Original')
                axs[k].plot(reconstructed[k], color='tab:orange', label='Reconstructed')
                axs[k].legend()  # Add legend for clarity

            fig.tight_layout()
            plt.xlabel('Time')  # Add x-axis label if needed
            plt.ylabel('mV')  # Add y-axis label if needed

            wandb.log({"reconstruction": wandb.Image(plt)})

            plt.close()
