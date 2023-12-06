import tensorflow as tf
from sklearn.manifold import TSNE
import wandb
from matplotlib import pyplot as plt
import numpy as np


class KLCoefficientScheduler(tf.keras.callbacks.Callback):

    def __init__(self, alpha_cycle, beta_cycle, gamma_cycle):
        super(KLCoefficientScheduler, self).__init__()
        self._alpha_cycle = alpha_cycle
        self._beta_cycle = beta_cycle
        self._gamma_cycle = gamma_cycle

    def on_epoch_begin(self, epoch, logs=None):
        self.model.alpha = self._alpha_cycle[epoch]
        self.model.beta = self._beta_cycle[epoch]
        self.model.gamma = self._gamma_cycle[epoch]


class LatentVectorSpaceSnapshot(tf.keras.callbacks.Callback):

    def __init__(self, data, labelling, period=10):
        super(LatentVectorSpaceSnapshot, self).__init__()
        self._data = data
        self._labelling = labelling
        self._period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._period == 0:
            encoded = np.nan_to_num(self.model.encode(self._data)[2])
            x_tsne = TSNE(n_components=2).fit_transform(encoded)
            plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=self._labelling, s=2)
            wandb.log({"lv_space": wandb.Image(plt)})
            plt.close()


class NearestNeighbourPerformance(tf.keras.callbacks.Callback):

    def __init__(self, data, labelling, period=10):
        super(NearestNeighbourPerformance, self).__init__()
        self._data = data
        self._labelling = labelling
        self._period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._period == 0:
            wandb.log({"nn_performance": None})


class ReconstructionPlot(tf.keras.callbacks.Callback):
    def __init__(self, data, period=10):
        super(ReconstructionPlot, self).__init__()
        self._data = data
        self._n = len(data)
        self._period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._period == 0:
            reconstructed = self.model.predict(self._data)
            fig, axs = plt.subplots(self._n, 1, sharex=True)

            for k in range(self._n):
                axs[k].plot(self._data[k], color='tab:blue', label='Original')
                axs[k].plot(reconstructed[k], color='tab:orange', label='Reconstructed')
                axs[k].legend()  # Add legend for clarity

            fig.tight_layout()
            plt.xlabel('Time')  # Add x-axis label if needed
            plt.ylabel('mV')  # Add y-axis label if needed

            wandb.log({"reconstruction": wandb.Image(plt)})

            plt.close()


class CollapseCallback(tf.keras.callbacks.Callback):
    '''
    CRED: As suggested in https://arxiv.org/pdf/1901.05534.pdf
    CRED: https://github.com/jxhe/vae-lagging-encoder/blob/cdc4eb9d9599a026bf277db74efc2ba1ec203b15/modules/encoders/encoder.py#L111
    '''

    def __init__(self, data, aggressive=True):
        super().__init__()
        self._aggressive = aggressive
        self._data = data
        self._temp = -np.inf

    def on_epoch_begin(self, epoch, logs=None):

        res = self.model.compute_information_gain(self._data)
        self.aggressive = res >= self._temp

        if self.aggressive:
            if self.model.decoder.trainable:
                self.model.decoder.trainable = True #False
            else:
                self.model.encoder.trainable = True
                self.model.decoder.trainable = True #False
            self._temp = res
        else:
            self.model.encoder.trainable = True
            self.model.decoder.trainable = True

    def on_epoch_end(self, epoch, logs=None):

        res = self.model.compute_information_gain(self._data)
        self._aggressive = res >= self._temp
        print(res)
        print(self._temp)
        if self.aggressive:
            if self.model.decoder.trainable:
                self.model.decoder.trainable = False
            else:
                self.model.encoder.trainable = True
                self.model.decoder.trainable = False
            self.temp = res
        else:
            self.model.encoder.trainable = True
            self.model.decoder.trainable = True
