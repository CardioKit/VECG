import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from metrics.loss import TCVAELoss

from utils.utils import Utils


class CoefficientSchedulerTCVAE(tf.keras.callbacks.Callback):

    def __init__(self, loss: TCVAELoss, epochs, coefficients_raise, coefficients):
        super(CoefficientSchedulerTCVAE, self).__init__()

        self._alpha_cycle = np.arange(epochs) // coefficients_raise * coefficients['alpha']
        self._beta_cycle = np.arange(epochs) // coefficients_raise * coefficients['beta']
        self._gamma_cycle = np.arange(epochs) // coefficients_raise * coefficients['gamma']

    def on_epoch_begin(self, epoch, logs=None):
        self.model._lossEntity.alpha = self._alpha_cycle[epoch]
        self.model._lossEntity.beta = self._beta_cycle[epoch]
        self.model._lossEntity.gamma = self._gamma_cycle[epoch]


class LatentVectorSpaceSnapshot(tf.keras.callbacks.Callback):

    def __init__(self, dataset, sample, path, period=10):
        super(LatentVectorSpaceSnapshot, self).__init__()
        self._data = Utils.get_sample(dataset, sample)
        self._period = period
        self._path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._period == 0:
            encoded = np.nan_to_num(self.model.encode(self._data)[2])
            x_tsne = TSNE(n_components=2).fit_transform(encoded)
            plt.figure(figsize=(8, 5))
            plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=self._labelling, s=2)
            plt.savefig(self._path + 'lv_' + str(epoch) + '.png')
            plt.close()


class ReconstructionPlot(tf.keras.callbacks.Callback):
    def __init__(self, dataset, sample, path, period=5):
        super(ReconstructionPlot, self).__init__()
        self._data = Utils.get_sample(dataset, sample)
        self._period = period
        self._path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._period == 0:
            _, _, z = self.model.encode(self._data)
            reconstructed = self.model.decode(z)
            plt.figure(figsize=(8, 5))
            plt.plot(self._data[0], color='tab:blue', label='Original')
            plt.plot(reconstructed[0], color='tab:orange', label='Reconstructed')
            plt.legend()
            plt.savefig(self._path + 'reconstruction_' + str(epoch) + '.png')
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
                self.model.decoder.trainable = True  # False
            else:
                self.model.encoder.trainable = True
                self.model.decoder.trainable = True  # False
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
