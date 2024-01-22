import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from utils.helper import Helper


class CoefficientScheduler(tf.keras.callbacks.Callback):

    def __init__(self, epochs, coefficients):
        super(CoefficientScheduler, self).__init__()

        self._alpha_cycle = np.ones(epochs) * coefficients['alpha']
        self._beta_cycle = np.ones(epochs) * coefficients['beta']
        self._gamma_cycle = np.ones(epochs) * coefficients['gamma']

        # np.arange(epochs) // coefficients_raise * coefficients['alpha']
        # np.arange(epochs) // coefficients_raise * coefficients['beta']
        # np.arange(epochs) // coefficients_raise * coefficients['gamma']

    def on_epoch_begin(self, epoch, logs=None):
        self.model.alpha = tf.cast(self._alpha_cycle[epoch], tf.float32)
        self.model.beta = tf.cast(self._beta_cycle[epoch], tf.float32)
        self.model.gamma = tf.cast(self._gamma_cycle[epoch], tf.float32)


class ReconstructionPlot(tf.keras.callbacks.Callback):
    def __init__(self, dataset, sample, path, period=20):
        super(ReconstructionPlot, self).__init__()
        self._data = Helper.get_sample(dataset, sample)
        self._period = period
        self._path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._period == 0:
            z, _, _ = self.model.encode(self._data)
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
    CRED: https://github.com/jxhe/vae-lagging-encoder/blob/cdc4eb9d9599a026bf277db74efc2ba1ec203b15/image.py#L133
    '''

    def __init__(self, data, aggressive=True):
        super().__init__()
        self._aggressive = aggressive
        self._data = data
        self._temp = -np.inf
        iterator = iter(data)
        batch = next(iterator)
        self._ecg = batch['ecg']['I']

    def on_epoch_begin(self, epoch, logs=None):
        res = self.model.compute_information_gain(self._ecg)
        self.aggressive = res >= self._temp
        if self.aggressive:
            if self.model._decoder.trainable:
                self.model._decoder.trainable = False
            else:
                self.model._encoder.trainable = True
                self.model._decoder.trainable = False
            self._temp = res
        else:
            self.model._encoder.trainable = True
            self.model._decoder.trainable = True

    def on_epoch_end(self, epoch, logs=None):

        res = self.model.compute_information_gain(self._ecg)
        tf.print(res)
        self._aggressive = res >= self._temp
        if self.aggressive:
            if self.model._decoder.trainable:
                self.model._decoder.trainable = False
            else:
                self.model._encoder.trainable = True
                self.model._decoder.trainable = False
            self.temp = res
        else:
            self.model._encoder.trainable = True
            self.model._decoder.trainable = True
