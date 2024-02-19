import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from utils.helper import Helper


class CoefficientScheduler(tf.keras.callbacks.Callback):

    def __init__(self, epochs, coefficients, coefficient_raise):
        super(CoefficientScheduler, self).__init__()

        alpha = coefficients['alpha']
        beta = coefficients['beta']
        gamma = coefficients['gamma']

        self._alpha_cycle = np.ones(epochs) * alpha
        self._beta_cycle = np.ones(epochs) * beta
        self._gamma_cycle = np.ones(epochs) * gamma

        #coef_raise = annealing['coefficients_raise']
        #self._alpha_cycle[0:coef_raise] = np.linspace(0, alpha, coef_raise)
        #self._beta_cycle[0:coef_raise] = np.linspace(0, beta, coef_raise)
        #self._gamma_cycle[0:coef_raise] = np.linspace(0, gamma, coef_raise)


    def on_epoch_begin(self, epoch, logs=None):
        self.model.alpha = tf.cast(self._alpha_cycle[epoch], tf.float32)
        self.model.beta = tf.cast(self._beta_cycle[epoch], tf.float32)
        self.model.gamma = tf.cast(self._gamma_cycle[epoch], tf.float32)


class ReconstructionPlot(tf.keras.callbacks.Callback):
    def __init__(self, dataset, path, params):

        sample = params['index_sample']
        period = params['period_plot']

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

    def __init__(self, data, path, aggressive=True):
        super().__init__()
        self._data = data
        self._path = path
        self._aggressive = aggressive
        self._last_mi = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        mi = self.model._mi_val
        aggr = mi >= self._last_mi
        tf.print('Epoch', epoch, 'MI', mi, 'LV', self._last_mi, 'Aggressive', aggr)
        if aggr:
            self.model._decoder.trainable = False
        else:
            self.model._decoder.trainable = True
        self._last_mi = tf.identity(mi)
