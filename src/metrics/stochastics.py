import tensorflow as tf
import math
import numpy as np


class Stochastics():
    @staticmethod
    def gaussian_log_density(samples, mean, log_var):
        # CRE: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/unsupervised/vae.py#L374
        pi = tf.constant(math.pi)
        normalization = tf.math.log(2. * pi)
        inv_sigma = tf.exp(-log_var)

        tmp = (samples - mean)
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    @staticmethod
    def log_sum_exp(value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m = np.max(value, axis=dim, keepdims=True)
            value0 = value - m
            if keepdim is False:
                m = np.squeeze(dim)
            return m + np.log(np.sum(np.exp(value0), axis=dim, keepdims=keepdim))
        else:
            m = np.max(value)
            sum_exp = np.sum(np.exp(value - m))
            return m + np.log(sum_exp)

    @staticmethod
    def compute_information_gain(data, vae):
        mu, logvar, z = vae.encode(data)
        x_batch, nz = mu.shape[0], mu.shape[1]
        neg_entropy = np.mean(-0.5 * nz * math.log(2 * math.pi) - 0.5 * np.sum(1 + logvar, axis=-1))
        mu, logvar = tf.expand_dims(mu, 1, name=None), tf.expand_dims(logvar, 1, name=None)
        var = np.exp(logvar)
        dev = z - mu
        log_density = -0.5 * np.sum((dev ** 2) / var, axis=-1) - 0.5 * (
                nz * math.log(2 * math.pi) + np.sum(logvar, axis=-1))
        log_qz = Stochastics.log_sum_exp(log_density, dim=1) - math.log(x_batch)
        res = neg_entropy - np.mean(log_qz, axis=-1)
        return res
