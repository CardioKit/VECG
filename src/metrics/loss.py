import tensorflow as tf
from abc import ABC, abstractmethod

from metrics.stochastics import Stochastics


class VAELoss(ABC):
    @abstractmethod
    def loss(self, reconstruction, x, mu, log_var, z):
        pass


class TCVAELoss(VAELoss):

    def __init__(self, size_dataset, coefficients, mss=True):
        self._n = size_dataset
        self._alpha = tf.Variable(coefficients['alpha'], name="alpha", trainable=False)
        self._beta = tf.Variable(coefficients['beta'], name="beta", trainable=False)
        self._gamma = tf.Variable(coefficients['gamma'], name="gamma", trainable=False)
        self._mss = mss

    def get_coefficients(self):
        return self._alpha, self._beta, self._gamma

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha.assign(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta.assign(value)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma.assign(value)

    def log_importance_weight_matrix(self, batch_size):
        """
        CRE: TF adapted version of (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """
        N = tf.constant(self._n)
        M = tf.math.subtract(batch_size, 1)
        strat_weight = tf.divide(tf.math.subtract(N, M), tf.math.multiply(N, M))
        new_column1 = tf.cast(tf.fill((batch_size, 1), 1 / N), tf.float32)
        new_column2 = tf.cast(tf.fill((batch_size, 1), strat_weight), tf.float32)

        W = tf.divide(tf.ones([batch_size, batch_size]), tf.cast(M, tf.float32))
        W = tf.concat([new_column1, new_column2, W[:, 2:]], axis=1)
        W = tf.tensor_scatter_nd_update(W, [[M - 1, 0]], [strat_weight])

        return tf.math.log(W)

    def loss(self, reconstruction, x, mu, log_var, z):
        size_batch = tf.shape(x)[0]
        recon_loss = tf.cast(tf.keras.losses.mean_absolute_error(x, reconstruction), tf.float32)

        log_q_z_given_x = tf.cast(
            tf.reduce_sum(Stochastics.gaussian_log_density(z, mu, log_var), axis=-1),
            tf.float64,
        )

        log_prior = tf.cast(
            tf.reduce_sum(Stochastics.gaussian_log_density(z, tf.zeros_like(z), tf.ones_like(z)), axis=-1),
            tf.float64,
        )

        log_qz_prob = tf.cast(
            Stochastics.gaussian_log_density(
                tf.expand_dims(z, 1),
                tf.expand_dims(mu, 0),
                tf.expand_dims(log_var, 0),
            ),
            tf.float64,
        )

        if self._mss:
            logiw_mat = tf.cast(self.log_importance_weight_matrix(size_batch), tf.float64)
            log_qz = tf.reduce_logsumexp(
                tf.reduce_sum(log_qz_prob, axis=2, keepdims=False) + logiw_mat,
                axis=1,
                keepdims=False,
            )
            log_qz_product = tf.reduce_sum(
                tf.reduce_logsumexp(log_qz_prob + tf.reshape(logiw_mat, shape=(tf.shape(z)[0], tf.shape(z)[0], -1)),
                                    axis=1,
                                    keepdims=False), axis=1, keepdims=False)
        else:
            log_qz = tf.reduce_logsumexp(
                tf.reduce_sum(log_qz_prob, axis=-1, keepdims=False),
                axis=1,
                keepdims=False,
            )
            log_qz_product = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False), axis=1,
                                           keepdims=False)

        mutual_info_loss = tf.cast(log_q_z_given_x - log_qz, tf.float32)
        tc_loss = tf.cast(log_qz - log_qz_product, tf.float32)
        dimension_wise_kl = tf.cast(log_qz_product - log_prior, tf.float32)

        kl_loss = ((tf.multiply(self._alpha, mutual_info_loss)
                    + tf.multiply(self._beta, tc_loss))
                   + tf.multiply(self._gamma, dimension_wise_kl))
        total_loss = tf.reduce_mean(recon_loss + kl_loss)

        return {
            "loss": total_loss,
            "recon": tf.reduce_mean(recon_loss),
            "mi": tf.reduce_mean(mutual_info_loss),
            "tc": tf.reduce_mean(tc_loss),
            "dw_kl": tf.reduce_mean(dimension_wise_kl),
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
        }
