import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions


class Loss(ABC):
    @abstractmethod
    def loss(self, reconstruction, x, mu, log_var, z):
        pass

    def log_importance_weight_matrix_iso(self, batch_size, dataset_size):
        """
        CRE: TF adapted version of (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """
        N = tf.constant(dataset_size)
        M = tf.math.subtract(batch_size, 1)
        strat_weight = tf.divide(tf.math.subtract(N, M), tf.math.multiply(N, M))
        new_column1 = tf.cast(tf.fill((batch_size, 1), tf.divide(1, N)), tf.float32)
        new_column2 = tf.cast(tf.fill((batch_size, 1), strat_weight), tf.float32)

        W = tf.divide(tf.ones([batch_size, batch_size]), tf.cast(M, tf.float32))
        W = tf.concat([new_column1, new_column2, W[:, 2:]], axis=1)
        W = tf.tensor_scatter_nd_update(W, [[M - 1, 0]], [strat_weight])

        return tf.math.log(W)

    def log_importance_weight_matrix(self, batch_size, dataset_size):
        N = tf.cast(tf.constant(dataset_size), dtype=tf.float32)
        B = tf.cast(tf.math.subtract(batch_size, 1), dtype=tf.float32)
        W = tf.multiply(tf.ones([batch_size, batch_size]),
                        tf.divide(tf.subtract(N, 1), tf.multiply(N, tf.subtract(B, 1))))
        W = tf.linalg.set_diag(W, tf.multiply(tf.divide(1.0, N), tf.ones([batch_size])))

        return tf.math.log(W)

    # q(z|x) = log_normal_pdf(z, mean, logvar)
    # p(z) = log_normal_pdf(z, 0., 0.)
    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)


class VAELoss(Loss):

    def __init__(self, size_dataset, coefficients, dist=tfd.Normal):
        self._size_dataset = size_dataset
        self._beta = tf.Variable(coefficients['beta'], name="beta", trainable=False)
        self._dist = dist

    def loss(self, reconstruction, x, mu, log_var, z):
        recon_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(x, reconstruction))
        logpz = tf.reduce_mean(self._dist(0.0, 1.0).log_prob(z))
        logqz_x = tf.reduce_mean(self._dist(mu, tf.exp(log_var)).log_prob(z))

        loss = recon_loss + tf.multiply(self._beta, (logpz - logqz_x))

        return {
            "loss": loss,
            "recon": recon_loss,
            "log p(z)": logpz,
            "log q(z|x)": logqz_x,
        }


class TCVAELoss(Loss):

    def __init__(self, size_dataset, coefficients, dist=tfd.Normal):
        self._dist = dist
        self._size_dataset = size_dataset
        self._alpha = tf.Variable(coefficients['alpha'], name="alpha", trainable=False)
        self._beta = tf.Variable(coefficients['beta'], name="beta", trainable=False)
        self._gamma = tf.Variable(coefficients['gamma'], name="gamma", trainable=False)

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

    def loss(self, reconstruction, x, mu, log_var, z):
        size_batch = tf.shape(x)[0]
        logiw_mat = self.log_importance_weight_matrix_iso(size_batch, self._size_dataset)

        recon_loss = tf.reduce_sum(tf.keras.losses.mean_absolute_error(x, reconstruction))
        # log(q(z|x))
        # log_qz_x = tf.reduce_sum(self._dist(mu, tf.exp(log_var)).log_prob(z), axis=-1)
        log_qz_x = tf.reduce_sum(self.log_normal_pdf(z, mu, log_var), axis=-1)

        # log(p(z))
        # log_prior = tf.reduce_sum(self._dist(tf.zeros_like(z), tf.ones_like(z)).log_prob(z), axis=-1)
        log_prior = tf.reduce_sum(
            self.log_normal_pdf(
                z,
                tf.zeros_like(z),
                tf.zeros_like(z),
            ), axis=-1)

        # log(q(z(x_j) | x_i))
        # log_qz_prob = self._dist(
        #    tf.expand_dims(mu, 0), tf.expand_dims(tf.exp(log_var), 0),
        # ).log_prob(tf.expand_dims(z, 1))

        log_qz_prob = self.log_normal_pdf(
            tf.expand_dims(z, 1),
            tf.expand_dims(mu, 0),
            tf.expand_dims(log_var, 1),
        )

        log_qz_prob = log_qz_prob + tf.expand_dims(logiw_mat, 2)
        # log(q(z))
        log_qz = tf.reduce_logsumexp(tf.reduce_sum(log_qz_prob, axis=2), axis=1)
        # log(PI_i q(z_i))
        log_qz_product = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob, axis=1), axis=1)

        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mutual_info_loss = tf.reduce_sum(tf.subtract(log_qz_x, log_qz))
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = tf.reduce_sum(tf.subtract(log_qz, log_qz_product))
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dimension_wise_kl = tf.reduce_sum(tf.subtract(log_qz_product, log_prior))

        kl_loss = tf.multiply(self._alpha, mutual_info_loss) + \
                  tf.multiply(self._beta, tc_loss) + \
                  tf.multiply(self._gamma, dimension_wise_kl)

        # kl_loss = tf.multiply(self._alpha, mutual_info_loss)

        total_loss = tf.add(recon_loss, kl_loss)

        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl_loss": kl_loss,
            "mi": mutual_info_loss,
            "tc": tc_loss,
            "dw_kl": dimension_wise_kl,
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
        }


class HFVAELoss(Loss):

    def __init__(self, size_dataset, coefficients, dist=tfd.Normal):
        self._dist = dist
        self._size_dataset = size_dataset

    def loss(self, reconstruction, x, mu, log_var, z):
        NotImplementedError
