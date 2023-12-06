import math
import numpy as np
import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class TCVAE(tf.keras.Model):

    def __init__(self, encoder, decoder, size_dataset, mss=True, coefficients=(1.0, 4.0, 1.0), **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.size_dataset = size_dataset
        self._alpha = tf.Variable(coefficients[0], name="weight_index", trainable=False)
        self._beta = tf.Variable(coefficients[1], name="weight_tc", trainable=False)
        self._gamma = tf.Variable(coefficients[2], name="weight_dimension", trainable=False)
        self._mss = mss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.mi_loss_tracker = tf.keras.metrics.Mean(name="mi_loss")
        self.tc_loss_tracker = tf.keras.metrics.Mean(name="tc_loss")
        self.dw_kl_tracker = tf.keras.metrics.Mean(name="dw_kl")
        self.infogainnn = -np.inf

        self.prior_params = tf.zeros(self.encoder.latent_dim, 2)

    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.shape()
        prior_params = self.prior_params.expand(expanded_size)
        return prior_params

    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def get_config(self):
        config = super().get_config()
        config['encoder'] = self.encoder
        config['decoder'] = self.decoder
        config['alpha'] = self._alpha.numpy()
        config['beta'] = self._beta.numpy()
        config['gamma'] = self._gamma.numpy()
        config['summary'] = self.summary()
        return config

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.mi_loss_tracker,
            self.tc_loss_tracker,
            self.dw_kl_tracker,
        ]

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

    '''
    @property
    def infogainnn(self):
        return self.infogainnn

    @infogainnn.setter
    def infogainnn(self, value):
        self.infogainnn.assign(value)
    '''

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def encode(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

    @tf.function
    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstruction = self.decode(z_mean)
        return reconstruction

    def gaussian_log_density(self, samples, mean, log_var):
        # CRE: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/unsupervised/vae.py#L374
        pi = tf.constant(math.pi)
        normalization = tf.math.log(2. * pi)
        inv_sigma = tf.exp(-log_var)

        tmp = (samples - mean)
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    def log_importance_weight_matrix(self, batch_size, dataset_size):
        """
        CRE: TF adapted version of (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """
        N = tf.constant(dataset_size)
        M = tf.math.subtract(batch_size, 1)
        strat_weight = tf.divide(tf.math.subtract(N, M), tf.math.multiply(N, M))
        new_column1 = tf.cast(tf.fill((batch_size, 1), 1 / N), tf.float32)
        new_column2 = tf.cast(tf.fill((batch_size, 1), strat_weight), tf.float32)

        W = tf.divide(tf.ones([batch_size, batch_size]), tf.cast(M, tf.float32))
        W = tf.concat([new_column1, new_column2, W[:, 2:]], axis=1)
        W = tf.tensor_scatter_nd_update(W, [[M - 1, 0]], [strat_weight])

        return tf.math.log(W)

    def log_sum_exp(self, value, dim=None, keepdim=False):
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

    def compute_information_gain(self, data):
        mu, logvar, z = self.encode(data)
        x_batch, nz = mu.shape[0], mu.shape[1]
        neg_entropy = np.mean(-0.5 * nz * math.log(2 * math.pi) - 0.5 * np.sum(1 + logvar, axis=-1))
        mu, logvar = tf.expand_dims(mu, 1, name=None), tf.expand_dims(logvar, 1, name=None)
        var = np.exp(logvar)
        dev = z - mu
        log_density = -0.5 * np.sum((dev ** 2) / var, axis=-1) - 0.5 * (
                nz * math.log(2 * math.pi) + np.sum(logvar, axis=-1))
        log_qz = self.log_sum_exp(log_density, dim=1) - math.log(x_batch)
        res = neg_entropy - np.mean(log_qz, axis=-1)
        return res

    def loss_function(self, reconstruction, x, mu, log_var, z, size_dataset):

        size_batch = tf.shape(x)[0]
        recon_loss = tf.cast(tf.keras.losses.mean_squared_error(x, reconstruction), tf.float32)

        log_q_z_given_x = tf.cast(
            tf.reduce_sum(self.gaussian_log_density(z, mu, log_var), axis=-1),
            tf.float64,
        )

        log_prior = tf.cast(
            tf.reduce_sum(self.gaussian_log_density(z, tf.zeros_like(z), tf.ones_like(z)), axis=-1),
            tf.float64,
        )

        log_qz_prob = tf.cast(
            self.gaussian_log_density(
                tf.expand_dims(z, 1),
                tf.expand_dims(mu, 0),
                tf.expand_dims(log_var, 0),
            ),
            tf.float64,
        )

        '''
        logiw_mat = self._log_importance_weight_matrix(z.shape[0], dataset_size).to(
            z.device
        )
        log_q_z = torch.logsumexp(
            logiw_mat + log_q_batch_perm.sum(dim=-1), dim=-1
        )  # MMS [B]
        log_prod_q_z = (
            torch.logsumexp(
                logiw_mat.reshape(z.shape[0], z.shape[0], -1) + log_q_batch_perm,
                dim=1,
            )
        ).sum(
            dim=-1
        )  # MMS [B]
        '''

        if self._mss:
            logiw_mat = tf.cast(self.log_importance_weight_matrix(size_batch, size_dataset), tf.float64)
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

        kl_loss = tf.multiply(self._alpha, mutual_info_loss) + tf.multiply(self._beta, tc_loss) + tf.multiply(self._gamma,
                                                                                                        dimension_wise_kl)
        total_loss = tf.reduce_mean(recon_loss + kl_loss)

        return total_loss, tf.reduce_mean(recon_loss), tf.reduce_mean(mutual_info_loss), tf.reduce_mean(
            tc_loss), tf.reduce_mean(dimension_wise_kl)

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            total_loss, recon_loss, mi_loss, tc_loss, dw_kl = self.loss_function(
                reconstruction, data, z_mean, z_log_var, z, self.size_dataset,
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.mi_loss_tracker.update_state(mi_loss)
        self.tc_loss_tracker.update_state(tc_loss)
        self.dw_kl_tracker.update_state(dw_kl)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon": self.reconstruction_loss_tracker.result(),
            "mi": self.mi_loss_tracker.result(),
            "tc": self.tc_loss_tracker.result(),
            "dw_kl": self.dw_kl_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        z_mean, z_log_var, z = self.encode(data)
        reconstruction = self.decode(z)
        total_loss, recon_loss, mi_loss, tc_loss, dw_kl = self.loss_function(
            reconstruction, data, z_mean, z_log_var, z, self.size_dataset,
        )

        return {
            "loss": total_loss,
            "recon": recon_loss,
            "mi": mi_loss,
            "tc": tc_loss,
            "dw_kl": dw_kl,
        }
