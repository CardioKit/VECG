import tensorflow as tf
import tensorflow_probability as tfp
import keras
import numpy as np
import math

tfd = tfp.distributions


class VAE(tf.keras.Model):

    def __init__(self, encoder, decoder, coefficients, dataset_size, dist=tfd.Normal, **kwargs):
        super().__init__(**kwargs)

        self._encoder = encoder
        self._decoder = decoder
        self._coefficients = coefficients
        self._dataset_size = dataset_size
        self._dist = dist
        self._loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self._loss_tracker,
        ]

    @staticmethod
    def reparameterize(mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return tf.add(mean, tf.multiply(eps, tf.exp(log_var * 0.5)), name="sampled_latent_variable")

    def get_config(self):
        config = super().get_config()
        config['encoder'] = keras.saving.serialize_keras_object(self._encoder)
        config['decoder'] = keras.saving.serialize_keras_object(self._decoder)
        config['coefficients'] = self._coefficients
        config['dist'] = self._dist
        config['dataset_size'] = self._dataset_size
        return config

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def encode(self, inputs):
        z_mean, z_log_var = self._encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

    @tf.function
    def decode(self, z):
        reconstructed = self._decoder(z)
        return reconstructed

    def call(self, inputs):
        _, _, z = self.encode(inputs)
        reconstruction = self.decode(z)
        return reconstruction

    @tf.function
    def train_step(self, data):
        with (tf.GradientTape() as tape):
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            loss_value = self._loss(reconstruction, data, z_mean, z_log_var, z)
        grads = tape.gradient(loss_value['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._loss_tracker.update_state(loss_value['loss'])
        return loss_value

    @tf.function
    def test_step(self, data):
        z_mean, z_log_var = self._encoder(data)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self._decoder(z)
        loss_value = self._loss(reconstruction, data, z_mean, z_log_var, z)
        return loss_value

    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

    def log_importance_weight_matrix_iso(self, batch_size):
        """
        CRE: TF adapted version of (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """
        N = tf.constant(self._dataset_size)
        M = tf.math.subtract(batch_size, 1)
        strat_weight = tf.divide(tf.math.subtract(N, M), tf.math.multiply(N, M))
        new_column1 = tf.cast(tf.fill((batch_size, 1), tf.divide(1, N)), tf.float32)
        new_column2 = tf.cast(tf.fill((batch_size, 1), strat_weight), tf.float32)

        W = tf.divide(tf.ones([batch_size, batch_size]), tf.cast(M, tf.float32))
        W = tf.concat([new_column1, new_column2, W[:, 2:]], axis=1)
        W = tf.tensor_scatter_nd_update(W, [[M - 1, 0]], [strat_weight])

        return tf.math.log(W)

    def log_importance_weight_matrix(self, batch_size):
        N = tf.cast(tf.constant(self._dataset_size), dtype=tf.float32)
        B = tf.cast(tf.math.subtract(batch_size, 1), dtype=tf.float32)
        W = tf.multiply(tf.ones([batch_size, batch_size]),
                        tf.divide(tf.subtract(N, 1), tf.multiply(N, tf.subtract(B, 1))))
        W = tf.linalg.set_diag(W, tf.multiply(tf.divide(1.0, N), tf.ones([batch_size])))

        return tf.math.log(W)

    def compute_information_gain(self, data):
        mu, log_var, z = self.encode(data)
        x_batch, nz = mu.shape[0], mu.shape[1]

        neg_entropy = np.mean(-0.5 * nz * math.log(2 * math.pi) - 0.5 * np.sum(1 + log_var, axis=-1))

        mu, log_var = tf.expand_dims(mu, 0, name=None), tf.expand_dims(log_var, 0, name=None)
        var = np.exp(log_var)
        dev = tf.expand_dims(z, 1, name=None) - mu

        log_density = -0.5 * np.sum((dev ** 2) / var, axis=-1) - 0.5 * (
                nz * math.log(2 * math.pi) + np.sum(log_var, axis=-1))
        log_qz = tf.reduce_sum(log_density, axis=1) - math.log(x_batch)
        res = neg_entropy - np.mean(log_qz, axis=-1)
        return res
