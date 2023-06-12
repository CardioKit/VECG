import tensorflow as tf
import numpy as np

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_inputs = tf.keras.Input(shape=(500,))
        self.x = tf.keras.layers.Dense(500, activation='relu')(self.encoder_inputs)
        self.x = tf.keras.layers.Reshape((500, 1))(self.x)
        self.x = tf.keras.layers.Conv1D(1024, 2, activation='relu')(self.x)
        self.x = tf.keras.layers.MaxPool1D(2)(self.x)
        self.x = tf.keras.layers.Conv1D(512, 2, activation='relu')(self.x)
        self.x = tf.keras.layers.MaxPool1D(2)(self.x)
        self.x = tf.keras.layers.Conv1D(128, 5, activation='relu')(self.x)
        self.x = tf.keras.layers.MaxPool1D(5)(self.x)
        self.x = tf.keras.layers.Conv1D(32, 5, activation='relu')(self.x)
        self.x = tf.keras.layers.MaxPool1D(5)(self.x)
        self.x = tf.keras.layers.Flatten()(self.x)
        self.x = tf.keras.layers.Dense(64, activation='relu')(self.x)
        self.x = tf.keras.layers.Dense(32, activation='relu')(self.x)
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(self.x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(self.x)
        self.z = Sampling()([self.z_mean, self.z_log_var])

        self.encoder = tf.keras.Model(self.encoder_inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        return config

    def call(self, inputs):
        return self.encoder(inputs)


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.latent_inputs = tf.keras.Input(shape=(latent_dim,))
        self.x = tf.keras.layers.Dense(32, activation='relu')(self.latent_inputs)
        self.x = tf.keras.layers.Dense(64, activation='relu')(self.x)
        self.x = tf.keras.layers.Reshape((2, 32))(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=1024, kernel_size=2, strides=2, padding='same',
                                                 activation='relu')(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=512, kernel_size=5, strides=5, padding='same',
                                                 activation='relu')(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=5, strides=5, padding='same',
                                                 activation='relu')(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=5, strides=5, padding='same',
                                                 activation='relu')(self.x)
        self.decoder_outputs = tf.keras.layers.Reshape((500,))(self.x)

        self.decoder = tf.keras.Model(self.latent_inputs, self.decoder_outputs, name="decoder")

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        return config

    def call(self, inputs):
        return self.decoder(inputs)


class TCVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, coefficients=(1.0, 1.0, 1.0), **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.alpha_ = tf.Variable(coefficients[0], name="weight_index", trainable=False)
        self.beta_ = tf.Variable(coefficients[1], name="weight_tc", trainable=False)
        self.gamma_ = tf.Variable(coefficients[2], name="weight_dimension", trainable=False)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.tc_loss_tracker = tf.keras.metrics.Mean(name="tc_loss")

    def get_config(self):
        config = super().get_config()
        config['encoder'] = self.encoder
        config['decoder'] = self.decoder
        config['alpha'] = self.alpha_.numpy()
        config['beta'] = self.beta_.numpy()
        config['gamma'] = self.gamma_.numpy()
        return config

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.tc_loss_tracker,
        ]

    @property
    def alpha(self):
        return self.alpha_

    @alpha.setter
    def alpha(self, value):
        self.alpha_.assign(value)

    @property
    def beta(self):
        return self.beta_

    @beta.setter
    def beta(self, value):
        self.beta_.assign(value)

    @property
    def gamma(self):
        return self.gamma_

    @gamma.setter
    def gamma(self, value):
        self.gamma_.assign(value)

    def encode(self, inputs):
        z = self.encoder(inputs)
        z_mean, z_logvar = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_logvar

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def gaussian_log_density(self, samples, mean, log_squared_scale):
        pi = tf.constant(np.pi)
        normalization = tf.math.log(2. * pi)
        inv_sigma = tf.math.exp(-log_squared_scale)
        tmp = (samples - mean)
        return -0.5 * (tmp * tmp * inv_sigma + log_squared_scale + normalization)

    def reconstruction_loss(self, data, reconstruction):
        return tf.reduce_mean(tf.keras.losses.mean_absolute_error(data, reconstruction))

    def total_correlation(self, z, z_mean, z_log_squared_scale):
        # TODO: declare source - https://github.com/julian-carpenter/beta-TCVAE
        """Estimate of total correlation on a batch.
        We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
        log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
        for the minimization. The constant should be equal to (num_latents - 1) *
        log(batch_size * dataset_size)
        Args:
          z: [batch_size, num_latents]-tensor with sampled representation.
          z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
          z_log_squared_scale: [batch_size, num_latents]-tensor with log variance of the encoder.
        Returns:
          Total correlation estimated on a batch.
        """
        # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
        # tensor of size [batch_size, batch_size, num_latents]. In the following
        # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].

        log_qz_prob = self.gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_log_squared_scale, 0))
        log_qz_product = tf.math.reduce_sum(
            tf.math.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
            axis=1,
            keepdims=False)
        # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
        # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
        log_qz = tf.math.reduce_logsumexp(
            tf.math.reduce_sum(log_qz_prob, axis=2, keepdims=False),
            axis=1,
            keepdims=False)
        return tf.math.reduce_mean(log_qz - log_qz_product) # * (self.beta_ - 1.)


    def kl_penalty(self, z_mean, z_log_squared_scale):
        summand = tf.math.square(z_mean) + tf.math.exp(z_log_squared_scale) - z_log_squared_scale - 1
        return tf.math.reduce_mean(0.5 * tf.math.reduce_sum(summand, [1]), name="kl_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = self.reconstruction_loss(data, reconstruction)
            kl_loss = self.kl_penalty(z_mean, z_log_var)
            tc_loss = self.total_correlation(z, z_mean, z_log_var)

            total_loss = tf.math.add(tf.math.add(reconstruction_loss, kl_loss), tc_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.tc_loss_tracker.update_state(tc_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "tc_loss": self.tc_loss_tracker.result(),
        }
