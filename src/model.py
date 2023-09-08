import math
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_inputs = tf.keras.Input(shape=(500,))
        self.x = tf.keras.layers.Dense(500, activation='elu')(self.encoder_inputs)
        self.x = tf.keras.layers.Reshape((500, 1))(self.x)
        self.x = tf.keras.layers.Conv1D(1024, 5, 5, activation='elu')(self.x)
        self.x = tf.keras.layers.BatchNormalization()(self.x)
        self.x = tf.keras.layers.Conv1D(512, 5, 5, activation='elu')(self.x)
        self.x = tf.keras.layers.BatchNormalization()(self.x)
        self.x = tf.keras.layers.Conv1D(128, 2, 2, activation='elu')(self.x)
        self.x = tf.keras.layers.BatchNormalization()(self.x)
        self.x = tf.keras.layers.Conv1D(64, 2, 2, activation='elu')(self.x)
        self.x = tf.keras.layers.BatchNormalization()(self.x)
        self.x = tf.keras.layers.Flatten()(self.x)
        self.x = tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4))(self.x)
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(self.x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(self.x)

        self.encoder = tf.keras.Model(self.encoder_inputs, [self.z_mean, self.z_log_var], name="encoder")

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        config['summary'] = self.summary()
        return config

    def call(self, inputs):
        return self.encoder(inputs)


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.latent_inputs = tf.keras.Input(shape=(latent_dim,))
        self.x = tf.keras.layers.Dense(64, activation='elu', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4))(self.latent_inputs)
        self.x = tf.keras.layers.Reshape((2, 32))(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=1024, kernel_size=2, strides=2, padding='same', activation='elu')(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=512, kernel_size=5, strides=5, padding='same', activation='elu')(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=5, strides=5, padding='same', activation='elu')(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=5, strides=5, padding='same', activation='elu')(self.x)
        self.decoder_outputs = tf.keras.layers.Reshape((500,))(self.x)

        self.decoder = tf.keras.Model(self.latent_inputs, self.decoder_outputs, name="decoder")

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        config['summary'] = self.summary()
        return config

    def call(self, inputs):
        return self.decoder(inputs)


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
        self.alpha_ = tf.Variable(coefficients[0], name="weight_index", trainable=False)
        self.beta_ = tf.Variable(coefficients[1], name="weight_tc", trainable=False)
        self.gamma_ = tf.Variable(coefficients[2], name="weight_dimension", trainable=False)
        self.mss = mss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.mi_loss_tracker = tf.keras.metrics.Mean(name="mi_loss")
        self.tc_loss_tracker = tf.keras.metrics.Mean(name="tc_loss")
        self.dw_kl_tracker = tf.keras.metrics.Mean(name="dw_kl")

        self.prior_params = tf.zeros(self.encoder.latent_dim, 2)

    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.shape()
        prior_params = self.prior_params.expand(expanded_size)
        return prior_params

    def get_config(self):
        config = super().get_config()
        config['encoder'] = self.encoder
        config['decoder'] = self.decoder
        config['alpha'] = self.alpha_.numpy()
        config['beta'] = self.beta_.numpy()
        config['gamma'] = self.gamma_.numpy()
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
        z_mean, z_log_var = self.encoder(inputs)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstruction = self.decode(z)
        return reconstruction

    def reconstruction_loss(self, data, reconstruction):
        return tf.reduce_mean(tf.keras.losses.mean_absolute_error(data, reconstruction))

    def gaussian_log_density(self, samples, mean, log_var):
        # source: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/unsupervised/vae.py#L374
        pi = tf.constant(math.pi)
        normalization = tf.math.log(2. * pi)
        inv_sigma = tf.exp(-log_var)

        tmp = (samples - mean) # x - mu
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    def log_importance_weight_matrix(self, batch_size, dataset_size):
        """
        TF adapted version of (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
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

    def loss_function(self, reconstruction, x, mu, log_var, z, size_dataset):

        size_batch = tf.shape(x)[0]
        recon_loss = self.reconstruction_loss(x, reconstruction)

        '''
        log_q_z_given_x = tf.cast(
            tf.reduce_sum(self.gaussian_log_density(z, mu, log_var), axis=-1, keepdims=False),
            tf.float64,
        )

        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_logvar, 0))

        log_qz_prob = tf.cast(
            self.gaussian_log_density(
                tf.expand_dims(z, 1),
                tf.expand_dims(mu, 0),
                tf.expand_dims(log_var, 0),
            ),
            tf.float64,
        )

        log_prior = tf.cast(
            tf.reduce_sum(
                self.gaussian_log_density(z, tf.zeros_like(z), tf.ones_like(z)),
                axis=1,
                keepdims=False),
            tf.float64,
        )

        #if self.mss:
        logiw_mat = tf.cast(self.log_importance_weight_matrix(size_batch, size_dataset), tf.float64)
        log_qz = tf.reduce_logsumexp(
            tf.reduce_sum(log_qz_prob, axis=2, keepdims=False), # + logiw_mat ,
            axis=1,
            keepdims=False,
        )

        log_qz_product = tf.reduce_sum(
            tf.reduce_logsumexp(
                log_qz_prob, # + tf.reshape(logiw_mat, shape=(tf.shape(z)[0], tf.shape(z)[0], -1)),
                axis=1,
                keepdims=False,
            ),
            axis=1,
            keepdims=False,
        )

        mutual_info_loss = tf.cast(tf.reduce_mean(log_q_z_given_x - log_qz), tf.float32)
        tc_loss = tf.cast(tf.reduce_mean(log_qz - log_qz_product), tf.float32)
        dimension_wise_kl = tf.cast(tf.reduce_mean(log_qz_product - log_prior), tf.float32)
        
        return recon_loss, mutual_info_loss, tc_loss, dimension_wise_kl
        '''

        kl_loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        return recon_loss, 0, kl_loss, 0

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            reconstruction_loss, mutual_info_loss, tc_loss, dimension_wise_kl = self.loss_function(
                reconstruction, data, z_mean, z_log_var, z, self.size_dataset,
            )
            kl_loss = self.alpha_ * mutual_info_loss + self.beta_ * tc_loss + self.gamma_ * dimension_wise_kl
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.mi_loss_tracker.update_state(mutual_info_loss)
        self.tc_loss_tracker.update_state(tc_loss)
        self.dw_kl_tracker.update_state(dimension_wise_kl)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "mi_loss": self.mi_loss_tracker.result(),
            "tc_loss": self.tc_loss_tracker.result(),
            "dw_kl": self.dw_kl_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        z_mean, z_log_var, z = self.encode(data)
        reconstruction = self.decode(z)
        reconstruction_loss, mutual_info_loss, tc_loss, dimension_wise_kl = self.loss_function(
            reconstruction, data, z_mean, z_log_var, z, self.size_dataset,
        )
        kl_loss = self.alpha_ * mutual_info_loss + self.beta_ * tc_loss + self.gamma_ * dimension_wise_kl
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "mi_loss": mutual_info_loss,
            "tc_loss": tc_loss,
            "dw_kl": dimension_wise_kl,
        }
