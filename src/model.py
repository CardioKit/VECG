import math
import numpy as np
import tensorflow as tf

class Encoder(tf.keras.Model):

    def conv_block_enc(self, input, filters, kernel_size, dilation_rate):
        low = 0
        high = 1000000
        seed = np.random.randint(low, high)
        initializer = tf.keras.initializers.Orthogonal(seed=seed)
        forward = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_initializer=initializer)(input)
        forward = tf.keras.layers.LeakyReLU()(forward)
        forward = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_initializer=initializer)(forward)
        forward = tf.keras.layers.LeakyReLU()(forward)
        return forward

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_inputs = tf.keras.Input(shape=(500,))
        self.x = tf.keras.layers.Reshape((500, 1))(self.encoder_inputs)
        self.x = self.conv_block_enc(self.x, 32, 5, 1)
        self.x = self.conv_block_enc(self.x, 32, 5, 2)
        self.x = self.conv_block_enc(self.x, 32, 5, 4)
        self.x = self.conv_block_enc(self.x, 32, 5, 8)
        self.x = self.conv_block_enc(self.x, 32, 5, 16)
        self.x = self.conv_block_enc(self.x, 16, 5, 32)
        self.x = tf.keras.layers.MaxPooling1D()(self.x)

        self.x = tf.keras.layers.Flatten()(self.x)
        #self.x = tf.keras.layers.Dropout(.4)(self.x)
        self.x = tf.keras.layers.Dense(64)(self.x)
        #self.x = tf.keras.layers.Dropout(.4)(self.x)
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(self.x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var", activation='softplus')(self.x)

        self.encoder = tf.keras.Model(self.encoder_inputs, [self.z_mean, self.z_log_var], name="encoder")

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        config['summary'] = self.summary()
        return config

    def call(self, inputs):
        return self.encoder(inputs)


class Decoder(tf.keras.Model):

    def conv_block_dec(self, input, filters, kernel_size, dilation_rate, seed=42):
        low = 0
        high = 1000000
        seed = np.random.randint(low, high)
        initializer = tf.keras.initializers.Orthogonal(seed=seed)
        forward = tf.keras.layers.Conv1DTranspose(filters, kernel_size, padding='same', dilation_rate=dilation_rate,kernel_initializer = initializer)(input)
        forward = tf.keras.layers.LeakyReLU()(forward)
        forward = tf.keras.layers.Conv1DTranspose(filters, kernel_size, padding='same', dilation_rate=dilation_rate,kernel_initializer = initializer)(forward)
        forward = tf.keras.layers.LeakyReLU()(forward)

        return forward

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.latent_inputs = tf.keras.Input(shape=(latent_dim,))
        self.x = tf.keras.layers.Dense(64)(self.latent_inputs)
        self.x = tf.keras.layers.Dense(32*500)(self.x)
        self.x = tf.keras.layers.Reshape((500, 32))(self.x)
        self.x = self.conv_block_dec(self.x, 32, 5, 32)
        self.x = self.conv_block_dec(self.x, 32, 5, 16)
        self.x = self.conv_block_dec(self.x, 32, 5, 8)
        self.x = self.conv_block_dec(self.x, 32, 5, 4)
        self.x = self.conv_block_dec(self.x, 32, 5, 2)
        self.x = self.conv_block_dec(self.x, 1, 5, 1)
        self.x = tf.keras.layers.Flatten()(self.x)
        #self.x = tf.keras.layers.Dropout(.4)(self.x)
        self.x = tf.keras.layers.Dense(500)(self.x)
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

    def reconstruction_loss(self, data, reconstruction):
        return 10000*tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction))

    def gaussian_log_density(self, samples, mean, log_var):
        # CRE: https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/unsupervised/vae.py#L374
        pi = tf.constant(math.pi)
        normalization = tf.math.log(2. * pi)
        inv_sigma = tf.exp(-log_var)

        tmp = (samples - mean) # x - mu
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
        log_density = -0.5 * np.sum((dev ** 2) / var, axis=-1) - 0.5 * (nz * math.log(2 * math.pi) + np.sum(logvar, axis=-1))
        log_qz = self.log_sum_exp(log_density, dim=1) - math.log(x_batch)
        res = neg_entropy - np.mean(log_qz, axis=-1)
        return res

    def loss_function(self, reconstruction, x, mu, log_var, z, size_dataset):

        size_batch = tf.shape(x)[0]
        recon_loss = self.reconstruction_loss(x, reconstruction)

        log_q_z_given_x = tf.cast(
            tf.reduce_sum(self.gaussian_log_density(z, mu, log_var), axis=-1, keepdims=False),
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

        log_prior = tf.cast(
            tf.reduce_sum(
                self.gaussian_log_density(z, tf.zeros_like(z), tf.ones_like(z)),
                axis=1,
                keepdims=False),
            tf.float64,
        )

        logiw_mat = tf.cast(self.log_importance_weight_matrix(size_batch, size_dataset), tf.float64)
        log_qz = tf.reduce_logsumexp(
            tf.reduce_sum(log_qz_prob, axis=2, keepdims=False) + logiw_mat,
            axis=1,
            keepdims=False,
        )

        log_qz_product = tf.reduce_sum(
            tf.reduce_logsumexp(
                log_qz_prob + tf.reshape(logiw_mat, shape=(tf.shape(z)[0], tf.shape(z)[0], -1)),
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

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            recon_loss, mi_loss, tc_loss, dw_kl = self.loss_function(
                reconstruction, data, z_mean, z_log_var, z, self.size_dataset,
            )
            kl_loss = self.alpha_ * mi_loss + self.beta_ * tc_loss + self.gamma_ * dw_kl
            total_loss = tf.cast(recon_loss, tf.float32) + tf.cast(kl_loss, tf.float32)

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
        reconstruction_loss, mutual_info_loss, tc_loss, dimension_wise_kl = self.loss_function(
            reconstruction, data, z_mean, z_log_var, z, self.size_dataset,
        )
        kl_loss = self.alpha_ * mutual_info_loss + self.beta_ * tc_loss + self.gamma_ * dimension_wise_kl
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "recon": reconstruction_loss,
            "mi": mutual_info_loss,
            "tc": tc_loss,
            "dw_kl": dimension_wise_kl,
        }
