import numpy as np
import tensorflow as tf


class Encoder(tf.keras.Model):

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
        # self.x = tf.keras.layers.Dropout(.4)(self.x)
        self.x = tf.keras.layers.Dense(64)(self.x)
        # self.x = tf.keras.layers.Dropout(.4)(self.x)
        self.z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(self.x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var", activation='softplus')(self.x)

        self.encoder = tf.keras.Model(self.encoder_inputs, [self.z_mean, self.z_log_var], name="encoder")

    def conv_block_enc(self, input, filters, kernel_size, dilation_rate):
        low = 0
        high = 1000000
        seed = np.random.randint(low, high)
        initializer = tf.keras.initializers.Orthogonal(seed=seed)
        forward = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                                         kernel_initializer=initializer)(input)
        forward = tf.keras.layers.LeakyReLU()(forward)
        forward = tf.keras.layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate,
                                         kernel_initializer=initializer)(forward)
        forward = tf.keras.layers.LeakyReLU()(forward)
        return forward

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self.latent_dim
        config['summary'] = self.summary()
        return config

    def call(self, inputs):
        return self.encoder(inputs)
