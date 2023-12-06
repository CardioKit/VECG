import numpy as np
import tensorflow as tf


class Decoder(tf.keras.Model):

    def conv_block_dec(self, input, filters, kernel_size, dilation_rate, seed=42):
        low = 0
        high = 1000000
        seed = np.random.randint(low, high)
        # initializer = tf.keras.initializers.Orthogonal(seed=seed)
        initializer = tf.keras.initializers.Orthogonal()
        forward = tf.keras.layers.Conv1DTranspose(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                                                  kernel_initializer=initializer)(input)
        forward = tf.keras.layers.LeakyReLU()(forward)
        forward = tf.keras.layers.Conv1DTranspose(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
                                                  kernel_initializer=initializer)(forward)
        forward = tf.keras.layers.LeakyReLU()(forward)

        return forward

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.latent_inputs = tf.keras.Input(shape=(latent_dim,))
        self.x = tf.keras.layers.Dense(64)(self.latent_inputs)
        self.x = tf.keras.layers.Dense(32 * 500)(self.x)
        self.x = tf.keras.layers.Reshape((500, 32))(self.x)
        self.x = self.conv_block_dec(self.x, 32, 5, 32)
        self.x = self.conv_block_dec(self.x, 32, 5, 16)
        self.x = self.conv_block_dec(self.x, 32, 5, 8)
        self.x = self.conv_block_dec(self.x, 32, 5, 4)
        self.x = self.conv_block_dec(self.x, 32, 5, 2)
        self.x = self.conv_block_dec(self.x, 1, 5, 1)
        self.x = tf.keras.layers.Flatten()(self.x)
        # self.x = tf.keras.layers.Dropout(.4)(self.x)
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
