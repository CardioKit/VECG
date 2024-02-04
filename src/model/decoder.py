import tensorflow as tf
import keras

class Decoder(tf.keras.Model):

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self._latent_dim = latent_dim

        self.latent_inputs = keras.Input(shape=(latent_dim,))
        self.x = tf.keras.layers.Dense(20)(self.latent_inputs)
        self.x = tf.keras.layers.Reshape((5, 4))(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=5, strides=1, padding='same')(self.x)
        self.x = tf.keras.layers.LeakyReLU()(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=5, strides=2, padding='same')(self.x)
        self.x = tf.keras.layers.LeakyReLU()(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(self.x)
        self.x = tf.keras.layers.LeakyReLU()(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=5, strides=5, padding='same')(self.x)
        self.x = tf.keras.layers.LeakyReLU()(self.x)
        self.x = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=5, strides=5, padding='same')(self.x)
        self.x = tf.keras.layers.LeakyReLU()(self.x)
        self.x = tf.keras.layers.Flatten()(self.x)
        self.decoder_outputs = tf.keras.layers.Reshape((500,))(self.x)

        self.decoder = tf.keras.Model(self.latent_inputs, self.decoder_outputs, name="decoder")

    def get_config(self):
        config = super().get_config()
        config['latent_dim'] = self._latent_dim
        config['summary'] = self.summary()
        return config

    def call(self, inputs):
        return self.decoder(inputs)
