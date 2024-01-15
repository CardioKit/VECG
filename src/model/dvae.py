import tensorflow as tf
import keras


class DVAE(tf.keras.Model):

    def __init__(self, encoder, decoder, loss_entity, **kwargs):
        super().__init__(**kwargs)

        self._encoder = encoder
        self._decoder = decoder
        self._loss_entity = loss_entity
        self._loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self._loss_tracker,
        ]

    @staticmethod
    def reparameterize(mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return tf.add(mean, tf.multiply(eps, tf.exp(logvar * 0.5)), name="sampled_latent_variable")

    def get_config(self):
        config = super().get_config()
        config['encoder'] = self._encoder
        config['decoder'] = self._decoder
        config['loss_entity'] = self._loss_entity
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
            loss_value = self._loss_entity.loss(reconstruction, data, z_mean, z_log_var, z)
        grads = tape.gradient(loss_value['loss'], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._loss_tracker.update_state(loss_value['loss'])
        return loss_value

    @tf.function
    def test_step(self, data):
        z_mean, z_log_var = self._encoder(data)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self._decoder(z)
        loss_value = self._loss_entity.loss(reconstruction, data, z_mean, z_log_var, z)
        return loss_value
