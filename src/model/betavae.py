from model.vae import VAE
import tensorflow as tf


class TCVAE(VAE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._beta = tf.Variable(self._coefficients['beta'], name="beta", trainable=False)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta.assign(value)

    def _loss(self, reconstruction, x, mu, log_var, z):
        recon_loss = tf.reduce_sum(tf.keras.losses.mean_absolute_error(x, reconstruction))

        kl_loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        loss = recon_loss + tf.multiply(self._beta, kl_loss)

        return {
            "loss": loss,
            "recon": recon_loss,
            "kl_loss": kl_loss,
            "beta": self._beta,
        }
