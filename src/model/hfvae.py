from model.vae import VAE
import tensorflow as tf


class HFVAE(VAE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = tf.Variable(self._coefficients['alpha'], name="alpha", trainable=False)
        self._beta = tf.Variable(self._coefficients['beta'], name="beta", trainable=False)
        self._gamma = tf.Variable(self._coefficients['gamma'], name="gamma", trainable=False)

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

    def _loss(self, reconstruction, x, mu, log_var, z):
        size_batch = tf.shape(x)[0]
        logiw_mat = self.log_importance_weight_matrix_iso(size_batch)

        recon_loss = tf.reduce_sum(tf.keras.losses.mean_absolute_error(x, reconstruction))
        # log(q(z|x))
        # log_qz_x = tf.reduce_sum(self._dist(mu, tf.exp(log_var)).log_prob(z), axis=-1)
        log_qz_x = tf.reduce_sum(self.log_normal_pdf(z, mu, log_var), axis=-1)

        # log(p(z))
        # log_prior = tf.reduce_sum(self._dist(tf.zeros_like(z), tf.ones_like(z)).log_prob(z), axis=-1)
        log_prior = tf.reduce_sum(
            self.log_normal_pdf(
                z,
                tf.zeros_like(z),
                tf.zeros_like(z),
            ), axis=-1)

        # log(q(z(x_j) | x_i))
        # log_qz_prob = self._dist(
        #    tf.expand_dims(mu, 0), tf.expand_dims(tf.exp(log_var), 0),
        # ).log_prob(tf.expand_dims(z, 1))

        log_qz_prob = self.log_normal_pdf(
            tf.expand_dims(z, 1),
            tf.expand_dims(mu, 0),
            tf.expand_dims(log_var, 1),
        )

        log_qz_prob = log_qz_prob + tf.expand_dims(logiw_mat, 2)
        # log(q(z))
        log_qz = tf.reduce_logsumexp(tf.reduce_sum(log_qz_prob, axis=2), axis=1)
        # log(PI_i q(z_i))
        log_qz_product = tf.reduce_sum(tf.reduce_logsumexp(log_qz_prob, axis=1), axis=1)

        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mutual_info_loss = tf.reduce_sum(tf.subtract(log_qz_x, log_qz))
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = tf.reduce_sum(tf.subtract(log_qz, log_qz_product))
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dimension_wise_kl = tf.reduce_sum(tf.subtract(log_qz_product, log_prior))

        kl_loss = tf.multiply(self._alpha, mutual_info_loss) + \
                  tf.multiply(self._beta, tc_loss) + \
                  tf.multiply(self._gamma, dimension_wise_kl)

        # kl_loss = tf.multiply(self._alpha, mutual_info_loss)

        total_loss = tf.add(recon_loss, kl_loss)

        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl_loss": kl_loss,
            "mi": mutual_info_loss,
            "tc": tc_loss,
            "dw_kl": dimension_wise_kl,
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
        }
