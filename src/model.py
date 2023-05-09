import tensorflow as tf


class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(128, 10, 2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dense(500, activation='sigmoid'),
            tf.keras.layers.Reshape((500, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class VariationalAutoEncoder(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, length_series=500, prior=False):

        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.prior = prior
        self.beta_ = tf.Variable(0.0, name="kl_weight", trainable=False)

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(length_series, 1)),
                tf.keras.layers.Conv1D(filters=8, kernel_size=20, strides=5, activation='relu'),
                tf.keras.layers.Conv1D(filters=16, kernel_size=10, strides=5, activation='relu'),
                tf.keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                tf.keras.layers.Dense(2*latent_dim),
            ], name="Encoder"
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=10 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(10, 32)),
                tf.keras.layers.Conv1DTranspose(
                    filters=32, kernel_size=2, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv1DTranspose(
                    filters=16, kernel_size=10, strides=5, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv1DTranspose(
                    filters=8, kernel_size=20, strides=5, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv1DTranspose(
                    filters=1, kernel_size=1, strides=1, padding='same'),
            ], name="Decoder"
        )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="rec"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def print_model(self):
        print(self.encoder.summary())
        print(self.decoder.summary())

    def sample(self, number_of_samples=100, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(number_of_samples, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        if self.prior == True:
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar
        return self.encoder(x)

    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        import numpy as np
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @property
    def beta(self):
        return self.beta_

    @beta.setter
    def beta(self, value):
        self.beta_.assign(value)

    # def call(self, inputs):
    #    #mean, logvar = self.encode(inputs)
    #    #z = self.reparameterize(mean, logvar)
    #    #reconstruction = self.decode(z)
    #    reconstruction = self.predict(inputs)
    #    return reconstruction #, mean, logvar, z

    def train_step(self, data):

        x = data

        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decode(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_absolute_error(x, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "rec": self.reconstruction_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result(),
            "beta": self.beta,
        }

    def test_step(self, data):
        mean, logvar = self.encode(data)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_absolute_error(data, reconstruction), axis=1
            )
        )

        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + 0.01 * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

'''
import tensorflow as tf

class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim, intermediate_dim=128, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=5, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=5, activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=5, activation='relu')

        self.flatten = tf.keras.layers.Flatten()
        self.denseIntermediate = tf.keras.layers.Dense(intermediate_dim, activation='sigmoid')
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.denseIntermediate(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):

    def __init__(self, intermediate_dim=128, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.dense256 = tf.keras.layers.Dense(intermediate_dim, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape(target_shape=(4, 32))
        self.conv64trans = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=5, strides=5, padding='same',
                                                        activation='relu')
        self.conv32trans = tf.keras.layers.Conv1DTranspose(filters=8, kernel_size=5, strides=5, padding='same',
                                                        activation='relu')
        self.conv16trans = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=5, strides=5, padding='same',
                                                        activation='relu')

    def call(self, inputs):
        x = self.dense256(inputs)
        x = self.reshape(x)
        x = self.conv64trans(x)
        x = self.conv32trans(x)
        x = self.conv16trans(x)

        return x


class VariationalAutoEncoder56(tf.keras.Model):

    def __init__(
            self,
            original_dim,
            latent_dim=32,
            name="autoencoder",
            **kwargs
    ):
        super(VariationalAutoEncoder56, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()

    def print_model(self):
        print(self.encoder)
        print(self.decoder)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # kl_loss = -0.5 * tf.reduce_mean(
        #    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        # )
        # self.add_loss(kl_loss)

        return reconstructed


class AutoencoderSimple(tf.keras.Model):
    def __init__(self, length_series=1000, latent_dim=24, channels=1):
        super(AutoencoderSimple, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.length_series = length_series
        self.encoder = tf.keras.Sequential([
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          tf.keras.layers.Dense(channels*length_series, activation='sigmoid'),
          tf.keras.layers.Reshape((channels, length_series))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=10, strides=10, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=10, strides=5, activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=5, activation='relu')

        self.flatten = tf.keras.layers.Flatten()
        self.denseIntermediate = tf.keras.layers.Dense(intermediate_dim, activation='sigmoid')
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.denseIntermediate(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(tf.keras.layers.Layer):

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.denseIntermediate = tf.keras.layers.Dense(intermediate_dim, activation='sigmoid')

        self.dense256 = tf.keras.layers.Dense(256, activation='sigmoid')
        self.dense640 = tf.keras.layers.Dense(640, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape(target_shape=(20, 32))
        self.conv64trans = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=5, strides=5, padding='same',
                                                        activation='relu')
        self.conv32trans = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=10, strides=5, padding='same',
                                                        activation='relu')
        self.conv16trans = tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=10, strides=10, padding='same',
                                                        activation='relu')

    def call(self, inputs):
        x = self.denseIntermediate(inputs)
        x = self.dense256(x)
        x = self.dense640(x)
        x = self.reshape(x)
        x = self.conv64trans(x)
        x = self.conv32trans(x)
        x = self.conv16trans(x)

        return x


class VariationalAutoEncoder1(tf.keras.Model):

    def __init__(
            self,
            original_dim,
            latent_dim=32,
            intermediate_dim=64,
            name="autoencoder",
            **kwargs
    ):
        super(VariationalAutoEncoder1, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim, intermediate_dim=intermediate_dim)

    def print_model(self):
        print(self.encoder)
        print(self.decoder)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # kl_loss = -0.5 * tf.reduce_mean(
        #    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        # )
        # self.add_loss(kl_loss)

        return reconstructed

class VariationalAutoEncoder(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, length_series=500, latent_dim=24, channels=1, prior=False):
        
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.prior = prior
        self.beta_ = tf.Variable(0.0, name="kl_weight", trainable=False)
        
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(length_series, channels)),
                tf.keras.layers.Conv1D(
                    filters=8, kernel_size=10, strides=10, activation='relu'),
                tf.keras.layers.Conv1D(
                    filters=16, kernel_size=10, strides=10, activation='relu'),
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=2, strides=2, activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ], name="Encoder"
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=10*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(10, 32)),
                tf.keras.layers.Conv1DTranspose(
                    filters=32, kernel_size=2, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv1DTranspose(
                    filters=16, kernel_size=10, strides=5, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv1DTranspose(
                    filters=8, kernel_size=20, strides=10, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv1DTranspose(
                    filters=1, kernel_size=1, strides=1, padding='same'),
            ], name="Decoder"
        )
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="rec"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def print_model(self):
        print(self.encoder.summary())
        print(self.decoder.summary())

    @tf.function
    def sample(self, number_of_samples=100, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(number_of_samples, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def encode(self, x):
        if self.prior == True:
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar
        return self.encoder(x)

    @tf.function
    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        import numpy as np
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)
    
    @property
    def beta(self):
        return self.beta_

    @beta.setter
    def beta(self, value):
        self.beta_.assign(value)

    @tf.function
    def call(self, inputs):
        mean, logvar = self.encoder(inputs)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        #reconstruction = self.predict(inputs)
        return reconstruction #mean, logvar, z

    @tf.function
    def train_step(self, data):
        
        x = data
        
        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            reconstruction = self.decode(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_absolute_error(x, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "rec": self.reconstruction_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result(),
            "beta": self.beta,
        }

    @tf.function
    def test_step(self, data):
        mean, logvar = self.encoder(data)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_absolute_error(data, reconstruction), axis=1
            )
        )
        
        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + 0.01*kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {m.name: m.result() for m in self.metrics}
'''