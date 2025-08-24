from keras import Model, ops
from keras.layers import Input, Dense
from keras.regularizers import l2
from keras.saving import register_keras_serializable

@register_keras_serializable()
class AE(Model):
    """Autoencoder with dense layers for dimensionality reduction."""
    def __init__(self, input_dim, latent_dim, optimizer, l2_penalty=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.encoder_net = self._build_encoder(input_dim, latent_dim, l2_penalty)
        self.decoder_net = self._build_decoder(latent_dim, input_dim, l2_penalty)
        self.compile(optimizer=optimizer, loss="mse")
        self.history = None

    def _build_encoder(self, input_dim, latent_dim, l2_penalty):
        """Build encoder network."""
        x = Input(shape=(input_dim,), name="input_vector")
        h = Dense(512, activation="relu", kernel_regularizer=l2(l2_penalty))(x)
        h = Dense(256, activation="relu", kernel_regularizer=l2(l2_penalty))(h)
        h = Dense(128, activation="relu", kernel_regularizer=l2(l2_penalty))(h)
        h = Dense(64, activation="relu", kernel_regularizer=l2(l2_penalty))(h)
        z = Dense(latent_dim, name="z")(h)
        return Model(x, z, name="ae_encoder")

    def _build_decoder(self, latent_dim, output_dim, l2_penalty):
        """Build decoder network."""
        z = Input(shape=(latent_dim,), name="latent_input")
        h = Dense(64, activation="relu", kernel_regularizer=l2(l2_penalty))(z)
        h = Dense(128, activation="relu", kernel_regularizer=l2(l2_penalty))(h)
        h = Dense(256, activation="relu", kernel_regularizer=l2(l2_penalty))(h)
        h = Dense(512, activation="relu", kernel_regularizer=l2(l2_penalty))(h)
        x_recon = Dense(output_dim, activation="linear", name="reconstructed")(h)
        return Model(z, x_recon, name="ae_decoder")

    def call(self, inputs):
        """Encode and decode input."""
        z = self.encoder_net(inputs)
        return self.decoder_net(z)

    def fit_transform(self, X, **kwargs):
        """Fit autoencoder and return encoded representation."""
        self.fit(X, X, **kwargs)
        return ops.convert_to_numpy(self.encoder_net(X))

    def inverse_transform(self, Z):
        """Decode latent representation back to input space."""
        return ops.convert_to_numpy(self.decoder_net(Z))
    def get_config(self):
        """Return model config."""
        return {
            "input_dim": self.encoder_net.input_shape[-1],
            "latent_dim": self.decoder_net.input_shape[-1],
            "optimizer": self.optimizer,
        }

    @classmethod
    def from_config(cls, config):
        """Create model from config."""
        return cls(**config)

