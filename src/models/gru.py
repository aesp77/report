import keras
from keras import ops
from keras.models import Model
from keras.layers import GRU, Dense, Add
from keras.optimizers import Adam

@keras.saving.register_keras_serializable()
class GRUForecasterAugmented(Model):
    """
    GRU-based forecaster with residual connection for latent code prediction.
    Architecture:
        - Three stacked GRU layers (tanh activation)
        - Dense layer for delta prediction
        - Residual addition for output
    Options:
        - center_input: subtract mean from input sequence
        - lr: learning rate for Adam optimizer
    """
    def __init__(self, lookback, input_dim, latent_dim, lr=None, center_input=False, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.center_input = center_input
        self.gru1 = GRU(64, return_sequences=True, activation="tanh")
        self.gru2 = GRU(64, return_sequences=True, activation="tanh")
        self.gru3 = GRU(32, activation="tanh")
        self.delta_z = Dense(latent_dim, name="delta_z")
        self.add = Add(name="residual_z_out")
        self.compile(optimizer=Adam(lr), loss="mse")

    def call(self, inputs):
        # Optionally center input sequence
        if self.center_input:
            inputs = inputs - ops.mean(inputs, axis=1, keepdims=True)
        # Stacked GRU layers
        x = self.gru1(inputs)
        x = self.gru2(x)
        x = self.gru3(x)
        # Residual connection: base + delta
        z_base = inputs[:, -1, :self.latent_dim]
        z_delta = self.delta_z(x)
        return self.add([z_base, z_delta])

    def get_config(self):
        return {
            "lookback": self.lookback,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "lr": self.optimizer.learning_rate.numpy() if hasattr(self, "optimizer") else None,
            "center_input": self.center_input,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_gru_forecaster_augmented(lookback, input_dim, latent_dim, lr=None):
    """Factory for GRUForecasterAugmented."""
    return GRUForecasterAugmented(lookback, input_dim, latent_dim, lr=lr)
