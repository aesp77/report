import keras
from keras import Model, layers
from keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Add, GlobalAveragePooling1D, Embedding
)
import keras.ops as ops
import numpy as np
from keras.saving import register_keras_serializable


def positional_encoding(length, d_model):
    """Generates positional encoding for transformer input.
    Uses sine/cosine functions for even/odd dimensions.
    """
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros((length, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return ops.convert_to_tensor(pos_encoding, dtype="float32")


@register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    """Transformer encoder block with multi-head attention and feedforward layers.
    Activation: relu in feedforward.
    """
    def __init__(self, d_model, ff_dim, n_heads, dropout):
        super().__init__()
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)
        self.ff1 = Dense(ff_dim, activation="relu")
        self.ff2 = Dense(d_model)

    def call(self, x, training=False):
        # Multi-head attention with residual connection
        attn_out = self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.drop1(attn_out, training=training)
        # Feedforward network with residual connection
        ff_out = self.ff2(self.ff1(self.norm2(x)))
        x = x + self.drop2(ff_out, training=training)
        return x


@register_keras_serializable()
class TransformerForecaster(Model):
    """Transformer forecaster for time series to latent representation.
    Encoder: multiple TransformerEncoderBlocks.
    Activation: relu in feedforward layers.
    Options:
        center_input: subtract mean from input sequence
        d_model, n_heads, ff_dim, n_layers, dropout: transformer hyperparameters
    Output: Dense(latent_dim)
    """
    def __init__(self, lookback, input_dim, latent_dim,
                 d_model=64, n_heads=4, ff_dim=128, n_layers=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.center_input = True
        self.lookback = lookback
        self.input_proj = Dense(d_model)
        self.pos_encoding = positional_encoding(lookback, d_model)
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, ff_dim, n_heads, dropout)
            for _ in range(n_layers)
        ]
        self.pool = GlobalAveragePooling1D()
        self.out_proj = Dense(latent_dim)

    def call(self, inputs, training=False):
        if self.center_input:
            inputs = inputs - ops.mean(inputs, axis=1, keepdims=True)
        x = self.input_proj(inputs)
        x = x + self.pos_encoding
        for block in self.encoder_blocks:
            x = block(x, training=training)
        x = self.pool(x)
        return self.out_proj(x)


    def get_config(self):
        return {
            "lookback": self.lookback,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }


    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class TransformerForecasterV2(Model):
    """Transformer forecaster V2 for time series to latent representation.
    Encoder: multiple attention blocks with embedding-based positional encoding.
    Activation: relu in feedforward layers.
    Options:
        d_model, n_heads, ff_dim, n_layers, dropout: transformer hyperparameters
    Output: Dense(latent_dim)
    """
    def __init__(self, lookback, input_dim, latent_dim,
                 d_model=128, n_heads=4, ff_dim=256, n_layers=6, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.center_input = False
        self.lookback = lookback
        self.input_proj = Dense(d_model)
        self.pos_embedding = Embedding(input_dim=lookback, output_dim=d_model)
        self.attn_layers = []
        self.norm1_layers = []
        self.norm2_layers = []
        self.ff1_layers = []
        self.ff2_layers = []
        self.drop1_layers = []
        self.drop2_layers = []
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(num_heads=n_heads, key_dim=d_model))
            self.norm1_layers.append(LayerNormalization())
            self.norm2_layers.append(LayerNormalization())
            self.ff1_layers.append(Dense(ff_dim, activation="relu"))
            self.ff2_layers.append(Dense(d_model))
            self.drop1_layers.append(Dropout(dropout))
            self.drop2_layers.append(Dropout(dropout))
        self.out_proj = Dense(latent_dim)

    def call(self, inputs, training=False):
        # Optionally center input sequence
        if self.center_input:
            inputs = inputs - ops.mean(inputs, axis=1, keepdims=True)
        B, T, _ = ops.shape(inputs)
        x = self.input_proj(inputs)
        positions = ops.arange(self.lookback)
        pos_embed = self.pos_embedding(positions)
        x = x + pos_embed
        for i in range(len(self.attn_layers)):
            x_norm = self.norm1_layers[i](x)
            attn_out = self.attn_layers[i](x_norm, x_norm, training=training)
            x = x + self.drop1_layers[i](attn_out, training=training)
            x_norm = self.norm2_layers[i](x)
            ff_out = self.ff2_layers[i](self.ff1_layers[i](x_norm))
            x = x + self.drop2_layers[i](ff_out, training=training)
        return self.out_proj(x[:, -1, :])


    def get_config(self):
        return {
            "lookback": self.lookback,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }


    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_attention_forecaster(lookback, input_dim, latent_dim, lr=5e-4, n_heads=4, dropout=0.2, use_layernorm=True):
    """Builds a simple attention-based forecaster model.
    Layers: LayerNorm (optional), MultiHeadAttention, Dense with relu, Dense(latent_dim).
    Output: latent representation.
    """
    inputs = layers.Input(shape=(lookback, input_dim), name="sequence_input")
    x = inputs
    if use_layernorm:
        x = LayerNormalization()(x)
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=input_dim)(x, x)
    x = Add()([inputs, attn_output])
    if use_layernorm:
        x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(latent_dim, name="z_output")(x)
    model = Model(inputs, output, name="attention_forecaster")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model
