# cnn_dec.py

import keras
from keras import ops
from keras.layers import Dense, Conv2DTranspose, Reshape, BatchNormalization, Activation, Conv2D, Input, Dropout
from keras.models import Model, Sequential
from keras.saving import register_keras_serializable
import numpy as np
import scipy.optimize


@register_keras_serializable()
class CNNDecoder(keras.Model):
    """CNN-based decoder for volatility surface reconstruction."""
    def __init__(self, latent_dim, M, K, feature_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.input_dim = latent_dim + feature_dim
        self.M = M
        self.K = K
        self.output_shape_2d = (M, K, 1)

        self.encoder = Sequential([
            Dense(2 * M * K, activation="gelu"),
            Dense(M * K, activation="gelu"),
        ])
        self.reshape = Reshape((M, K, 1))

        self.deconv = Sequential([
            Conv2DTranspose(128, 3, padding="same", activation="gelu"),
            Conv2DTranspose(64, 3, padding="same", activation="gelu"),
            Conv2DTranspose(32, 3, padding="same", activation="gelu"),
            Conv2DTranspose(1, 1, padding="same", activation="softplus"),
            Reshape(self.output_shape_2d)
        ])

    def call(self, zf, training=False):
        """Forward pass for CNN decoder."""
        x = self.encoder(zf)
        x = self.reshape(x)
        out = self.deconv(x, training=training)
        return out[:, :, :, 0]


    def predict_surface(self, z_vec, f_vec=None):
        """Predict volatility surface from latent and feature vectors."""
        z_vec = ops.convert_to_numpy(z_vec).reshape(1, -1).astype(np.float32)
        if self.feature_dim > 0 and f_vec is not None:
            f_vec = ops.convert_to_numpy(f_vec).reshape(1, -1).astype(np.float32)
            zf = np.concatenate([z_vec, f_vec], axis=-1)
        else:
            zf = z_vec

        zf_tensor = ops.convert_to_tensor(zf, dtype="float32")
        iv_surface = self(zf_tensor, training=False)
        return ops.convert_to_numpy(iv_surface)[0]

    def refine_surface(self, surface, lambda_cal=0.1, lambda_smile=0.1):
        """Refine surface with calendar and smile penalties."""
        M, K = self.M, self.K
        flat = surface.flatten()

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            sigma_sq = surf ** 2
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0), 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(surf, n=2, axis=1), 0, None))
            return lambda_cal * cal_penalty + lambda_smile * smile_penalty

        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def get_config(self):
        """Return model config."""
        return {
            "latent_dim": self.latent_dim,
            "M": self.M,
            "K": self.K,
            "feature_dim": self.feature_dim
        }

    @classmethod
    def from_config(cls, config):
        """Create model from config."""
        return cls(**config)
@register_keras_serializable()
class CNNDecoderImproved(keras.Model):
    """Improved CNN decoder with dropout and more layers."""
    def __init__(self, latent_dim, M, K, hidden_dim=512, dropout_rate=0.2, feature_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.input_dim = latent_dim + feature_dim
        self.M = M
        self.K = K
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.encoder = Sequential([
            Dense(hidden_dim, activation="elu"),
            Dropout(dropout_rate),
            Dense(M * K, activation="linear"),
            Reshape((M, K, 1))
        ])

        self.deconv = Sequential([
            Conv2D(64, kernel_size=3, padding="same", activation="elu"),
            # BatchNormalization(),
            Conv2D(32, kernel_size=3, padding="same", activation="elu"),
            # BatchNormalization(),
            Conv2D(16, kernel_size=3, padding="same", activation="elu"),
            Conv2D(1, kernel_size=1, padding="same", activation="softplus"),
        ])

    def call(self, z, training=False):
        """Forward pass for improved CNN decoder."""
        x = self.encoder(z)
        x = self.deconv(x, training=training)
        return x[:, :, :, 0]


    def predict_surface(self, z_vec, f_vec=None):
        """Predict volatility surface from latent and feature vectors."""
        z_vec = np.asarray(z_vec).reshape(1, -1).astype(np.float32)
        if self.feature_dim > 0 and f_vec is not None:
            f_vec = np.asarray(f_vec).reshape(1, -1).astype(np.float32)
            z_vec = np.concatenate([z_vec, f_vec], axis=-1)
        z_tensor = ops.convert_to_tensor(z_vec, dtype="float32")
        iv_surface = self(z_tensor, training=False)
        return ops.convert_to_numpy(iv_surface)[0]

    def refine_surface(self, surface, lambda_cal=1.0, lambda_smile=1.0):
        """Refine surface with calendar and smile penalties."""
        M, K = self.M, self.K
        flat = surface.flatten()

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            sigma_sq = surf ** 2
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0), 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(surf, n=2, axis=1), 0, None))
            return lambda_cal * cal_penalty + lambda_smile * smile_penalty

        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def get_config(self):
        """Return model config."""
        return {
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "M": self.M,
            "K": self.K,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate
        }

    @classmethod
    def from_config(cls, config):
        """Create model from config."""
        return cls(**config)
