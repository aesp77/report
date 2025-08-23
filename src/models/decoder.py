import os
import numpy as np
import scipy
import keras
from keras.layers import TimeDistributed

from keras import ops
from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
from keras.metrics import Mean


def build_decoder_dataset(Z_latent, Y_surface, strike_tensor, tau_tensor, M, K, F_features=None, splits=(0.7, 0.15, 0.15)):
    """Splits decoder dataset into train/val/test sets for surface models."""
    N = len(Z_latent)
    n_train = int(splits[0] * N)
    n_val   = int(splits[1] * N)
    n_test  = N - n_train - n_val
    Z_train, Z_val, Z_test = np.split(Z_latent, [n_train, n_train + n_val])
    Y_train, Y_val, Y_test = np.split(Y_surface, [n_train, n_train + n_val])
    if F_features is not None:
        F_train, F_val, F_test = np.split(F_features, [n_train, n_train + n_val])
        ZF_train = np.concatenate([Z_train, F_train], axis=-1)
        ZF_val   = np.concatenate([Z_val,   F_val],   axis=-1)
        ZF_test  = np.concatenate([Z_test,  F_test],  axis=-1)
    else:
        ZF_train, ZF_val, ZF_test = Z_train, Z_val, Z_test
    return {
        "train": (ZF_train, Y_train),
        "val":   (ZF_val, Y_val),
        "test":  (ZF_test, Y_test)
    }


@register_keras_serializable()
class SimpleSurfaceDecoder(Model):
    """Decoder model for volatility surface reconstruction.
    Hidden layers: Dense with relu activation.
    Output: Dense(M*K) with linear activation.
    Options:
        lambda_cal, lambda_smile: penalty weights
        use_penalties: enable/disable smoothness penalties
    """
    def __init__(self, latent_dim, M, K, feature_dim=0,
                 lambda_cal=1, lambda_smile=1, use_penalties=True, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.M = M
        self.K = K
        self.lambda_cal = lambda_cal
        self.lambda_smile = lambda_smile
        self.use_penalties = use_penalties
        self.hidden = [
            Dense(64, activation="relu"),
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
            Dense(512, activation="relu"),
        ]
        self.output_layer = Dense(M * K, activation="linear")
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, zf):
        x = zf
        for layer in self.hidden:
            x = layer(x)
        return self.output_layer(x)

    def train_step(self, data):
        x, y_true = data

        y_pred_flat = self(x, training=True)
        y_pred = ops.reshape(y_pred_flat, (-1, self.M, self.K))

        base_loss = ops.square(y_pred_flat - y_true)

        # --- Optional ATM weighting (center strike)
        weight_mask = ops.ones_like(y_pred)
        #weight_mask[..., self.K // 2] *= 2.0
        base_loss_weighted = weight_mask * ops.square(y_pred - ops.reshape(y_true, (-1, self.M, self.K)))
        recon_loss = ops.mean(base_loss_weighted)

        # --- Smooth penalties
        sigma_sq = ops.square(y_pred)
        cal_penalty = ops.mean(ops.square(ops.diff(-ops.diff(sigma_sq, axis=1))))
        smile_penalty = ops.mean(ops.square(ops.diff(-ops.diff(y_pred, n=2, axis=2))))

        # --- L2 regularization
        l2_penalty = ops.sum([ops.sum(ops.square(v.value)) for v in self.trainable_weights])

        # --- Total loss
        loss = recon_loss
        if self.use_penalties:
            loss += self.lambda_cal * cal_penalty + self.lambda_smile * smile_penalty
        loss += 1e-6 * l2_penalty

        # --- Backprop (Torch-style)
        loss.backward()
        gradients = [v.value.grad for v in self.trainable_weights]

        import torch
        with torch.no_grad():
            self.optimizer.apply(gradients, self.trainable_weights)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}



    def test_step(self, data):
        x, y_true = data
        y_pred_flat = self(x, training=False)
        y_pred = ops.reshape(y_pred_flat, (-1, self.M, self.K))

        # --- Weighted ATM reconstruction loss
        weight_mask = ops.ones_like(y_pred)
        
        base_loss_weighted = weight_mask * ops.square(y_pred - ops.reshape(y_true, (-1, self.M, self.K)))
        recon_loss = ops.mean(base_loss_weighted)

        # --- Smooth penalties
        sigma_sq = ops.square(y_pred)
        cal_penalty = ops.mean(ops.square(ops.diff(-ops.diff(sigma_sq, axis=1))))
        smile_penalty = ops.mean(ops.square(ops.diff(-ops.diff(y_pred, n=2, axis=2))))

        # --- L2 regularization (eval only)
        l2_penalty = ops.sum([ops.sum(ops.square(w.value)) for w in self.trainable_weights])

        # --- Total loss
        loss = recon_loss
        if self.use_penalties:
            loss += self.lambda_cal * cal_penalty + self.lambda_smile * smile_penalty
        loss += 1e-6 * l2_penalty

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


    @property
    def metrics(self):
        return [self.loss_tracker]

    def predict_surface(self, z_vec, f_vec=None):
        z_vec = np.asarray(z_vec, dtype=np.float32).reshape(1, -1)
        if self.feature_dim > 0 and f_vec is not None:
            f_vec = np.asarray(f_vec, dtype=np.float32).reshape(1, -1)
            zf = np.concatenate([z_vec, f_vec], axis=-1)
        else:
            zf = z_vec
        zf_tensor = ops.convert_to_tensor(zf, dtype="float32")
        iv_flat = self(zf_tensor, training=False)
        return ops.convert_to_numpy(iv_flat).reshape(self.M, self.K)

    def refine_surface(self, surface, lambda_cal=1, lambda_smile=1):
        import scipy.optimize
        M, K = self.M, self.K
        flat = surface.flatten()

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            sigma_sq = np.square(surf)
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0), 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(surf, n=2, axis=1), 0, None))
            return lambda_cal * cal_penalty + lambda_smile * smile_penalty

        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "M": self.M,
            "K": self.K,
            "feature_dim": self.feature_dim,
            "lambda_cal": self.lambda_cal,
            "lambda_smile": self.lambda_smile,
            "use_penalties": self.use_penalties,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)





from keras import Model, ops
from keras.layers import Input, Dense, TimeDistributed, RepeatVector, Reshape
from keras.saving import register_keras_serializable

@register_keras_serializable()
class SliceSurfaceDecoder(Model):
    """Decoder model for volatility surface using slice-wise approach.
    Encoder: Dense layers with elu activation.
    Output: TimeDistributed(Dense(K, softplus)).
    """
    def __init__(self, latent_dim, M, K, taus, feature_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.input_dim = latent_dim + feature_dim
        self.M = M
        self.K = K
        self.taus = ops.convert_to_tensor(np.array(taus).reshape(M, 1), dtype="float32")  # (M, 1)
        self.encoder = keras.Sequential([
            Dense(512, activation="elu"),
            Dense(256, activation="elu"),
            Dense(128, activation="elu"),
        ])
        self.slice_net = TimeDistributed(Dense(K, activation="softplus"))

    def call(self, zf):
        B = ops.shape(zf)[0]
        taus_tiled = ops.tile(self.taus, (B, 1))             # (B*M, 1)
        zf_tiled = ops.repeat(zf, self.M, axis=0)            # (B*M, D)
        zf_tau = ops.concatenate([zf_tiled, taus_tiled], -1) # (B*M, D+1)
        x = self.encoder(zf_tau)                             # (B*M, hidden)
        x = ops.reshape(x, (B, self.M, -1))                  # (B, M, hidden)
        return self.slice_net(x)                             # (B, M, K)

    def predict_surface(self, z, f, *args):
        z = ops.convert_to_numpy(z).reshape(1, -1).astype(np.float32)
        f = ops.convert_to_numpy(f).reshape(1, -1).astype(np.float32)
        zf = np.concatenate([z, f], axis=-1) if self.feature_dim > 0 else z
        zf_tensor = ops.convert_to_tensor(zf, dtype="float32")
        iv_surface = self(zf_tensor, training=False)
        return ops.convert_to_numpy(iv_surface)[0]

    def refine_surface(self, surface, lambda_cal=0.1, lambda_smile=0.1):
        import scipy.optimize
        M, K = self.M, self.K
        flat = surface.flatten()

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            sigma_sq = np.square(surf)
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0), 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(surf, n=2, axis=1), 0, None))
            return lambda_cal * cal_penalty + lambda_smile * smile_penalty

        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "M": self.M,
            "K": self.K,
            "taus": ops.convert_to_numpy(self.taus).flatten().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        taus = config.pop("taus")
        return cls(taus=taus, **config)


@register_keras_serializable()
class PiecewiseSurfaceDecoder(Model):
    """Piecewise decoder for volatility surface.
    Hidden layers: Dense with gelu activation.
    Output: Dense(1) with softplus activation.
    """
    def __init__(self, latent_dim, M, K, feature_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.M = M
        self.K = K
        self.hidden = [
            Dense(256, activation="gelu"),
            Dense(128, activation="gelu"),
            Dense(64, activation="gelu"),
        ]
        self.output_layer = Dense(1, activation="softplus")

    def call(self, inputs):
        zf, m, tau = inputs  # zf = [z] or [z || f]
        x = ops.concatenate([zf, m, tau], axis=-1)
        for layer in self.hidden:
            x = layer(x)
        return self.output_layer(x)

    def predict_surface(self, z, f, m_flat, tau_flat):
        z = ops.convert_to_numpy(z).reshape(1, -1).astype(np.float32)
        f = ops.convert_to_numpy(f).reshape(1, -1).astype(np.float32)
        zf = np.concatenate([z, f], axis=-1) if self.feature_dim > 0 else z

        z_repeat = np.repeat(zf, self.M * self.K, axis=0)
        m_tensor = ops.convert_to_tensor(m_flat.astype("float32"))
        tau_tensor = ops.convert_to_tensor(tau_flat.astype("float32"))
        iv = self([z_repeat, m_tensor, tau_tensor], training=False)
        return ops.convert_to_numpy(iv).reshape(self.M, self.K)


    def refine_surface(self, surface, lambda_cal=0.1, lambda_smile=0.1, lambda_extreme=0.1):
        import scipy.optimize
        M, K = self.M, self.K
        flat = surface.flatten()

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            sigma_sq = np.square(surf)
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0), 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(surf, n=2, axis=1), 0, None))
            d_left = surf[:, 1] - surf[:, 0]
            d_right = surf[:, -1] - surf[:, -2]
            extreme_penalty = np.mean(np.square(d_left)) + np.mean(np.square(d_right))
            return lambda_cal * cal_penalty + lambda_smile * smile_penalty + lambda_extreme * extreme_penalty

        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def build_training_data_from_surfaces(self, Z_latent, Y_surface_flat, strike_tensor, tau_tensor, F_features=None):
        M, K = self.M, self.K
        m_grid, tau_grid = np.meshgrid(strike_tensor, tau_tensor)
        m_flat = m_grid.reshape(-1, 1)
        tau_flat = tau_grid.reshape(-1, 1)

        X_zf, X_m, X_tau, y = [], [], [], []
        for i, (z_vec, surf_flat) in enumerate(zip(Z_latent, Y_surface_flat)):
            z = z_vec if self.feature_dim == 0 else ops.convert_to_numpy(ops.concatenate([z_vec, F_features[i]], axis=-1))
            z_repeat = np.repeat(z[None, :], M * K, axis=0)
            X_zf.append(z_repeat)
            X_m.append(m_flat)
            X_tau.append(tau_flat)
            y.append(surf_flat.reshape(-1, 1))

        return [np.vstack(X_zf), np.vstack(X_m), np.vstack(X_tau)], np.vstack(y)

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "M": self.M,
            "K": self.K,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

from keras.layers import Activation
from keras.layers import LayerNormalization

@register_keras_serializable()
class PiecewiseSurfaceDecoderModular(keras.Model):
    """Modular piecewise decoder for volatility surface.
    Hidden layers: Dense with configurable activation (elu, relu, gelu, tanh, sigmoid).
    Options:
        tau_expand, m_expand: expand tau/m input features
        use_layernorm: apply layer normalization
        dropout_rate: dropout after each layer
        atm_weighting: apply ATM weighting in penalty
    Output: Dense(1) with softplus activation.
    """
    def __init__(self, latent_dim, M, K,
                 feature_dim=0,
                 activation="elu",
                 tau_expand=False,
                 m_expand=False,
                 use_layernorm=False,
                 dropout_rate=0.0,
                 atm_weighting=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.M = M
        self.K = K
        self.activation = activation
        self.tau_expand = tau_expand
        self.m_expand = m_expand
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout_rate
        self.atm_weighting = atm_weighting
        self.dense_layers = []
        for units in [256, 128, 64]:
            self.dense_layers.append(Dense(units))
            if self.use_layernorm:
                self.dense_layers.append(LayerNormalization())
            self.dense_layers.append(Activation(self._get_activation(self.activation)))
            if self.dropout_rate > 0:
                self.dense_layers.append(Dropout(self.dropout_rate))
        self.output_layer = Dense(1, activation="softplus")

    def _get_activation(self, name):
        return {
            "relu": keras.activations.relu,
            "elu": keras.activations.elu,
            "gelu": keras.activations.gelu,
            "tanh": keras.activations.tanh,
            "sigmoid": keras.activations.sigmoid
        }.get(name, keras.activations.elu)

    def call(self, inputs):
        if self.feature_dim > 0 and len(inputs) == 4:
            z, m, tau, f = inputs
            z = ops.concatenate([z, f], axis=-1)
        else:
            z, m, tau = inputs

        if self.m_expand:
            m = ops.concatenate([m, ops.square(m)], axis=-1)
        if self.tau_expand:
            tau = ops.concatenate([tau, ops.square(tau), ops.log(ops.add(tau, 1e-3))], axis=-1)

        x = ops.concatenate([z, m, tau], axis=-1)
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

    def predict_surface(self, z, f, m_flat, tau_flat):
        z = ops.convert_to_numpy(z).reshape(1, -1).astype(np.float32)
        if self.feature_dim > 0:
            f = ops.convert_to_numpy(f).reshape(1, -1).astype(np.float32)
            zf = np.concatenate([z, f], axis=-1)
        else:
            zf = z

        z_repeat = np.repeat(zf, self.M * self.K, axis=0)
        m_tensor = ops.convert_to_tensor(m_flat.astype("float32"))
        tau_tensor = ops.convert_to_tensor(tau_flat.astype("float32"))
        

        iv = self([z_repeat, m_tensor, tau_tensor], training=False)
        return ops.convert_to_numpy(iv).reshape(self.M, self.K)

    def refine_surface(self, surface, lambda_cal=0.1, lambda_smile=0.1):
        import scipy.optimize
        M, K = self.M, self.K
        flat = surface.flatten()

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            sigma_sq = np.square(surf)
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0), 0, None))
            smile_diff2 = np.diff(surf, n=2, axis=1)
            if self.atm_weighting:
                weights = np.abs(np.arange(K) - K // 2) / (K // 2)
                weighted_smile = (smile_diff2 ** 2) * weights[1:-1][None, :]
                smile_pen = np.mean(weighted_smile)
            else:
                smile_pen = np.mean(np.clip(-smile_diff2, 0, None))
            return lambda_cal * cal_penalty + lambda_smile * smile_pen

        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def build_training_data_from_surfaces(self, Z_latent, Y_surface_flat, strike_tensor, tau_tensor, F_features=None):
        M, K = self.M, self.K
        m_grid, tau_grid = np.meshgrid(strike_tensor, tau_tensor)
        m_flat = m_grid.reshape(-1, 1)
        tau_flat = tau_grid.reshape(-1, 1)

        X_f, X_m, X_tau, y = [], [], [], []
        for i, (z_vec, surf_flat) in enumerate(zip(Z_latent, Y_surface_flat)):
            if self.feature_dim > 0 and F_features is not None:
                zf = ops.convert_to_numpy(ops.concatenate([z_vec, F_features[i]], axis=-1))
            else:
                zf = z_vec
            z_repeat = np.repeat(zf[None, :], M * K, axis=0)
            X_f.append(z_repeat)
            X_m.append(m_flat)
            X_tau.append(tau_flat)
            y.append(surf_flat.reshape(-1, 1))

        return [np.vstack(X_f), np.vstack(X_m), np.vstack(X_tau)], np.vstack(y)

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "M": self.M,
            "K": self.K,
            "activation": self.activation,
            "tau_expand": self.tau_expand,
            "m_expand": self.m_expand,
            "use_layernorm": self.use_layernorm,
            "dropout_rate": self.dropout_rate,
            "atm_weighting": self.atm_weighting
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)




@register_keras_serializable()
class ArbitrageDecoder2_pointwise(keras.Model):
    """Arbitrage decoder (pointwise) for volatility surface.
    Hidden layers: Dense with relu activation.
    Output: Dense(1) with softplus activation.
    Options:
        lambda_cal, lambda_smile: penalty weights
    """
    def __init__(self, latent_dim, M, K, m_flat, tau_flat, lambda_cal=0.1, lambda_smile=0.1):
        super().__init__()
        self.M = M
        self.K = K
        self.m_flat = m_flat
        self.tau_flat = tau_flat
        self.lambda_cal = lambda_cal
        self.lambda_smile = lambda_smile
        self.model = self._build_model(latent_dim)

    def _build_model(self, latent_dim):
        feature_input = Input(shape=(latent_dim,), name="features")
        m_input = Input(shape=(1,), name="m")
        tau_input = Input(shape=(1,), name="tau")

        x = Concatenate()([feature_input, m_input, tau_input])
        x = Dense(512, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        iv_out = Dense(1, activation="softplus", name="iv")(x)
        return Model(inputs=[feature_input, m_input, tau_input], outputs=iv_out)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def _calendar_penalty(self, y_hat):
        B = ops.shape(y_hat)[0] // (self.M * self.K)
        y_hat_2d = ops.reshape(y_hat, (B, self.M, self.K))
        sigma_sq = ops.square(y_hat_2d)
        diff = ops.diff(sigma_sq, axis=1)
        return ops.mean(ops.relu(-diff))

    def _smile_penalty(self, y_hat):
        B = ops.shape(y_hat)[0] // (self.M * self.K)
        y_hat_2d = ops.reshape(y_hat, (B, self.M, self.K))
        diff = ops.diff(y_hat_2d, n=2, axis=2)
        return ops.mean(ops.relu(-diff))

    def _loss_with_penalties(self, y_true, y_pred):
        mse = ops.mean(ops.square(y_true - y_pred))
        cal_penalty = self._calendar_penalty(y_pred)
        smile_penalty = self._smile_penalty(y_pred)
        return mse + self.lambda_cal * cal_penalty + self.lambda_smile * smile_penalty

    def build_training_data(self, X_decoded, surface_tensor, test_idx, *_):
        M, K = self.M, self.K
        X_f, X_m, X_tau, y = [], [], [], []

        for i, f_vec in enumerate(X_decoded):
            d_idx = test_idx[i]
            iv_surface = ops.convert_to_numpy(surface_tensor[d_idx])[..., -1].flatten()
            f_repeat = np.repeat(f_vec[None, :], M * K, axis=0)
            m_repeat = np.repeat(self.m_flat, M, axis=0)
            tau_repeat = np.tile(self.tau_flat, (K, 1))
            X_f.append(f_repeat)
            X_m.append(m_repeat)
            X_tau.append(tau_repeat)
            y.append(iv_surface.reshape(-1, 1))

        return [np.vstack(X_f), np.vstack(X_m), np.vstack(X_tau)], np.vstack(y)

    def train_step(self, data):
        (f_input, m_input, tau_input), y_true = data
        with keras.backend.GradientTape() as tape:
            y_pred = self([f_input, m_input, tau_input], training=True)
            loss = self._loss_with_penalties(y_true, y_pred)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(grads, self.trainable_variables)
        self.loss_tracker_train.update_state(loss)
        return {"loss": self.loss_tracker_train.result()}

    def compile_with_loss_tracking(self, learning_rate=1e-5):
        self.optimizer = Adam(learning_rate)
        self.loss_tracker_train = Mean(name="loss")
        self.loss_tracker_val = Mean(name="val_loss")
        self.val_losses = []
        self.compile(run_eagerly=True)

    def test_step(self, data):
        (f_input, m_input, tau_input), y_true = data
        y_pred = self([f_input, m_input, tau_input], training=False)
        loss = self._loss_with_penalties(y_true, y_pred)
        self.val_losses.append(loss)
        return {}

    def test_epoch_end(self, outputs):
        mean_loss = ops.mean(ops.stack(self.val_losses))
        self.val_losses.clear()
        return {"loss": mean_loss}

    @property
    def metrics(self):
        return [self.loss_tracker_train, self.loss_tracker_val]

    def predict_surface(self, f_vec):
        repeated_f = np.repeat(f_vec[None, :], self.M * self.K, axis=0)
        m_grid = np.tile(self.m_flat, (self.M, 1))
        tau_grid = np.repeat(self.tau_flat, self.K, axis=0)
        iv = self.model.predict([repeated_f, m_grid, tau_grid], verbose=0)
        return iv.reshape(self.M, self.K)

    def refine_surface(self, surface):
        import scipy.optimize
        M, K = self.M, self.K
        flat = surface.flatten()

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            sigma_sq = np.square(surf)
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0), 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(surf, n=2, axis=1), 0, None))
            return self.lambda_cal * cal_penalty + self.lambda_smile * smile_penalty

        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def get_config(self):
        return {
            "latent_dim": self.model.input_shape[0][-1],
            "M": self.M,
            "K": self.K,
            "lambda_cal": self.lambda_cal,
            "lambda_smile": self.lambda_smile
        }

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError("Reload not supported (m_flat and tau_flat are not serializable).")













######################### OLD ARBITRAGE DECODERS #########################
@register_keras_serializable()
class ArbitrageDecoder:
    """Arbitrage decoder for volatility surface.
    Hidden layers: TimeDistributed(Dense) with relu activation (surface mode), Dense with relu (pointwise mode).
    Output: Dense(1) with linear activation.
    Options:
        use_split_loss: use split MSE loss
        smile_weight, term_weight: penalty weights
    """
    def __init__(self, latent_dim, M, K, m_flat, tau_flat,
                 use_split_loss=True,
                 smile_weight=1.0, term_weight=1.0,
                 optimizer=None):
        self.M = M
        self.K = K
        self.m_flat = m_flat
        self.tau_flat = tau_flat
        self.smile_weight = smile_weight
        self.term_weight = term_weight
        self.use_split_loss = use_split_loss
        self.model = self._build_model(latent_dim)
        if optimizer is None:
            optimizer = Adam(1e-4)
        loss = self._split_mse_loss if use_split_loss else "mse"
        if self.use_split_loss:
            self.model.compile(optimizer=optimizer, loss=self._split_mse_loss)
        else:
            self.model.compile(optimizer=optimizer, loss="mse")



    def _build_model(self, latent_dim):
        if self.use_split_loss:
            # --- Surface mode (N, M*K, latent_dim + m + tau)
            z_input = Input(shape=(self.M * self.K, latent_dim), name="features")
            m_input = Input(shape=(self.M * self.K, 1), name="m")
            tau_input = Input(shape=(self.M * self.K, 1), name="tau")

            x = Concatenate(axis=-1)([z_input, m_input, tau_input])
            for _ in range(2): x = keras.layers.TimeDistributed(Dense(256, activation="relu"))(x)
            for _ in range(2): x = keras.layers.TimeDistributed(Dense(128, activation="relu"))(x)
            for _ in range(2): x = keras.layers.TimeDistributed(Dense(64, activation="relu"))(x)
            iv_out = keras.layers.TimeDistributed(Dense(1, activation="linear"))(x)

            return Model(inputs=[z_input, m_input, tau_input], outputs=iv_out)
        
        else:
            # --- Pointwise mode (flat vector inputs)
            z_input = Input(shape=(latent_dim,), name="features")
            m_input = Input(shape=(1,), name="m")
            tau_input = Input(shape=(1,), name="tau")

            x = Concatenate()([z_input, m_input, tau_input])
            for _ in range(2): x = Dense(128, activation="relu")(x)
            for _ in range(2): x = Dense(64, activation="relu")(x)
            iv_out = Dense(1, activation="linear")(x)

            return Model(inputs=[z_input, m_input, tau_input], outputs=iv_out)


    def _split_mse_loss(self, y_true, y_pred):
        # y_true, y_pred: shape (B, M*K, 1)
        B = ops.shape(y_true)[0]
        reshaped_y_true = ops.reshape(y_true, (B, self.M, self.K))
        reshaped_y_pred = ops.reshape(y_pred, (B, self.M, self.K))
        smile_mse = ops.mean(ops.square(reshaped_y_true - reshaped_y_pred), axis=-1)
        term_mse = ops.mean(ops.square(reshaped_y_true - reshaped_y_pred), axis=-2)
        return self.smile_weight * ops.mean(smile_mse) + self.term_weight * ops.mean(term_mse)



    def fit(self, X_train, y_train, **kwargs):
        return self.model.fit(X_train, y_train, **kwargs)

    def predict_surface(self, f_vec):
        if self.use_split_loss:
            f_repeated = np.repeat(f_vec[None, None, :], self.M * self.K, axis=1)  # shape (1, M*K, latent_dim)
            m = np.expand_dims(self.m_flat, axis=0)  # shape (1, M*K, 1)
            tau = np.expand_dims(self.tau_flat, axis=0)  # shape (1, M*K, 1)
            iv = self.model.predict([f_repeated, m, tau], verbose=0)
            return iv[0].reshape(self.M, self.K)
        else:
            f_repeated = np.repeat(f_vec[None, :], self.M * self.K, axis=0)
            iv = self.model.predict([f_repeated, self.m_flat, self.tau_flat], verbose=0)
            return iv.reshape(self.M, self.K)


    def refine_surface(self, surface, lambda_cal=0.0, lambda_smile=0.0, lambda_l2=0.0, lambda_extreme=0.0, use_sqrt_tau=False):
        M, K = self.M, self.K
        flat = surface.flatten()
        tau_vec = self.tau_flat[:M*K].reshape(M, K)[:, 0]
        delta_tau = np.diff(tau_vec)
        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            scaled = surf / np.sqrt(tau_vec).reshape(-1, 1) if use_sqrt_tau else surf
            sigma_sq = np.square(scaled)
            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0) / delta_tau[:, None], 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(scaled, n=2, axis=1), 0, None))
            l2_strike = np.mean(np.square(np.diff(scaled, n=2, axis=1)))
            l2_term = np.mean(np.square(np.diff(scaled, n=2, axis=0)))
            d_left = scaled[:, 1] - scaled[:, 0]
            d_right = scaled[:, -1] - scaled[:, -2]
            extreme_penalty = np.mean(np.square(d_left)) + np.mean(np.square(d_right))
            return (
                lambda_cal * cal_penalty +
                lambda_smile * smile_penalty +
                lambda_l2 * (l2_strike + l2_term) +
                lambda_extreme * extreme_penalty
            )
        result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
        return result.x.reshape(M, K)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))

    @classmethod
    def load(cls, path, latent_dim, M, K, m_flat, tau_flat):
        instance = cls(latent_dim, M, K, m_flat, tau_flat)
        instance.model = load_model(os.path.join(path, "model.keras"))
        return instance

    def get_config(self):
        return {
            "latent_dim": self.model.input_shape[0][-1],
            "M": self.M,
            "K": self.K
        }



@register_keras_serializable()

class ArbitrageDecoder2:
    """Arbitrage decoder v2 for volatility surface.
    Hidden layers: Dense with elu activation.
    Output: Dense(1) with relu activation.
    Options:
        lambda_smile, lambda_term, lambda_cal, lambda_extreme, lambda_l2: penalty weights
    """
    def __init__(self, latent_dim, M, K, m_flat, tau_flat,
                 lambda_smile=1.0, lambda_term=1.0,
                 lambda_cal=1.0, lambda_extreme=1.0, lambda_l2=1.0):
        self.M = M
        self.K = K
        self.m_flat = m_flat
        self.tau_flat = tau_flat
        self.lambda_smile = lambda_smile
        self.lambda_term = lambda_term
        self.lambda_cal = lambda_cal
        self.lambda_extreme = lambda_extreme
        self.lambda_l2 = lambda_l2
        self.model = self._build_model(latent_dim)
        self.model.compile(optimizer=Adam(1e-4), loss=self._split_loss_with_penalties)

    def _build_model(self, latent_dim):
        f_input = Input(shape=(latent_dim,))
        m_input = Input(shape=(1,))
        tau_input = Input(shape=(1,))
        x = Concatenate()([f_input, m_input, tau_input])
        x = Dense(128, activation="elu")(x)
        x = Dense(128, activation="elu")(x)
        x = Dense(64, activation="elu")(x)
        x = Dense(64, activation="elu")(x)
        iv = Dense(1, activation="relu")(x)
        return Model([f_input, m_input, tau_input], iv)

    def _calendar_penalty(self, y):
        sigma_sq = ops.square(y)
        return ops.mean(ops.relu(-ops.diff(sigma_sq, axis=1)))

    def _smile_penalty(self, y):
        return ops.mean(ops.relu(-ops.diff(y, n=2, axis=2)))

    def _extreme_strike_penalty(self, y):
        left = y[:, :, 1] - y[:, :, 0]
        right = y[:, :, -1] - y[:, :, -2]
        return ops.mean(ops.square(left)) + ops.mean(ops.square(right))

    def _split_loss_with_penalties(self, y_true, y_pred):
        y = ops.reshape(y_pred, (-1, self.M, self.K))
        y_t = ops.reshape(y_true, (-1, self.M, self.K))

        # Split fit loss
        strike_idx = self.K // 2
        tau_idx = self.M // 2
        mse_smile = ops.mean(ops.square(y[:, tau_idx, :] - y_t[:, tau_idx, :]))
        mse_term = ops.mean(ops.square(y[:, :, strike_idx] - y_t[:, :, strike_idx]))

        # Penalties
        cal = self._calendar_penalty(y)
        smile = self._smile_penalty(y)
        ext = self._extreme_strike_penalty(y)

        l2_strike = ops.mean(ops.square(ops.diff(y, n=2, axis=2)))
        l2_term = ops.mean(ops.square(ops.diff(y, n=2, axis=1)))

        return (self.lambda_smile * mse_smile +
                self.lambda_term * mse_term +
                self.lambda_cal * cal +
                self.lambda_extreme * ext +
                self.lambda_l2 * (l2_strike + l2_term))
    @classmethod
    def predict_surface(self, f_vec):
        repeated = np.repeat(f_vec[None, :], self.M * self.K, axis=0)
        iv = self.model.predict([repeated, self.m_flat, self.tau_flat], verbose=0)
        return iv.reshape(self.M, self.K)

    
    def get_config(self):
        return {
            "M": self.M,
            "K": self.K,
            "lambda_smile": self.lambda_smile,
            "lambda_term": self.lambda_term,
            "lambda_cal": self.lambda_cal,
            "lambda_extreme": self.lambda_extreme,
            "lambda_l2": self.lambda_l2
        }


@register_keras_serializable()
class NoArbDecoder(Model):
    """No-arbitrage decoder for volatility surface.
    Hidden layers: Dense with elu activation.
    Output: Dense(1) with linear activation.
    Options:
        lambda_cal, lambda_smile, lambda_extreme: penalty weights
        warmup_epochs: number of epochs before penalties are applied
    """
    def __init__(self, latent_dim, M, K, m_flat, tau_flat, 
                 lambda_cal=1.0, lambda_smile=1.0, lambda_extreme=1.0, warmup_epochs=0, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.K = K
        self.m_flat = m_flat
        self.tau_flat = tau_flat
        self.lambda_cal = lambda_cal
        self.lambda_smile = lambda_smile
        self.lambda_extreme = lambda_extreme
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self._build(latent_dim)

    def _build(self, latent_dim):
        self.feature_input = Input(shape=(latent_dim,))
        self.m_input = Input(shape=(1,))
        self.tau_input = Input(shape=(1,))
        x = Concatenate()([self.feature_input, self.m_input, self.tau_input])
        x = Dense(128, activation="elu")(x)
        x = Dense(128, activation="elu")(x)
        x = Dense(64, activation="elu")(x)
        x = Dense(64, activation="elu")(x)
        iv_out = Dense(1, activation="linear")(x)
        self._model = Model(inputs=[self.feature_input, self.m_input, self.tau_input], outputs=iv_out)

    def call(self, inputs):
        return self._model(inputs)

    def compile_with_penalties(self, learning_rate=1e-4):
        self.compile(
            optimizer=Adam(learning_rate),
            loss=self.loss_with_penalties
        )

    def loss_with_penalties(self, y_true, y_pred):
        mse = ops.mean(ops.square(y_true - y_pred))
        if self.warmup_epochs and self.current_epoch < self.warmup_epochs:
            return mse
        cal = self._calendar_penalty(y_pred)
        smile = self._smile_penalty(y_pred)
        extreme = self._extreme_strike_penalty(y_pred)
        return mse + self.lambda_cal * cal + self.lambda_smile * smile + self.lambda_extreme * extreme

    def _calendar_penalty(self, y_hat):
        y_hat = ops.reshape(y_hat, (-1, self.M, self.K))
        return ops.mean(ops.relu(-ops.diff(ops.square(y_hat), axis=1)))

    def _smile_penalty(self, y_hat):
        y_hat = ops.reshape(y_hat, (-1, self.M, self.K))
        return ops.mean(ops.relu(-ops.diff(y_hat, n=2, axis=2)))

    def _extreme_strike_penalty(self, y_hat):
        y_hat = ops.reshape(y_hat, (-1, self.M, self.K))
        d_left = y_hat[:, :, 1] - y_hat[:, :, 0]
        d_right = y_hat[:, :, -1] - y_hat[:, :, -2]
        return ops.mean(ops.square(d_left)) + ops.mean(ops.square(d_right))

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch


from keras import Model, ops
from keras.layers import Dense, Concatenate, Input
from keras.optimizers import Adam
from keras.initializers import Constant

@register_keras_serializable()
class PenalizedDecoder(Model):
    """Penalized decoder for volatility surface.
    Hidden layers: Dense with elu activation.
    Output: Dense(1) with softplus activation.
    Options:
        lambda_cal, lambda_smile: learnable penalty weights
    """
    def __init__(self, latent_dim, M, K, m_flat, tau_flat):
        super().__init__()
        self.latent_dim = latent_dim
        self.M, self.K = M, K
        self.m_flat = m_flat
        self.tau_flat = tau_flat
        self.loss_tracker_train = keras.metrics.mean(name="loss")
        self.loss_tracker_val = keras.metrics.mean(name="val_loss")
        self.lambda_cal = self.add_weight(name="lambda_cal", shape=(), initializer=Constant(0.1), trainable=True)
        self.lambda_smile = self.add_weight(name="lambda_smile", shape=(), initializer=Constant(0.1), trainable=True)
        self.concat = Concatenate()
        self.hidden = [
            Dense(128, activation="elu"),
            Dense(128, activation="elu"),
            Dense(64, activation="elu"),
            Dense(64, activation="elu")
        ]
        self.output_layer = Dense(1, activation="softplus")

    def call(self, inputs):
        z, m, tau = inputs  # shapes: (B, M*K, latent_dim), (B, M*K, 1), (B, M*K, 1)
        x = self.concat([z, m, tau])  # shape: (B, M*K, latent_dim + 2)
        for layer in self.hidden:
            x = layer(x)
        raw_iv = self.output_layer(x)  # shape: (B, M*K, 1)
        return raw_iv * ops.sqrt(tau)  # scaled output


    def compute_loss(self, inputs, y_true, y_pred):
        mse = ops.mean(ops.square(y_true - y_pred))
        B = ops.shape(y_pred)[0]
        flat_size = ops.shape(y_pred)[1]

        if flat_size != self.M * self.K:
            cal = ops.sum(y_pred) * 0.0
            smile = ops.sum(y_pred) * 0.0
            extreme = ops.sum(y_pred) * 0.0
        else:
            reshaped = ops.reshape(y_pred, (B, self.M, self.K))
            cal = self._calendar_penalty(reshaped)
            smile = self._smile_penalty(reshaped)



      
        # import torch
        # print("lambda_cal grad:", self.lambda_cal.value.grad)
        # print("lambda_smile grad:", self.lambda_smile.value.grad)


        
        # import torch
        # print("λ_cal grad:", self.lambda_cal.value.grad)
        # print("λ_smile grad:", self.lambda_smile.value.grad)
        # print("cal.requires_grad:", isinstance(cal, torch.Tensor) and cal.requires_grad)

        return mse + self.lambda_cal * cal + self.lambda_smile * smile


    def _calendar_penalty(self, y_hat):
        # y_hat is already (B, M, K)
        sigma_sq = ops.square(y_hat)
        diff = ops.diff(sigma_sq, axis=1)
        return ops.mean(ops.relu(-diff))

    def _smile_penalty(self, y_hat):
        # y_hat is already (B, M, K)
        diff = ops.diff(y_hat, n=2, axis=2)
        return ops.mean(ops.relu(-diff))
    
    def _extreme_strike_penalty(self, y_hat):
        # y_hat is already (B, M, K)
        d_left = y_hat[:, :, 1] - y_hat[:, :, 0]
        d_right = y_hat[:, :, -1] - y_hat[:, :, -2]
        return ops.mean(ops.square(d_left)) + ops.mean(ops.square(d_right))

    def train_step(self, data):
        (z, m, tau), y_true = data
        self.zero_grad()

        y_pred = self([z, m, tau], training=True)
        y_pred = ops.reshape(y_pred, (-1, self.M * self.K, 1))  # ensure (B, M*K, 1)

  

        loss = self.compute_loss([z, m, tau], y_true, y_pred)
        loss.backward()
        
        # import torch
        # for v in self.trainable_variables:
        #     if v.name in ["lambda_cal", "lambda_smile"]:
        #         print(f"{v.name} grad:", v.value.grad)


        grads = []
        for v in self.trainable_variables:
            grad = v.value.grad
            if grad is not None:
                grads.append(grad)
            else:
                grads.append(torch.zeros_like(v.value))

        import torch
        with torch.no_grad():
            self.optimizer.apply(grads, self.trainable_variables)

        self.loss_tracker_train.update_state(loss)
        return {"loss": self.loss_tracker_train.result()}



    def predict_surface(self, f_vec):
        z = np.repeat(f_vec[None, None, :], self.M * self.K, axis=1)     # (1, M*K, latent_dim)
        m = np.expand_dims(self.m_flat, axis=0)                          # (1, M*K, 1)
        tau = np.expand_dims(self.tau_flat, axis=0)                      # (1, M*K, 1)
        
        iv = self([z, m, tau], training=False)                           # (1, M*K, 1)
        return ops.convert_to_numpy(iv).reshape(self.M, self.K)

    
    def refine_surface(self, surface, 
                    lambda_data=0.0, 
                    lambda_cal=1.0, 
                    lambda_smile=1.0, 
                    use_sqrt_tau=True):
        import numpy as np
        import scipy.optimize

        M, K = self.M, self.K
        flat = surface.flatten()

        # Tau structure as used before
        tau_vec = ops.convert_to_numpy(self.tau_flat[:M*K].reshape(M, K))[:, 0]
        delta_tau = np.diff(tau_vec)

        def loss_fn(flat_iv):
            surf = flat_iv.reshape(M, K)
            scaled = surf / np.sqrt(tau_vec).reshape(-1, 1) if use_sqrt_tau else surf
            sigma_sq = np.square(scaled)

            cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0) / delta_tau[:, None], 0, None))
            smile_penalty = np.mean(np.clip(-np.diff(scaled, n=2, axis=1), 0, None))
            fidelity = np.mean(np.square(surf - surface))  # original decoder surface

            #print(f"[refine] loss={fidelity:.4e} cal={cal_penalty:.4e} smile={smile_penalty:.4e}")
            return (
                lambda_data * fidelity +
                lambda_cal * cal_penalty +
                lambda_smile * smile_penalty
            )

        result = scipy.optimize.minimize(
            loss_fn,
            flat,
            method="L-BFGS-B",
            options={"disp": True, "maxiter": 50}
        )

        print("Refinement status:", result.message)
        return result.x.reshape(M, K)
    




    @property
    def metrics(self):
        return [self.loss_tracker_train, self.loss_tracker_val]



    def test_step(self, data):
        (z, m, tau), y_true = data
        y_pred = self([z, m, tau], training=False)
        loss = self.compute_loss([z, m, tau], y_true, y_pred)
        self.loss_tracker_val.update_state(loss)
        return {"loss": self.loss_tracker_val.result()}



    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "M": self.M,
            "K": self.K,
            "lambda_cal": float(self.lambda_cal.numpy()),
            "lambda_smile": float(self.lambda_smile.numpy())
        }


    @classmethod
    def from_config(cls, config):
        return cls(**config)

from keras.callbacks import Callback
from keras import ops


class PenaltyDebugCallback(Callback):
    def __init__(self, model, X_val, y_val, M, K, name="debug"):
        super().__init__()
        self.model_to_debug = model
        self.X_val = X_val
        self.y_val = ops.convert_to_tensor(y_val, dtype="float32")
        self.name = name
        self.M = M
        self.K = K

    def on_epoch_end(self, epoch, logs=None):
        z, m, tau = self.X_val
        y_pred = self.model_to_debug([z, m, tau], training=False)
        mse = ops.mean(ops.square(self.y_val - y_pred))
        

        out = f"[{self.name} | Epoch {epoch + 1:03d}]  MSE: {ops.convert_to_numpy(mse):.6f}"

        if ops.shape(y_pred)[1] != self.M * self.K:
            out += "  | Shape mismatch: skipping penalties"
            print(out)
            return

        B = ops.shape(y_pred)[0]
        y_hat = ops.reshape(y_pred, (B, self.M, self.K))


    
        sigma_sq = ops.square(y_hat)
        cal = ops.mean(ops.relu(-ops.diff(sigma_sq, axis=1)))
        smile = ops.mean(ops.relu(-ops.diff(y_hat, n=2, axis=2)))
        left = y_hat[:, :, 1] - y_hat[:, :, 0]
        right = y_hat[:, :, -1] - y_hat[:, :, -2]
        extreme = ops.mean(ops.square(left)) + ops.mean(ops.square(right))
        l2_strike = ops.mean(ops.square(ops.diff(y_hat, n=2, axis=2)))
        l2_term = ops.mean(ops.square(ops.diff(y_hat, n=2, axis=1)))

        lambda_cal = self.model_to_debug.lambda_cal.value
        lambda_smile = self.model_to_debug.lambda_smile.value

        out += (
            f"  | Cal: {ops.convert_to_numpy(cal):.4f}"
            f"  Smile: {ops.convert_to_numpy(smile):.4f}"
            f"  Extr: {ops.convert_to_numpy(extreme):.4f}"
            f"  L2s: {ops.convert_to_numpy(l2_strike):.4f}"
            f"  L2t: {ops.convert_to_numpy(l2_term):.4f}"
            f"  lambda_cal: {ops.convert_to_numpy(lambda_cal):.4f}"
            f"  lambda_smile: {ops.convert_to_numpy(lambda_smile):.4f}"
        )
        print(out)






