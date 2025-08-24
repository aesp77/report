import os
import numpy as np
import scipy
import keras
from keras.layers import TimeDistributed
from keras import ops
from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, Activation, LayerNormalization
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
from keras.metrics import Mean
from keras.callbacks import Callback


def build_decoder_dataset(Z_latent, Y_surface, strike_tensor, tau_tensor, M, K, F_features=None, splits=(0.7, 0.15, 0.15)):
   """Build train/val/test splits for decoder training."""
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
   """Basic decoder: z+features -> flattened surface.
   
   Simplest approach - direct mapping from latent+features to surface.
   Uses optional arbitrage penalties (calendar/smile) for surface consistency.
   Good baseline, fast training.
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

       # weighted reconstruction loss (optional atm emphasis)
       weight_mask = ops.ones_like(y_pred)
       base_loss_weighted = weight_mask * ops.square(y_pred - ops.reshape(y_true, (-1, self.M, self.K)))
       recon_loss = ops.mean(base_loss_weighted)

       # arbitrage penalties
       sigma_sq = ops.square(y_pred)
       cal_penalty = ops.mean(ops.square(ops.diff(-ops.diff(sigma_sq, axis=1))))
       smile_penalty = ops.mean(ops.square(ops.diff(-ops.diff(y_pred, n=2, axis=2))))

       # l2 regularization
       l2_penalty = ops.sum([ops.sum(ops.square(v.value)) for v in self.trainable_weights])

       # total loss
       loss = recon_loss
       if self.use_penalties:
           loss += self.lambda_cal * cal_penalty + self.lambda_smile * smile_penalty
       loss += 1e-6 * l2_penalty

       # backprop (torch-style)
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

       weight_mask = ops.ones_like(y_pred)
       base_loss_weighted = weight_mask * ops.square(y_pred - ops.reshape(y_true, (-1, self.M, self.K)))
       recon_loss = ops.mean(base_loss_weighted)

       sigma_sq = ops.square(y_pred)
       cal_penalty = ops.mean(ops.square(ops.diff(-ops.diff(sigma_sq, axis=1))))
       smile_penalty = ops.mean(ops.square(ops.diff(-ops.diff(y_pred, n=2, axis=2))))
       l2_penalty = ops.sum([ops.sum(ops.square(w.value)) for w in self.trainable_weights])

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
       """Post-process surface to enforce arbitrage constraints."""
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


@register_keras_serializable()
class SliceSurfaceDecoder(Model):
   """Maturity-aware decoder: generates surface slice by slice.
   
   Uses TimeDistributed layers to process each maturity independently.
   Includes tau (maturity) as explicit input for each slice.
   Better captures term structure dynamics.
   """
   def __init__(self, latent_dim, M, K, taus, feature_dim=0, **kwargs):
       super().__init__(**kwargs)
       self.latent_dim = latent_dim
       self.feature_dim = feature_dim
       self.input_dim = latent_dim + feature_dim
       self.M = M
       self.K = K
       self.taus = ops.convert_to_tensor(np.array(taus).reshape(M, 1), dtype="float32")

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
   """Pointwise decoder: predicts each (strike, maturity) point independently.
   
   Most flexible - each surface point gets its own prediction.
   Takes (z, m, tau) as input for each point.
   Slower training but can capture complex local patterns.
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
           # extreme strikes penalty
           d_left = surf[:, 1] - surf[:, 0]
           d_right = surf[:, -1] - surf[:, -2]
           extreme_penalty = np.mean(np.square(d_left)) + np.mean(np.square(d_right))
           return lambda_cal * cal_penalty + lambda_smile * smile_penalty + lambda_extreme * extreme_penalty

       result = scipy.optimize.minimize(loss_fn, flat, method="L-BFGS-B")
       return result.x.reshape(M, K)

   def build_training_data_from_surfaces(self, Z_latent, Y_surface_flat, strike_tensor, tau_tensor, F_features=None):
       """Convert surfaces to pointwise training data."""
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


@register_keras_serializable()
class PiecewiseSurfaceDecoderModular(keras.Model):
   """Enhanced pointwise decoder with modular features.
   
   Most configurable version of pointwise decoder:
   - Optional feature expansion (tau^2, log(tau), m^2)
   - LayerNorm, dropout for regularization
   - ATM weighting for smile penalty
   - Multiple activation choices
   Best for experimentation and hyperparameter tuning.
   """
   def __init__(self, latent_dim, M, K,
                feature_dim=0,
                activation="elu",
                tau_expand=False,      # add tau^2, log(tau)
                m_expand=False,         # add m^2
                use_layernorm=False,
                dropout_rate=0.0,
                atm_weighting=False,    # weight smile penalty by distance from atm
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

       # feature expansion
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
               # weight by distance from atm
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
class PenalizedDecoder(Model):
   """Decoder with learnable penalty weights.
   
   Key innovation: lambda_cal and lambda_smile are trainable parameters.
   Network learns optimal balance between reconstruction and arbitrage.
   Uses sqrt(tau) scaling for better numerical stability.
   Good when penalty weights are hard to tune manually.
   """
   def __init__(self, latent_dim, M, K, m_flat, tau_flat):
       super().__init__()
       self.latent_dim = latent_dim
       self.M, self.K = M, K
       self.m_flat = m_flat
       self.tau_flat = tau_flat
       self.loss_tracker_train = keras.metrics.Mean(name="loss")
       self.loss_tracker_val = keras.metrics.Mean(name="val_loss")

       # learnable penalty weights
       self.lambda_cal = self.add_weight(name="lambda_cal", shape=(), initializer=keras.initializers.Constant(0.1), trainable=True)
       self.lambda_smile = self.add_weight(name="lambda_smile", shape=(), initializer=keras.initializers.Constant(0.1), trainable=True)

       # network architecture
       self.concat = Concatenate()
       self.hidden = [
           Dense(128, activation="elu"),
           Dense(128, activation="elu"),
           Dense(64, activation="elu"),
           Dense(64, activation="elu")
       ]
       self.output_layer = Dense(1, activation="softplus")

   def call(self, inputs):
       z, m, tau = inputs
       x = self.concat([z, m, tau])
       for layer in self.hidden:
           x = layer(x)
       raw_iv = self.output_layer(x)
       return raw_iv * ops.sqrt(tau)  # scaled by sqrt(tau)

   def compute_loss(self, inputs, y_true, y_pred):
       mse = ops.mean(ops.square(y_true - y_pred))
       B = ops.shape(y_pred)[0]
       flat_size = ops.shape(y_pred)[1]

       if flat_size != self.M * self.K:
           cal = ops.sum(y_pred) * 0.0
           smile = ops.sum(y_pred) * 0.0
       else:
           reshaped = ops.reshape(y_pred, (B, self.M, self.K))
           cal = self._calendar_penalty(reshaped)
           smile = self._smile_penalty(reshaped)

       return mse + self.lambda_cal * cal + self.lambda_smile * smile

   def _calendar_penalty(self, y_hat):
       sigma_sq = ops.square(y_hat)
       diff = ops.diff(sigma_sq, axis=1)
       return ops.mean(ops.relu(-diff))

   def _smile_penalty(self, y_hat):
       diff = ops.diff(y_hat, n=2, axis=2)
       return ops.mean(ops.relu(-diff))

   def train_step(self, data):
       (z, m, tau), y_true = data
       self.zero_grad()

       y_pred = self([z, m, tau], training=True)
       y_pred = ops.reshape(y_pred, (-1, self.M * self.K, 1))

       loss = self.compute_loss([z, m, tau], y_true, y_pred)
       loss.backward()

       grads = []
       for v in self.trainable_variables:
           grad = v.value.grad
           if grad is not None:
               grads.append(grad)
           else:
               import torch
               grads.append(torch.zeros_like(v.value))

       import torch
       with torch.no_grad():
           self.optimizer.apply(grads, self.trainable_variables)

       self.loss_tracker_train.update_state(loss)
       return {"loss": self.loss_tracker_train.result()}

   def predict_surface(self, f_vec):
       z = np.repeat(f_vec[None, None, :], self.M * self.K, axis=1)
       m = np.expand_dims(self.m_flat, axis=0)
       tau = np.expand_dims(self.tau_flat, axis=0)
       
       iv = self([z, m, tau], training=False)
       return ops.convert_to_numpy(iv).reshape(self.M, self.K)

   def refine_surface(self, surface, lambda_data=0.0, lambda_cal=1.0, lambda_smile=1.0, use_sqrt_tau=True):
       """Refine with optional data fidelity term."""
       import scipy.optimize
       M, K = self.M, self.K
       flat = surface.flatten()

       tau_vec = ops.convert_to_numpy(self.tau_flat[:M*K].reshape(M, K))[:, 0]
       delta_tau = np.diff(tau_vec)

       def loss_fn(flat_iv):
           surf = flat_iv.reshape(M, K)
           scaled = surf / np.sqrt(tau_vec).reshape(-1, 1) if use_sqrt_tau else surf
           sigma_sq = np.square(scaled)

           cal_penalty = np.mean(np.clip(-np.diff(sigma_sq, axis=0) / delta_tau[:, None], 0, None))
           smile_penalty = np.mean(np.clip(-np.diff(scaled, n=2, axis=1), 0, None))
           fidelity = np.mean(np.square(surf - surface))  # keep close to original

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

       print("refinement status:", result.message)
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


class PenaltyDebugCallback(Callback):
   """Debug callback to monitor penalty values during training."""
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
       
       out = f"[{self.name} | epoch {epoch + 1:03d}]  mse: {ops.convert_to_numpy(mse):.6f}"

       if ops.shape(y_pred)[1] != self.M * self.K:
           out += "  | shape mismatch: skipping penalties"
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

       # get learnable weights if they exist
       if hasattr(self.model_to_debug, 'lambda_cal'):
           lambda_cal = self.model_to_debug.lambda_cal.value
           lambda_smile = self.model_to_debug.lambda_smile.value
           out += (
               f"  | cal: {ops.convert_to_numpy(cal):.4f}"
               f"  smile: {ops.convert_to_numpy(smile):.4f}"
               f"  extr: {ops.convert_to_numpy(extreme):.4f}"
               f"  l2s: {ops.convert_to_numpy(l2_strike):.4f}"
               f"  l2t: {ops.convert_to_numpy(l2_term):.4f}"
               f"  λ_cal: {ops.convert_to_numpy(lambda_cal):.4f}"
               f"  λ_smile: {ops.convert_to_numpy(lambda_smile):.4f}"
           )
       else:
           out += (
               f"  | cal: {ops.convert_to_numpy(cal):.4f}"
               f"  smile: {ops.convert_to_numpy(smile):.4f}"
               f"  extr: {ops.convert_to_numpy(extreme):.4f}"
               f"  l2s: {ops.convert_to_numpy(l2_strike):.4f}"
               f"  l2t: {ops.convert_to_numpy(l2_term):.4f}"
           )
       
       print(out)