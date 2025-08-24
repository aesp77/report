import keras
from keras import Model, ops, Sequential
from keras.layers import Conv1D, Input, Dense, Flatten, Lambda, Conv2D
from keras.metrics import Mean
from keras.saving import register_keras_serializable
import torch
import numpy as np


@keras.saving.register_keras_serializable()
class SliceRLDecoder(Model):
   """Residual learning decoder for individual smile slices.
   
   Processes each maturity slice (smile) independently.
   Learns corrections to base predictions rather than full surface.
   
   Key features:
   - Works on (K, 1) slices (single maturity)
   - CNN or MLP architecture options
   - Residual mode: adds corrections to base predictions
   - Convexity penalty for smile consistency
   - Delta penalty to keep corrections small
   
   Best for: Refining individual smile shapes when base decoder
   captures general structure but misses local details.
   """
   def __init__(self, K, use_cnn=True, mode="residual", lambda_convexity=0.0, alpha=1e-2, **kwargs):
       super().__init__(**kwargs)
       self.K = K
       self.use_cnn = use_cnn
       self.mode = mode  # "residual" or "direct"
       self.lambda_convexity = lambda_convexity  # smile convexity penalty
       self.alpha = alpha  # delta penalty weight

       if use_cnn:
           # 1d cnn for smile refinement
           self.net = Sequential([
               Input(shape=(K, 1)),  # [iv_pred, tau]
               Conv1D(128, 3, padding="same", activation="elu"),
               Conv1D(64, 3, padding="same", activation="elu"),
               Conv1D(32, 3, padding="same", activation="elu"),
               Conv1D(1, 1, activation="linear")
           ])
       else:
           # mlp alternative
           self.net = Sequential([
               Input(shape=(K, 1)),
               Flatten(),
               Dense(128, activation="relu"),
               Dense(64, activation="relu"),
               Dense(32, activation="relu"),
               Dense(K, activation="linear"),
               Lambda(lambda x: ops.reshape(x, (-1, K, 1)))
           ])

       self.loss_tracker = Mean(name="loss")

   def call(self, x):
       return self.net(x)

   def compute_loss(self, x, y_true):
       iv_pred = x[..., 0:1]  # base prediction
       delta = self(x)  # learned correction

       # apply correction
       iv_out = iv_pred + delta if self.mode == "residual" else delta
       fit_loss = ops.mean(ops.square(iv_out - y_true))
       delta_penalty = ops.mean(ops.square(delta))  # keep corrections small
       total_loss = fit_loss + self.alpha * delta_penalty

       # optional convexity penalty for smile shape
       if self.lambda_convexity > 0.0:
           iv = iv_out[..., 0]
           second_diff = iv[:, :-2] - 2 * iv[:, 1:-1] + iv[:, 2:]
           penalty = ops.mean(ops.square(ops.clip(-second_diff, 0.0, float("inf"))))
           total_loss += self.lambda_convexity * penalty

       return total_loss

   def train_step(self, data):
       x, y = data
       loss = self.compute_loss(x, y)
       loss.backward()
       grads = [v.value.grad for v in self.trainable_weights]
       with torch.no_grad():
           self.optimizer.apply(grads, self.trainable_weights)
       self.loss_tracker.update_state(loss)
       return {"loss": self.loss_tracker.result()}

   def test_step(self, data):
       x, y = data
       loss = self.compute_loss(x, y)
       self.loss_tracker.update_state(loss)
       return {"loss": float(ops.convert_to_numpy(loss))}

   @property
   def metrics(self):
       return [self.loss_tracker]

   def get_config(self):
       return {
           "K": self.K,
           "use_cnn": self.use_cnn,
           "mode": self.mode,
           "lambda_convexity": self.lambda_convexity,
           "alpha": self.alpha,
       }

   @classmethod
   def from_config(cls, config):
       return cls(**config)


def build_slice_dataset(Y_surface, selected_taus=None):
   """Convert (N, M, K) surfaces into (N*M, K, 1) slice-wise pairs.
   
   Prepares data for SliceRLDecoder by extracting individual smiles.
   """
   N, M, K = Y_surface.shape
   X, Y = [], []

   for surface in Y_surface:
       for m in range(M):
           if selected_taus is not None and m not in selected_taus:
               continue
           smile = surface[m]
           X.append(smile.reshape(K, 1))
           Y.append(smile.reshape(K, 1))

   X = np.stack(X)
   Y = np.stack(Y)

   # validate shape
   expected_slices = N * len(selected_taus) if selected_taus else N * M
   assert X.shape[0] == expected_slices, f"expected {expected_slices} slices, got {X.shape[0]}"

   return X, Y


@register_keras_serializable()
class PointwiseRLDecoder(Model):
   """Pointwise residual learning decoder.
   
   Predicts corrections for individual (strike, maturity) points.
   Most granular approach - each point gets its own correction.
   
   Key features:
   - Works on individual points (not slices or surfaces)
   - Can predict deltas (corrections) or direct values
   - Optional normalization for stable training
   - Optional Black-Scholes loss for option pricing consistency
   - Clipping to prevent extreme corrections
   
   Best for: Fine-grained local adjustments when base predictions
   have point-specific errors (e.g., certain strikes consistently off).
   """
   def __init__(self, use_delta=False, normalize_delta=False, correction_scale=1, 
                use_bs_loss=False, taus=None, rel_strikes=None, M=None, K=None, **kwargs):
       super().__init__(**kwargs)
       self.use_delta = use_delta  # predict corrections vs direct values
       self.normalize_delta = normalize_delta  # normalize residuals
       self.correction_scale = correction_scale  # scale corrections
       self.use_bs_loss = use_bs_loss  # use black-scholes consistency loss
       
       # lightweight network for point predictions
       self.net = Sequential([
           Dense(32, activation="gelu", kernel_regularizer="l2"),
           Dense(16, activation="gelu", kernel_regularizer="l2"),
           Dense(1, activation="linear")
       ])
       self.loss_tracker = Mean(name="loss")
       
       # create black-scholes loss if requested
       if self.use_bs_loss and all(x is not None for x in [taus, rel_strikes, M, K]):
           from models.decoder_losses import create_black_scholes_hybrid_loss
           self.bs_loss_fn, _, _ = create_black_scholes_hybrid_loss(
               taus=taus, rel_strikes=rel_strikes, M=M, K=K
           )
       else:
           self.bs_loss_fn = None

   def call(self, inputs):
       x_zf, x_m, x_tau = inputs  # features, strike, maturity
       x_combined = ops.concatenate([x_zf, x_m, x_tau], axis=1)
       output = self.net(x_combined)
       scaled_output = self.correction_scale * output
       
       if self.use_delta:
           return ops.clip(scaled_output, -0.02, 0.02)  # clip corrections only
       else:
           return scaled_output  # no clip for direct predictions

   def compute_loss(self, inputs, y_true):
       y_pred = self(inputs)
       
       # use black-scholes loss if configured
       if self.use_bs_loss and self.bs_loss_fn is not None:
           return self.bs_loss_fn(y_true, y_pred)
       
       # otherwise use mse-based loss
       if self.use_delta:
           if self.normalize_delta:
               # normalize targets for stable training
               mu = ops.mean(y_true)
               std = ops.std(y_true) + 1e-6
               y_norm = (y_true - mu) / std
               loss = ops.mean(ops.square(y_pred - y_norm))
           else:
               loss = ops.mean(ops.square(y_pred - y_true))
       else:
           # direct prediction
           loss = ops.mean(ops.square(y_pred - y_true))

       return loss

   def train_step(self, data):
       x, y = data
       loss = self.compute_loss(x, y)
       loss.backward()
       grads = [v.value.grad for v in self.trainable_weights]
       with torch.no_grad():
           self.optimizer.apply(grads, self.trainable_weights)
       self.loss_tracker.update_state(loss)
       return {"loss": self.loss_tracker.result()}

   def test_step(self, data):
       x, y = data
       loss = self.compute_loss(x, y)
       self.loss_tracker.update_state(loss)
       return {"loss": float(ops.convert_to_numpy(loss))}
   
   def predict_surface(self, z_vec, f_vec, m_flat, tau_flat):
       """Predict corrections for a full surface."""
       # prepare input features
       if f_vec is not None:
           zf_vec = ops.concatenate([z_vec, f_vec], axis=0)
       else:
           zf_vec = z_vec
       
       # repeat for each point on the surface
       n_points = len(m_flat)
       zf_repeat = ops.repeat(zf_vec[None, :], n_points, axis=0)
       
       # prepare inputs in expected format
       inputs = [zf_repeat, m_flat, tau_flat]
       
       # get predictions
       predictions = self(inputs, training=False)
       
       return ops.convert_to_numpy(predictions).flatten()

   @property
   def metrics(self):
       return [self.loss_tracker]

   def get_config(self):
       config = super().get_config()
       config.update({
           "use_delta": self.use_delta,
           "normalize_delta": self.normalize_delta,
           "correction_scale": self.correction_scale,
           "use_bs_loss": self.use_bs_loss
       })
       return config

   @classmethod
   def from_config(cls, config):
       return cls(**config)


@register_keras_serializable()
class SurfaceRLDecoder(Model):
   """Full surface residual learning decoder.
   
   Processes entire (M, K) surface at once using 2D convolutions.
   Learns spatially coherent corrections across strikes and maturities.
   
   Key features:
   - Works on full (M, K, 1) surfaces
   - 2D CNN architecture for spatial patterns
   - Strike weighting (focus on ATM region)
   - Delta and negativity penalties
   - Residual or full prediction modes
   
   Best for: Global surface corrections when errors have spatial
   structure (e.g., systematic biases in certain regions).
   Maintains consistency across neighboring points.
   """
   def __init__(
       self,
       M,
       K,
       alpha=0.01,  # delta penalty weight
       beta=10.0,   # negativity penalty weight
       mode="residual",
       use_strike_weight=True,
       use_cnn=True,
       **kwargs
   ):
       super().__init__(**kwargs)
       self.M = M
       self.K = K
       self.alpha = alpha
       self.beta = beta
       self.mode = mode  # "residual" or "full"
       self.use_strike_weight = use_strike_weight  # weight atm region more
       self.use_cnn = use_cnn

       if self.use_cnn:
           # 2d cnn for surface processing
           self.net = Sequential([
               Input(shape=(M, K, 1)),
               Conv2D(64, 3, padding="same", activation="relu"),
               Conv2D(128, 3, padding="same", activation="relu"),
               Conv2D(64, 3, padding="same", activation="relu"),
               Conv2D(16, 3, padding="same", activation="relu"),
               Conv2D(1, 1, activation="linear"),  # output corrections
           ])
       else:
           # mlp alternative (flattened)
           self.net = Sequential([
               Input(shape=(M * K,)),
               Dense(512, activation="relu"),
               Dense(512, activation="relu"),
               Dense(256, activation="relu"),
               Dense(M * K, activation="linear"),
               Lambda(lambda x: ops.reshape(x, (-1, M, K, 1)))
           ])

       self.loss_tracker = Mean(name="loss")

   def call(self, x):
       if not self.use_cnn:
           x = ops.reshape(x, (-1, self.M * self.K))
       return self.net(x)

   def compute_loss(self, x, y):
       pred = self(x)  # predicted corrections
       iv_pred = x if self.use_cnn else ops.reshape(x, (-1, self.M, self.K, 1))

       if self.mode == "residual":
           # add corrections to base predictions
           iv_corrected = iv_pred + pred
           y_target = iv_pred + y
       else:  # full
           iv_corrected = pred
           y_target = y

       loss = ops.square(iv_corrected - y_target)

       if self.use_strike_weight:
           # gaussian weight centered at atm
           weight = ops.exp(-ops.square(ops.arange(self.K) - self.K // 2) / (2 * 1.0**2))
           weight = ops.reshape(weight, (1, 1, self.K, 1))
           loss = loss * weight

       loss = ops.mean(loss)
       delta_penalty = ops.mean(ops.square(pred))  # keep corrections small
       neg_penalty = ops.mean(ops.square(ops.relu(-iv_corrected)))  # penalize negative ivs
       total = loss + self.alpha * delta_penalty + self.beta * neg_penalty

       return total, loss

   def train_step(self, data):
       x, y = data
       loss, core_loss = self.compute_loss(x, y)
       loss.backward()
       grads = [v.value.grad for v in self.trainable_weights]
       with torch.no_grad():
           self.optimizer.apply(grads, self.trainable_weights)
       self.loss_tracker.update_state(core_loss)
       return {"loss": self.loss_tracker.result()}

   def test_step(self, data):
       x, y = data
       _, core_loss = self.compute_loss(x, y)
       self.loss_tracker.update_state(core_loss)
       return {"loss": float(ops.convert_to_numpy(core_loss))}

   @property
   def metrics(self):
       return [self.loss_tracker]

   def get_config(self):
       return {
           "M": self.M, "K": self.K,
           "alpha": self.alpha, "beta": self.beta,
           "mode": self.mode,
           "use_strike_weight": self.use_strike_weight,
           "use_cnn": self.use_cnn
       }

   @classmethod
   def from_config(cls, config):
       return cls(**config)