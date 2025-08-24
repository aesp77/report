from keras import ops, Model
from keras.layers import Input, Dense, Lambda
from keras.regularizers import L2
from keras.saving import register_keras_serializable
from keras.callbacks import Callback
import keras
import numpy as np


# encoder standard architecture
def build_encoder(input_dim, latent_dim, l2_penalty=1e-5):
   """Create encoder network for VAE."""
   x = Input(shape=(input_dim,), name="input_vector")
   h = Dense(512, activation="elu", kernel_regularizer=L2(l2_penalty))(x)
   h = Dense(256, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(128, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(64, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   z_mean = Dense(latent_dim, name="z_mean")(h)
   z_log_var = Dense(latent_dim, name="z_log_var")(h)

   def sample_z(args):
       z_mu, z_logvar = args
       eps = keras.random.normal(ops.shape(z_mu))
       return z_mu + ops.exp(0.5 * z_logvar) * eps

   z = Lambda(sample_z)([z_mean, z_log_var])
   return Model(x, [z, z_mean, z_log_var], name="vae_encoder")


# decoder with fixed variance (diag-d: diagonal decoder with deterministic variance)
def build_decoder_diag_d(output_dim, latent_dim, l2_penalty=1e-5):
   """Create decoder with fixed unit variance (logvar=0)."""
   z = Input(shape=(latent_dim,), name="latent_input")
   h = Dense(64, activation="elu", kernel_regularizer=L2(l2_penalty))(z)
   h = Dense(128, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(256, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(512, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   mu = Dense(output_dim, activation="linear", name="recon_mu")(h)
   
   # fixed variance: logvar = log(1) = 0 for all outputs
   # this simplifies the loss to standard mse
   logvar = Lambda(
       lambda x: ops.ones_like(x) * np.log(1),
       name="fixed_logvar",
       output_shape=lambda input_shape: input_shape
   )(mu)

   return Model(z, [mu, logvar], name="vae_decoder_diag_d")


class VAE(Model):
   """VAE with fixed variance decoder."""
   def __init__(self, encoder, decoder, beta=1.0, kl_warmup=True, **kwargs):
       super().__init__(**kwargs)
       self.encoder = encoder
       self.decoder = decoder
       self.beta = beta
       self.kl_warmup = kl_warmup
       self.current_epoch = 0

       self.total_loss_tracker = keras.metrics.Mean(name="loss")
       self.kl_tracker = keras.metrics.Mean(name="kl_loss")
       self.recon_tracker = keras.metrics.Mean(name="reconstruction_loss")
       self.z_std_tracker = keras.metrics.Mean(name="z_std")
       self.z_logvar_mean_tracker = keras.metrics.Mean(name="z_logvar_mean")

   def call(self, x, training=False):
       """Forward pass returns mean prediction only."""
       z, _, _ = self.encoder(x, training=training)
       mu, _ = self.decoder(z, training=training)
       return mu

   def compute_loss_components(self, x, mu, logvar, z_mean, z_log_var):
       """Compute MLE reconstruction and KL losses."""
       # with fixed variance, this simplifies to mse
       sigma2 = ops.exp(ops.clip(logvar, -6.0, 2.0))
       recon_term = (ops.square(x - mu) / sigma2) + ops.clip(logvar, -6.0, 2.0)
       recon_loss = ops.mean(recon_term)

       # kl divergence
       safe_z_log_var = ops.clip(z_log_var, -6.0, 6.0)
       kl = 0.5 * ops.mean(
           ops.exp(safe_z_log_var) + ops.square(z_mean) - 1 - safe_z_log_var
       )

       # kl warmup schedule
       kl_weight = self.beta
       if self.kl_warmup:
           kl_weight = ops.minimum(self.current_epoch / 50.0, 1.0) * self.beta

       total = recon_loss + kl_weight * kl
       return total, recon_loss, kl

   def train_step(self, data):
       """Custom training step for VAE."""
       x, y = data
       self.zero_grad()
       z, z_mean, z_log_var = self.encoder(x)
       mu, logvar = self.decoder(z)
       loss, recon, kl = self.compute_loss_components(y, mu, logvar, z_mean, z_log_var)

       loss.backward()
       grads = [v.value.grad for v in self.trainable_weights]
       import torch
       with torch.no_grad():
           self.optimizer.apply(grads, self.trainable_weights)

       self.total_loss_tracker.update_state(loss)
       self.kl_tracker.update_state(kl)
       self.recon_tracker.update_state(recon)
       self.z_std_tracker.update_state(ops.std(z_mean))
       self.z_logvar_mean_tracker.update_state(ops.mean(z_log_var))

       return {
           "loss": self.total_loss_tracker.result(),
           "reconstruction_loss": self.recon_tracker.result(),
           "kl_loss": self.kl_tracker.result(),
           "z_std": self.z_std_tracker.result(),
           "z_logvar_mean": self.z_logvar_mean_tracker.result(),
       }

   def test_step(self, data):
       """Custom test step for VAE."""
       x, y = data
       z, z_mean, z_log_var = self.encoder(x)
       mu, logvar = self.decoder(z)
       loss, recon, kl = self.compute_loss_components(y, mu, logvar, z_mean, z_log_var)

       self.total_loss_tracker.update_state(loss)
       self.kl_tracker.update_state(kl)
       self.recon_tracker.update_state(recon)
       self.z_std_tracker.update_state(ops.std(z_mean))
       self.z_logvar_mean_tracker.update_state(ops.mean(z_log_var))

       return {
           "loss": self.total_loss_tracker.result(),
           "reconstruction_loss": self.recon_tracker.result(),
           "kl_loss": self.kl_tracker.result(),
           "z_std": self.z_std_tracker.result(),
           "val_z_std": self.z_std_tracker.result(),
           "z_logvar_mean": self.z_logvar_mean_tracker.result(),
           "val_z_logvar_mean": self.z_logvar_mean_tracker.result()
       }

   def fit_transform(self, X, **kwargs):
       """Fit VAE and return encoded mean."""
       self.fit(X, X, **kwargs)
       _, z_mean, _ = self.encoder(X)
       return ops.convert_to_numpy(z_mean)

   def inverse_transform(self, Z):
       """Decode latent codes to reconstructions."""
       mu, _ = self.decoder(Z)
       return ops.convert_to_numpy(mu)

   @property
   def metrics(self):
       """Return tracked metrics."""
       return [
           self.total_loss_tracker,
           self.recon_tracker,
           self.kl_tracker,
           self.z_std_tracker,
           self.z_logvar_mean_tracker,
       ]


@register_keras_serializable()
def sample_z(args):
   """Reparameterization trick for sampling."""
   z_mu, z_logvar = args
   eps = keras.random.normal(ops.shape(z_mu))
   return z_mu + ops.exp(0.5 * z_logvar) * eps


class KLEpochTracker(Callback):
   """Callback to track KL warmup epoch."""
   def __init__(self, vae_model):
       super().__init__()
       self.vae_model = vae_model

   def on_epoch_begin(self, epoch, logs=None):
       """Update current epoch for KL warmup."""
       self.vae_model.current_epoch = epoch


class VAETrainingDebugCallback(Callback):
   """Callback for VAE training debug output."""
   def __init__(self, vae_model, X_val, name="vae_mle_debug"):
       super().__init__()
       self.vae = vae_model
       self.X_val = ops.convert_to_tensor(X_val, dtype="float32")
       self.name = name

   def on_epoch_end(self, epoch, logs=None):
       """Print debug info at end of each epoch."""
       self.vae.current_epoch = epoch
       z, z_mean, z_log_var = self.vae.encoder(self.X_val, training=False)
       mu, _ = self.vae.decoder(z, training=False)

       z_std = ops.std(z_mean)
       z_logvar_mean = ops.mean(z_log_var)
       z_mean_abs = ops.mean(ops.abs(z_mean))
       recon_l1 = ops.mean(ops.abs(self.X_val - mu))

       print(
           f"[{self.name} | epoch {epoch + 1:03d}] "
           f"recon l1: {ops.convert_to_numpy(recon_l1):.6f} | "
           f"z_std: {ops.convert_to_numpy(z_std):.4f} | "
           f"z_mean_abs: {ops.convert_to_numpy(z_mean_abs):.4f} | "
           f"z_logvar_mean: {ops.convert_to_numpy(z_logvar_mean):.4f}"
       )