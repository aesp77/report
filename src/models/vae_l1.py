from keras import ops, Model
from keras.layers import Input, Dense, Lambda
from keras.regularizers import L2
from keras.saving import register_keras_serializable
from keras.callbacks import Callback
import keras

# encoder with l1 reconstruction loss focus
def build_encoder(input_dim, latent_dim, l2_penalty=1e-4):
   """Create encoder network for VAE."""
   x_in = Input(shape=(input_dim,), name="input_vector")
   h = Dense(256, activation="elu", kernel_regularizer=L2(l2_penalty))(x_in)
   h = Dense(128, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(64, activation="elu", kernel_regularizer=L2(l2_penalty))(h)

   z_mean = Dense(latent_dim, name="z_mean")(h)
   z_log_var = Dense(latent_dim, name="z_log_var")(h)

   def sample_z(args):
       z_mu, z_logvar = args
       eps = keras.random.normal(ops.shape(z_mu))
       return z_mu + ops.exp(0.5 * z_logvar) * eps

   z = Lambda(sample_z)([z_mean, z_log_var])
   return Model(inputs=x_in, outputs=[z, z_mean, z_log_var], name="vae_encoder")


# decoder mirrors encoder architecture
def build_decoder(output_dim, latent_dim, l2_penalty=1e-4):
   """Create decoder network for VAE."""
   z_in = Input(shape=(latent_dim,), name="latent_input")
   h = Dense(64, activation="elu", kernel_regularizer=L2(l2_penalty))(z_in)
   h = Dense(128, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(256, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   x_out = Dense(output_dim, activation="linear", name="reconstructed")(h)
   return Model(z_in, x_out, name="vae_decoder")


class VAE(Model):
   """Variational autoencoder with L1 reconstruction loss."""
   def __init__(self, encoder, decoder, beta=1.0, lambda_smile=0.0, lambda_term=0.0, kl_warmup=True, **kwargs):
       super().__init__(**kwargs)
       self.encoder = encoder
       self.decoder = decoder
       self.beta = beta
       self.lambda_smile = lambda_smile
       self.lambda_term = lambda_term
       self.kl_warmup = kl_warmup
       self.current_epoch = 0

       self.total_loss_tracker = keras.metrics.Mean(name="loss")
       self.kl_tracker = keras.metrics.Mean(name="kl_loss")
       self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction_loss")
       self.z_std_tracker = keras.metrics.Mean(name="z_mean_std")
       self.z_logvar_mean_tracker = keras.metrics.Mean(name="z_logvar_mean")

   def call(self, inputs):
       """Encode and decode input."""
       z, _, _ = self.encoder(inputs)
       return self.decoder(z)

   def compute_loss_components(self, x, x_hat, z_mean, z_log_var):
       """Compute reconstruction, KL, and optional penalties."""
       # l1 reconstruction loss - more robust to outliers than l2
       recon = ops.abs(x - x_hat)
       recon_loss = ops.mean(ops.where(ops.isnan(recon), ops.zeros_like(recon), recon))

       # kl divergence regularization
       eps = 1e-6
       safe_log_var = ops.clip(z_log_var, -10.0, 10.0)
       safe_exp = ops.exp(safe_log_var)
       kl_term = ops.square(z_mean) + safe_exp - 1 - safe_log_var
       kl_term = ops.where(ops.isnan(kl_term), ops.zeros_like(kl_term), kl_term)
       kl = 0.5 * ops.mean(ops.maximum(kl_term, eps))

       # no smile/term penalties for l1 variant
       smile_penalty = ops.zeros(())
       term_penalty = ops.zeros(())

       # kl warmup schedule
       kl_weight = self.beta
       if self.kl_warmup:
           kl_weight = ops.minimum(self.current_epoch / 50.0, 1.0) * self.beta

       total = recon_loss + kl_weight * kl + self.lambda_smile * smile_penalty + self.lambda_term * term_penalty
       return total, recon_loss, kl, smile_penalty, term_penalty

   def train_step(self, data):
       """Custom training step for VAE."""
       x, y = data
       assert keras.backend.backend() == "torch"
       self.zero_grad()

       z, z_mean, z_log_var = self.encoder(x)
       x_hat = self.decoder(z)
       loss, recon, kl, _, _ = self.compute_loss_components(y, x_hat, z_mean, z_log_var)

       loss.backward()
       gradients = [v.value.grad for v in self.trainable_weights]

       import torch
       with torch.no_grad():
           self.optimizer.apply(gradients, self.trainable_weights)

       self.total_loss_tracker.update_state(loss)
       self.kl_tracker.update_state(kl)
       self.reconstruction_tracker.update_state(recon)
       self.z_std_tracker.update_state(ops.std(z_mean))
       self.z_logvar_mean_tracker.update_state(ops.mean(z_log_var))

       return {
           "loss": self.total_loss_tracker.result(),
           "reconstruction_loss": self.reconstruction_tracker.result(),
           "kl_loss": self.kl_tracker.result(),
           "z_mean_std": self.z_std_tracker.result(),
           "z_logvar_mean": self.z_logvar_mean_tracker.result()
       }

   def test_step(self, data):
       """Custom test step for VAE."""
       x, y = data
       z, z_mean, z_log_var = self.encoder(x)
       x_hat = self.decoder(z)
       loss, recon, kl, _, _ = self.compute_loss_components(y, x_hat, z_mean, z_log_var)

       self.total_loss_tracker.update_state(loss)
       self.kl_tracker.update_state(kl)
       self.reconstruction_tracker.update_state(recon)
       self.z_std_tracker.update_state(ops.std(z_mean))
       self.z_logvar_mean_tracker.update_state(ops.mean(z_log_var))

       return {
           "loss": self.total_loss_tracker.result(),
           "reconstruction_loss": self.reconstruction_tracker.result(),
           "kl_loss": self.kl_tracker.result(),
           "z_mean_std": self.z_std_tracker.result(),
           "z_logvar_mean": self.z_logvar_mean_tracker.result()
       }

   def fit_transform(self, X, **kwargs):
       """Fit VAE and return encoded mean."""
       self.fit(X, X, **kwargs)
       _, z_mean, _ = self.encoder(X)
       return ops.convert_to_numpy(z_mean)

   def inverse_transform(self, Z):
       """Decode latent representation back to input space."""
       return ops.convert_to_numpy(self.decoder(Z))

   @property
   def metrics(self):
       """Return tracked metrics."""
       return [
           self.total_loss_tracker,
           self.reconstruction_tracker,
           self.kl_tracker,
           self.z_std_tracker,
           self.z_logvar_mean_tracker,
       ]

   def get_config(self):
       """Return model config."""
       return {
           "encoder": self.encoder,
           "decoder": self.decoder,
           "beta": self.beta,
           "lambda_smile": self.lambda_smile,
           "lambda_term": self.lambda_term,
           "kl_warmup": self.kl_warmup,
       }

   @classmethod
   def from_config(cls, config):
       """Create VAE from config."""
       encoder = config.pop("encoder")
       decoder = config.pop("decoder")
       return cls(encoder=encoder, decoder=decoder, **config)


@register_keras_serializable()
def sample_z(args):
   """Reparameterization trick for sampling."""
   z_mu, z_logvar = args
   eps = keras.random.normal(ops.shape(z_mu))
   return z_mu + ops.exp(0.5 * z_logvar) * eps


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
       mu = self.vae.decoder(z, training=False)

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


class KLEpochTracker(Callback):
   """Callback to track KL warmup epoch for VAE."""
   def __init__(self, vae_model):
       super().__init__()
       self.vae_model = vae_model

   def on_epoch_begin(self, epoch, logs=None):
       """Update current epoch for KL warmup."""
       self.vae_model.current_epoch = epoch