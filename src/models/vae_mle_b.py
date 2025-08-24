from keras import ops, Model
from keras.layers import Input, Dense, Lambda
from keras.regularizers import L2
from keras.saving import register_keras_serializable
from keras.callbacks import Callback
import keras


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


# decoder with learned variance
def build_decoder(output_dim, latent_dim, l2_penalty=1e-5):
   """Create decoder network with variance output."""
   z = Input(shape=(latent_dim,), name="latent_input")
   h = Dense(64, activation="elu", kernel_regularizer=L2(l2_penalty))(z)
   h = Dense(128, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(256, activation="elu", kernel_regularizer=L2(l2_penalty))(h)
   h = Dense(512, activation="elu", kernel_regularizer=L2(l2_penalty))(h)

   mu = Dense(output_dim, activation="linear", name="recon_mu")(h)
   logvar = ops.log(ops.softplus(Dense(output_dim)(h)) + 1e-6)
   return Model(z, [mu, logvar], name="vae_decoder")


@register_keras_serializable()
class VAE(Model):
   """VAE with batch-wise training support."""
   def __init__(self, encoder, decoder, beta=1.0, kl_warmup=True, **kwargs):
       super().__init__(**kwargs)
       self.encoder = encoder
       self.decoder = decoder
       self.beta = beta
       self.kl_warmup = kl_warmup
       self.current_epoch = 0

       self.total_loss_tracker = keras.metrics.Mean(name="loss")
       self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction_loss")
       self.kl_tracker = keras.metrics.Mean(name="kl_loss")
       self.z_std_tracker = keras.metrics.Mean(name="z_std")
       self.z_logvar_mean_tracker = keras.metrics.Mean(name="z_logvar_mean")

   def call(self, x):
       """Forward pass returns mean prediction only."""
       z, _, _ = self.encoder(x)
       mu, _ = self.decoder(z)
       return mu

   def compute_loss_components(self, x, mu, logvar, z_mean, z_log_var):
       """Compute MLE reconstruction and KL losses."""
       safe_logvar = ops.clip(logvar, -6.0, 2.0)
       sigma2 = ops.maximum(ops.exp(safe_logvar), 1e-6)
       recon_term = (ops.square(x - mu) / sigma2) + safe_logvar
       recon_loss = ops.mean(recon_term)

       safe_z_log_var = ops.clip(z_log_var, -6.0, 6.0)
       kl = 0.5 * ops.mean(
           ops.exp(safe_z_log_var) + ops.square(z_mean) - 1 - safe_z_log_var
       )

       kl_weight = self.beta
       if self.kl_warmup:
           kl_weight = ops.minimum(self.current_epoch / 60.0, 1.0) * self.beta

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
       self.reconstruction_tracker.update_state(recon)
       self.z_std_tracker.update_state(ops.std(z_mean))
       self.z_logvar_mean_tracker.update_state(ops.mean(z_log_var))

       return {
           "loss": self.total_loss_tracker.result(),
           "reconstruction_loss": self.reconstruction_tracker.result(),
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

       return {
           "loss": loss,
           "reconstruction_loss": recon,
           "kl_loss": kl,
           "z_std": ops.std(z_mean),
           "z_logvar_mean": ops.mean(z_log_var),
       }

   def train_on_batch(self, x):
       """Train on single batch."""
       return self.train_step((x, x))

   def fit_batches(self, dataset, val_dataset=None, epochs=100, verbose=True, callbacks=None):
       """Custom batch training loop.
       
       Note: this optimizes per batch, not per epoch. Each batch gets its own gradient
       update, which can lead to more frequent but noisier updates compared to 
       full-batch training. This is standard SGD behavior - the loss is computed
       and backpropagated for each batch independently.
       """
       history = {m.name: [] for m in self.metrics}

       for epoch in range(epochs):
           self.current_epoch = epoch

           # initialize callbacks on first epoch
           if epoch == 0 and callbacks:
               for cb in callbacks:
                   if not hasattr(cb, "model") or cb.model is None:
                       cb.set_model(self)
                   if hasattr(cb, "on_train_begin"):
                       cb.on_train_begin(logs={})

           if verbose:
               print(f"\nepoch {epoch + 1}/{epochs}")

           # train on each batch - gradient update happens per batch
           for i in range(len(dataset)):
               x, _ = dataset[i]
               self.train_on_batch(x)  # this does gradient update

           # training metrics
           current_metrics = {m.name: float(m.result()) for m in self.metrics}

           # validation metrics
           val_metrics = {}
           if val_dataset is not None:
               x_val, _ = val_dataset[0]
               val_logs = self.test_step((x_val, x_val))
               val_metrics = {
                   (k if k.startswith("val_") else f"val_{k}"): float(v)
                   for k, v in val_logs.items()
               }

           # print all metrics
           if verbose:
               logs = {k: round(v, 5) for k, v in {**current_metrics, **val_metrics}.items()}
               print(logs, flush=True)

           # save into history
           for m in self.metrics:
               history[m.name].append(float(m.result()))
           for k, v in val_metrics.items():
               history.setdefault(k, []).append(v)

           if callbacks:
               logs = {**current_metrics, **val_metrics}
               for cb in callbacks:
                   cb.on_epoch_end(epoch, logs=logs)

           # reset metrics for next epoch
           for m in self.metrics:
               m.reset_state()

       return history

   def inverse_transform(self, Z):
       """Decode latent codes to reconstructions."""
       mu, _ = self.decoder(Z)
       return ops.convert_to_numpy(mu)

   def fit_transform(self, X, **kwargs):
       """Fit VAE and return encoded mean."""
       self.fit(X, X, **kwargs)
       _, z_mean, _ = self.encoder(X)
       return ops.convert_to_numpy(z_mean)

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
       z, z_mean, z_log_var = self.vae.encoder(self.X_val, training=False)
       mu, _ = self.vae.decoder(z, training=False)

       z_std = ops.std(z_mean)
       z_logvar_mean = ops.mean(z_log_var)
       z_mean_abs = ops.mean(ops.abs(z_mean))
       recon_l1 = ops.mean(ops.abs(self.X_val - mu))

       print(
           f"[{self.name} | epoch {epoch + 1:03d}] "
           f"recon mle: {ops.convert_to_numpy(recon_l1):.6f} | "
           f"z_std: {ops.convert_to_numpy(z_std):.4f} | "
           f"z_mean_abs: {ops.convert_to_numpy(z_mean_abs):.4f} | "
           f"z_logvar_mean: {ops.convert_to_numpy(z_logvar_mean):.4f}",
           flush=True
       )