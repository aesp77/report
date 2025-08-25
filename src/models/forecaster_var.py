from statsmodels.tsa.api import VAR
import numpy as np
from models.encoder_pca import PCAEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler


class VARForecaster:
   def __init__(self, maxlags=5):
       self.maxlags = maxlags
       self.model = None
       self.results = None

   def fit(self, Z_train: np.ndarray):
       self.model = VAR(Z_train)
       self.results = self.model.fit(maxlags=self.maxlags)

   def forecast(self, Z_hist: np.ndarray, steps: int = 1) -> np.ndarray:
       return self.results.forecast(Z_hist[-self.results.k_ar:], steps=steps)

   def summary(self):
       return self.results.summary()


def run_pca_var_forecast(
   X_train, X_test, latent_dim=5, use_delta=False, maxlags=5, 
   standardize=False, demean_surface=False, mean_mode="global"
):
   """runs pca + var forecasting on surface data"""
   
   X_mean = None
   X_train_means = X_train.mean(axis=1, keepdims=True)
   X_test_means = X_test.mean(axis=1, keepdims=True)

   if demean_surface:
       X_train = X_train - X_train_means
       X_test = X_test - X_test_means

   # fit pca
   pca = PCAEncoder(n_components=latent_dim)
   Z_train = pca.fit_transform(X_train)
   Z_test = pca.fit_transform(X_test)

   # optional standardize
   if standardize:
       scaler = StandardScaler()
       Z_train = scaler.fit_transform(Z_train)
       Z_test = scaler.transform(Z_test)

   var_model = VARForecaster(maxlags=maxlags)

   if use_delta:
       Z_delta_train = Z_train[1:] - Z_train[:-1]
       var_model.fit(Z_delta_train)

       steps = Z_test.shape[0]
       Z_delta_forecast = var_model.forecast(Z_delta_train, steps=steps)

       Z_forecast = [Z_train[-1]]
       for t in range(steps):
           Z_forecast.append(Z_forecast[-1] + Z_delta_forecast[t])
       Z_forecast = np.stack(Z_forecast[1:])
   else:
       var_model.fit(Z_train)
       Z_forecast = var_model.forecast(Z_train, steps=Z_test.shape[0])

   # decode
   X_recon = pca.inverse_transform(Z_forecast)
   X_true = X_test[:len(X_recon)]

   # add mean back
   if demean_surface:
       if mean_mode == "true":
           X_mean = X_test_means[:len(X_recon)]
       elif mean_mode == "last":
           X_mean = np.full((len(X_recon), 1), X_train_means[-1])
       elif mean_mode == "global":
           X_mean = np.full((len(X_recon), 1), X_train_means.mean())
       elif mean_mode == "predicted":
           y = X_train_means.squeeze()
           phi, alpha = np.polyfit(y[:-1], y[1:], 1)
           forecasted_means = [y[-1]]
           for _ in range(len(X_recon)):
               next_mean = alpha + phi * forecasted_means[-1]
               forecasted_means.append(next_mean)
           X_mean = np.array(forecasted_means[1:]).reshape(-1, 1)
       else:
           raise ValueError(f"invalid mean_mode: {mean_mode}")

       print(f"using mean_mode='{mean_mode}' for demeaning: {X_mean[0, 0]:.4f}")
       X_recon += X_mean

   # rmse
   rmse_latent = root_mean_squared_error(Z_test[:len(Z_forecast)], Z_forecast)
   rmse_surface = root_mean_squared_error(X_true, X_recon)

   print(f"rmse_z: {rmse_latent:.4f} | rmse_σ: {rmse_surface:.4f}")

   return Z_forecast, Z_test, Z_train, X_recon, X_true, (rmse_latent, rmse_surface), pca, var_model, X_mean


def run_pca_var_forecast_with_features(
   X_train, X_test, F_train, F_test, latent_dim=5, use_delta=False, maxlags=5,
   standardize=False, demean_surface=False, mean_mode="global"
):
   """forecasts surfaces using pca on surfaces + auxiliary features"""
   
   X_mean = None
   X_train_means = X_train.mean(axis=1, keepdims=True)
   X_test_means = X_test.mean(axis=1, keepdims=True)

   if demean_surface:
       X_train = X_train - X_train_means
       X_test = X_test - X_test_means

   # pca only on surface
   pca = PCAEncoder(n_components=latent_dim)
   Z_train = pca.fit_transform(X_train)
   Z_test = pca.fit_transform(X_test)

   # concatenate features
   Z_aug_train = Z_train if F_train is None else np.concatenate([Z_train, F_train], axis=1)
   Z_aug_test = Z_test if F_test is None else np.concatenate([Z_test, F_test], axis=1)

   # standardize z space (optional)
   if standardize:
       scaler = StandardScaler()
       Z_aug_train = scaler.fit_transform(Z_aug_train)
       Z_aug_test = scaler.transform(Z_aug_test)

   # fit var
   var_model = VARForecaster(maxlags=maxlags)
   if use_delta:
       Z_delta_train = Z_aug_train[1:] - Z_aug_train[:-1]
       var_model.fit(Z_delta_train)

       steps = Z_aug_test.shape[0]
       Z_delta_forecast = var_model.forecast(Z_delta_train, steps=steps)

       Z_forecast = [Z_aug_train[-1]]
       for t in range(steps):
           Z_forecast.append(Z_forecast[-1] + Z_delta_forecast[t])
       Z_forecast = np.stack(Z_forecast[1:])
   else:
       var_model.fit(Z_aug_train)
       Z_forecast = var_model.forecast(Z_aug_train, steps=Z_aug_test.shape[0])

   # decode only the pca part
   Z_forecast_trimmed = Z_forecast[:, :latent_dim]
   X_recon = pca.inverse_transform(Z_forecast_trimmed)
   X_true = X_test[:len(X_recon)]

   # add surface mean back
   if demean_surface:
       if mean_mode == "true":
           X_mean = X_test_means[:len(X_recon)]
       elif mean_mode == "last":
           X_mean = np.full((len(X_recon), 1), X_train_means[-1])
       elif mean_mode == "global":
           X_mean = np.full((len(X_recon), 1), X_train_means.mean())
       elif mean_mode == "predicted":
           y = X_train_means.squeeze()
           phi, alpha = np.polyfit(y[:-1], y[1:], 1)
           forecasted_means = [y[-1]]
           for _ in range(len(X_recon)):
               forecasted_means.append(alpha + phi * forecasted_means[-1])
           X_mean = np.array(forecasted_means[1:]).reshape(-1, 1)
       else:
           raise ValueError(f"invalid mean_mode: {mean_mode}")
       print(f"using mean_mode='{mean_mode}' for demeaning: {X_mean[0, 0]:.4f}")
       X_recon += X_mean

   # evaluate
   rmse_latent = root_mean_squared_error(Z_test[:len(Z_forecast_trimmed)], Z_forecast_trimmed)
   rmse_surface = root_mean_squared_error(X_true, X_recon)
   print(f"rmse_z: {rmse_latent:.4f} | rmse_σ: {rmse_surface:.4f}")

   return Z_forecast_trimmed, Z_test, Z_train, X_recon, X_true, (rmse_latent, rmse_surface), pca, var_model, X_mean