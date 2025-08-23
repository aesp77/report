import keras
from keras import ops
import numpy as np

class VolSurfaceDataset(keras.utils.Sequence):
    # dataset for volatility surface, spot, curve, and date tensors
    def __init__(self, surface_tensor, spot_tensor, curve_tensor, date_tensor,
                 lookback=1, batch_size=None, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        self.surface = surface_tensor  # (t, m, k, c)
        self.spot = spot_tensor       # (t,)
        self.curve = curve_tensor     # (t, ...)
        self.dates = date_tensor      # (t,)
        self.lookback = lookback
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.T = len(date_tensor)
        self.indices = np.arange(self.lookback, self.T)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # number of batches
        if self.batch_size is None:
            return 1
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        # get batch idxs
        if self.batch_size is None:
            idxs = self.indices
        else:
            idxs = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, dates_out = [], []
        for t in idxs:
            surface_seq, log_ret_seq, curve_seq = [], [], []
            for lag in range(t - self.lookback, t):
                surface_iv = ops.convert_to_numpy(self.surface[lag][..., -1])  # (m, k)
                flat_surface = surface_iv.flatten()  # (m*k,)
                surface_seq.append(flat_surface)
                if lag > 0:
                    s_now = ops.convert_to_numpy(self.spot[lag])
                    s_prev = ops.convert_to_numpy(self.spot[lag - 1])
                    log_ret = np.log(s_now / s_prev)
                else:
                    log_ret = 0.0
                log_ret_seq.append(np.array([log_ret], dtype=np.float32))
                curve = ops.convert_to_numpy(self.curve[lag])
                curve_seq.append(curve)
            flat_input = np.concatenate(surface_seq + log_ret_seq + curve_seq)
            X.append(flat_input)
            dates_out.append(self.dates[t])
        x_tensor = ops.convert_to_tensor(np.array(X, dtype=np.float32))
        if getattr(self, "return_dates", False):
            return (x_tensor, x_tensor), dates_out
        else:
            return (x_tensor, x_tensor)

    def on_epoch_end(self):
        # shuffle indices at epoch end
        if self.shuffle:
            np.random.shuffle(self.indices)
            
import keras
from keras import ops
import numpy as np
from sklearn.preprocessing import StandardScaler


def _batch_generator(X_scaled, batch_size):
    # yield batches of (x, x)
    for start in range(0, len(X_scaled), batch_size):
        end = start + batch_size
        x = X_scaled[start:end]
        yield (x, x)
        
def prepare_vae_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, normalize=False):
    train_dataset.return_dates = False
    val_dataset.return_dates = False
    test_dataset.return_dates = False

    # --- Concatenate all batches
    def extract_all(dataset):
        # concatenate all batches
        all_x = []
        for i in range(len(dataset)):
            x_batch, _ = dataset[i]
            all_x.append(ops.convert_to_numpy(x_batch))
        return np.concatenate(all_x, axis=0)

    X_train = extract_all(train_dataset)
    X_val   = extract_all(val_dataset)
    X_test  = extract_all(test_dataset)

    # normalize to [0, 1] based on train set
    if normalize:
        min_val = X_train.min(axis=0)
        max_val = X_train.max(axis=0)
        range_val = np.clip(max_val - min_val, 1e-6, None)
        X_train = (X_train - min_val) / range_val
        X_val   = (X_val - min_val) / range_val
        X_test  = (X_test - min_val) / range_val

    # --- Batch generator
    def _batch_generator(X_scaled):
        # yield batches of (x, x)
        for start in range(0, len(X_scaled), batch_size):
            end = start + batch_size
            x = X_scaled[start:end]
            yield (x, x)

    return (
        _batch_generator(X_train),
        _batch_generator(X_val),
        _batch_generator(X_test),
        None  # no scaler
    )

class VolSurfaceDatasetTau(keras.utils.Sequence):
    # dataset for volatility surface, tau, spot, curve, and date tensors
    def __init__(self, surface_tensor, tau_tensor, spot_tensor, curve_tensor, date_tensor,
                 lookback=1, batch_size=None, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        self.surface = surface_tensor  # (t, m, k, c)
        self.tau = tau_tensor         # (t, m, k)
        self.spot = spot_tensor       # (t,)
        self.curve = curve_tensor     # (t, ...)
        self.dates = date_tensor      # (t,)
        self.lookback = lookback
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(lookback, len(date_tensor))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # number of batches
        return 1 if self.batch_size is None else int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        # get batch idxs
        if self.batch_size is None:
            idxs = self.indices
        else:
            idxs = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, dates_out = [], []
        for t in idxs:
            surface_seq, tau_seq, log_ret_seq, curve_seq = [], [], [], []
            for lag in range(t - self.lookback, t):
                surface_iv = ops.convert_to_numpy(self.surface[lag][..., -1]).flatten()  # (m*k,)
                tau_vals = ops.convert_to_numpy(self.tau[lag]).flatten()                 # (m*k,)
                surface_seq.append(surface_iv)
                tau_seq.append(tau_vals)
                if lag > 0:
                    log_ret = np.log(float(self.spot[lag]) / float(self.spot[lag - 1]))
                else:
                    log_ret = 0.0
                log_ret_seq.append([log_ret])
                curve_seq.append(ops.convert_to_numpy(self.curve[lag]))
            flat_input = np.concatenate(surface_seq + tau_seq + log_ret_seq + curve_seq)
            X.append(flat_input)
            dates_out.append(self.dates[t])
        x_tensor = ops.convert_to_tensor(np.array(X, dtype=np.float32))
        return (x_tensor, x_tensor)
    
    

import numpy as np
from keras import ops

def prepare_decoder_datasets(Z_scaled, surface_tensor, index_dict, m_flat, tau_flat, M, K):
    """
    Prepares (X, y) tuples for decoder training from latent Z and surface tensors.
    Assumes Z_scaled aligns with index_dict["train"] (e.g. after lookback).
    """
    output = {}
    for key in index_dict:
        idx = index_dict[key]
        X_z, y_surface = [], []

        for j, i in enumerate(idx):
            if key == "train":
                z = Z_scaled[j]
            else:
                z = np.zeros_like(Z_scaled[0])  # dummy Z for val/test

            iv = ops.convert_to_numpy(surface_tensor[i])[..., -1]
            X_z.append(z)
            y_surface.append(iv.reshape(M * K))

        X_z = np.array(X_z)[:, None, :]  # (N, 1, latent_dim)
        X_z = np.repeat(X_z, M * K, axis=1)
        m_input = np.broadcast_to(m_flat[None, ...], (X_z.shape[0], M * K, 1))
        tau_input = np.broadcast_to(tau_flat[None, ...], (X_z.shape[0], M * K, 1))

        X = [X_z, m_input, tau_input]
        y = np.array(y_surface).reshape(-1, M * K, 1)
        output[key] = (X, y)

    return output


class FeatureToSurfaceDataset(keras.utils.Sequence):
    # dataset for surface and feature tensors
    def __init__(self, surface_tensor, feature_tensor, date_tensor,
                 batch_size=None, shuffle=False, global_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.surface = surface_tensor  # (t, m, k, c)
        self.features = feature_tensor # (t, d)
        self.dates = date_tensor      # (t,)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.T = len(date_tensor)
        self.global_indices = global_indices if global_indices is not None else np.arange(self.T)
        self.indices = np.arange(self.T)
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __len__(self):
        # number of batches
        if self.batch_size is None:
            return 1
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        # get batch idxs
        if self.batch_size is None:
            idxs = self.indices
        else:
            idxs = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        X_combined, dates_out = [], []
        for t in idxs:
            iv_flat = ops.convert_to_numpy(self.surface[t][..., -1]).flatten()  # (m*k,)
            x_feat = ops.convert_to_numpy(self.features[t])                    # (d,)
            x_combined = np.concatenate([iv_flat, x_feat])
            X_combined.append(x_combined)
            dates_out.append(self.dates[t])
        x_tensor = ops.convert_to_tensor(np.array(X_combined, dtype=np.float32))
        if getattr(self, "return_dates", False):
            return (x_tensor, x_tensor), dates_out
        else:
            return (x_tensor, x_tensor)

    def on_epoch_end(self):
        # shuffle indices at epoch end
        if self.shuffle:
            np.random.shuffle(self.indices)

    def to_tensor(self):
        # concatenate all batches to tensor
        X_batches = []
        for i in range(len(self)):
            x_batch, _ = self[i]
            X_batches.append(ops.convert_to_numpy(x_batch))
        return ops.convert_to_tensor(np.concatenate(X_batches, axis=0))
    
    
    
from keras.utils import Sequence


from keras import ops
from keras.utils import Sequence
from keras import ops

class FeatureToLatentSequenceDataset(Sequence):
    # dataset for feature and surface sequences
    def __init__(self, surface_tensor, feature_tensor, lookback, batch_size, step=1, global_indices=None):
        self.surface = surface_tensor       # (t, m, k, c)
        self.features = feature_tensor      # (t, d)
        self.lookback = lookback
        self.batch_size = batch_size
        self.step = step
        self.T = surface_tensor.shape[0]
        self.indices = ops.arange(lookback, self.T, step)
        self.indices_np = ops.convert_to_numpy(self.indices).astype(int)
        self.global_indices = (
            ops.convert_to_numpy(global_indices)[self.indices_np]
            if global_indices is not None else self.indices_np
        )

    def __len__(self):
        # number of batches
        return int(len(self.indices_np) / self.batch_size)

    def __getitem__(self, idx):
        # get batch indices
        idx_start = idx * self.batch_size
        idx_end = (idx + 1) * self.batch_size
        raw_indices = self.indices_np[idx_start:idx_end]
        batch_indices = [i for i in raw_indices if i >= self.lookback]
        if len(batch_indices) == 0:
            raise IndexError(f"no valid indices in batch {idx}")
        surf_seq = ops.stack([self.surface[i - self.lookback:i] for i in batch_indices])  # (batch, lookback, m, k, c)
        feat_seq = ops.stack([self.features[i - self.lookback:i] for i in batch_indices]) # (batch, lookback, d)
        surf_target = ops.stack([self.surface[i] for i in batch_indices])                  # (batch, m, k, c)
        return (surf_seq, feat_seq), surf_target

#-------SSVI FUNCTIONS ------
# the experiments with SSVI failed and werent completely used . this needs to go into fruther works section of the discussion as another possible avenue 

import keras
from keras import ops
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SSVISequenceDataset(keras.utils.Sequence):
    """
    SSVI-based dataset for temporal volatility surface forecasting
    Predicts SSVI parameters from surface + feature sequences
    """
    
    def __init__(self, surface_tensor, feature_tensor, ssvi_tensor, lookback, 
                 batch_size, step=1, shuffle=False, global_indices=None, **kwargs):
        super().__init__(**kwargs)
        
        self.surface = surface_tensor       # (T, M, K, C) - reconstructed surfaces
        self.features = feature_tensor      # (T, D) - market features 
        self.ssvi_params = ssvi_tensor      # (T, N_maturities, 3) - theta, rho, beta
        self.lookback = lookback
        self.batch_size = batch_size
        self.step = step
        self.shuffle = shuffle
        self.T = surface_tensor.shape[0]

        # Create valid indices (need lookback history)
        self.indices = ops.arange(lookback, self.T, step)
        self.indices_np = ops.convert_to_numpy(self.indices).astype(int)
        
        if shuffle:
            np.random.shuffle(self.indices_np)

        # Global index tracking for debugging/validation
        self.global_indices = (
            ops.convert_to_numpy(global_indices)[self.indices_np]
            if global_indices is not None else self.indices_np
        )

    def __len__(self):
        return int(np.ceil(len(self.indices_np) / self.batch_size))

    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = min((idx + 1) * self.batch_size, len(self.indices_np))
        
        batch_indices = self.indices_np[idx_start:idx_end]
        
        # Ensure all indices have sufficient lookback
        batch_indices = [i for i in batch_indices if i >= self.lookback]
        
        if len(batch_indices) == 0:
            raise IndexError(f"No valid indices in batch {idx}")

        # Collect sequences
        surf_sequences = []
        feat_sequences = []
        ssvi_targets = []
        
        for i in batch_indices:
            # Input: lookback window of surfaces and features
            surf_seq = self.surface[i - self.lookback:i]  # (lookback, M, K, C)
            feat_seq = self.features[i - self.lookback:i]  # (lookback, D)
            
            # Target: SSVI parameters at time i
            ssvi_target = self.ssvi_params[i]  # (N_maturities, 3)
            
            surf_sequences.append(surf_seq)
            feat_sequences.append(feat_seq)
            ssvi_targets.append(ssvi_target)
        
        # Stack into batch tensors
        surf_seq_batch = ops.stack(surf_sequences)  # (batch, lookback, M, K, C)
        feat_seq_batch = ops.stack(feat_sequences)  # (batch, lookback, D)
        ssvi_target_batch = ops.stack(ssvi_targets)  # (batch, N_maturities, 3)

        return (surf_seq_batch, feat_seq_batch), ssvi_target_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices_np)

class SSVIFeatureDataset(keras.utils.Sequence):
    """
    Simple SSVI dataset combining surface + features -> SSVI parameters
    For encoder training (no temporal sequence)
    """
    
    def __init__(self, surface_tensor, feature_tensor, ssvi_tensor,
                 batch_size=None, shuffle=False, global_indices=None, **kwargs):
        super().__init__(**kwargs)
        
        self.surface = surface_tensor
        self.features = feature_tensor  
        self.ssvi_params = ssvi_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.T = len(surface_tensor)
        
        self.global_indices = (
            global_indices if global_indices is not None 
            else np.arange(self.T)
        )
        
        self.indices = np.arange(self.T)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        if self.batch_size is None:
            return 1
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        if self.batch_size is None:
            idxs = self.indices
        else:
            idxs = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_combined, y_ssvi = [], []
        
        for t in idxs:
            # Flatten surface and concatenate with features
            iv_flat = ops.convert_to_numpy(self.surface[t][..., -1]).flatten()
            x_feat = ops.convert_to_numpy(self.features[t])
            x_combined = np.concatenate([iv_flat, x_feat])
            
            # SSVI parameters as target
            ssvi_target = ops.convert_to_numpy(self.ssvi_params[t])
            
            X_combined.append(x_combined)
            y_ssvi.append(ssvi_target)

        X_tensor = ops.convert_to_tensor(np.array(X_combined, dtype=np.float32))
        y_tensor = ops.convert_to_tensor(np.array(y_ssvi, dtype=np.float32))

        return X_tensor, y_tensor

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def to_tensor(self):
        """Extract all data as single tensors"""
        X_batches, y_batches = [], []
        for i in range(len(self)):
            x_batch, y_batch = self[i]
            X_batches.append(ops.convert_to_numpy(x_batch))
            y_batches.append(ops.convert_to_numpy(y_batch))
        
        X_all = ops.convert_to_tensor(np.concatenate(X_batches, axis=0))
        y_all = ops.convert_to_tensor(np.concatenate(y_batches, axis=0))
        return X_all, y_all

class SSVIPointwiseDataset(keras.utils.Sequence):
    """
    Pointwise SSVI dataset for decoder training
    Predicts IV at specific (strike, maturity) points from SSVI parameters
    """
    
    def __init__(self, ssvi_params, surface_tensor, m_flat, tau_flat, M, K,
                 batch_size=32, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        
        self.ssvi_params = ssvi_params  # (T, N_maturities, 3)
        self.surface = surface_tensor   # (T, M, K, C)
        self.m_flat = m_flat           # (M*K,) - log moneyness
        self.tau_flat = tau_flat       # (M*K,) - time to maturity
        self.M, self.K = M, K
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.T = len(ssvi_params)
        self.indices = np.arange(self.T)
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.T / self.batch_size))

    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = min((idx + 1) * self.batch_size, self.T)
        batch_indices = self.indices[idx_start:idx_end]
        
        X_ssvi, X_m, X_tau, y_iv = [], [], [], []
        
        for t in batch_indices:
            # SSVI parameters for all maturities
            ssvi_t = self.ssvi_params[t]  # (N_maturities, 3)
            
            # Target IV surface
            iv_surface = ops.convert_to_numpy(self.surface[t])[..., -1]  # (M, K)
            iv_flat = iv_surface.reshape(self.M * self.K)
            
            # Repeat SSVI params for each strike/maturity combination
            ssvi_repeated = np.repeat(ssvi_t[None, :, :], self.M * self.K, axis=0)
            
            X_ssvi.append(ssvi_repeated)
            X_m.append(self.m_flat)
            X_tau.append(self.tau_flat)
            y_iv.append(iv_flat)
        
        # Convert to tensors
        X_ssvi_batch = ops.convert_to_tensor(np.array(X_ssvi))
        X_m_batch = ops.convert_to_tensor(np.array(X_m))
        X_tau_batch = ops.convert_to_tensor(np.array(X_tau))
        y_batch = ops.convert_to_tensor(np.array(y_iv))
        
        return [X_ssvi_batch, X_m_batch, X_tau_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def prepare_ssvi_dataloaders(train_dataset, val_dataset, test_dataset, 
                            batch_size, normalize=False):
    """
    Prepare data loaders with optional normalization
    Similar to prepare_vae_dataloaders but for SSVI parameters
    """
    
    def extract_all_ssvi(dataset):
        all_x, all_y = [], []
        for i in range(len(dataset)):
            if isinstance(dataset[i], tuple) and len(dataset[i]) == 2:
                x_batch, y_batch = dataset[i]
                all_x.append(ops.convert_to_numpy(x_batch))
                all_y.append(ops.convert_to_numpy(y_batch))
        return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

    X_train, y_train = extract_all_ssvi(train_dataset)
    X_val, y_val = extract_all_ssvi(val_dataset) 
    X_test, y_test = extract_all_ssvi(test_dataset)

    if normalize:
        # Normalize features
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        
        # Normalize SSVI parameters
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1]))
        y_val = scaler_y.transform(y_val.reshape(-1, y_val.shape[-1]))
        y_test = scaler_y.transform(y_test.reshape(-1, y_test.shape[-1]))
        
        y_train = y_train.reshape(y_train.shape[0], -1, 3)
        y_val = y_val.reshape(y_val.shape[0], -1, 3)
        y_test = y_test.reshape(y_test.shape[0], -1, 3)
        
        print("Applied StandardScaler normalization")
    else:
        scaler_X, scaler_y = None, None

    def _batch_generator_ssvi(X, y):
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield X[start:end], y[start:end]

    return (
        _batch_generator_ssvi(X_train, y_train),
        _batch_generator_ssvi(X_val, y_val),
        _batch_generator_ssvi(X_test, y_test),
        (scaler_X, scaler_y)
    )