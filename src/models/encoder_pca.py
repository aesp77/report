from sklearn.decomposition import PCA
import numpy as np
import keras 
from keras import ops

class PCAEncoder:
    """
    Principal Component Analysis encoder for dimensionality reduction.
    Parameters:
        n_components: Number of principal components to retain.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(Z)

    def explained_variance(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def components(self) -> np.ndarray:
        return self.pca.components_

#-------------- FOR PIPELINE COLLECTION USE ONLY ------------#
class PCAEncoderWrapper:
    """
    Wrapper to make PCA encoder compatible with pipeline interface.
    Handles input reshaping and conversion for PCA.
    """
    def __init__(self, pca_model, M, K):
        self.pca = pca_model
        self.M = M
        self.K = K

    def predict(self, X, **kwargs):
        import keras.ops as ops
        X = ops.convert_to_numpy(X)
        # Reshape input to 2D for PCA
        if len(X.shape) == 4:  # (batch, M, K, timesteps)
            X = X[..., -1].reshape(X.shape[0], -1)
        elif len(X.shape) == 3:  # (batch, timesteps, features)
            X = X.reshape(X.shape[0], -1)
        return self.pca.pca.transform(X)

    def __call__(self, X, training=False):
        return self.predict(X)

class PCADecoderWrapper:
    """
    Wrapper to make PCA decoder compatible with pipeline interface.
    """
    def __init__(self, encoder_wrapper):
        self.encoder = encoder_wrapper

    def predict(self, Z, **kwargs):
        import keras.ops as ops
        Z = ops.convert_to_numpy(Z)
        return self.encoder.pca.inverse_transform(Z)
    
# PCA models were trained on surface only initially and need to be adjusted for a new sequence to compare the output of later models

class PCAEncodedLatentSequence:
    """
    Special encoder sequence handler for PCA that only passes surfaces.
    Iterates over dataset and encodes surface sequences and targets.
    """
    def __init__(self, dataset, encoder):
        self.dataset = dataset
        self.encoder = encoder

    def __iter__(self):
        for batch in self.dataset:
            (surface_seq, feat_seq), target_surface = batch
            batch_size = surface_seq.shape[0]
            lookback = surface_seq.shape[1]
            # Encode sequences - PCA only needs surfaces
            z_seq = []
            for t in range(lookback):
                surface_t = ops.convert_to_numpy(surface_seq[:, t, :, :, -1].reshape(batch_size, -1))
                z_t = self.encoder.predict(surface_t)
                z_seq.append(z_t)
            z_seq = np.stack(z_seq, axis=1)
            # Encode target
            target_flat = ops.convert_to_numpy(target_surface[:, :, :, -1].reshape(batch_size, -1))
            z_target = self.encoder.predict(target_flat)
            yield z_seq, z_target

    def __len__(self):
        return len(self.dataset)