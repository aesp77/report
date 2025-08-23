# gbo.py (Keras 3 compatible, tensor-safe, serializable)
from keras import ops
import numpy as np

class GeneralizedBasisOperator:
    """
    Generalized Basis Operator (GBO) for dimensionality reduction and surface reconstruction.
    Parameters:
        n_components: Number of basis vectors to retain.
        regularization: L2 regularization strength for inversion stability.
    """
    def __init__(self, n_components=5, regularization=1e-3):
        self.n_components = n_components
        self.regularization = regularization
        self.basis = None
        self.mean = None

    def fit(self, X_surface, X_features):
        self.mean = ops.mean(X_surface, axis=0, keepdims=True)
        Xc = X_surface - self.mean
        Phi = ops.matmul(ops.transpose(X_features), X_features)
        S = ops.matmul(ops.transpose(Xc), X_features)
        reg = self.regularization * ops.eye(Phi.shape[0])
        inv = ops.linalg.inv(Phi + reg)
        A = ops.matmul(S, inv)
        # SVD for basis extraction
        U, _, _ = ops.linalg.svd(ops.matmul(A, ops.transpose(A)))
        self.basis = U[:, :self.n_components]

    def transform(self, X_surface):
        Xc = X_surface - self.mean
        return ops.matmul(Xc, self.basis)

    def inverse_transform(self, Z):
        Z_tensor = ops.convert_to_tensor(Z, dtype="float32")
        return ops.matmul(Z_tensor, ops.transpose(self.basis)) + self.mean

    def fit_transform(self, X_surface, X_features):
        self.fit(X_surface, X_features)
        return self.transform(X_surface)

    def get_config(self):
        config = {
            "n_components": self.n_components,
            "regularization": self.regularization
        }
        if self.basis is not None:
            config["basis"] = ops.convert_to_numpy(self.basis).tolist()
        if self.mean is not None:
            config["mean"] = ops.convert_to_numpy(self.mean).tolist()
        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(
            n_components=config["n_components"],
            regularization=config["regularization"]
        )
        if "basis" in config:
            instance.basis = ops.convert_to_tensor(config["basis"], dtype="float32")
        if "mean" in config:
            instance.mean = ops.convert_to_tensor(config["mean"], dtype="float32")
        return instance


class GBOEncoderWrapper:
    """
    Wrapper for GeneralizedBasisOperator encoder.
    Parameters:
        gbo: Fitted GeneralizedBasisOperator instance.
        X_features: Feature matrix used for fitting.
        M, K: Surface/feature dimensions.
    """
    def __init__(self, gbo, X_features, M, K):
        self.gbo = gbo
        self.X_features = X_features
        self.M = M
        self.K = K

    def fit_transform(self, X_surface_combined):
        surface_part = X_surface_combined[:, :self.M * self.K]
        return ops.convert_to_numpy(
            self.gbo.transform(
                ops.convert_to_tensor(surface_part, dtype="float32")
            )
        )

    def predict(self, X_surface_combined, verbose=0):
        surface_part = X_surface_combined[:, :self.M * self.K]
        return ops.convert_to_numpy(
            self.gbo.transform(
                ops.convert_to_tensor(surface_part, dtype="float32")
            )
        )

    def inverse_transform(self, Z):
        return ops.convert_to_numpy(self.gbo.inverse_transform(Z))

    def get_config(self):
        config = {
            "M": self.M,
            "K": self.K,
            "gbo_config": self.gbo.get_config() if self.gbo else None
        }
        if self.X_features is not None:
            config["X_features_shape"] = list(ops.shape(self.X_features))
            config["X_features_data"] = ops.convert_to_numpy(self.X_features).tolist()
        return config

    @classmethod
    def from_config(cls, config):
        gbo = None
        if config.get("gbo_config"):
            gbo = GeneralizedBasisOperator.from_config(config["gbo_config"])
        X_features = None
        if config.get("X_features_data"):
            X_features = ops.convert_to_tensor(config["X_features_data"], dtype="float32")
        return cls(gbo, X_features, config["M"], config["K"])


class GBODecoderWrapper:
    """
    Decoder wrapper for GBO matching pipeline interface.
    Decodes latent codes to surface reconstruction and pads features.
    """
    def __init__(self, gbo_encoder):
        self.encoder = gbo_encoder
        self.M = gbo_encoder.M
        self.K = gbo_encoder.K
        self.gbo = gbo_encoder.gbo

    def __call__(self, z, training=False):
        z_tensor = ops.convert_to_tensor(z, dtype="float32")
        surface_recon = self.gbo.inverse_transform(z_tensor)
        # Ensure surface_recon is 2D: (batch_size, M*K)
        if len(ops.shape(surface_recon)) == 1:
            surface_recon = ops.reshape(surface_recon, (1, -1))
        batch_size = ops.shape(surface_recon)[0]
        n_features = ops.shape(self.encoder.X_features)[1]
        feature_padding = ops.zeros((batch_size, n_features))
        output = ops.concatenate([surface_recon, feature_padding], axis=-1)
        return ops.convert_to_numpy(output)

    def predict(self, z):
        return self(z, training=False)

    def get_config(self):
        return {
            'encoder_config': self.encoder.get_config()
        }

    @classmethod
    def from_config(cls, config):
        encoder = GBOEncoderWrapper.from_config(config['encoder_config'])
        return cls(encoder)