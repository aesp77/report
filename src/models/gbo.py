import warnings

warnings.warn(
    "gbo.py is deprecated. Please import from fw_pca instead:\n"
    "  from src.models.fw_pca import FeatureWeightedPCA, FWPCAEncoderWrapper, FWPCADecoderWrapper",
    DeprecationWarning,
    stacklevel=2
)

from .fw_pca import (
    FeatureWeightedPCA,
    FWPCAEncoderWrapper, 
    FWPCADecoderWrapper
)

# Deprecated aliases for backward compatibility
GeneralizedBasisOperator = FeatureWeightedPCA
GBOEncoderWrapper = FWPCAEncoderWrapper
GBODecoderWrapper = FWPCADecoderWrapper