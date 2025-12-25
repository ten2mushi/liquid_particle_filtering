"""Observation models for particle filter inference.

Provides a suite of observation models for computing p(y | h):
- Gaussian: Linear projection with Gaussian noise
- Heteroscedastic Gaussian: State-dependent noise variance
- Classification: Categorical/multinomial observations
- Learned: MLP-based likelihood estimation
- Energy-based: Learned energy function
"""

from .base import (
    ObservationModel,
    IdentityObservationModel,
)
from .gaussian import (
    GaussianObservationModel,
    HeteroscedasticGaussianObservationModel,
)
from .learned import (
    LearnedMLPObservationModel,
    EnergyBasedObservationModel,
    AttentionObservationModel,
)
from .classification import (
    ClassificationObservationModel,
    MLPClassificationObservationModel,
    OrdinalObservationModel,
)

__all__ = [
    # Base
    "ObservationModel",
    "IdentityObservationModel",
    # Gaussian
    "GaussianObservationModel",
    "HeteroscedasticGaussianObservationModel",
    # Learned
    "LearnedMLPObservationModel",
    "EnergyBasedObservationModel",
    "AttentionObservationModel",
    # Classification
    "ClassificationObservationModel",
    "MLPClassificationObservationModel",
    "OrdinalObservationModel",
]
