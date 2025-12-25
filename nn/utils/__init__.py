"""Core utilities for Particle Filter NCPs."""

from .resampling import (
    SoftResampler,
    AlphaMode,
    soft_resample,
    compute_proposal,
)
from .weights import (
    log_weight_update,
    normalize_log_weights,
    compute_ess,
    safe_logsumexp,
    weighted_mean,
    weighted_variance,
)
from .noise import (
    NoiseInjector,
    NoiseType,
    ConstantNoise,
    TimeScaledNoise,
    LearnedNoise,
    StateDependentNoise,
)
from .monitoring import (
    ParticleMonitor,
    compute_particle_diversity,
    check_numerical_health,
)
from .functional import batched_linear

__all__ = [
    # Resampling
    "SoftResampler",
    "AlphaMode",
    "soft_resample",
    "compute_proposal",
    # Weights
    "log_weight_update",
    "normalize_log_weights",
    "compute_ess",
    "safe_logsumexp",
    "weighted_mean",
    "weighted_variance",
    # Noise
    "NoiseInjector",
    "NoiseType",
    "ConstantNoise",
    "TimeScaledNoise",
    "LearnedNoise",
    "StateDependentNoise",
    # Monitoring
    "ParticleMonitor",
    "compute_particle_diversity",
    "check_numerical_health",
    # Functional
    "batched_linear",
]
