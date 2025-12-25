"""Particle Filter Neural Circuit Policies - PyTorch implementation.

This module provides particle filter extensions for Neural Circuit Policies (NCPs),
supporting four approaches:
- Approach A (State-Level): Particles over hidden states
- Approach B (Parameter-Level): Particles over model parameters
- Approach C (Dual): Joint particles over states and parameters
- Approach D (SDE): Stochastic differential equation formulation (LTC only)

Example:
    >>> from pfncps.nn import PFCfC, PFLTC, PFNCP
    >>> from ncps.wirings import AutoNCP
    >>>
    >>> # Simple CfC with state-level particle filter
    >>> model = PFCfC(input_size=20, hidden_size=64, n_particles=32)
    >>>
    >>> # Wired NCP with SDE approach
    >>> wiring = AutoNCP(units=64, output_size=10)
    >>> model = PFNCP(wiring=wiring, input_size=20, n_particles=32, approach='sde')
"""

# Core utilities
from .utils import (
    AlphaMode,
    NoiseType,
    SoftResampler,
    NoiseInjector,
    ConstantNoise,
    TimeScaledNoise,
    LearnedNoise,
    StateDependentNoise,
    ParticleMonitor,
    compute_ess,
    normalize_log_weights,
    log_weight_update,
    weighted_mean,
    weighted_variance,
)

# Observation models
from .observation import (
    ObservationModel,
    IdentityObservationModel,
    GaussianObservationModel,
    HeteroscedasticGaussianObservationModel,
    LearnedMLPObservationModel,
    EnergyBasedObservationModel,
    AttentionObservationModel,
    ClassificationObservationModel,
    MLPClassificationObservationModel,
    OrdinalObservationModel,
)

# Approach A: State-Level PF cells
from .state_level import (
    StateLevelPFCell,
    PFCfCCell,
    PFLTCCell,
    PFWiredCfCCell,
)

# Approach B: Parameter-Level PF cells
from .param_level import (
    ParamLevelPFCell,
    ParamPFCfCCell,
    ParamPFLTCCell,
    ParamPFWiredCfCCell,
    ParameterRegistry,
)

# Approach C: Dual PF cells
from .dual import (
    DualPFCell,
    DualPFCfCCell,
    DualPFLTCCell,
    DualPFWiredCfCCell,
    RaoBlackwellEstimator,
)

# Approach D: SDE cells (LTC only)
from .sde import (
    SDEPFCell,
    SDELTCCell,
    SDEWiredLTCCell,
    DiffusionCoefficient,
    ConstantDiffusion,
    LearnedDiffusion,
    StateDependentDiffusion,
    EulerMaruyamaSolver,
    MilsteinSolver,
)

# Mixed memory (LSTM augmentation)
from .mixed_memory import (
    LSTMAugmentedPFCell,
    MixedMemoryPFCell,
)

# High-level sequence wrappers
from .wrappers import (
    PFCfC,
    PFLTC,
    PFNCP,
)

# Spatial heatmap generation
from .spatial import (
    SpatialProjectionHead,
    SoftSpatialRenderer,
    SpatialPFNCP,
)

__all__ = [
    # Utils
    "AlphaMode",
    "NoiseType",
    "SoftResampler",
    "NoiseInjector",
    "ConstantNoise",
    "TimeScaledNoise",
    "LearnedNoise",
    "StateDependentNoise",
    "ParticleMonitor",
    "compute_ess",
    "normalize_log_weights",
    "log_weight_update",
    "weighted_mean",
    "weighted_variance",
    # Observation models
    "ObservationModel",
    "IdentityObservationModel",
    "GaussianObservationModel",
    "HeteroscedasticGaussianObservationModel",
    "LearnedMLPObservationModel",
    "EnergyBasedObservationModel",
    "AttentionObservationModel",
    "ClassificationObservationModel",
    "MLPClassificationObservationModel",
    "OrdinalObservationModel",
    # State-Level (Approach A)
    "StateLevelPFCell",
    "PFCfCCell",
    "PFLTCCell",
    "PFWiredCfCCell",
    # Parameter-Level (Approach B)
    "ParamLevelPFCell",
    "ParamPFCfCCell",
    "ParamPFLTCCell",
    "ParamPFWiredCfCCell",
    "ParameterRegistry",
    # Dual (Approach C)
    "DualPFCell",
    "DualPFCfCCell",
    "DualPFLTCCell",
    "DualPFWiredCfCCell",
    "RaoBlackwellEstimator",
    # SDE (Approach D)
    "SDEPFCell",
    "SDELTCCell",
    "SDEWiredLTCCell",
    "DiffusionCoefficient",
    "ConstantDiffusion",
    "LearnedDiffusion",
    "StateDependentDiffusion",
    "EulerMaruyamaSolver",
    "MilsteinSolver",
    # Mixed Memory
    "LSTMAugmentedPFCell",
    "MixedMemoryPFCell",
    # Wrappers
    "PFCfC",
    "PFLTC",
    "PFNCP",
    # Spatial
    "SpatialProjectionHead",
    "SoftSpatialRenderer",
    "SpatialPFNCP",
]