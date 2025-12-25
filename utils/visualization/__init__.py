"""PFNCPS Visualization Submodule.

Comprehensive visualization toolkit for deep investigation of latent space
components across all PFNCPS model architectures.

Quick Start:
    >>> from pfncps.utils.visualization import PFVisualizer
    >>> from pfncps.nn import PFCfC
    >>>
    >>> # Create model and visualizer
    >>> model = PFCfC(input_size=10, hidden_size=64, n_particles=32)
    >>> viz = PFVisualizer.from_model(model)
    >>>
    >>> # Collect data during forward pass
    >>> with viz.collect():
    ...     for x, y in dataloader:
    ...         output, state = model(x, state, observation=y)
    >>>
    >>> # Generate visualizations
    >>> viz.plot_ess_timeline()
    >>> viz.plot_particle_trajectories(dims=[0, 1, 2], style="fan")
    >>> viz.dashboard("health")
    >>> viz.save_all("./figures/")

Supported Architectures:
    - State-Level PF: PFCfCCell, PFLTCCell, PFWiredCfCCell
    - Parameter-Level PF: ParamPFCfCCell, ParamPFLTCCell
    - Dual PF: DualPFCfCCell, DualPFLTCCell
    - SDE PF: SDELTCCell, SDEWiredLTCCell

Core Visualizations (C1-C10):
    - plot_ess_timeline: Track particle diversity health
    - plot_weight_distribution: Show weight concentration
    - plot_weight_entropy: Measure weight uniformity
    - plot_particle_trajectories: Visualize state evolution
    - plot_particle_diversity: Multi-metric diversity analysis
    - plot_resampling_events: Show resampling triggers
    - plot_observation_likelihoods: Per-particle likelihoods
    - plot_numerical_health: NaN/Inf/bounds detection
    - plot_weighted_output: Predictions with uncertainty
    - animate_particles_2d: 2D projection animation

Dashboard Types:
    - "health": ESS, weights, numerical health, diversity
    - "research": Trajectories, calibration, uncertainty
    - "debug": All diagnostic plots

TensorBoard Integration:
    >>> from torch.utils.tensorboard import SummaryWriter
    >>> writer = SummaryWriter()
    >>> viz.to_tensorboard(writer, step=epoch)
"""

from .core.base import (
    PFVisualizer,
    ArchitectureInfo,
    PFApproach,
    BaseArchitecture,
    detect_architecture,
)
from .core.themes import (
    Theme,
    get_theme,
    register_theme,
    AVAILABLE_THEMES,
)
from .collectors import (
    BaseDataCollector,
    CollectedStep,
    StateCollector,
    ParamCollector,
    DualCollector,
    SDECollector,
)

__all__ = [
    # Main interface
    "PFVisualizer",
    # Architecture detection
    "ArchitectureInfo",
    "PFApproach",
    "BaseArchitecture",
    "detect_architecture",
    # Themes
    "Theme",
    "get_theme",
    "register_theme",
    "AVAILABLE_THEMES",
    # Collectors
    "BaseDataCollector",
    "CollectedStep",
    "StateCollector",
    "ParamCollector",
    "DualCollector",
    "SDECollector",
]

__version__ = "0.1.0"
