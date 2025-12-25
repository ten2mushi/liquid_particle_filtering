"""Core visualization infrastructure for PFNCPS models.

Provides the main PFVisualizer class and architecture detection utilities.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

if TYPE_CHECKING:
    from ..collectors.base_collector import BaseDataCollector


class PFApproach(Enum):
    """Particle filtering approach type."""
    STATE_LEVEL = "state_level"
    PARAM_LEVEL = "param_level"
    DUAL = "dual"
    SDE = "sde"
    UNKNOWN = "unknown"


class BaseArchitecture(Enum):
    """Base NCP architecture type."""
    CFC = "cfc"
    LTC = "ltc"
    WIRED_CFC = "wired_cfc"
    WIRED_LTC = "wired_ltc"
    UNKNOWN = "unknown"


@dataclass
class ArchitectureInfo:
    """Information about the detected model architecture.

    Attributes:
        pf_approach: The particle filtering approach used
        base_arch: The underlying NCP architecture
        has_wiring: Whether the model uses NCP wiring
        n_particles: Number of particles K
        hidden_size: Hidden state dimension
        input_size: Input dimension
        output_size: Output dimension (if detectable)
        tracked_params: List of tracked parameter names (for param/dual approaches)
        extra: Additional architecture-specific information
    """
    pf_approach: PFApproach
    base_arch: BaseArchitecture
    has_wiring: bool
    n_particles: int
    hidden_size: int
    input_size: int = 0
    output_size: int = 0
    tracked_params: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_state_level(self) -> bool:
        return self.pf_approach == PFApproach.STATE_LEVEL

    @property
    def is_param_level(self) -> bool:
        return self.pf_approach == PFApproach.PARAM_LEVEL

    @property
    def is_dual(self) -> bool:
        return self.pf_approach == PFApproach.DUAL

    @property
    def is_sde(self) -> bool:
        return self.pf_approach == PFApproach.SDE

    @property
    def is_ltc_based(self) -> bool:
        return self.base_arch in (BaseArchitecture.LTC, BaseArchitecture.WIRED_LTC)

    @property
    def is_cfc_based(self) -> bool:
        return self.base_arch in (BaseArchitecture.CFC, BaseArchitecture.WIRED_CFC)


def detect_architecture(model: nn.Module) -> ArchitectureInfo:
    """Detect the architecture of a PFNCPS model.

    Uses isinstance checks and attribute inspection to determine
    the particle filtering approach and base architecture.

    Args:
        model: The PFNCPS model to analyze

    Returns:
        ArchitectureInfo with detected architecture details
    """
    pf_approach = PFApproach.UNKNOWN
    base_arch = BaseArchitecture.UNKNOWN
    has_wiring = False
    n_particles = 0
    hidden_size = 0
    input_size = 0
    output_size = 0
    tracked_params: List[str] = []
    extra: Dict[str, Any] = {}

    # Find the PF cell (may be wrapped in a sequence model)
    pf_cell = _find_pf_cell(model)

    if pf_cell is None:
        # Model itself might be the cell
        pf_cell = model

    # Detect PF approach by class name patterns
    cell_class_name = pf_cell.__class__.__name__

    # Check for approach type
    if "SDE" in cell_class_name:
        pf_approach = PFApproach.SDE
    elif "Dual" in cell_class_name:
        pf_approach = PFApproach.DUAL
    elif "Param" in cell_class_name:
        pf_approach = PFApproach.PARAM_LEVEL
    elif any(name in cell_class_name for name in ["StateLevelPF", "PFCfC", "PFLTC", "PFWired"]):
        pf_approach = PFApproach.STATE_LEVEL

    # Check for base architecture
    if "Wired" in cell_class_name:
        has_wiring = True
        if "LTC" in cell_class_name:
            base_arch = BaseArchitecture.WIRED_LTC
        elif "CfC" in cell_class_name:
            base_arch = BaseArchitecture.WIRED_CFC
    else:
        if "LTC" in cell_class_name:
            base_arch = BaseArchitecture.LTC
        elif "CfC" in cell_class_name:
            base_arch = BaseArchitecture.CFC

    # Extract common attributes
    if hasattr(pf_cell, "n_particles"):
        n_particles = pf_cell.n_particles

    if hasattr(pf_cell, "hidden_size"):
        hidden_size = pf_cell.hidden_size
    elif hasattr(pf_cell, "state_size"):
        hidden_size = pf_cell.state_size

    if hasattr(pf_cell, "input_size"):
        input_size = pf_cell.input_size

    if hasattr(pf_cell, "output_size"):
        output_size = pf_cell.output_size

    # Extract tracked params for param-level and dual approaches
    if pf_approach in (PFApproach.PARAM_LEVEL, PFApproach.DUAL):
        if hasattr(pf_cell, "param_registry"):
            registry = pf_cell.param_registry
            if hasattr(registry, "group_names"):
                tracked_params = list(registry.group_names)

    # Extract extra info
    if hasattr(pf_cell, "ode_unfolds"):
        extra["ode_unfolds"] = pf_cell.ode_unfolds

    if hasattr(pf_cell, "mode"):
        extra["cfc_mode"] = pf_cell.mode

    if has_wiring and hasattr(pf_cell, "wiring"):
        wiring = pf_cell.wiring
        if hasattr(wiring, "units"):
            extra["wiring_units"] = wiring.units

    return ArchitectureInfo(
        pf_approach=pf_approach,
        base_arch=base_arch,
        has_wiring=has_wiring,
        n_particles=n_particles,
        hidden_size=hidden_size,
        input_size=input_size,
        output_size=output_size,
        tracked_params=tracked_params,
        extra=extra,
    )


def _find_pf_cell(model: nn.Module) -> Optional[nn.Module]:
    """Find the particle filter cell within a model.

    Handles wrapper models like PFCfC, PFLTC that contain a cell.

    Args:
        model: The model to search

    Returns:
        The PF cell module or None if not found
    """
    # Check for common cell attribute names
    cell_attr_names = ["cell", "pf_cell", "rnn_cell", "base_cell"]

    for name in cell_attr_names:
        if hasattr(model, name):
            cell = getattr(model, name)
            if cell is not None and isinstance(cell, nn.Module):
                return cell

    # Check if model itself has n_particles (is a cell)
    if hasattr(model, "n_particles"):
        return model

    # Search in children
    for name, child in model.named_children():
        if hasattr(child, "n_particles"):
            return child
        # Recursive search
        found = _find_pf_cell(child)
        if found is not None:
            return found

    return None


class PFVisualizer:
    """Main visualization interface for PFNCPS models.

    Provides comprehensive visualization of particle filter dynamics,
    including particle trajectories, weight evolution, ESS, and
    architecture-specific metrics.

    Example:
        >>> model = PFCfC(input_size=10, hidden_size=64, n_particles=32)
        >>> viz = PFVisualizer.from_model(model)
        >>>
        >>> # Run model and collect data
        >>> with viz.collect():
        ...     for x, y in dataloader:
        ...         output, state = model(x, state, observation=y)
        >>>
        >>> # Generate visualizations
        >>> viz.plot_ess_timeline()
        >>> viz.plot_particle_trajectories(dims=[0, 1, 2])
        >>> viz.dashboard("health")

    Attributes:
        model: The PFNCPS model being visualized
        arch_info: Detected architecture information
        collector: Data collector for this model type
        theme: Visual theme configuration
    """

    def __init__(
        self,
        model: nn.Module,
        theme: str = "default",
        max_history: int = 10000,
        downsample_strategy: str = "lttb",
        batch_idx: int = 0,
    ):
        """Initialize visualizer.

        Args:
            model: PFNCPS model instance
            theme: Visual theme ("default", "paper", "dark")
            max_history: Maximum timesteps to store
            downsample_strategy: How to downsample long sequences ("lttb", "uniform")
            batch_idx: Which batch element to track (default 0)
        """
        self.model = model
        self.arch_info = detect_architecture(model)
        self.theme_name = theme
        self.max_history = max_history
        self.downsample_strategy = downsample_strategy
        self.batch_idx = batch_idx

        # Import here to avoid circular imports
        from .themes import get_theme
        self.theme = get_theme(theme)

        # Create appropriate collector
        self.collector = self._create_collector()

        # Hooks for data collection
        self._hooks: List[RemovableHandle] = []
        self._collecting = False

    @classmethod
    def from_model(cls, model: nn.Module, **kwargs) -> "PFVisualizer":
        """Factory method with auto-detection.

        Args:
            model: PFNCPS model instance
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured PFVisualizer instance
        """
        return cls(model, **kwargs)

    def _create_collector(self) -> "BaseDataCollector":
        """Create appropriate data collector based on architecture."""
        from ..collectors import (
            StateCollector,
            ParamCollector,
            DualCollector,
            SDECollector,
        )

        collector_kwargs = {
            "max_history": self.max_history,
            "downsample_strategy": self.downsample_strategy,
            "batch_idx": self.batch_idx,
            "arch_info": self.arch_info,
        }

        if self.arch_info.is_sde:
            return SDECollector(**collector_kwargs)
        elif self.arch_info.is_dual:
            return DualCollector(**collector_kwargs)
        elif self.arch_info.is_param_level:
            return ParamCollector(**collector_kwargs)
        else:
            # Default to state-level
            return StateCollector(**collector_kwargs)

    @contextmanager
    def collect(self):
        """Context manager for data collection.

        Registers forward hooks to capture particle filter state
        at each timestep.

        Usage:
            with viz.collect():
                output, state = model(x, state)

        Yields:
            The data collector instance
        """
        if self._collecting:
            raise RuntimeError("Already collecting data. Cannot nest collect() calls.")

        self._collecting = True
        self._register_hooks()

        try:
            yield self.collector
        finally:
            self._remove_hooks()
            self._collecting = False

    def _register_hooks(self) -> None:
        """Register forward hooks for data collection."""
        pf_cell = _find_pf_cell(self.model)
        if pf_cell is None:
            pf_cell = self.model

        def pf_forward_hook(module, input, output):
            """Hook to capture PF cell output."""
            # Expected output format: (output, (particles, log_weights))
            # or: (output, state_tuple)
            if isinstance(output, tuple) and len(output) >= 2:
                out = output[0]
                state = output[1]

                if isinstance(state, tuple) and len(state) >= 2:
                    particles = state[0]
                    log_weights = state[1]

                    # Extract extra data if available
                    extra = {}
                    if len(state) > 2:
                        extra["additional_state"] = state[2:]

                    # Get observation if passed in input
                    observation = None
                    if isinstance(input, tuple):
                        for inp in input:
                            if isinstance(inp, dict) and "observation" in inp:
                                observation = inp["observation"]

                    self.collector.log_step(
                        particles=particles,
                        log_weights=log_weights,
                        outputs=out,
                        observation=observation,
                        **extra,
                    )

        handle = pf_cell.register_forward_hook(pf_forward_hook)
        self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def reset(self) -> None:
        """Clear collected data."""
        self.collector.reset()

    @property
    def n_steps(self) -> int:
        """Number of collected timesteps."""
        return len(self.collector)

    @property
    def has_data(self) -> bool:
        """Whether any data has been collected."""
        return self.n_steps > 0

    def _check_data(self) -> None:
        """Raise error if no data collected."""
        if not self.has_data:
            raise RuntimeError(
                "No data collected. Use viz.collect() context manager to collect data first."
            )

    # =========================================================================
    # Core Plots (C1-C10)
    # =========================================================================

    def plot_ess_timeline(
        self,
        ax=None,
        show_threshold: bool = True,
        show_resampling: bool = True,
        **kwargs,
    ):
        """Plot Effective Sample Size over time.

        Args:
            ax: Matplotlib axes (created if None)
            show_threshold: Whether to show resampling threshold line
            show_resampling: Whether to mark resampling events
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_ess_timeline
        return plot_ess_timeline(
            self.collector,
            ax=ax,
            theme=self.theme,
            show_threshold=show_threshold,
            show_resampling=show_resampling,
            **kwargs,
        )

    def plot_weight_distribution(
        self,
        mode: str = "heatmap",
        ax=None,
        **kwargs,
    ):
        """Visualize particle weight evolution.

        Args:
            mode: "heatmap" or "stacked_area"
            ax: Matplotlib axes (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_weight_distribution
        return plot_weight_distribution(
            self.collector,
            mode=mode,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def plot_weight_entropy(self, ax=None, **kwargs):
        """Plot weight distribution entropy over time.

        Args:
            ax: Matplotlib axes (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_weight_entropy
        return plot_weight_entropy(
            self.collector,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def plot_particle_trajectories(
        self,
        dims: Optional[List[int]] = None,
        n_particles: Optional[int] = None,
        style: str = "fan",
        ax=None,
        **kwargs,
    ):
        """Plot state trajectories with uncertainty.

        Args:
            dims: Which hidden dimensions to plot (default: first 3)
            n_particles: Number of particles to show in spaghetti mode
            style: "fan" (mean with quantiles), "spaghetti" (individual lines),
                   or "quantiles" (percentile bands)
            ax: Matplotlib axes (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_particle_trajectories
        return plot_particle_trajectories(
            self.collector,
            dims=dims,
            n_particles=n_particles,
            style=style,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def plot_particle_diversity(self, ax=None, **kwargs):
        """Multi-panel plot of diversity metrics.

        Includes variance, pairwise distance, effective dimension,
        and collapse ratio over time.

        Args:
            ax: Matplotlib axes array (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_particle_diversity
        return plot_particle_diversity(
            self.collector,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def plot_resampling_events(self, ax=None, **kwargs):
        """Visualize when and why resampling was triggered.

        Args:
            ax: Matplotlib axes (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_resampling_events
        return plot_resampling_events(
            self.collector,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def plot_observation_likelihoods(
        self,
        mode: str = "box",
        ax=None,
        **kwargs,
    ):
        """Plot observation log-likelihoods per particle.

        Args:
            mode: "box" for box plots, "violin" for violin plots
            ax: Matplotlib axes (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_observation_likelihoods
        return plot_observation_likelihoods(
            self.collector,
            mode=mode,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def plot_numerical_health(self, ax=None, **kwargs):
        """Plot numerical health indicators (NaN/Inf/bounds).

        Args:
            ax: Matplotlib axes (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_numerical_health
        return plot_numerical_health(
            self.collector,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def plot_weighted_output(
        self,
        output_dims: Optional[List[int]] = None,
        ax=None,
        **kwargs,
    ):
        """Plot predictions with uncertainty bands.

        Args:
            output_dims: Which output dimensions to plot
            ax: Matplotlib axes (created if None)
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure and axes
        """
        self._check_data()
        from ..plots.core_plots import plot_weighted_output
        return plot_weighted_output(
            self.collector,
            output_dims=output_dims,
            ax=ax,
            theme=self.theme,
            **kwargs,
        )

    def animate_particles_2d(
        self,
        dims: Tuple[int, int] = (0, 1),
        projection: str = "raw",
        interval: int = 100,
        **kwargs,
    ):
        """Animate 2D projection of particle cloud.

        Args:
            dims: Which dimensions to project to (default: first 2)
            projection: "raw", "pca", or "tsne"
            interval: Animation interval in ms
            **kwargs: Additional animation arguments

        Returns:
            Matplotlib animation object
        """
        self._check_data()
        from ..plots.core_plots import animate_particles_2d
        return animate_particles_2d(
            self.collector,
            dims=dims,
            projection=projection,
            interval=interval,
            theme=self.theme,
            **kwargs,
        )

    # =========================================================================
    # Dashboard Generation
    # =========================================================================

    def dashboard(self, which: str = "health", figsize: Tuple[int, int] = (16, 12)):
        """Generate multi-panel diagnostic dashboard.

        Args:
            which: Dashboard type
                - "health": ESS, weights, numerical health, diversity
                - "research": Trajectories, calibration, uncertainty
                - "debug": All diagnostic plots
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        self._check_data()
        from ..plots.diagnostic_plots import create_dashboard
        return create_dashboard(
            self.collector,
            which=which,
            figsize=figsize,
            theme=self.theme,
            arch_info=self.arch_info,
        )

    # =========================================================================
    # Export Utilities
    # =========================================================================

    def save_all(
        self,
        output_dir: str,
        format: str = "png",
        dpi: int = 150,
    ) -> List[str]:
        """Save all relevant plots to directory.

        Args:
            output_dir: Output directory path
            format: Image format ("png", "pdf", "svg")
            dpi: Resolution for raster formats

        Returns:
            List of saved file paths
        """
        self._check_data()
        from ..utils.export import save_all_plots
        return save_all_plots(
            self,
            output_dir=output_dir,
            format=format,
            dpi=dpi,
        )

    def to_tensorboard(self, writer, step: int, prefix: str = "pf_viz"):
        """Log visualizations to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter
            step: Global step number
            prefix: Tag prefix for logged items
        """
        self._check_data()
        from ..backends.tensorboard_backend import log_to_tensorboard
        log_to_tensorboard(
            self,
            writer=writer,
            step=step,
            prefix=prefix,
        )

    # =========================================================================
    # Architecture-Specific Plots (stubs for future implementation)
    # =========================================================================

    def plot_ltc_voltage_traces(self, neuron_ids=None, ax=None, **kwargs):
        """LTC-specific: Plot voltage traces per neuron."""
        if not self.arch_info.is_ltc_based:
            raise ValueError(f"LTC plots not available for {self.arch_info.base_arch}")
        self._check_data()
        from ..plots.ltc_plots import plot_voltage_traces
        return plot_voltage_traces(self.collector, neuron_ids=neuron_ids, ax=ax, theme=self.theme, **kwargs)

    def plot_cfc_interpolation(self, ax=None, **kwargs):
        """CfC-specific: Plot time interpolation weights."""
        if not self.arch_info.is_cfc_based:
            raise ValueError(f"CfC plots not available for {self.arch_info.base_arch}")
        self._check_data()
        from ..plots.cfc_plots import plot_interpolation_weights
        return plot_interpolation_weights(self.collector, ax=ax, theme=self.theme, **kwargs)

    def plot_sde_diffusion(self, ax=None, **kwargs):
        """SDE-specific: Plot diffusion coefficient evolution."""
        if not self.arch_info.is_sde:
            raise ValueError(f"SDE plots not available for {self.arch_info.pf_approach}")
        self._check_data()
        from ..plots.sde_plots import plot_diffusion_magnitude
        return plot_diffusion_magnitude(self.collector, ax=ax, theme=self.theme, **kwargs)

    def plot_param_posteriors(self, param_names=None, ax=None, **kwargs):
        """Param-level specific: Plot parameter posterior distributions."""
        if not self.arch_info.is_param_level and not self.arch_info.is_dual:
            raise ValueError(f"Param plots not available for {self.arch_info.pf_approach}")
        self._check_data()
        from ..plots.param_plots import plot_param_posterior_marginals
        return plot_param_posterior_marginals(self.collector, param_names=param_names, ax=ax, theme=self.theme, **kwargs)
