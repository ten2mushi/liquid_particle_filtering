"""Core visualization plots for all PFNCPS architectures.

Implements plots C1-C10:
- C1: ESS timeline
- C2: Weight distribution
- C3: Weight entropy
- C4: Particle trajectories
- C5: Particle diversity
- C6: Resampling events
- C7: Observation likelihoods
- C8: Numerical health
- C9: Weighted output
- C10: 2D particle animation
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

# Matplotlib imports with lazy loading
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..collectors.base_collector import BaseDataCollector
    from ..core.themes import Theme


def _check_matplotlib():
    """Check matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def _create_figure(ax=None, figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, Axes]:
    """Create or get figure and axes."""
    _check_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_ess_timeline(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    show_threshold: bool = True,
    show_resampling: bool = True,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (10, 4),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot Effective Sample Size over time (C1).

    Args:
        collector: Data collector with logged steps
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        show_threshold: Whether to show resampling threshold
        show_resampling: Whether to mark resampling events
        threshold: ESS threshold as fraction of K
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get data
    timesteps = collector.get_timesteps().numpy()
    ess = collector.get_ess().numpy()
    n_particles = collector.arch_info.n_particles if collector.arch_info else ess.max()

    # Apply theme
    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("ess", "#9467bd")
        threshold_color = theme.colors.get("threshold", "#e377c2")
        resampling_color = theme.colors.get("resampling", "#bcbd22")
        lw = theme.line_widths.get("main", 2.0)
    else:
        color = "#9467bd"
        threshold_color = "#e377c2"
        resampling_color = "#bcbd22"
        lw = 2.0

    # Plot ESS
    ax.plot(timesteps, ess, color=color, linewidth=lw, label="ESS", **kwargs)

    # Show threshold
    if show_threshold:
        threshold_val = threshold * n_particles
        ax.axhline(
            y=threshold_val,
            color=threshold_color,
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.0%}K)",
        )

    # Mark resampling events
    if show_resampling:
        resampling_steps = collector.get_resampling_events()
        if resampling_steps:
            resampling_times = [t for t in resampling_steps if t in timesteps]
            resampling_ess = [ess[np.where(timesteps == t)[0][0]] for t in resampling_times if t in timesteps]
            ax.scatter(
                resampling_times,
                resampling_ess,
                color=resampling_color,
                marker="v",
                s=50,
                zorder=5,
                label="Resampling",
            )

    # Labels and legend
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Effective Sample Size (ESS)")
    ax.set_title("ESS Timeline")
    ax.set_ylim(0, n_particles * 1.1)
    ax.legend(loc="upper right")

    return fig, ax


def plot_weight_distribution(
    collector: "BaseDataCollector",
    mode: str = "heatmap",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Visualize particle weight evolution (C2).

    Args:
        collector: Data collector with logged steps
        mode: "heatmap" or "stacked_area"
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get data
    timesteps = collector.get_timesteps().numpy()
    weights = collector.get_weights().numpy()  # [time, K]

    # Apply theme
    if theme:
        theme.apply_to_axes(ax)
        cmap = "viridis"
    else:
        cmap = "viridis"

    if mode == "heatmap":
        # Sort weights per timestep for better visualization
        sorted_weights = np.sort(weights, axis=1)[:, ::-1]

        im = ax.imshow(
            sorted_weights.T,
            aspect="auto",
            origin="lower",
            extent=[timesteps[0], timesteps[-1], 0, weights.shape[1]],
            cmap=cmap,
            **kwargs,
        )
        fig.colorbar(im, ax=ax, label="Weight")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Particle (sorted by weight)")
        ax.set_title("Weight Distribution (Heatmap)")

    elif mode == "stacked_area":
        # Sort weights and create stacked area
        sorted_weights = np.sort(weights, axis=1)[:, ::-1]

        # Plot top-K particles
        top_k = min(10, weights.shape[1])
        colors = plt.cm.viridis(np.linspace(0, 1, top_k))

        ax.stackplot(
            timesteps,
            sorted_weights[:, :top_k].T,
            colors=colors,
            labels=[f"Particle {i+1}" for i in range(top_k)],
            **kwargs,
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cumulative Weight")
        ax.set_title(f"Weight Distribution (Top {top_k} Particles)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1.05)

    return fig, ax


def plot_weight_entropy(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 4),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot weight distribution entropy over time (C3).

    Higher entropy = more uniform weights (healthy).
    Lower entropy = weight concentration (degeneracy).

    Args:
        collector: Data collector with logged steps
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get data
    timesteps = collector.get_timesteps().numpy()
    entropy = collector.get_weight_entropy().numpy()
    n_particles = collector.arch_info.n_particles if collector.arch_info else 32
    max_entropy = np.log(n_particles)

    # Apply theme
    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("primary", "#1f77b4")
        lw = theme.line_widths.get("main", 2.0)
    else:
        color = "#1f77b4"
        lw = 2.0

    # Plot entropy
    ax.plot(timesteps, entropy, color=color, linewidth=lw, **kwargs)

    # Show max entropy line
    ax.axhline(
        y=max_entropy,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"Max entropy (log K = {max_entropy:.2f})",
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Weight Entropy")
    ax.set_title("Weight Distribution Entropy")
    ax.set_ylim(0, max_entropy * 1.1)
    ax.legend(loc="upper right")

    return fig, ax


def plot_particle_trajectories(
    collector: "BaseDataCollector",
    dims: Optional[List[int]] = None,
    n_particles: Optional[int] = None,
    style: str = "fan",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot state trajectories with uncertainty (C4).

    Args:
        collector: Data collector with logged steps
        dims: Which hidden dimensions to plot (default: first 3)
        n_particles: Number of particles to show in spaghetti mode
        style: "fan" (mean with quantiles), "spaghetti" (individual lines),
               or "quantiles" (percentile bands)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    # Get data
    timesteps = collector.get_timesteps().numpy()
    particles = collector.get_particles().numpy()  # [time, K, H]
    weighted_mean = collector.get_weighted_mean().numpy()  # [time, H]

    # Default dimensions
    if dims is None:
        dims = list(range(min(3, particles.shape[-1])))

    # Create subplots for each dimension
    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=(figsize[0], figsize[1] * n_dims / 3), sharex=True)
    if n_dims == 1:
        axes = [axes]

    # Apply theme and plot each dimension
    for i, (ax, dim) in enumerate(zip(axes, dims)):
        if theme:
            theme.apply_to_axes(ax)
            mean_color = theme.colors.get("mean", "#1f77b4")
            particle_color = theme.colors.get("particles", "#7f7f7f")
            alpha = theme.alpha_values.get("particle_line", 0.3)
            band_alpha = theme.alpha_values.get("confidence_band", 0.3)
        else:
            mean_color = "#1f77b4"
            particle_color = "#7f7f7f"
            alpha = 0.3
            band_alpha = 0.3

        dim_data = particles[:, :, dim]  # [time, K]

        if style == "spaghetti":
            # Plot individual particle trajectories
            n_show = n_particles if n_particles else min(10, dim_data.shape[1])
            for k in range(n_show):
                ax.plot(
                    timesteps,
                    dim_data[:, k],
                    color=particle_color,
                    alpha=alpha,
                    linewidth=0.5,
                )
            # Plot mean
            ax.plot(
                timesteps,
                weighted_mean[:, dim],
                color=mean_color,
                linewidth=2,
                label="Weighted Mean",
            )

        elif style == "quantiles":
            # Plot percentile bands
            p5 = np.percentile(dim_data, 5, axis=1)
            p25 = np.percentile(dim_data, 25, axis=1)
            p50 = np.percentile(dim_data, 50, axis=1)
            p75 = np.percentile(dim_data, 75, axis=1)
            p95 = np.percentile(dim_data, 95, axis=1)

            ax.fill_between(timesteps, p5, p95, alpha=band_alpha * 0.5, color=mean_color, label="5-95%")
            ax.fill_between(timesteps, p25, p75, alpha=band_alpha, color=mean_color, label="25-75%")
            ax.plot(timesteps, p50, color=mean_color, linewidth=2, label="Median")

        else:  # fan (default)
            # Mean with std bands
            mean = weighted_mean[:, dim]
            std = dim_data.std(axis=1)

            ax.fill_between(
                timesteps,
                mean - 2 * std,
                mean + 2 * std,
                alpha=band_alpha * 0.5,
                color=mean_color,
                label="±2σ",
            )
            ax.fill_between(
                timesteps,
                mean - std,
                mean + std,
                alpha=band_alpha,
                color=mean_color,
                label="±1σ",
            )
            ax.plot(timesteps, mean, color=mean_color, linewidth=2, label="Mean")

        ax.set_ylabel(f"Dim {dim}")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Particle Trajectories", y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_particle_diversity(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs,
) -> Tuple[Figure, List[Axes]]:
    """Multi-panel plot of diversity metrics (C5).

    Includes:
    - Average variance across dimensions
    - Average pairwise distance
    - Effective dimension
    - Collapse ratio

    Args:
        collector: Data collector with logged steps
        ax: Array of 4 axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and list of axes
    """
    _check_matplotlib()

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Get data
    timesteps = collector.get_timesteps().numpy()
    particles = collector.get_particles()  # [time, K, H]

    # Compute metrics
    variance = collector.get_particle_variance().mean(dim=-1).numpy()  # Avg variance
    pairwise = collector.get_pairwise_distances().numpy()

    # Effective dimension
    particles_np = particles.numpy()
    time_steps, K, H = particles_np.shape
    eff_dims = []
    for t in range(time_steps):
        var_per_dim = particles_np[t].var(axis=0)
        total_var = var_per_dim.sum()
        sum_sq_var = (var_per_dim ** 2).sum()
        eff_dim = total_var ** 2 / (sum_sq_var + 1e-8)
        eff_dims.append(eff_dim)
    eff_dim = np.array(eff_dims)

    # Collapse ratio
    collapse = collector.history[0].extra.get("collapse_ratio", None)
    if collapse is None:
        # Compute it
        mean_state = particles_np.mean(axis=1)
        var_mean = particles_np.var(axis=1).mean(axis=-1)
        mean_sq = (mean_state ** 2).mean(axis=-1)
        collapse = var_mean / (mean_sq + 1e-8)

    # Theme colors
    if theme:
        colors = [
            theme.colors.get("primary", "#1f77b4"),
            theme.colors.get("secondary", "#ff7f0e"),
            theme.colors.get("tertiary", "#2ca02c"),
            theme.colors.get("ess", "#9467bd"),
        ]
    else:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    # Plot 1: Variance
    axes[0].plot(timesteps, variance, color=colors[0], linewidth=2)
    axes[0].set_ylabel("Avg Variance")
    axes[0].set_title("Particle Variance")
    axes[0].set_xlabel("Timestep")

    # Plot 2: Pairwise distance
    axes[1].plot(timesteps, pairwise, color=colors[1], linewidth=2)
    axes[1].set_ylabel("Avg Pairwise Distance")
    axes[1].set_title("Particle Spread")
    axes[1].set_xlabel("Timestep")

    # Plot 3: Effective dimension
    axes[2].plot(timesteps, eff_dim, color=colors[2], linewidth=2)
    axes[2].axhline(y=H, color="gray", linestyle="--", label=f"Max ({H})")
    axes[2].set_ylabel("Effective Dimension")
    axes[2].set_title("Effective Dimension")
    axes[2].set_xlabel("Timestep")
    axes[2].legend()

    # Plot 4: Collapse ratio
    if isinstance(collapse, Tensor):
        collapse = collapse.numpy()
    axes[3].plot(timesteps, collapse, color=colors[3], linewidth=2)
    axes[3].set_ylabel("Collapse Ratio")
    axes[3].set_title("Collapse Detection")
    axes[3].set_xlabel("Timestep")

    # Apply theme
    if theme:
        for ax in axes:
            theme.apply_to_axes(ax)

    plt.tight_layout()
    return fig, list(axes)


def plot_resampling_events(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 4),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Visualize when and why resampling was triggered (C6).

    Args:
        collector: Data collector with logged steps
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get data
    timesteps = collector.get_timesteps().numpy()
    ess = collector.get_ess().numpy()
    n_particles = collector.arch_info.n_particles if collector.arch_info else ess.max()

    # Get resampling events
    resampling_events = collector.get_resampling_events()

    # Theme
    if theme:
        theme.apply_to_axes(ax)
        ess_color = theme.colors.get("ess", "#9467bd")
        event_color = theme.colors.get("resampling", "#bcbd22")
    else:
        ess_color = "#9467bd"
        event_color = "#bcbd22"

    # Plot ESS
    ax.plot(timesteps, ess, color=ess_color, linewidth=1.5, alpha=0.7, label="ESS")

    # Mark resampling events with vertical lines
    for event_time in resampling_events:
        ax.axvline(x=event_time, color=event_color, linestyle="-", alpha=0.5, linewidth=1)

    # Add event count annotation
    n_events = len(resampling_events)
    ax.text(
        0.02, 0.98,
        f"Resampling events: {n_events}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("ESS")
    ax.set_title("Resampling Events")
    ax.set_ylim(0, n_particles * 1.1)

    return fig, ax


def plot_observation_likelihoods(
    collector: "BaseDataCollector",
    mode: str = "box",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 5),
    max_timesteps: int = 50,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot observation log-likelihoods per particle (C7).

    Args:
        collector: Data collector with logged steps
        mode: "box" for box plots, "violin" for violin plots
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        max_timesteps: Maximum timesteps to show
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get log weights as proxy for likelihoods
    log_weights = collector.get_log_weights().numpy()  # [time, K]
    timesteps = collector.get_timesteps().numpy()

    # Subsample if too many timesteps
    if len(timesteps) > max_timesteps:
        indices = np.linspace(0, len(timesteps) - 1, max_timesteps, dtype=int)
        log_weights = log_weights[indices]
        timesteps = timesteps[indices]

    # Theme
    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("weights", "#d62728")
    else:
        color = "#d62728"

    if mode == "violin":
        parts = ax.violinplot(
            [log_weights[t] for t in range(len(timesteps))],
            positions=timesteps,
            showmeans=True,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    else:  # box
        bp = ax.boxplot(
            [log_weights[t] for t in range(len(timesteps))],
            positions=timesteps,
            widths=0.8 * (timesteps[1] - timesteps[0]) if len(timesteps) > 1 else 0.8,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Log Weight (proxy for log-likelihood)")
    ax.set_title("Observation Likelihoods Distribution")

    return fig, ax


def plot_numerical_health(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 5),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot numerical health indicators (C8).

    Shows NaN/Inf detection and particle norm bounds.

    Args:
        collector: Data collector with logged steps
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get health data
    health = collector.get_numerical_health()
    timesteps = collector.get_timesteps().numpy()

    has_nan = health["has_nan"].numpy().astype(float)
    has_inf = health["has_inf"].numpy().astype(float)
    max_norm = health["max_norm"].numpy()

    # Theme
    if theme:
        theme.apply_to_axes(ax)
        colors = {
            "healthy": theme.colors.get("healthy", "#2ca02c"),
            "warning": theme.colors.get("warning", "#ff7f0e"),
            "error": theme.colors.get("error", "#d62728"),
        }
    else:
        colors = {"healthy": "#2ca02c", "warning": "#ff7f0e", "error": "#d62728"}

    # Create twin axis for norm
    ax2 = ax.twinx()

    # Plot max norm
    ax2.plot(timesteps, max_norm, color=colors["warning"], linewidth=1.5, label="Max Norm")
    ax2.set_ylabel("Max Particle Norm", color=colors["warning"])

    # Mark NaN events
    nan_events = timesteps[has_nan > 0]
    if len(nan_events) > 0:
        ax.scatter(
            nan_events,
            [1] * len(nan_events),
            color=colors["error"],
            marker="x",
            s=100,
            label="NaN detected",
            zorder=5,
        )

    # Mark Inf events
    inf_events = timesteps[has_inf > 0]
    if len(inf_events) > 0:
        ax.scatter(
            inf_events,
            [0.5] * len(inf_events),
            color=colors["error"],
            marker="^",
            s=100,
            label="Inf detected",
            zorder=5,
        )

    # Health indicator
    is_healthy = (has_nan == 0) & (has_inf == 0)
    ax.fill_between(
        timesteps,
        0,
        is_healthy.astype(float),
        alpha=0.3,
        color=colors["healthy"],
        label="Healthy",
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Health Status")
    ax.set_ylim(-0.1, 1.5)
    ax.set_title("Numerical Health")
    ax.legend(loc="upper left")

    return fig, ax


def plot_weighted_output(
    collector: "BaseDataCollector",
    output_dims: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot predictions with uncertainty bands (C9).

    Args:
        collector: Data collector with logged steps
        output_dims: Which output dimensions to plot
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size if creating new figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    # Get weighted mean and variance
    mean = collector.get_weighted_mean().numpy()  # [time, H]
    variance = collector.get_weighted_variance().numpy()  # [time, H]
    std = np.sqrt(variance)
    timesteps = collector.get_timesteps().numpy()

    # Default dimensions
    if output_dims is None:
        output_dims = list(range(min(3, mean.shape[-1])))

    n_dims = len(output_dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=(figsize[0], figsize[1] * n_dims / 3), sharex=True)
    if n_dims == 1:
        axes = [axes]

    # Get observations if available
    observations = collector.get_observations()
    if observations is not None:
        observations = observations.numpy()

    for i, (ax, dim) in enumerate(zip(axes, output_dims)):
        if theme:
            theme.apply_to_axes(ax)
            color = theme.colors.get("primary", "#1f77b4")
            alpha = theme.alpha_values.get("confidence_band", 0.3)
        else:
            color = "#1f77b4"
            alpha = 0.3

        # Plot mean with uncertainty bands
        ax.fill_between(
            timesteps,
            mean[:, dim] - 2 * std[:, dim],
            mean[:, dim] + 2 * std[:, dim],
            alpha=alpha * 0.5,
            color=color,
            label="±2σ",
        )
        ax.fill_between(
            timesteps,
            mean[:, dim] - std[:, dim],
            mean[:, dim] + std[:, dim],
            alpha=alpha,
            color=color,
            label="±1σ",
        )
        ax.plot(timesteps, mean[:, dim], color=color, linewidth=2, label="Prediction")

        # Plot observations if available
        if observations is not None and dim < observations.shape[-1]:
            ax.scatter(
                timesteps,
                observations[:, dim],
                color="red",
                s=20,
                alpha=0.5,
                label="Observation",
            )

        ax.set_ylabel(f"Dim {dim}")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Weighted Output with Uncertainty", y=1.02)
    plt.tight_layout()

    return fig, axes


def animate_particles_2d(
    collector: "BaseDataCollector",
    dims: Tuple[int, int] = (0, 1),
    projection: str = "raw",
    interval: int = 100,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (8, 8),
    **kwargs,
) -> animation.FuncAnimation:
    """Animate 2D projection of particle cloud (C10).

    Args:
        collector: Data collector with logged steps
        dims: Which dimensions to project to (default: first 2)
        projection: "raw", "pca", or "tsne" (future)
        interval: Animation interval in ms
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional animation arguments

    Returns:
        Matplotlib animation object
    """
    _check_matplotlib()

    particles = collector.get_particles().numpy()  # [time, K, H]
    weights = collector.get_weights().numpy()  # [time, K]
    timesteps = collector.get_timesteps().numpy()

    # Select dimensions or project
    if projection == "raw":
        x_data = particles[:, :, dims[0]]
        y_data = particles[:, :, dims[1]]
    elif projection == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        flat_particles = particles.reshape(-1, particles.shape[-1])
        projected = pca.fit_transform(flat_particles)
        projected = projected.reshape(particles.shape[0], particles.shape[1], 2)
        x_data = projected[:, :, 0]
        y_data = projected[:, :, 1]
    else:
        x_data = particles[:, :, dims[0]]
        y_data = particles[:, :, dims[1]]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if theme:
        theme.apply_to_axes(ax)
        particle_color = theme.colors.get("primary", "#1f77b4")
    else:
        particle_color = "#1f77b4"

    # Initialize scatter plot
    scatter = ax.scatter([], [], s=50, alpha=0.6, c=particle_color)

    # Set axis limits
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    margin = 0.1
    ax.set_xlim(x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
    ax.set_ylim(y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))

    ax.set_xlabel(f"Dim {dims[0]}" if projection == "raw" else "PC 1")
    ax.set_ylabel(f"Dim {dims[1]}" if projection == "raw" else "PC 2")
    title = ax.set_title(f"Timestep 0")

    def init():
        scatter.set_offsets(np.c_[[], []])
        return scatter, title

    def update(frame):
        # Update scatter positions
        scatter.set_offsets(np.c_[x_data[frame], y_data[frame]])
        # Update sizes based on weights
        sizes = 50 + 200 * weights[frame]
        scatter.set_sizes(sizes)
        # Update title
        title.set_text(f"Timestep {int(timesteps[frame])}")
        return scatter, title

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(timesteps),
        interval=interval,
        blit=True,
    )

    return anim
