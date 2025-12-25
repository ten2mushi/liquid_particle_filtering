"""Parameter-level specific visualization plots (P1-P5).

Implements:
- P1: Parameter posterior marginals
- P2: Parameter uncertainty timeline
- P3: Parameter correlation matrix
- P4: Tracked vs base parameters
- P5: Parameter evolution trajectory
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..collectors.param_collector import ParamCollector
    from ..core.themes import Theme


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")


def _create_figure(ax=None, figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, Axes]:
    _check_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_param_posterior_marginals(
    collector: "ParamCollector",
    param_names: Optional[List[str]] = None,
    timestep: int = -1,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot per-parameter posterior distributions (P1).

    Shows violin or KDE plots of parameter particles at a given timestep.

    Args:
        collector: Parameter-level data collector
        param_names: Which parameters to plot (default: all)
        timestep: Which timestep to visualize (-1 for last)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get parameter particles
    param_particles = collector.get_param_particles()
    if param_particles is None:
        ax.text(0.5, 0.5, "No parameter particles collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    param_particles = param_particles.numpy()  # [time, K, P]
    weights = collector.get_weights().numpy()  # [time, K]

    # Select timestep
    if timestep < 0:
        timestep = param_particles.shape[0] + timestep
    timestep = min(timestep, param_particles.shape[0] - 1)

    particles_t = param_particles[timestep]  # [K, P]
    weights_t = weights[timestep]  # [K]

    # Get param names
    all_param_names = collector.get_param_names()
    if all_param_names is None:
        all_param_names = [f"param_{i}" for i in range(particles_t.shape[1])]

    if param_names is None:
        param_names = all_param_names[:min(10, len(all_param_names))]  # Limit to 10

    # Get indices of selected parameters
    param_indices = [all_param_names.index(n) if n in all_param_names else i for i, n in enumerate(param_names)]

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("primary", "#1f77b4")
    else:
        color = "#1f77b4"

    # Create violin plot
    data = [particles_t[:, idx] for idx in param_indices if idx < particles_t.shape[1]]
    positions = range(len(data))

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels([param_names[i] for i in range(len(data))], rotation=45, ha="right")
    ax.set_ylabel("Parameter Value")
    ax.set_title(f"Parameter Posterior Marginals (t={timestep})")

    plt.tight_layout()
    return fig, ax


def plot_param_uncertainty_timeline(
    collector: "ParamCollector",
    param_names: Optional[List[str]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot parameter uncertainty evolution over time (P2).

    Shows how parameter posterior uncertainty changes during inference.

    Args:
        collector: Parameter-level data collector
        param_names: Which parameters to plot (default: first 5)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get data
    param_particles = collector.get_param_particles()
    if param_particles is None:
        ax.text(0.5, 0.5, "No parameter particles collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    timesteps = collector.get_timesteps().numpy()
    stats = collector.get_param_posterior_stats()

    if stats is None:
        ax.text(0.5, 0.5, "Could not compute posterior stats", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    mean = stats["mean"].numpy()  # [time, P]
    std = stats["std"].numpy()  # [time, P]

    # Get param names
    all_param_names = collector.get_param_names()
    if all_param_names is None:
        all_param_names = [f"param_{i}" for i in range(mean.shape[1])]

    if param_names is None:
        param_names = all_param_names[:min(5, len(all_param_names))]

    param_indices = [all_param_names.index(n) if n in all_param_names else i for i, n in enumerate(param_names)]

    if theme:
        theme.apply_to_axes(ax)
        colors = theme.get_particle_colors(len(param_indices))
        alpha = theme.alpha_values.get("confidence_band", 0.3)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_indices)))
        alpha = 0.3

    # Plot each parameter with uncertainty bands
    for i, (idx, color) in enumerate(zip(param_indices, colors)):
        if idx >= mean.shape[1]:
            continue
        name = param_names[i] if i < len(param_names) else f"param_{idx}"

        ax.fill_between(
            timesteps,
            mean[:, idx] - 2 * std[:, idx],
            mean[:, idx] + 2 * std[:, idx],
            alpha=alpha * 0.5,
            color=color,
        )
        ax.fill_between(
            timesteps,
            mean[:, idx] - std[:, idx],
            mean[:, idx] + std[:, idx],
            alpha=alpha,
            color=color,
        )
        ax.plot(timesteps, mean[:, idx], color=color, linewidth=2, label=name)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Parameter Uncertainty Evolution")
    ax.legend(loc="upper right", fontsize=8)

    return fig, ax


def plot_param_correlation_matrix(
    collector: "ParamCollector",
    timestep: int = -1,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot inter-parameter correlation matrix (P3).

    Shows correlation between tracked parameters at a given timestep.

    Args:
        collector: Parameter-level data collector
        timestep: Which timestep to visualize (-1 for last)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get correlation matrix
    correlation = collector.get_param_correlation()
    if correlation is None:
        ax.text(0.5, 0.5, "No parameter correlation data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    correlation = correlation.numpy()  # [time, P, P]

    # Select timestep
    if timestep < 0:
        timestep = correlation.shape[0] + timestep
    timestep = min(timestep, correlation.shape[0] - 1)

    corr_t = correlation[timestep]  # [P, P]

    # Get param names
    all_param_names = collector.get_param_names()
    if all_param_names is None:
        all_param_names = [f"p{i}" for i in range(corr_t.shape[0])]

    # Limit to reasonable size for display
    max_params = min(20, corr_t.shape[0])
    corr_t = corr_t[:max_params, :max_params]
    param_labels = all_param_names[:max_params]

    if theme:
        theme.apply_to_axes(ax)

    # Plot heatmap
    im = ax.imshow(corr_t, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Correlation")

    # Set labels
    ax.set_xticks(range(len(param_labels)))
    ax.set_yticks(range(len(param_labels)))
    ax.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(param_labels, fontsize=8)

    ax.set_title(f"Parameter Correlation Matrix (t={timestep})")

    plt.tight_layout()
    return fig, ax


def plot_tracked_vs_base_params(
    collector: "ParamCollector",
    param_names: Optional[List[str]] = None,
    base_params: Optional[dict] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot particle parameter deviations from base (P4).

    Shows how parameter particles deviate from the base (initial) parameters.

    Args:
        collector: Parameter-level data collector
        param_names: Which parameters to plot
        base_params: Base parameter values (if not stored in collector)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    param_particles = collector.get_param_particles()
    if param_particles is None:
        ax.text(0.5, 0.5, "No parameter particles collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    param_particles = param_particles.numpy()  # [time, K, P]

    # Use first timestep as "base" if not provided
    if base_params is None:
        base = param_particles[0].mean(axis=0)  # [P]
    else:
        # Convert dict to array (assuming order matches)
        base = np.array(list(base_params.values()))

    # Get final timestep particles
    final_particles = param_particles[-1]  # [K, P]
    deviation = final_particles - base  # [K, P]

    # Get param names
    all_param_names = collector.get_param_names()
    if all_param_names is None:
        all_param_names = [f"param_{i}" for i in range(deviation.shape[1])]

    if param_names is None:
        param_names = all_param_names[:min(10, len(all_param_names))]

    param_indices = [all_param_names.index(n) if n in all_param_names else i for i, n in enumerate(param_names)]

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("primary", "#1f77b4")
    else:
        color = "#1f77b4"

    # Box plot of deviations
    data = [deviation[:, idx] for idx in param_indices if idx < deviation.shape[1]]
    positions = range(len(data))

    bp = ax.boxplot(data, positions=positions, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Zero line
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([param_names[i] for i in range(len(data))], rotation=45, ha="right")
    ax.set_ylabel("Deviation from Base")
    ax.set_title("Parameter Particle Deviations from Base")

    plt.tight_layout()
    return fig, ax


def plot_param_evolution_trajectory(
    collector: "ParamCollector",
    param_pair: Tuple[str, str] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 8),
    n_particles_show: int = 5,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot 2D parameter space evolution trajectory (P5).

    Shows how parameter particles evolve in a 2D projection.

    Args:
        collector: Parameter-level data collector
        param_pair: Tuple of two parameter names to plot
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        n_particles_show: Number of particle trajectories to show
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    param_particles = collector.get_param_particles()
    if param_particles is None:
        ax.text(0.5, 0.5, "No parameter particles collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    param_particles = param_particles.numpy()  # [time, K, P]
    timesteps = collector.get_timesteps().numpy()

    # Get param names and indices
    all_param_names = collector.get_param_names()
    if all_param_names is None:
        all_param_names = [f"param_{i}" for i in range(param_particles.shape[-1])]

    if param_pair is None:
        # Use first two parameters
        idx1, idx2 = 0, min(1, param_particles.shape[-1] - 1)
        name1, name2 = all_param_names[idx1], all_param_names[idx2]
    else:
        name1, name2 = param_pair
        idx1 = all_param_names.index(name1) if name1 in all_param_names else 0
        idx2 = all_param_names.index(name2) if name2 in all_param_names else 1

    x_data = param_particles[:, :, idx1]  # [time, K]
    y_data = param_particles[:, :, idx2]  # [time, K]

    if theme:
        theme.apply_to_axes(ax)
        colors = theme.get_particle_colors(n_particles_show)
        alpha = theme.alpha_values.get("particle_line", 0.3)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, n_particles_show))
        alpha = 0.3

    # Plot trajectories for selected particles
    n_show = min(n_particles_show, param_particles.shape[1])
    for k in range(n_show):
        ax.plot(
            x_data[:, k],
            y_data[:, k],
            color=colors[k],
            alpha=alpha,
            linewidth=1,
        )
        # Mark start and end
        ax.scatter(x_data[0, k], y_data[0, k], color=colors[k], marker="o", s=50, zorder=5)
        ax.scatter(x_data[-1, k], y_data[-1, k], color=colors[k], marker="s", s=50, zorder=5)

    # Plot mean trajectory
    mean_x = x_data.mean(axis=1)
    mean_y = y_data.mean(axis=1)
    ax.plot(mean_x, mean_y, color="black", linewidth=2, label="Mean Trajectory", zorder=10)
    ax.scatter(mean_x[0], mean_y[0], color="black", marker="o", s=100, zorder=10, label="Start")
    ax.scatter(mean_x[-1], mean_y[-1], color="black", marker="s", s=100, zorder=10, label="End")

    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_title("Parameter Evolution Trajectory")
    ax.legend(loc="upper right")

    return fig, ax
