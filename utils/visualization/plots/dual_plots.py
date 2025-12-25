"""Dual particle filter specific visualization plots (D1-D4).

Implements:
- D1: Joint state-parameter scatter
- D2: Rao-Blackwell variance comparison
- D3: State-parameter correlation
- D4: Marginal posteriors (state and param)
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
    from ..collectors.dual_collector import DualCollector
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


def plot_joint_state_param_scatter(
    collector: "DualCollector",
    state_dim: int = 0,
    param_dim: int = 0,
    timestep: int = -1,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot joint state-parameter particle distribution (D1).

    Shows 2D projection of joint (state, param) particles.

    Args:
        collector: Dual data collector
        state_dim: Which state dimension to plot
        param_dim: Which parameter dimension to plot
        timestep: Which timestep to visualize (-1 for last)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get data
    state_particles = collector.get_particles()  # [time, K, H]
    param_particles = collector.get_param_particles()
    weights = collector.get_weights()

    if param_particles is None:
        ax.text(0.5, 0.5, "No parameter particles collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    state_particles = state_particles.numpy()
    param_particles = param_particles.numpy()
    weights = weights.numpy()

    # Select timestep
    if timestep < 0:
        timestep = state_particles.shape[0] + timestep
    timestep = min(timestep, state_particles.shape[0] - 1)

    state_t = state_particles[timestep, :, state_dim]  # [K]
    param_t = param_particles[timestep, :, param_dim]  # [K]
    weights_t = weights[timestep]  # [K]

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("primary", "#1f77b4")
    else:
        color = "#1f77b4"

    # Scatter plot with size based on weight
    sizes = 50 + 200 * weights_t
    scatter = ax.scatter(
        state_t,
        param_t,
        c=weights_t,
        s=sizes,
        cmap="viridis",
        alpha=0.7,
        **kwargs,
    )
    fig.colorbar(scatter, ax=ax, label="Particle Weight")

    # Mark weighted mean
    state_mean = (weights_t * state_t).sum()
    param_mean = (weights_t * param_t).sum()
    ax.scatter(state_mean, param_mean, color="red", marker="x", s=200, linewidths=3, zorder=10, label="Weighted Mean")

    ax.set_xlabel(f"State (dim {state_dim})")
    ax.set_ylabel(f"Parameter (dim {param_dim})")
    ax.set_title(f"Joint State-Parameter Distribution (t={timestep})")
    ax.legend(loc="upper right")

    return fig, ax


def plot_rao_blackwell_variance(
    collector: "DualCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 5),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot Rao-Blackwell variance reduction comparison (D2).

    Shows variance before and after Rao-Blackwellization,
    and the variance reduction ratio over time.

    Args:
        collector: Dual data collector
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get Rao-Blackwell variance data
    rb_data = collector.get_rao_blackwell_variance()
    timesteps = collector.get_timesteps().numpy()

    if rb_data is None:
        # If no RB data, compute variance reduction estimate
        ax.text(0.5, 0.5, "No Rao-Blackwell variance data available.\nShowing state variance instead.",
                ha="center", va="center", transform=ax.transAxes)

        # Fall back to showing state variance
        variance = collector.get_weighted_variance().mean(dim=-1).numpy()
        ax.plot(timesteps, variance, linewidth=2, label="State Variance")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Variance")
        ax.set_title("State Variance (Rao-Blackwell data not available)")
        ax.legend()
        return fig, ax

    before = rb_data["before"].mean(dim=-1).numpy()  # Average over dimensions
    after = rb_data["after"].mean(dim=-1).numpy()
    reduction = rb_data["reduction"].numpy()

    if theme:
        theme.apply_to_axes(ax)
        color1 = theme.colors.get("warning", "#ff7f0e")
        color2 = theme.colors.get("healthy", "#2ca02c")
    else:
        color1 = "#ff7f0e"
        color2 = "#2ca02c"

    # Plot before and after
    ax.plot(timesteps, before, color=color1, linewidth=2, label="Before RB", linestyle="--")
    ax.plot(timesteps, after, color=color2, linewidth=2, label="After RB")

    # Fill between to show reduction
    ax.fill_between(timesteps, after, before, alpha=0.3, color=color2, label="Reduction")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Variance")
    ax.set_title("Rao-Blackwell Variance Reduction")
    ax.legend(loc="upper right")

    # Add secondary axis for reduction percentage
    ax2 = ax.twinx()
    ax2.plot(timesteps, reduction * 100, color="gray", alpha=0.5, linestyle=":", label="Reduction %")
    ax2.set_ylabel("Reduction %", color="gray")
    ax2.set_ylim(0, 100)

    return fig, ax


def plot_state_param_correlation(
    collector: "DualCollector",
    timestep: int = -1,
    max_dims: int = 10,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot state-parameter cross-correlation matrix (D3).

    Shows correlation between state dimensions and parameter dimensions.

    Args:
        collector: Dual data collector
        timestep: Which timestep to visualize (-1 for last)
        max_dims: Maximum dimensions to show per axis
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    # Get correlation
    correlation = collector.get_state_param_correlation()

    if correlation is None:
        ax.text(0.5, 0.5, "No state-parameter correlation data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    correlation = correlation.numpy()  # [time, H, P]

    # Select timestep
    if timestep < 0:
        timestep = correlation.shape[0] + timestep
    timestep = min(timestep, correlation.shape[0] - 1)

    corr_t = correlation[timestep]  # [H, P]

    # Limit dimensions for display
    H, P = corr_t.shape
    corr_t = corr_t[:min(max_dims, H), :min(max_dims, P)]

    if theme:
        theme.apply_to_axes(ax)

    # Plot heatmap
    im = ax.imshow(corr_t, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Correlation")

    # Labels
    ax.set_xlabel("Parameter Dimension")
    ax.set_ylabel("State Dimension")
    ax.set_title(f"State-Parameter Correlation (t={timestep})")

    ax.set_xticks(range(corr_t.shape[1]))
    ax.set_yticks(range(corr_t.shape[0]))
    ax.set_xticklabels([f"p{i}" for i in range(corr_t.shape[1])], fontsize=8)
    ax.set_yticklabels([f"h{i}" for i in range(corr_t.shape[0])], fontsize=8)

    return fig, ax


def plot_marginal_posteriors(
    collector: "DualCollector",
    state_dims: Optional[List[int]] = None,
    param_dims: Optional[List[int]] = None,
    timestep: int = -1,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (14, 6),
    **kwargs,
) -> Tuple[Figure, List[Axes]]:
    """Plot marginalized state and parameter posteriors side by side (D4).

    Args:
        collector: Dual data collector
        state_dims: Which state dimensions to plot
        param_dims: Which parameter dimensions to plot
        timestep: Which timestep to visualize (-1 for last)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and list of axes
    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get data
    state_particles = collector.get_particles().numpy()  # [time, K, H]
    param_particles = collector.get_param_particles()
    weights = collector.get_weights().numpy()

    # Select timestep
    if timestep < 0:
        timestep = state_particles.shape[0] + timestep
    timestep = min(timestep, state_particles.shape[0] - 1)

    state_t = state_particles[timestep]  # [K, H]
    weights_t = weights[timestep]

    # Default dimensions
    if state_dims is None:
        state_dims = list(range(min(5, state_t.shape[1])))

    if theme:
        theme.apply_to_axes(ax1)
        theme.apply_to_axes(ax2)
        color1 = theme.colors.get("primary", "#1f77b4")
        color2 = theme.colors.get("secondary", "#ff7f0e")
    else:
        color1 = "#1f77b4"
        color2 = "#ff7f0e"

    # Plot state marginals
    state_data = [state_t[:, d] for d in state_dims if d < state_t.shape[1]]
    if state_data:
        parts1 = ax1.violinplot(state_data, positions=range(len(state_data)), showmeans=True)
        for pc in parts1["bodies"]:
            pc.set_facecolor(color1)
            pc.set_alpha(0.7)

    ax1.set_xticks(range(len(state_dims)))
    ax1.set_xticklabels([f"h{d}" for d in state_dims], fontsize=10)
    ax1.set_ylabel("Value")
    ax1.set_title(f"State Marginal Posteriors (t={timestep})")

    # Plot param marginals
    if param_particles is not None:
        param_particles = param_particles.numpy()
        param_t = param_particles[timestep]  # [K, P]

        if param_dims is None:
            param_dims = list(range(min(5, param_t.shape[1])))

        param_data = [param_t[:, d] for d in param_dims if d < param_t.shape[1]]
        if param_data:
            parts2 = ax2.violinplot(param_data, positions=range(len(param_data)), showmeans=True)
            for pc in parts2["bodies"]:
                pc.set_facecolor(color2)
                pc.set_alpha(0.7)

        ax2.set_xticks(range(len(param_dims)))
        ax2.set_xticklabels([f"p{d}" for d in param_dims], fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No parameter particles", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_ylabel("Value")
    ax2.set_title(f"Parameter Marginal Posteriors (t={timestep})")

    plt.tight_layout()
    return fig, [ax1, ax2]


def plot_joint_evolution(
    collector: "DualCollector",
    state_dim: int = 0,
    param_dim: int = 0,
    n_timesteps: int = 10,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot evolution of joint distribution over time.

    Shows how the joint state-parameter distribution evolves.

    Args:
        collector: Dual data collector
        state_dim: Which state dimension to plot
        param_dim: Which parameter dimension to plot
        n_timesteps: Number of timesteps to show
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    state_particles = collector.get_particles().numpy()
    param_particles = collector.get_param_particles()
    timesteps = collector.get_timesteps().numpy()

    if param_particles is None:
        ax.text(0.5, 0.5, "No parameter particles", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    param_particles = param_particles.numpy()

    # Select timesteps
    total_steps = len(timesteps)
    if total_steps <= n_timesteps:
        indices = list(range(total_steps))
    else:
        indices = np.linspace(0, total_steps - 1, n_timesteps, dtype=int)

    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    if theme:
        theme.apply_to_axes(ax)

    # Plot joint distribution at each timestep
    for i, (idx, color) in enumerate(zip(indices, colors)):
        state_t = state_particles[idx, :, state_dim]
        param_t = param_particles[idx, :, param_dim]

        ax.scatter(
            state_t,
            param_t,
            c=[color],
            s=30,
            alpha=0.5,
            label=f"t={int(timesteps[idx])}" if i % max(1, len(indices) // 5) == 0 else None,
        )

        # Draw ellipse for 1-std contour (approximation)
        state_mean, param_mean = state_t.mean(), param_t.mean()
        state_std, param_std = state_t.std(), param_t.std()

        from matplotlib.patches import Ellipse
        ellipse = Ellipse(
            (state_mean, param_mean),
            width=2 * state_std,
            height=2 * param_std,
            fill=False,
            color=color,
            alpha=0.7,
            linewidth=1,
        )
        ax.add_patch(ellipse)

    ax.set_xlabel(f"State (dim {state_dim})")
    ax.set_ylabel(f"Parameter (dim {param_dim})")
    ax.set_title("Joint Distribution Evolution")
    ax.legend(loc="upper right", fontsize=8)

    return fig, ax
