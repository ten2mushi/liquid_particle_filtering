"""State-level specific visualization plots (S1-S4).

Implements:
- S1: Noise injection magnitude
- S2: State-dependent noise visualization
- S3: Particle pairwise distances
- S4: Particle cloud evolution (3D)
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..collectors.state_collector import StateCollector
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


def plot_noise_injection_magnitude(
    collector: "StateCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 4),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot noise injection scale over time (S1).

    Shows how much exploration noise is being injected at each timestep.

    Args:
        collector: State-level data collector
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()

    # Try to get noise scale from collector
    noise_scale = collector.get_noise_scale()
    noise_magnitude = collector.get_noise_magnitude()

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("secondary", "#ff7f0e")
        lw = theme.line_widths.get("main", 2.0)
    else:
        color = "#ff7f0e"
        lw = 2.0

    if noise_magnitude is not None:
        # Plot actual noise magnitude
        ax.plot(
            timesteps,
            noise_magnitude.numpy(),
            color=color,
            linewidth=lw,
            label="Noise Magnitude (L2)",
            **kwargs,
        )
    elif noise_scale is not None:
        # Plot noise scale parameter
        if isinstance(noise_scale, Tensor):
            noise_scale = noise_scale.numpy()
        if noise_scale.ndim == 1:
            ax.plot(timesteps, noise_scale, color=color, linewidth=lw, label="Noise Scale", **kwargs)
        else:
            # Average over dimensions
            ax.plot(timesteps, noise_scale.mean(axis=-1), color=color, linewidth=lw, label="Avg Noise Scale", **kwargs)
    else:
        # Estimate from particle variance change
        particles = collector.get_particles().numpy()
        variance = particles.var(axis=1).mean(axis=-1)
        variance_diff = np.diff(variance, prepend=variance[0])
        ax.plot(timesteps, np.abs(variance_diff), color=color, linewidth=lw, label="Variance Change (proxy)", **kwargs)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Noise Magnitude")
    ax.set_title("Noise Injection Magnitude")
    ax.legend(loc="upper right")

    return fig, ax


def plot_state_dependent_noise(
    collector: "StateCollector",
    dims: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot state-dependent noise g(h) if available (S2).

    Shows how noise varies with state for state-dependent noise models.

    Args:
        collector: State-level data collector
        dims: Which dimensions to visualize (default: first 10)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    particles = collector.get_particles().numpy()  # [time, K, H]

    # Get noise scale - for state-dependent, this varies per particle
    noise_scale = collector.get_noise_scale()

    if dims is None:
        dims = list(range(min(10, particles.shape[-1])))

    if theme:
        theme.apply_to_axes(ax)
        cmap = "viridis"
    else:
        cmap = "viridis"

    if noise_scale is not None and isinstance(noise_scale, Tensor):
        noise_np = noise_scale.numpy()
        if noise_np.ndim >= 2:
            # noise_scale: [time, K, H] or [time, H]
            if noise_np.ndim == 3:
                # Average over particles, select dimensions
                noise_to_plot = noise_np[:, :, dims].mean(axis=1)  # [time, dims]
            else:
                noise_to_plot = noise_np[:, dims]  # [time, dims]

            im = ax.imshow(
                noise_to_plot.T,
                aspect="auto",
                origin="lower",
                extent=[timesteps[0], timesteps[-1], 0, len(dims)],
                cmap=cmap,
                **kwargs,
            )
            fig.colorbar(im, ax=ax, label="Noise Scale")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Dimension Index")
            ax.set_title("State-Dependent Noise (g(h))")
        else:
            # Scalar noise - just plot line
            ax.plot(timesteps, noise_np, linewidth=2)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Noise Scale")
            ax.set_title("Noise Scale (Constant)")
    else:
        # Estimate from particle spread
        particle_std = particles[:, :, dims].std(axis=1)  # [time, dims]
        im = ax.imshow(
            particle_std.T,
            aspect="auto",
            origin="lower",
            extent=[timesteps[0], timesteps[-1], 0, len(dims)],
            cmap=cmap,
            **kwargs,
        )
        fig.colorbar(im, ax=ax, label="Particle Std (proxy)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Dimension Index")
        ax.set_title("Particle Spread by Dimension")

    return fig, ax


def plot_particle_pairwise_distances(
    collector: "StateCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    show_distribution: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot pairwise distances between particles for collapse detection (S3).

    Low pairwise distances indicate particle collapse.

    Args:
        collector: State-level data collector
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        show_distribution: Whether to show distribution over time
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    pairwise_distances = collector.get_pairwise_distances().numpy()

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("tertiary", "#2ca02c")
        warning_color = theme.colors.get("warning", "#ff7f0e")
        lw = theme.line_widths.get("main", 2.0)
    else:
        color = "#2ca02c"
        warning_color = "#ff7f0e"
        lw = 2.0

    # Plot mean pairwise distance
    ax.plot(timesteps, pairwise_distances, color=color, linewidth=lw, label="Avg Pairwise Distance", **kwargs)

    # Mark potential collapse (low distance)
    collapse_threshold = pairwise_distances.max() * 0.1
    collapse_mask = pairwise_distances < collapse_threshold
    if collapse_mask.any():
        ax.scatter(
            timesteps[collapse_mask],
            pairwise_distances[collapse_mask],
            color=warning_color,
            marker="x",
            s=50,
            zorder=5,
            label="Potential Collapse",
        )

    ax.axhline(y=collapse_threshold, color=warning_color, linestyle="--", alpha=0.5, label="Collapse Threshold (10%)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Pairwise Distance")
    ax.set_title("Particle Pairwise Distances (Collapse Detection)")
    ax.legend(loc="upper right")

    # Add collapse ratio on secondary axis
    ax2 = ax.twinx()
    collapse_ratio = collector.get_collapse_ratio().numpy()
    ax2.plot(timesteps, collapse_ratio, color="gray", alpha=0.5, linestyle=":", label="Collapse Ratio")
    ax2.set_ylabel("Collapse Ratio", color="gray")

    return fig, ax


def plot_particle_cloud_evolution(
    collector: "StateCollector",
    dims: Tuple[int, int, int] = (0, 1, 2),
    n_timesteps: int = 10,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot 3D particle cloud evolution over time (S4).

    Shows how the particle cloud moves through state space.

    Args:
        collector: State-level data collector
        dims: Which 3 dimensions to project onto
        n_timesteps: Number of timesteps to show
        ax: Matplotlib 3D axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and 3D axes
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    particles = collector.get_particles().numpy()  # [time, K, H]
    weights = collector.get_weights().numpy()  # [time, K]
    timesteps = collector.get_timesteps().numpy()

    # Select timesteps to visualize
    total_steps = len(timesteps)
    if total_steps <= n_timesteps:
        indices = list(range(total_steps))
    else:
        indices = np.linspace(0, total_steps - 1, n_timesteps, dtype=int)

    # Color map for time
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for i, (idx, color) in enumerate(zip(indices, colors)):
        x = particles[idx, :, dims[0]]
        y = particles[idx, :, dims[1]]
        z = particles[idx, :, dims[2]]

        # Size based on weights
        sizes = 20 + 100 * weights[idx]

        ax.scatter(
            x, y, z,
            c=[color],
            s=sizes,
            alpha=0.6,
            label=f"t={int(timesteps[idx])}" if i % max(1, len(indices) // 5) == 0 else None,
        )

        # Draw lines connecting centroids
        if i > 0:
            prev_idx = indices[i - 1]
            prev_centroid = particles[prev_idx].mean(axis=0)
            curr_centroid = particles[idx].mean(axis=0)
            ax.plot(
                [prev_centroid[dims[0]], curr_centroid[dims[0]]],
                [prev_centroid[dims[1]], curr_centroid[dims[1]]],
                [prev_centroid[dims[2]], curr_centroid[dims[2]]],
                color="gray",
                alpha=0.5,
                linewidth=1,
            )

    ax.set_xlabel(f"Dim {dims[0]}")
    ax.set_ylabel(f"Dim {dims[1]}")
    ax.set_zlabel(f"Dim {dims[2]}")
    ax.set_title("Particle Cloud Evolution (3D)")
    ax.legend(loc="upper left", fontsize=8)

    return fig, ax


def animate_particle_cloud_3d(
    collector: "StateCollector",
    dims: Tuple[int, int, int] = (0, 1, 2),
    interval: int = 100,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> animation.FuncAnimation:
    """Animate 3D particle cloud evolution (S4 animated version).

    Args:
        collector: State-level data collector
        dims: Which 3 dimensions to project onto
        interval: Animation interval in ms
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional arguments

    Returns:
        Matplotlib animation object
    """
    _check_matplotlib()

    particles = collector.get_particles().numpy()  # [time, K, H]
    weights = collector.get_weights().numpy()  # [time, K]
    timesteps = collector.get_timesteps().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Set axis limits
    x_data = particles[:, :, dims[0]]
    y_data = particles[:, :, dims[1]]
    z_data = particles[:, :, dims[2]]

    margin = 0.1
    ax.set_xlim(x_data.min() - margin * np.ptp(x_data), x_data.max() + margin * np.ptp(x_data))
    ax.set_ylim(y_data.min() - margin * np.ptp(y_data), y_data.max() + margin * np.ptp(y_data))
    ax.set_zlim(z_data.min() - margin * np.ptp(z_data), z_data.max() + margin * np.ptp(z_data))

    ax.set_xlabel(f"Dim {dims[0]}")
    ax.set_ylabel(f"Dim {dims[1]}")
    ax.set_zlabel(f"Dim {dims[2]}")

    scatter = ax.scatter([], [], [], s=50, alpha=0.6)
    title = ax.set_title("Timestep 0")

    def init():
        scatter._offsets3d = ([], [], [])
        return scatter, title

    def update(frame):
        x = particles[frame, :, dims[0]]
        y = particles[frame, :, dims[1]]
        z = particles[frame, :, dims[2]]

        scatter._offsets3d = (x, y, z)
        sizes = 20 + 100 * weights[frame]
        scatter.set_sizes(sizes)

        title.set_text(f"Timestep {int(timesteps[frame])}")
        return scatter, title

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(timesteps),
        interval=interval,
        blit=False,  # 3D doesn't support blit
    )

    return anim
