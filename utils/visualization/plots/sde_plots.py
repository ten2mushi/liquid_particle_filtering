"""SDE-specific visualization plots (E1-E6).

Implements:
- E1: Diffusion magnitude g(h) over time
- E2: Drift/diffusion ratio f(h)/g(h)
- E3: Per-unfold convergence analysis
- E4: Brownian increment distribution
- E5: State clamping event detection
- E6: Euler-Maruyama stability metrics
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
    from ..collectors.sde_collector import SDECollector
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


def plot_diffusion_magnitude(
    collector: "SDECollector",
    dims: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot diffusion coefficient g(h) over time (E1).

    Shows how the diffusion magnitude evolves during inference,
    which controls the noise injection in the SDE.

    Args:
        collector: SDE data collector
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
    diffusion = collector.get_diffusion_values()

    if diffusion is None:
        ax.text(0.5, 0.5, "No diffusion data collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    diffusion = diffusion.numpy()  # [time, K, H] or [time, H]

    if dims is None:
        if diffusion.ndim == 3:
            dims = list(range(min(10, diffusion.shape[-1])))
        else:
            dims = list(range(min(10, diffusion.shape[-1])))

    if theme:
        theme.apply_to_axes(ax)
        cmap = "viridis"
    else:
        cmap = "viridis"

    if diffusion.ndim == 3:
        # Average over particles, select dimensions
        diff_to_plot = diffusion[:, :, dims].mean(axis=1)  # [time, dims]
    else:
        diff_to_plot = diffusion[:, dims]  # [time, dims]

    # Heatmap visualization
    im = ax.imshow(
        diff_to_plot.T,
        aspect="auto",
        origin="lower",
        extent=[timesteps[0], timesteps[-1], 0, len(dims)],
        cmap=cmap,
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="Diffusion g(h)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Dimension Index")
    ax.set_title("Diffusion Coefficient Evolution g(h)")

    # Set dimension tick labels
    ax.set_yticks(np.arange(len(dims)) + 0.5)
    ax.set_yticklabels([f"d{d}" for d in dims], fontsize=8)

    return fig, ax


def plot_drift_diffusion_ratio(
    collector: "SDECollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 5),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot drift/diffusion ratio f(h)/g(h) balance (E2).

    The ratio indicates whether drift (deterministic) or diffusion
    (stochastic) dominates the SDE dynamics.

    Args:
        collector: SDE data collector
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    ratio = collector.get_drift_diffusion_ratio()

    if ratio is None:
        # Try to compute from raw values
        drift = collector.get_drift_values()
        diffusion = collector.get_diffusion_values()

        if drift is None or diffusion is None:
            ax.text(0.5, 0.5, "No drift/diffusion data available", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        drift = drift.numpy()
        diffusion = diffusion.numpy()

        # Compute ratio (mean over particles and dimensions)
        if drift.ndim == 3:
            drift_mag = np.abs(drift).mean(axis=(1, 2))  # [time]
            diff_mag = np.abs(diffusion).mean(axis=(1, 2))  # [time]
        else:
            drift_mag = np.abs(drift).mean(axis=-1)  # [time]
            diff_mag = np.abs(diffusion).mean(axis=-1)  # [time]

        # Avoid division by zero
        diff_mag = np.maximum(diff_mag, 1e-10)
        ratio = drift_mag / diff_mag
    else:
        ratio = ratio.numpy()

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("primary", "#1f77b4")
        warning_color = theme.colors.get("warning", "#ff7f0e")
        lw = theme.line_widths.get("main", 2.0)
    else:
        color = "#1f77b4"
        warning_color = "#ff7f0e"
        lw = 2.0

    # Plot ratio
    ax.plot(timesteps, ratio, color=color, linewidth=lw, label="|f(h)|/|g(h)|", **kwargs)

    # Reference lines
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Balanced (ratio=1)")
    ax.axhline(y=10.0, color=warning_color, linestyle=":", alpha=0.5, label="Drift dominant (>10)")
    ax.axhline(y=0.1, color=warning_color, linestyle=":", alpha=0.5, label="Diffusion dominant (<0.1)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Drift/Diffusion Ratio")
    ax.set_title("Drift vs Diffusion Balance")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)

    # Fill regions to show dominance
    ax.fill_between(timesteps, ratio, 1.0, where=ratio > 1.0, alpha=0.2, color="blue", label="_nolegend_")
    ax.fill_between(timesteps, ratio, 1.0, where=ratio < 1.0, alpha=0.2, color="green", label="_nolegend_")

    return fig, ax


def plot_unfold_convergence(
    collector: "SDECollector",
    dims: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot per-unfold state change for convergence analysis (E3).

    Shows how much the state changes at each ODE unfold step,
    indicating convergence of the numerical integration.

    Args:
        collector: SDE data collector
        dims: Which state dimensions to track
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    per_unfold_states = collector.get_per_unfold_states()

    if per_unfold_states is None:
        ax.text(0.5, 0.5, "No per-unfold state data collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    per_unfold_states = per_unfold_states.numpy()  # [time, unfolds, K, H] or [time, unfolds, H]

    if dims is None:
        if per_unfold_states.ndim == 4:
            dims = list(range(min(5, per_unfold_states.shape[-1])))
        else:
            dims = list(range(min(5, per_unfold_states.shape[-1])))

    if theme:
        theme.apply_to_axes(ax)
        colors = theme.get_particle_colors(len(dims))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))

    # Compute per-unfold change (difference between consecutive unfolds)
    if per_unfold_states.ndim == 4:
        # Average over particles: [time, unfolds, K, H] -> [time, unfolds, H]
        states_avg = per_unfold_states.mean(axis=2)
    else:
        states_avg = per_unfold_states

    # Get number of unfolds
    n_unfolds = states_avg.shape[1]
    unfold_indices = np.arange(n_unfolds)

    # Average over time for overall convergence pattern
    states_time_avg = states_avg.mean(axis=0)  # [unfolds, H]

    # Compute deltas between consecutive unfolds
    deltas = np.diff(states_time_avg, axis=0)  # [unfolds-1, H]
    delta_magnitudes = np.abs(deltas)  # [unfolds-1, H]

    for i, (dim, color) in enumerate(zip(dims, colors)):
        if dim >= delta_magnitudes.shape[1]:
            continue
        ax.plot(
            unfold_indices[:-1],
            delta_magnitudes[:, dim],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"Dim {dim}",
        )

    ax.set_xlabel("Unfold Step")
    ax.set_ylabel("State Change Magnitude")
    ax.set_title("Per-Unfold Convergence (State Change)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")

    # Add convergence threshold line
    if delta_magnitudes.size > 0:
        threshold = delta_magnitudes.mean() * 0.1
        ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label="Convergence threshold")

    return fig, ax


def plot_brownian_increments(
    collector: "SDECollector",
    timestep: int = -1,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot Brownian increment dW distribution (E4).

    Shows the distribution of noise increments, which should
    follow N(0, sqrt(dt)) for correct Euler-Maruyama integration.

    Args:
        collector: SDE data collector
        timestep: Which timestep to analyze (-1 for last)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    brownian = collector.get_brownian_increments()

    if brownian is None:
        ax.text(0.5, 0.5, "No Brownian increment data collected", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    brownian = brownian.numpy()  # [time, K, H] or similar

    # Select timestep
    if timestep < 0:
        timestep = brownian.shape[0] + timestep
    timestep = min(timestep, brownian.shape[0] - 1)

    dw_t = brownian[timestep]  # [K, H] or [H]
    if dw_t.ndim == 2:
        dw_flat = dw_t.flatten()
    else:
        dw_flat = dw_t

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("primary", "#1f77b4")
    else:
        color = "#1f77b4"

    # Histogram of increments
    n, bins, patches = ax.hist(
        dw_flat, bins=50, density=True, alpha=0.7, color=color, label="Observed dW"
    )

    # Overlay theoretical N(0, sigma) distribution
    dt = collector.get_dt() if hasattr(collector, "get_dt") else 0.01
    sigma = np.sqrt(dt)
    x = np.linspace(dw_flat.min(), dw_flat.max(), 100)
    theoretical = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)
    ax.plot(x, theoretical, "r--", linewidth=2, label=f"N(0, sqrt(dt)), dt={dt:.4f}")

    # Statistics
    mean = dw_flat.mean()
    std = dw_flat.std()
    ax.axvline(x=mean, color="green", linestyle="-", alpha=0.7, label=f"Mean={mean:.4f}")

    ax.set_xlabel("dW (Brownian Increment)")
    ax.set_ylabel("Density")
    ax.set_title(f"Brownian Increment Distribution (t={timestep})")
    ax.legend(loc="upper right")

    # Add text with statistics
    stats_text = f"Observed:\nmean={mean:.4f}\nstd={std:.4f}\n\nExpected:\nmean=0\nstd={sigma:.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    return fig, ax


def plot_state_clamping_events(
    collector: "SDECollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 5),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot state clamping/clipping events over time (E5).

    Shows when states hit bounds and are clamped, which can
    indicate numerical instability or inappropriate bounds.

    Args:
        collector: SDE data collector
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    clamping_events = collector.get_clamping_events()

    if clamping_events is None:
        # Try to detect from particles exceeding bounds
        particles = collector.get_particles().numpy()  # [time, K, H]
        bounds = collector.get_state_bounds()

        if bounds is None:
            ax.text(0.5, 0.5, "No clamping event data available", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        lower, upper = bounds
        # Count particles at bounds per timestep
        at_lower = (np.abs(particles - lower) < 1e-6).sum(axis=(1, 2))  # [time]
        at_upper = (np.abs(particles - upper) < 1e-6).sum(axis=(1, 2))  # [time]
        clamping_count = at_lower + at_upper
    else:
        clamping_events = clamping_events.numpy()
        clamping_count = clamping_events.sum(axis=-1) if clamping_events.ndim > 1 else clamping_events

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("warning", "#ff7f0e")
        healthy_color = theme.colors.get("healthy", "#2ca02c")
        lw = theme.line_widths.get("main", 2.0)
    else:
        color = "#ff7f0e"
        healthy_color = "#2ca02c"
        lw = 2.0

    # Plot clamping count
    ax.fill_between(timesteps, 0, clamping_count, alpha=0.3, color=color)
    ax.plot(timesteps, clamping_count, color=color, linewidth=lw, label="Clamping Events")

    # Mark high-clamping regions
    threshold = clamping_count.max() * 0.5 if clamping_count.max() > 0 else 1
    high_clamping = clamping_count > threshold
    if high_clamping.any():
        ax.scatter(
            timesteps[high_clamping],
            clamping_count[high_clamping],
            color="red",
            marker="x",
            s=50,
            zorder=5,
            label="High clamping",
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Clamping Event Count")
    ax.set_title("State Clamping Events (Bound Hits)")
    ax.legend(loc="upper right")

    # Add secondary axis showing clamping rate
    if len(timesteps) > 1:
        ax2 = ax.twinx()
        total_possible = collector.get_particles().shape[1] * collector.get_particles().shape[2]
        clamping_rate = clamping_count / total_possible * 100
        ax2.plot(timesteps, clamping_rate, color="gray", alpha=0.5, linestyle=":", label="Clamping %")
        ax2.set_ylabel("Clamping Rate %", color="gray")
        ax2.set_ylim(0, max(100, clamping_rate.max() * 1.1))

    return fig, ax


def plot_euler_maruyama_stability(
    collector: "SDECollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot Euler-Maruyama stability metrics (E6).

    Shows dt*|f(h)| vs sqrt(dt)*|g(h)| ratio to assess
    numerical stability of the EM integration scheme.

    For stability, typically want dt*|f| < C and sqrt(dt)*|g| < C
    for some reasonable constant C.

    Args:
        collector: SDE data collector
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    drift = collector.get_drift_values()
    diffusion = collector.get_diffusion_values()

    if drift is None or diffusion is None:
        ax.text(0.5, 0.5, "No drift/diffusion data for stability analysis", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    drift = drift.numpy()
    diffusion = diffusion.numpy()

    # Get dt (assuming constant or get from collector)
    dt = collector.get_dt() if hasattr(collector, "get_dt") else 0.01

    # Compute stability metrics
    if drift.ndim == 3:
        drift_mag = np.abs(drift).mean(axis=(1, 2))  # [time]
        diff_mag = np.abs(diffusion).mean(axis=(1, 2))  # [time]
    else:
        drift_mag = np.abs(drift).mean(axis=-1)  # [time]
        diff_mag = np.abs(diffusion).mean(axis=-1)  # [time]

    # EM terms
    drift_term = dt * drift_mag  # dt * |f(h)|
    diffusion_term = np.sqrt(dt) * diff_mag  # sqrt(dt) * |g(h)|

    if theme:
        theme.apply_to_axes(ax)
        color1 = theme.colors.get("primary", "#1f77b4")
        color2 = theme.colors.get("secondary", "#ff7f0e")
        warning_color = theme.colors.get("warning", "#d62728")
        lw = theme.line_widths.get("main", 2.0)
    else:
        color1 = "#1f77b4"
        color2 = "#ff7f0e"
        warning_color = "#d62728"
        lw = 2.0

    # Plot both terms
    ax.plot(timesteps, drift_term, color=color1, linewidth=lw, label=f"dt*|f(h)| (dt={dt:.4f})")
    ax.plot(timesteps, diffusion_term, color=color2, linewidth=lw, label=f"sqrt(dt)*|g(h)|")

    # Stability thresholds (heuristic)
    stability_threshold = 1.0
    ax.axhline(y=stability_threshold, color=warning_color, linestyle="--", alpha=0.7, label="Stability threshold")

    # Mark unstable regions
    unstable_drift = drift_term > stability_threshold
    unstable_diff = diffusion_term > stability_threshold

    if unstable_drift.any():
        ax.scatter(
            timesteps[unstable_drift],
            drift_term[unstable_drift],
            color=warning_color,
            marker="^",
            s=50,
            zorder=5,
            label="Drift unstable",
        )

    if unstable_diff.any():
        ax.scatter(
            timesteps[unstable_diff],
            diffusion_term[unstable_diff],
            color=warning_color,
            marker="v",
            s=50,
            zorder=5,
            label="Diffusion unstable",
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("EM Term Magnitude")
    ax.set_title("Euler-Maruyama Stability Analysis")
    ax.legend(loc="upper right", fontsize=8)

    # Add stability indicator
    pct_stable = 100 * (1 - (unstable_drift | unstable_diff).mean())
    stability_text = f"Stability: {pct_stable:.1f}%"
    ax.text(0.02, 0.98, stability_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(
                boxstyle="round",
                facecolor="lightgreen" if pct_stable > 95 else "lightyellow" if pct_stable > 80 else "lightcoral",
                alpha=0.8
            ))

    return fig, ax


def plot_sde_summary_dashboard(
    collector: "SDECollector",
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (16, 12),
    **kwargs,
) -> Tuple[Figure, List[Axes]]:
    """Create a summary dashboard of SDE-specific metrics.

    Combines E1-E6 into a single multi-panel figure.

    Args:
        collector: SDE data collector
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and list of axes
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # E1: Diffusion magnitude
    plot_diffusion_magnitude(collector, ax=axes[0], theme=theme)

    # E2: Drift/diffusion ratio
    plot_drift_diffusion_ratio(collector, ax=axes[1], theme=theme)

    # E3: Unfold convergence
    plot_unfold_convergence(collector, ax=axes[2], theme=theme)

    # E4: Brownian increments
    plot_brownian_increments(collector, ax=axes[3], theme=theme)

    # E5: Clamping events
    plot_state_clamping_events(collector, ax=axes[4], theme=theme)

    # E6: EM stability
    plot_euler_maruyama_stability(collector, ax=axes[5], theme=theme)

    plt.tight_layout()
    return fig, list(axes)
