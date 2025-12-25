"""CfC-specific visualization plots (F1-F5).

Implements:
- F1: Time interpolation weights sigma_t = sigmoid(time_a*dt + time_b)
- F2: ff1/ff2 feed-forward contributions
- F3: Learned time constants (time_a, time_b parameters)
- F4: Backbone layer activations
- F5: CfC mode comparison (default, pure, no_gate)
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
    from ..collectors.base_collector import BaseDataCollector
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


def plot_interpolation_weights(
    collector: "BaseDataCollector",
    dims: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot time interpolation weights sigma_t (F1).

    Shows sigma_t = sigmoid(time_a*dt + time_b) which controls
    the interpolation between ff1 (asymptotic) and ff2 (instantaneous).

    Args:
        collector: Data collector with CfC internal data
        dims: Which hidden dimensions to plot (default: first 10)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    sigma_t = collector.get_interpolation_weights()

    if sigma_t is None:
        ax.text(0.5, 0.5, "No interpolation weight (sigma_t) data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    sigma_t = sigma_t.numpy()  # [time, H] or [time, K, H]

    if dims is None:
        if sigma_t.ndim >= 2:
            dims = list(range(min(10, sigma_t.shape[-1])))
        else:
            dims = [0]

    if theme:
        theme.apply_to_axes(ax)
        cmap = "viridis"
    else:
        cmap = "viridis"

    # Average over particles if present
    if sigma_t.ndim == 3:
        sigma_t = sigma_t.mean(axis=1)  # [time, H]

    # Select dimensions
    sigma_plot = sigma_t[:, dims]  # [time, dims]

    # Heatmap visualization
    im = ax.imshow(
        sigma_plot.T,
        aspect="auto",
        origin="lower",
        extent=[timesteps[0], timesteps[-1], 0, len(dims)],
        cmap=cmap,
        vmin=0,
        vmax=1,
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="sigma_t (0=ff1, 1=ff2)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Hidden Dimension")
    ax.set_title("CfC Time Interpolation Weights (sigma_t)")

    # Set dimension tick labels
    ax.set_yticks(np.arange(len(dims)) + 0.5)
    ax.set_yticklabels([f"h{d}" for d in dims], fontsize=8)

    return fig, ax


def plot_ff1_ff2_contributions(
    collector: "BaseDataCollector",
    dims: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (14, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot ff1/ff2 feed-forward contributions (F2).

    Shows the relative contribution of ff1 (1-sigma)*ff1 and ff2 sigma*ff2
    to the final hidden state update.

    Args:
        collector: Data collector with CfC internal data
        dims: Which hidden dimensions to analyze (default: first 5)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    ff1_values = collector.get_ff1_output()
    ff2_values = collector.get_ff2_output()
    sigma_t = collector.get_interpolation_weights()

    if ff1_values is None or ff2_values is None:
        ax.text(0.5, 0.5, "No ff1/ff2 output data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    ff1 = ff1_values.numpy()
    ff2 = ff2_values.numpy()

    # Average over particles if present
    if ff1.ndim == 3:
        ff1 = ff1.mean(axis=1)
    if ff2.ndim == 3:
        ff2 = ff2.mean(axis=1)

    if sigma_t is not None:
        sigma = sigma_t.numpy()
        if sigma.ndim == 3:
            sigma = sigma.mean(axis=1)

        # Compute actual contributions
        ff1_contrib = np.abs((1 - sigma) * ff1).mean(axis=-1)  # [time]
        ff2_contrib = np.abs(sigma * ff2).mean(axis=-1)  # [time]
    else:
        # Just plot raw values
        ff1_contrib = np.abs(ff1).mean(axis=-1)
        ff2_contrib = np.abs(ff2).mean(axis=-1)

    if theme:
        theme.apply_to_axes(ax)
        color1 = theme.colors.get("primary", "#1f77b4")
        color2 = theme.colors.get("secondary", "#ff7f0e")
    else:
        color1 = "#1f77b4"
        color2 = "#ff7f0e"

    # Stacked area plot
    ax.fill_between(timesteps, 0, ff1_contrib, alpha=0.7, color=color1, label="(1-sigma)*ff1 (asymptotic)")
    ax.fill_between(timesteps, ff1_contrib, ff1_contrib + ff2_contrib, alpha=0.7, color=color2, label="sigma*ff2 (instantaneous)")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Contribution Magnitude")
    ax.set_title("CfC ff1/ff2 Contributions")
    ax.legend(loc="upper right")

    # Add ratio on secondary axis
    ax2 = ax.twinx()
    total = ff1_contrib + ff2_contrib + 1e-10
    ratio = ff2_contrib / total * 100
    ax2.plot(timesteps, ratio, color="gray", alpha=0.5, linestyle=":", linewidth=2)
    ax2.set_ylabel("ff2 %", color="gray")
    ax2.set_ylim(0, 100)

    return fig, ax


def plot_time_constants_learned(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, List[Axes]]:
    """Plot learned time constants time_a, time_b (F3).

    Shows the learned parameters that control temporal dynamics:
    - time_a: controls rate of adaptation
    - time_b: controls base time constant

    Args:
        collector: Data collector with CfC parameter data
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and list of axes
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    time_a = collector.get_time_a()
    time_b = collector.get_time_b()

    if time_a is None and time_b is None:
        axes[0].text(0.5, 0.5, "No time_a data", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No time_b data", ha="center", va="center", transform=axes[1].transAxes)
        return fig, list(axes)

    if theme:
        theme.apply_to_axes(axes[0])
        theme.apply_to_axes(axes[1])
        color = theme.colors.get("primary", "#1f77b4")
    else:
        color = "#1f77b4"

    # Left: time_a distribution
    if time_a is not None:
        ta = time_a.numpy() if isinstance(time_a, Tensor) else time_a
        ta_flat = ta.flatten()
        axes[0].hist(ta_flat, bins=30, alpha=0.7, color=color, edgecolor="black")
        axes[0].axvline(x=ta_flat.mean(), color="red", linestyle="--", label=f"Mean={ta_flat.mean():.3f}")
        axes[0].set_xlabel("time_a Value")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"time_a Distribution (std={ta_flat.std():.3f})")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No time_a data", ha="center", va="center", transform=axes[0].transAxes)

    # Right: time_b distribution
    if time_b is not None:
        tb = time_b.numpy() if isinstance(time_b, Tensor) else time_b
        tb_flat = tb.flatten()
        axes[1].hist(tb_flat, bins=30, alpha=0.7, color=color, edgecolor="black")
        axes[1].axvline(x=tb_flat.mean(), color="red", linestyle="--", label=f"Mean={tb_flat.mean():.3f}")
        axes[1].set_xlabel("time_b Value")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"time_b Distribution (std={tb_flat.std():.3f})")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No time_b data", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    return fig, list(axes)


def plot_backbone_activations(
    collector: "BaseDataCollector",
    layer_names: Optional[List[str]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (14, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot backbone layer activations over time (F4).

    Shows activation patterns in the backbone network layers.

    Args:
        collector: Data collector with backbone activation data
        layer_names: Which layers to visualize
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    backbone_activations = collector.get_backbone_activations()

    if backbone_activations is None:
        ax.text(0.5, 0.5, "No backbone activation data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    if theme:
        theme.apply_to_axes(ax)

    # backbone_activations can be dict {layer_name: [time, ...]} or tensor [time, layers, units]
    if isinstance(backbone_activations, dict):
        if layer_names is None:
            layer_names = list(backbone_activations.keys())[:5]

        n_layers = len(layer_names)
        colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

        for layer_name, color in zip(layer_names, colors):
            if layer_name not in backbone_activations:
                continue
            act = backbone_activations[layer_name]
            if isinstance(act, Tensor):
                act = act.numpy()

            # Compute mean activation magnitude
            if act.ndim >= 2:
                act_mean = np.abs(act).mean(axis=tuple(range(1, act.ndim)))
            else:
                act_mean = np.abs(act)

            ax.plot(timesteps, act_mean, color=color, linewidth=2, label=layer_name)
    else:
        # Tensor format
        activations = backbone_activations.numpy() if isinstance(backbone_activations, Tensor) else backbone_activations

        if activations.ndim == 3:
            # [time, layers, units]
            n_layers = activations.shape[1]
            colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

            for i, color in enumerate(colors):
                act_mean = np.abs(activations[:, i, :]).mean(axis=-1)
                ax.plot(timesteps, act_mean, color=color, linewidth=2, label=f"Layer {i}")
        else:
            # [time, units] - single layer
            act_mean = np.abs(activations).mean(axis=-1) if activations.ndim == 2 else activations
            ax.plot(timesteps, act_mean, linewidth=2, label="Backbone")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Activation Magnitude")
    ax.set_title("CfC Backbone Layer Activations")
    ax.legend(loc="upper right", fontsize=8)

    return fig, ax


def plot_mode_comparison(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot CfC mode comparison metrics (F5).

    Compares behavior across different CfC modes:
    - default: Full CfC with gating
    - pure: Pure closed-form solution
    - no_gate: CfC without gating mechanism

    Args:
        collector: Data collector with mode comparison data
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    mode_data = collector.get_mode_comparison_data()

    if mode_data is None:
        # If no explicit mode comparison, show current mode behavior
        sigma_t = collector.get_interpolation_weights()
        ff1 = collector.get_ff1_output()
        ff2 = collector.get_ff2_output()

        if sigma_t is None:
            ax.text(0.5, 0.5, "No mode comparison data available", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        sigma = sigma_t.numpy()
        if sigma.ndim >= 2:
            sigma_avg = sigma.mean(axis=tuple(range(1, sigma.ndim)))
        else:
            sigma_avg = sigma

        if theme:
            theme.apply_to_axes(ax)
            color = theme.colors.get("primary", "#1f77b4")
        else:
            color = "#1f77b4"

        ax.plot(timesteps, sigma_avg, color=color, linewidth=2, label="sigma_t (avg)")

        # Add mode interpretation
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        # Annotate regions
        pure_region = sigma_avg < 0.2
        gate_region = sigma_avg > 0.8

        if pure_region.any():
            ax.fill_between(timesteps, 0, 1, where=pure_region, alpha=0.1, color="blue", label="Pure-like (sigma<0.2)")
        if gate_region.any():
            ax.fill_between(timesteps, 0, 1, where=gate_region, alpha=0.1, color="red", label="Gated (sigma>0.8)")

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Interpolation Weight")
        ax.set_title("CfC Mode Behavior (sigma_t)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")

    else:
        # Explicit mode comparison data available
        if theme:
            theme.apply_to_axes(ax)
            colors = [theme.colors.get("primary", "#1f77b4"),
                     theme.colors.get("secondary", "#ff7f0e"),
                     theme.colors.get("tertiary", "#2ca02c")]
        else:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        mode_names = list(mode_data.keys())[:3]
        for mode_name, color in zip(mode_names, colors):
            data = mode_data[mode_name]
            if isinstance(data, Tensor):
                data = data.numpy()

            if data.ndim >= 2:
                data_avg = data.mean(axis=tuple(range(1, data.ndim)))
            else:
                data_avg = data

            ax.plot(timesteps, data_avg, color=color, linewidth=2, label=mode_name)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Output Magnitude")
        ax.set_title("CfC Mode Comparison")
        ax.legend(loc="upper right")

    return fig, ax


def plot_effective_time_constant_cfc(
    collector: "BaseDataCollector",
    dims: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot effective time constant from CfC parameters.

    For CfC, the effective time constant is related to time_a and time_b
    via: tau_eff ∝ 1/(time_a * dt + epsilon)

    Args:
        collector: Data collector
        dims: Which dimensions to visualize
        ax: Matplotlib axes
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    time_a = collector.get_time_a()
    time_b = collector.get_time_b()
    dt = collector.get_dt() if hasattr(collector, "get_dt") else 1.0

    if time_a is None:
        ax.text(0.5, 0.5, "No time_a parameter data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    ta = time_a.numpy() if isinstance(time_a, Tensor) else time_a
    tb = time_b.numpy() if isinstance(time_b, Tensor) else (time_b if time_b is not None else np.zeros_like(ta))

    # Compute effective time constant
    # sigma_t = sigmoid(time_a * dt + time_b)
    # Effective tau ~ 1 / (time_a * sigmoid_derivative)
    tau_eff = 1.0 / (np.abs(ta) + 1e-6)

    if dims is None:
        dims = list(range(min(20, len(tau_eff.flatten()))))

    tau_flat = tau_eff.flatten()

    if theme:
        theme.apply_to_axes(ax)
        color = theme.colors.get("primary", "#1f77b4")
    else:
        color = "#1f77b4"

    # Bar plot of effective time constants
    x = np.arange(min(len(dims), len(tau_flat)))
    values = tau_flat[:len(x)]

    ax.bar(x, values, color=color, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Hidden Unit Index")
    ax.set_ylabel("Effective Time Constant (1/|time_a|)")
    ax.set_title("CfC Effective Time Constants per Hidden Unit")

    # Add statistics
    stats_text = f"Mean tau: {values.mean():.3f}\nStd: {values.std():.3f}\nMin: {values.min():.3f}\nMax: {values.max():.3f}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    return fig, ax


def plot_cfc_summary_dashboard(
    collector: "BaseDataCollector",
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (16, 10),
    **kwargs,
) -> Tuple[Figure, List[Axes]]:
    """Create summary dashboard of CfC-specific metrics.

    Combines key CfC visualizations into a single figure.

    Args:
        collector: Data collector
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and list of axes
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # F1: Interpolation weights
    try:
        plot_interpolation_weights(collector, ax=ax1, theme=theme)
    except Exception as e:
        ax1.text(0.5, 0.5, f"Interpolation weights unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax1.transAxes, fontsize=8)
        ax1.set_title("F1: Interpolation Weights")

    # F2: ff1/ff2 contributions
    try:
        plot_ff1_ff2_contributions(collector, ax=ax2, theme=theme)
    except Exception as e:
        ax2.text(0.5, 0.5, f"ff1/ff2 data unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax2.transAxes, fontsize=8)
        ax2.set_title("F2: ff1/ff2 Contributions")

    # F4: Backbone activations
    try:
        plot_backbone_activations(collector, ax=ax3, theme=theme)
    except Exception as e:
        ax3.text(0.5, 0.5, f"Backbone data unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax3.transAxes, fontsize=8)
        ax3.set_title("F4: Backbone Activations")

    # F5: Mode comparison
    try:
        plot_mode_comparison(collector, ax=ax4, theme=theme)
    except Exception as e:
        ax4.text(0.5, 0.5, f"Mode data unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax4.transAxes, fontsize=8)
        ax4.set_title("F5: Mode Comparison")

    # Effective time constant
    try:
        plot_effective_time_constant_cfc(collector, ax=ax5, theme=theme)
    except Exception as e:
        ax5.text(0.5, 0.5, f"Time constant unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax5.transAxes, fontsize=8)
        ax5.set_title("Effective Time Constants")

    # ESS from core plots (bonus)
    from .core_plots import plot_ess_timeline
    try:
        plot_ess_timeline(collector, ax=ax6, theme=theme)
    except Exception as e:
        ax6.text(0.5, 0.5, f"ESS unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax6.transAxes, fontsize=8)
        ax6.set_title("ESS Timeline")

    fig.suptitle("CfC Architecture Summary Dashboard", fontsize=14, fontweight="bold")

    return fig, axes
