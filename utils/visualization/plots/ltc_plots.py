"""LTC-specific visualization plots (L1-L7).

Implements:
- L1: Voltage traces per neuron
- L2: Time constants tau = cm/(gleak+...) dynamics
- L3: Synapse activation patterns w * sigmoid(v-mu)
- L4: Leak vs synaptic current decomposition
- L5: ODE unfold convergence dynamics
- L6: Reversal potential flow analysis
- L7: Sparsity mask utilization
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib.patches as mpatches
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


def plot_voltage_traces(
    collector: "BaseDataCollector",
    neuron_ids: Optional[List[int]] = None,
    n_particles: int = 5,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (14, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot voltage traces per neuron over time (L1).

    Shows the evolution of neural membrane potentials for LTC neurons,
    with particle uncertainty bands if available.

    Args:
        collector: Data collector with LTC voltage data
        neuron_ids: Which neurons to plot (default: first 10)
        n_particles: Number of particle trajectories to show
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()

    # Get voltage data - this is typically the particle state for LTC
    voltages = collector.get_voltages()
    if voltages is None:
        # Fall back to particles which represent voltages in LTC
        voltages = collector.get_particles()

    if voltages is None:
        ax.text(0.5, 0.5, "No voltage data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    voltages = voltages.numpy()  # [time, K, state_size]

    if neuron_ids is None:
        neuron_ids = list(range(min(10, voltages.shape[-1])))

    if theme:
        theme.apply_to_axes(ax)
        colors = theme.get_particle_colors(len(neuron_ids))
        alpha = theme.alpha_values.get("particle_line", 0.3)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_ids)))
        alpha = 0.3

    for i, (neuron_id, color) in enumerate(zip(neuron_ids, colors)):
        if neuron_id >= voltages.shape[-1]:
            continue

        # Get voltage for this neuron across all particles
        v_neuron = voltages[:, :, neuron_id]  # [time, K]

        # Plot mean trajectory
        v_mean = v_neuron.mean(axis=1)
        v_std = v_neuron.std(axis=1)

        ax.plot(timesteps, v_mean, color=color, linewidth=2, label=f"Neuron {neuron_id}")

        # Uncertainty band
        ax.fill_between(
            timesteps,
            v_mean - v_std,
            v_mean + v_std,
            color=color,
            alpha=alpha,
        )

        # Show a few individual particle trajectories
        n_show = min(n_particles, v_neuron.shape[1])
        for k in range(n_show):
            ax.plot(timesteps, v_neuron[:, k], color=color, alpha=0.1, linewidth=0.5)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Voltage (Membrane Potential)")
    ax.set_title("LTC Neuron Voltage Traces")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    return fig, ax


def plot_time_constants(
    collector: "BaseDataCollector",
    neuron_ids: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot time constants tau = cm/(gleak + g_syn) dynamics (L2).

    Shows how effective time constants evolve based on synaptic activity.

    Args:
        collector: Data collector with LTC internal data
        neuron_ids: Which neurons to plot
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    time_constants = collector.get_time_constants()

    if time_constants is None:
        # Try to estimate from gleak and synaptic conductances
        gleak = collector.get_gleak()
        cm = collector.get_cm()
        g_syn = collector.get_synaptic_conductance()

        if gleak is not None and cm is not None:
            if g_syn is not None:
                total_g = gleak + g_syn.mean(axis=-1)  # Sum over presynaptic
            else:
                total_g = gleak
            time_constants = cm / (total_g + 1e-8)
        else:
            ax.text(0.5, 0.5, "No time constant data available", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

    time_constants = time_constants.numpy() if isinstance(time_constants, Tensor) else time_constants

    if neuron_ids is None:
        if time_constants.ndim >= 2:
            neuron_ids = list(range(min(10, time_constants.shape[-1])))
        else:
            neuron_ids = [0]

    if theme:
        theme.apply_to_axes(ax)
        colors = theme.get_particle_colors(len(neuron_ids))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(neuron_ids)))

    # Handle different shapes
    if time_constants.ndim == 1:
        # Single value per timestep
        ax.plot(timesteps, time_constants, linewidth=2, label="tau")
    elif time_constants.ndim == 2:
        # [time, neurons] or [time, K]
        for i, (neuron_id, color) in enumerate(zip(neuron_ids, colors)):
            if neuron_id >= time_constants.shape[-1]:
                continue
            ax.plot(timesteps, time_constants[:, neuron_id], color=color, linewidth=2, label=f"tau_{neuron_id}")
    else:
        # [time, K, neurons] - average over particles
        tc_avg = time_constants.mean(axis=1)
        for i, (neuron_id, color) in enumerate(zip(neuron_ids, colors)):
            if neuron_id >= tc_avg.shape[-1]:
                continue
            ax.plot(timesteps, tc_avg[:, neuron_id], color=color, linewidth=2, label=f"tau_{neuron_id}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Time Constant tau (ms)")
    ax.set_title("LTC Effective Time Constants")
    ax.legend(loc="upper right", fontsize=8)

    # Add reference lines for typical biological ranges
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")
    ax.text(timesteps[0], 10, "10ms", fontsize=8, color="gray", va="bottom")
    ax.text(timesteps[0], 100, "100ms", fontsize=8, color="gray", va="bottom")

    return fig, ax


def plot_synapse_activations(
    collector: "BaseDataCollector",
    timestep: int = -1,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot synapse activation patterns w * sigmoid(v-mu) (L3).

    Shows connectivity heatmap weighted by activation strength.

    Args:
        collector: Data collector with synapse data
        timestep: Which timestep to visualize (-1 for last)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    synapse_activations = collector.get_synapse_activations()

    if synapse_activations is None:
        ax.text(0.5, 0.5, "No synapse activation data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    synapse_activations = synapse_activations.numpy()  # [time, pre, post] or [time, K, pre, post]

    # Select timestep
    if timestep < 0:
        timestep = synapse_activations.shape[0] + timestep
    timestep = min(timestep, synapse_activations.shape[0] - 1)

    if synapse_activations.ndim == 4:
        # Average over particles
        activation_t = synapse_activations[timestep].mean(axis=0)  # [pre, post]
    else:
        activation_t = synapse_activations[timestep]  # [pre, post]

    if theme:
        theme.apply_to_axes(ax)

    # Heatmap of activations
    im = ax.imshow(
        activation_t,
        cmap="viridis",
        aspect="auto",
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="Activation w*sigmoid(v-mu)")

    ax.set_xlabel("Post-synaptic Neuron")
    ax.set_ylabel("Pre-synaptic Neuron")
    ax.set_title(f"Synapse Activations (t={timestep})")

    # Add grid for clarity
    ax.set_xticks(np.arange(activation_t.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(activation_t.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.3)

    return fig, ax


def plot_leak_vs_synaptic(
    collector: "BaseDataCollector",
    neuron_ids: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot current decomposition: leak vs synaptic (L4).

    Shows the relative contribution of leak and synaptic currents.

    Args:
        collector: Data collector with current data
        neuron_ids: Which neurons to analyze
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()

    leak_current = collector.get_leak_current()
    synaptic_current = collector.get_synaptic_current()

    if leak_current is None and synaptic_current is None:
        ax.text(0.5, 0.5, "No current decomposition data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    if theme:
        theme.apply_to_axes(ax)
        leak_color = theme.colors.get("primary", "#1f77b4")
        syn_color = theme.colors.get("secondary", "#ff7f0e")
    else:
        leak_color = "#1f77b4"
        syn_color = "#ff7f0e"

    if leak_current is not None:
        leak_current = leak_current.numpy()
        if leak_current.ndim >= 2:
            leak_avg = np.abs(leak_current).mean(axis=tuple(range(1, leak_current.ndim)))
        else:
            leak_avg = np.abs(leak_current)
    else:
        leak_avg = np.zeros(len(timesteps))

    if synaptic_current is not None:
        synaptic_current = synaptic_current.numpy()
        if synaptic_current.ndim >= 2:
            syn_avg = np.abs(synaptic_current).mean(axis=tuple(range(1, synaptic_current.ndim)))
        else:
            syn_avg = np.abs(synaptic_current)
    else:
        syn_avg = np.zeros(len(timesteps))

    # Stacked area plot
    ax.fill_between(timesteps, 0, leak_avg, alpha=0.7, color=leak_color, label="Leak Current")
    ax.fill_between(timesteps, leak_avg, leak_avg + syn_avg, alpha=0.7, color=syn_color, label="Synaptic Current")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Current Magnitude")
    ax.set_title("Leak vs Synaptic Current Decomposition")
    ax.legend(loc="upper right")

    # Add ratio on secondary axis
    ax2 = ax.twinx()
    total = leak_avg + syn_avg + 1e-10
    ratio = syn_avg / total * 100
    ax2.plot(timesteps, ratio, color="gray", alpha=0.5, linestyle=":", linewidth=2)
    ax2.set_ylabel("Synaptic %", color="gray")
    ax2.set_ylim(0, 100)

    return fig, ax


def plot_ode_unfold_dynamics(
    collector: "BaseDataCollector",
    neuron_ids: Optional[List[int]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot ODE unfold convergence dynamics (L5).

    Shows how state changes across unfold iterations within each timestep.

    Args:
        collector: Data collector with per-unfold states
        neuron_ids: Which neurons to track
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

    if neuron_ids is None:
        if per_unfold_states.ndim == 4:
            neuron_ids = list(range(min(5, per_unfold_states.shape[-1])))
        else:
            neuron_ids = list(range(min(5, per_unfold_states.shape[-1])))

    if theme:
        theme.apply_to_axes(ax)
        colors = theme.get_particle_colors(len(neuron_ids))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_ids)))

    n_unfolds = per_unfold_states.shape[1]
    unfold_indices = np.arange(n_unfolds)

    # Average over time and particles
    if per_unfold_states.ndim == 4:
        states_avg = per_unfold_states.mean(axis=(0, 2))  # [unfolds, H]
    else:
        states_avg = per_unfold_states.mean(axis=0)  # [unfolds, H]

    # Compute convergence metric: absolute change per unfold
    deltas = np.abs(np.diff(states_avg, axis=0))  # [unfolds-1, H]

    for i, (neuron_id, color) in enumerate(zip(neuron_ids, colors)):
        if neuron_id >= deltas.shape[-1]:
            continue
        ax.plot(
            unfold_indices[:-1],
            deltas[:, neuron_id],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"Neuron {neuron_id}",
        )

    ax.set_xlabel("Unfold Iteration")
    ax.set_ylabel("State Change |dh|")
    ax.set_title("ODE Unfold Convergence")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)

    # Mark convergence region
    if deltas.size > 0:
        mean_final_delta = deltas[-1].mean()
        ax.axhline(y=mean_final_delta, color="red", linestyle="--", alpha=0.5, label="Final delta")

    return fig, ax


def plot_reversal_potential_flow(
    collector: "BaseDataCollector",
    timestep: int = -1,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 10),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot reversal potential flow as weighted connectivity (L6).

    Shows E_rev weighted by activation to visualize information flow direction.

    Args:
        collector: Data collector with reversal potential data
        timestep: Which timestep to visualize (-1 for last)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    reversal_potentials = collector.get_reversal_potentials()
    synapse_activations = collector.get_synapse_activations()

    if reversal_potentials is None:
        ax.text(0.5, 0.5, "No reversal potential data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    e_rev = reversal_potentials.numpy() if isinstance(reversal_potentials, Tensor) else reversal_potentials

    # Select timestep
    if e_rev.ndim >= 2:
        if timestep < 0:
            timestep = e_rev.shape[0] + timestep
        timestep = min(timestep, e_rev.shape[0] - 1)
        e_rev_t = e_rev[timestep]
    else:
        e_rev_t = e_rev

    # Weight by activation if available
    if synapse_activations is not None:
        syn_act = synapse_activations.numpy()
        if syn_act.ndim >= 2:
            if timestep < syn_act.shape[0]:
                if syn_act.ndim == 4:
                    weight = syn_act[timestep].mean(axis=0)  # [pre, post]
                else:
                    weight = syn_act[timestep]
            else:
                weight = syn_act[-1] if syn_act.ndim == 3 else syn_act[-1].mean(axis=0)
        else:
            weight = np.ones_like(e_rev_t)
    else:
        weight = np.ones_like(e_rev_t) if e_rev_t.ndim == 2 else np.ones((e_rev_t.shape[0], e_rev_t.shape[0]))

    # Weighted reversal potential
    if e_rev_t.ndim == 1:
        # E_rev per connection type - broadcast to matrix
        n_neurons = len(e_rev_t)
        weighted_flow = np.outer(e_rev_t, np.ones(n_neurons)) * weight
    else:
        weighted_flow = e_rev_t * weight

    if theme:
        theme.apply_to_axes(ax)

    # Diverging colormap: blue (inhibitory, negative) to red (excitatory, positive)
    vmax = max(abs(weighted_flow.min()), abs(weighted_flow.max()))
    im = ax.imshow(
        weighted_flow,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="Weighted E_rev (Activation * Reversal)")

    ax.set_xlabel("Post-synaptic Neuron")
    ax.set_ylabel("Pre-synaptic Neuron")
    ax.set_title(f"Reversal Potential Flow (t={timestep})")

    return fig, ax


def plot_sparsity_mask_utilization(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot sparsity mask utilization over time (L7).

    Shows which connections are active and their utilization patterns.

    Args:
        collector: Data collector with sparsity mask data
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sparsity_mask = collector.get_sparsity_mask()
    synapse_activations = collector.get_synapse_activations()

    if sparsity_mask is None:
        axes[0].text(0.5, 0.5, "No sparsity mask data", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No sparsity mask data", ha="center", va="center", transform=axes[1].transAxes)
        return fig, axes

    mask = sparsity_mask.numpy() if isinstance(sparsity_mask, Tensor) else sparsity_mask

    if theme:
        theme.apply_to_axes(axes[0])
        theme.apply_to_axes(axes[1])

    # Left panel: Binary sparsity mask
    axes[0].imshow(mask, cmap="binary", aspect="auto")
    axes[0].set_xlabel("Post-synaptic Neuron")
    axes[0].set_ylabel("Pre-synaptic Neuron")
    axes[0].set_title(f"Sparsity Mask (density={mask.mean()*100:.1f}%)")

    # Right panel: Utilization over time (if activations available)
    if synapse_activations is not None:
        syn_act = synapse_activations.numpy()
        if syn_act.ndim >= 3:
            # Average over particles if present
            if syn_act.ndim == 4:
                syn_act = syn_act.mean(axis=2)  # [time, pre, post]

            # Compute utilization: how much each connection is used
            utilization = np.abs(syn_act).mean(axis=0)  # [pre, post]
            masked_utilization = utilization * mask

            im = axes[1].imshow(masked_utilization, cmap="viridis", aspect="auto")
            fig.colorbar(im, ax=axes[1], label="Avg Activation")
            axes[1].set_xlabel("Post-synaptic Neuron")
            axes[1].set_ylabel("Pre-synaptic Neuron")
            axes[1].set_title("Connection Utilization")
        else:
            axes[1].text(0.5, 0.5, "Insufficient activation data", ha="center", va="center", transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "No activation data for utilization", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    return fig, axes


def plot_ltc_summary_dashboard(
    collector: "BaseDataCollector",
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (18, 12),
    **kwargs,
) -> Tuple[Figure, List[Axes]]:
    """Create summary dashboard of LTC-specific metrics.

    Combines key LTC visualizations into a single figure.

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

    # Create grid: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # L1: Voltage traces
    try:
        plot_voltage_traces(collector, ax=ax1, theme=theme)
    except Exception as e:
        ax1.text(0.5, 0.5, f"Voltage traces unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax1.transAxes, fontsize=8)
        ax1.set_title("L1: Voltage Traces")

    # L2: Time constants
    try:
        plot_time_constants(collector, ax=ax2, theme=theme)
    except Exception as e:
        ax2.text(0.5, 0.5, f"Time constants unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax2.transAxes, fontsize=8)
        ax2.set_title("L2: Time Constants")

    # L3: Synapse activations
    try:
        plot_synapse_activations(collector, ax=ax3, theme=theme)
    except Exception as e:
        ax3.text(0.5, 0.5, f"Synapse activations unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax3.transAxes, fontsize=8)
        ax3.set_title("L3: Synapse Activations")

    # L4: Leak vs Synaptic
    try:
        plot_leak_vs_synaptic(collector, ax=ax4, theme=theme)
    except Exception as e:
        ax4.text(0.5, 0.5, f"Current decomposition unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax4.transAxes, fontsize=8)
        ax4.set_title("L4: Leak vs Synaptic")

    # L5: ODE unfold dynamics
    try:
        plot_ode_unfold_dynamics(collector, ax=ax5, theme=theme)
    except Exception as e:
        ax5.text(0.5, 0.5, f"ODE unfold data unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax5.transAxes, fontsize=8)
        ax5.set_title("L5: ODE Unfold Dynamics")

    # L6: Reversal potential flow
    try:
        plot_reversal_potential_flow(collector, ax=ax6, theme=theme)
    except Exception as e:
        ax6.text(0.5, 0.5, f"Reversal potential unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax6.transAxes, fontsize=8)
        ax6.set_title("L6: Reversal Potential Flow")

    fig.suptitle("LTC Architecture Summary Dashboard", fontsize=14, fontweight="bold")

    return fig, axes
