"""Wired/NCP-specific visualization plots (W1-W5).

Implements:
- W1: Per-layer state activations
- W2: NCP connectivity graph visualization
- W3: Layer-to-layer information flow
- W4: Layer-wise ESS (if separable)
- W5: Sensory-to-motor pathway attribution
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
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


def plot_layer_activations(
    collector: "BaseDataCollector",
    layer_names: Optional[List[str]] = None,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (14, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot per-layer state activations over time (W1).

    Shows how each layer (sensory, inter, command, motor) activates.

    Args:
        collector: Data collector with layer activation data
        layer_names: Which layers to plot (default: all available)
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    _check_matplotlib()

    layer_activations = collector.get_layer_activations()

    if layer_activations is None:
        fig, ax = _create_figure(ax, figsize)
        ax.text(0.5, 0.5, "No layer activation data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    timesteps = collector.get_timesteps().numpy()

    # Determine available layers
    if isinstance(layer_activations, dict):
        available_layers = list(layer_activations.keys())
    else:
        # Assume tensor format [time, layers, neurons]
        n_layers = layer_activations.shape[1] if layer_activations.ndim >= 2 else 1
        available_layers = [f"layer_{i}" for i in range(n_layers)]

    if layer_names is None:
        layer_names = available_layers[:4]  # sensory, inter, command, motor typically

    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=True)
    if n_layers == 1:
        axes = [axes]

    if theme:
        colors = theme.get_particle_colors(n_layers)
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for i, (layer_name, color, ax) in enumerate(zip(layer_names, colors, axes)):
        if theme:
            theme.apply_to_axes(ax)

        if isinstance(layer_activations, dict):
            if layer_name not in layer_activations:
                ax.text(0.5, 0.5, f"No data for {layer_name}", ha="center", va="center", transform=ax.transAxes)
                continue
            act = layer_activations[layer_name]
            if isinstance(act, Tensor):
                act = act.numpy()
        else:
            act = layer_activations.numpy() if isinstance(layer_activations, Tensor) else layer_activations
            if act.ndim >= 2 and i < act.shape[1]:
                act = act[:, i, :]
            else:
                continue

        # Average over particles if present
        if act.ndim == 3:
            act = act.mean(axis=1)  # [time, neurons]

        # Heatmap of neuron activations
        if act.ndim == 2:
            im = ax.imshow(
                act.T,
                aspect="auto",
                origin="lower",
                extent=[timesteps[0], timesteps[-1], 0, act.shape[1]],
                cmap="viridis",
            )
            ax.set_ylabel(f"{layer_name}\n(neurons)")
        else:
            ax.plot(timesteps, act, color=color, linewidth=2)
            ax.set_ylabel(layer_name)

        ax.set_title(f"{layer_name} Activations", fontsize=10)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("NCP Layer Activations", fontsize=12, fontweight="bold")
    plt.tight_layout()

    return fig, axes


def plot_ncp_connectivity_graph(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 10),
    layout: str = "hierarchical",
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot NCP network topology visualization (W2).

    Shows the connectivity structure between neuron populations.

    Args:
        collector: Data collector with connectivity data
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        layout: Layout style ("hierarchical", "circular", "spring")
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    adjacency = collector.get_adjacency_matrix()
    wiring_config = collector.get_wiring_config()

    if adjacency is None:
        ax.text(0.5, 0.5, "No connectivity data available", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    adj = adjacency.numpy() if isinstance(adjacency, Tensor) else adjacency

    if theme:
        theme.apply_to_axes(ax)

    # Get layer info if available
    if wiring_config is not None:
        sensory_size = wiring_config.get("sensory_size", 0)
        inter_size = wiring_config.get("inter_size", 0)
        command_size = wiring_config.get("command_size", 0)
        motor_size = wiring_config.get("motor_size", 0)

        layer_sizes = [sensory_size, inter_size, command_size, motor_size]
        layer_names = ["Sensory", "Inter", "Command", "Motor"]
        layer_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    else:
        # Assume equal division
        n_neurons = adj.shape[0]
        layer_sizes = [n_neurons // 4] * 4
        layer_sizes[-1] += n_neurons % 4
        layer_names = ["Layer 0", "Layer 1", "Layer 2", "Layer 3"]
        layer_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Compute positions based on layout
    n_total = sum(layer_sizes)
    positions = {}
    node_colors = []

    if layout == "hierarchical":
        y_positions = [3, 2, 1, 0]  # Top to bottom
        idx = 0
        for layer_idx, (size, y, color) in enumerate(zip(layer_sizes, y_positions, layer_colors)):
            if size == 0:
                continue
            x_start = -size / 2
            for i in range(size):
                positions[idx] = (x_start + i + 0.5, y)
                node_colors.append(color)
                idx += 1
    else:  # circular
        idx = 0
        for layer_idx, (size, color) in enumerate(zip(layer_sizes, layer_colors)):
            if size == 0:
                continue
            radius = 1 + layer_idx * 0.5
            for i in range(size):
                angle = 2 * np.pi * i / size
                positions[idx] = (radius * np.cos(angle), radius * np.sin(angle))
                node_colors.append(color)
                idx += 1

    # Draw edges
    for i in range(min(n_total, adj.shape[0])):
        for j in range(min(n_total, adj.shape[1])):
            if adj[i, j] != 0 and i in positions and j in positions:
                weight = np.abs(adj[i, j])
                color = "green" if adj[i, j] > 0 else "red"
                alpha = min(0.8, 0.1 + 0.7 * weight / (np.abs(adj).max() + 1e-6))
                ax.plot(
                    [positions[i][0], positions[j][0]],
                    [positions[i][1], positions[j][1]],
                    color=color,
                    alpha=alpha,
                    linewidth=0.5 + weight,
                    zorder=1,
                )

    # Draw nodes
    for idx, (pos, color) in enumerate(zip(positions.values(), node_colors)):
        ax.scatter(pos[0], pos[1], c=color, s=100, zorder=2, edgecolors="black", linewidths=0.5)

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=name)
        for name, color, size in zip(layer_names, layer_colors, layer_sizes)
        if size > 0
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_xlim(-max(layer_sizes) / 2 - 1, max(layer_sizes) / 2 + 1)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("NCP Connectivity Graph", fontsize=12)

    return fig, ax


def plot_information_flow(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot layer-to-layer information flow (W3).

    Shows signal flow magnitude between NCP layers using a Sankey-like diagram.

    Args:
        collector: Data collector with layer flow data
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    layer_flow = collector.get_layer_flow()
    timesteps = collector.get_timesteps().numpy()

    if layer_flow is None:
        # Try to compute from layer activations
        layer_activations = collector.get_layer_activations()
        if layer_activations is None:
            ax.text(0.5, 0.5, "No layer flow data available", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        # Compute approximate flow as activation magnitude
        if isinstance(layer_activations, dict):
            layer_names = list(layer_activations.keys())
            flow_data = {}
            for name in layer_names:
                act = layer_activations[name]
                if isinstance(act, Tensor):
                    act = act.numpy()
                if act.ndim >= 2:
                    flow_data[name] = np.abs(act).mean(axis=tuple(range(1, act.ndim)))
                else:
                    flow_data[name] = np.abs(act)
        else:
            layer_activations = layer_activations.numpy() if isinstance(layer_activations, Tensor) else layer_activations
            n_layers = layer_activations.shape[1] if layer_activations.ndim >= 2 else 1
            layer_names = ["Sensory", "Inter", "Command", "Motor"][:n_layers]
            flow_data = {}
            for i, name in enumerate(layer_names):
                if i < layer_activations.shape[1]:
                    flow_data[name] = np.abs(layer_activations[:, i, :]).mean(axis=-1)
    else:
        flow_data = layer_flow

    if theme:
        theme.apply_to_axes(ax)
        colors = [theme.colors.get("primary", "#1f77b4"),
                 theme.colors.get("secondary", "#ff7f0e"),
                 theme.colors.get("tertiary", "#2ca02c"),
                 theme.colors.get("quaternary", "#d62728")]
    else:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Stacked area plot showing flow through each layer
    layer_names = list(flow_data.keys())
    n_layers = len(layer_names)

    cumulative = np.zeros(len(timesteps))
    for i, (name, color) in enumerate(zip(layer_names, colors[:n_layers])):
        if isinstance(flow_data[name], Tensor):
            flow = flow_data[name].numpy()
        else:
            flow = flow_data[name]

        ax.fill_between(timesteps, cumulative, cumulative + flow, alpha=0.7, color=color, label=name)
        cumulative = cumulative + flow

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Information Flow Magnitude")
    ax.set_title("Layer-to-Layer Information Flow")
    ax.legend(loc="upper right")

    return fig, ax


def plot_layer_wise_ess(
    collector: "BaseDataCollector",
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot layer-wise Effective Sample Size (W4).

    Shows ESS computed separately for each NCP layer.

    Args:
        collector: Data collector with per-layer ESS data
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    layer_ess = collector.get_layer_wise_ess()

    if layer_ess is None:
        # Compute from layer-wise particle variance as proxy
        layer_activations = collector.get_layer_activations()
        weights = collector.get_weights()

        if layer_activations is None or weights is None:
            ax.text(0.5, 0.5, "No layer-wise ESS data available", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        # Use overall ESS as fallback
        ess = collector.get_ess()
        if ess is not None:
            ess = ess.numpy()
            ax.plot(timesteps, ess, linewidth=2, label="Overall ESS")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Effective Sample Size")
            ax.set_title("ESS (Layer-wise ESS not available)")
            ax.legend()
            return fig, ax
    else:
        if isinstance(layer_ess, dict):
            layer_names = list(layer_ess.keys())
        else:
            layer_ess_np = layer_ess.numpy() if isinstance(layer_ess, Tensor) else layer_ess
            n_layers = layer_ess_np.shape[1] if layer_ess_np.ndim >= 2 else 1
            layer_names = ["Sensory", "Inter", "Command", "Motor"][:n_layers]

    if theme:
        theme.apply_to_axes(ax)
        colors = theme.get_particle_colors(len(layer_names))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))

    for i, (name, color) in enumerate(zip(layer_names, colors)):
        if isinstance(layer_ess, dict):
            ess_data = layer_ess[name]
            if isinstance(ess_data, Tensor):
                ess_data = ess_data.numpy()
        else:
            layer_ess_np = layer_ess.numpy() if isinstance(layer_ess, Tensor) else layer_ess
            if i < layer_ess_np.shape[1]:
                ess_data = layer_ess_np[:, i]
            else:
                continue

        ax.plot(timesteps, ess_data, color=color, linewidth=2, label=name)

    # Add ESS threshold
    n_particles = collector.get_particles().shape[1] if hasattr(collector, "get_particles") else 32
    threshold = n_particles * 0.5
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({threshold:.0f})")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Effective Sample Size")
    ax.set_title("Layer-wise ESS")
    ax.legend(loc="upper right", fontsize=8)

    return fig, ax


def plot_sensory_to_motor_path(
    collector: "BaseDataCollector",
    input_idx: int = 0,
    output_idx: int = 0,
    ax=None,
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (14, 8),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot sensory-to-motor pathway attribution (W5).

    Shows signal propagation from a specific input through to output.

    Args:
        collector: Data collector with pathway data
        input_idx: Which input to trace
        output_idx: Which output to analyze
        ax: Matplotlib axes (created if None)
        theme: Visual theme
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure and axes
    """
    fig, ax = _create_figure(ax, figsize)

    timesteps = collector.get_timesteps().numpy()
    pathway_attribution = collector.get_pathway_attribution()

    if pathway_attribution is None:
        # Try to compute from layer activations and adjacency
        layer_activations = collector.get_layer_activations()
        adjacency = collector.get_adjacency_matrix()

        if layer_activations is None:
            ax.text(0.5, 0.5, "No pathway attribution data available", ha="center", va="center", transform=ax.transAxes)
            return fig, ax

        # Simple visualization: show activation flow
        if isinstance(layer_activations, dict):
            layer_names = list(layer_activations.keys())
            pathway_data = []
            for name in layer_names:
                act = layer_activations[name]
                if isinstance(act, Tensor):
                    act = act.numpy()
                if act.ndim >= 2:
                    pathway_data.append(np.abs(act).mean(axis=tuple(range(1, act.ndim))))
                else:
                    pathway_data.append(np.abs(act))
        else:
            layer_activations = layer_activations.numpy() if isinstance(layer_activations, Tensor) else layer_activations
            n_layers = layer_activations.shape[1] if layer_activations.ndim >= 2 else 1
            layer_names = ["Sensory", "Inter", "Command", "Motor"][:n_layers]
            pathway_data = [np.abs(layer_activations[:, i, :]).mean(axis=-1) for i in range(n_layers)]

    else:
        if isinstance(pathway_attribution, dict):
            layer_names = list(pathway_attribution.keys())
            pathway_data = [
                pathway_attribution[name].numpy() if isinstance(pathway_attribution[name], Tensor) else pathway_attribution[name]
                for name in layer_names
            ]
        else:
            pathway_np = pathway_attribution.numpy() if isinstance(pathway_attribution, Tensor) else pathway_attribution
            n_layers = pathway_np.shape[1] if pathway_np.ndim >= 2 else 1
            layer_names = ["Sensory", "Inter", "Command", "Motor"][:n_layers]
            pathway_data = [pathway_np[:, i] for i in range(n_layers)]

    if theme:
        theme.apply_to_axes(ax)
        colors = [theme.colors.get("primary", "#1f77b4"),
                 theme.colors.get("secondary", "#ff7f0e"),
                 theme.colors.get("tertiary", "#2ca02c"),
                 theme.colors.get("quaternary", "#d62728")]
    else:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Plot pathway activations with arrows indicating flow direction
    for i, (name, data, color) in enumerate(zip(layer_names, pathway_data, colors[:len(layer_names)])):
        ax.plot(timesteps, data, color=color, linewidth=2, label=f"{name}", marker="o", markersize=3)

    # Add flow arrows between layers
    for i in range(len(layer_names) - 1):
        mid_t = timesteps[len(timesteps) // 2]
        y1 = pathway_data[i][len(timesteps) // 2]
        y2 = pathway_data[i + 1][len(timesteps) // 2]
        ax.annotate(
            "",
            xy=(mid_t + 2, y2),
            xytext=(mid_t, y1),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Pathway Activation")
    ax.set_title(f"Sensory-to-Motor Pathway (Input {input_idx} -> Output {output_idx})")
    ax.legend(loc="upper right", fontsize=9)

    return fig, ax


def plot_wired_summary_dashboard(
    collector: "BaseDataCollector",
    theme: Optional["Theme"] = None,
    figsize: Tuple[int, int] = (18, 12),
    **kwargs,
) -> Tuple[Figure, List[Axes]]:
    """Create summary dashboard of Wired/NCP-specific metrics.

    Combines key NCP visualizations into a single figure.

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

    # W2: Connectivity graph
    try:
        plot_ncp_connectivity_graph(collector, ax=ax1, theme=theme)
    except Exception as e:
        ax1.text(0.5, 0.5, f"Connectivity unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax1.transAxes, fontsize=8)
        ax1.set_title("W2: Connectivity Graph")

    # W3: Information flow
    try:
        plot_information_flow(collector, ax=ax2, theme=theme)
    except Exception as e:
        ax2.text(0.5, 0.5, f"Flow data unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax2.transAxes, fontsize=8)
        ax2.set_title("W3: Information Flow")

    # W4: Layer-wise ESS
    try:
        plot_layer_wise_ess(collector, ax=ax3, theme=theme)
    except Exception as e:
        ax3.text(0.5, 0.5, f"Layer ESS unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax3.transAxes, fontsize=8)
        ax3.set_title("W4: Layer-wise ESS")

    # W5: Sensory to motor path
    try:
        plot_sensory_to_motor_path(collector, ax=ax4, theme=theme)
    except Exception as e:
        ax4.text(0.5, 0.5, f"Pathway data unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax4.transAxes, fontsize=8)
        ax4.set_title("W5: Sensory-Motor Path")

    # ESS from core plots
    from .core_plots import plot_ess_timeline
    try:
        plot_ess_timeline(collector, ax=ax5, theme=theme)
    except Exception as e:
        ax5.text(0.5, 0.5, f"ESS unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax5.transAxes, fontsize=8)
        ax5.set_title("ESS Timeline")

    # Weight distribution from core plots
    from .core_plots import plot_weight_distribution
    try:
        plot_weight_distribution(collector, ax=ax6, theme=theme)
    except Exception as e:
        ax6.text(0.5, 0.5, f"Weight dist unavailable:\n{str(e)[:50]}", ha="center", va="center", transform=ax6.transAxes, fontsize=8)
        ax6.set_title("Weight Distribution")

    fig.suptitle("Wired/NCP Architecture Summary Dashboard", fontsize=14, fontweight="bold")

    return fig, axes
