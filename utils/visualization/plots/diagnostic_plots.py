"""Diagnostic visualization dashboards for PFNCPS."""

from typing import Optional, Tuple, TYPE_CHECKING

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..collectors.base_collector import BaseDataCollector
    from ..core.themes import Theme
    from ..core.base import ArchitectureInfo


def create_dashboard(
    collector: "BaseDataCollector",
    which: str = "health",
    figsize: Tuple[int, int] = (16, 12),
    theme: Optional["Theme"] = None,
    arch_info: Optional["ArchitectureInfo"] = None,
) -> Figure:
    """Create multi-panel diagnostic dashboard.

    Args:
        collector: Data collector with logged steps
        which: Dashboard type
            - "health": ESS, weights, numerical health, diversity
            - "research": Trajectories, calibration, uncertainty
            - "debug": All diagnostic plots
        figsize: Figure size
        theme: Visual theme
        arch_info: Architecture information

    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for dashboards")

    from .core_plots import (
        plot_ess_timeline,
        plot_weight_distribution,
        plot_particle_diversity,
        plot_numerical_health,
        plot_particle_trajectories,
        plot_weighted_output,
    )

    if which == "health":
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

        # ESS timeline
        ax1 = fig.add_subplot(gs[0, 0])
        plot_ess_timeline(collector, ax=ax1, theme=theme)

        # Weight distribution
        ax2 = fig.add_subplot(gs[0, 1])
        plot_weight_distribution(collector, ax=ax2, theme=theme, mode="heatmap")

        # Numerical health
        ax3 = fig.add_subplot(gs[1, 0])
        plot_numerical_health(collector, ax=ax3, theme=theme)

        # Particle diversity (just variance)
        ax4 = fig.add_subplot(gs[1, 1])
        timesteps = collector.get_timesteps().numpy()
        variance = collector.get_particle_variance().mean(dim=-1).numpy()
        ax4.plot(timesteps, variance, linewidth=2)
        ax4.set_xlabel("Timestep")
        ax4.set_ylabel("Avg Variance")
        ax4.set_title("Particle Variance")
        if theme:
            theme.apply_to_axes(ax4)

        # Pairwise distance
        ax5 = fig.add_subplot(gs[2, 0])
        pairwise = collector.get_pairwise_distances().numpy()
        ax5.plot(timesteps, pairwise, linewidth=2)
        ax5.set_xlabel("Timestep")
        ax5.set_ylabel("Avg Distance")
        ax5.set_title("Particle Spread")
        if theme:
            theme.apply_to_axes(ax5)

        # Weight entropy
        ax6 = fig.add_subplot(gs[2, 1])
        from .core_plots import plot_weight_entropy
        plot_weight_entropy(collector, ax=ax6, theme=theme)

        fig.suptitle("Particle Filter Health Dashboard", fontsize=14, y=1.02)

    elif which == "research":
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

        # Particle trajectories
        ax1 = fig.add_subplot(gs[0, :])
        plot_particle_trajectories(collector, dims=[0], style="fan", theme=theme)
        ax1.set_title("State Trajectory (Dim 0)")

        # Weighted output
        ax2 = fig.add_subplot(gs[1, 0])
        plot_weighted_output(collector, output_dims=[0], theme=theme)

        # ESS
        ax3 = fig.add_subplot(gs[1, 1])
        plot_ess_timeline(collector, ax=ax3, theme=theme)

        fig.suptitle("Research Dashboard", fontsize=14, y=1.02)

    else:  # debug
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.5))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.2)

        # ESS
        ax1 = fig.add_subplot(gs[0, 0])
        plot_ess_timeline(collector, ax=ax1, theme=theme)

        # Weights
        ax2 = fig.add_subplot(gs[0, 1])
        plot_weight_distribution(collector, ax=ax2, theme=theme)

        # Health
        ax3 = fig.add_subplot(gs[1, 0])
        plot_numerical_health(collector, ax=ax3, theme=theme)

        # Variance
        ax4 = fig.add_subplot(gs[1, 1])
        timesteps = collector.get_timesteps().numpy()
        variance = collector.get_particle_variance().mean(dim=-1).numpy()
        ax4.plot(timesteps, variance, linewidth=2)
        ax4.set_xlabel("Timestep")
        ax4.set_ylabel("Variance")
        ax4.set_title("Particle Variance")

        # Pairwise
        ax5 = fig.add_subplot(gs[2, 0])
        pairwise = collector.get_pairwise_distances().numpy()
        ax5.plot(timesteps, pairwise, linewidth=2)
        ax5.set_xlabel("Timestep")
        ax5.set_ylabel("Distance")
        ax5.set_title("Pairwise Distance")

        # Entropy
        ax6 = fig.add_subplot(gs[2, 1])
        from .core_plots import plot_weight_entropy
        plot_weight_entropy(collector, ax=ax6, theme=theme)

        # Trajectories
        ax7 = fig.add_subplot(gs[3, :])
        mean = collector.get_weighted_mean().numpy()
        ax7.plot(timesteps, mean[:, 0], linewidth=2, label="Dim 0")
        if mean.shape[1] > 1:
            ax7.plot(timesteps, mean[:, 1], linewidth=2, label="Dim 1")
        ax7.set_xlabel("Timestep")
        ax7.set_ylabel("State")
        ax7.set_title("Weighted Mean State")
        ax7.legend()

        fig.suptitle("Debug Dashboard", fontsize=14, y=1.01)

    plt.tight_layout()
    return fig
