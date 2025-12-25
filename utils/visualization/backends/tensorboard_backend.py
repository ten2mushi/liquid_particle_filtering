"""TensorBoard backend for PFNCPS visualization.

Logs particle filter metrics and visualizations to TensorBoard
for training monitoring.
"""

from typing import TYPE_CHECKING
import io

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..core.base import PFVisualizer


def log_to_tensorboard(
    visualizer: "PFVisualizer",
    writer,
    step: int,
    prefix: str = "pf_viz",
) -> None:
    """Log visualizations and metrics to TensorBoard.

    Args:
        visualizer: PFVisualizer instance with collected data
        writer: TensorBoard SummaryWriter
        step: Global step number
        prefix: Tag prefix for logged items
    """
    if not visualizer.has_data:
        return

    collector = visualizer.collector

    # Log scalar metrics
    ess = collector.get_ess()
    writer.add_scalar(f"{prefix}/ess_mean", ess.mean().item(), step)
    writer.add_scalar(f"{prefix}/ess_min", ess.min().item(), step)
    writer.add_scalar(f"{prefix}/ess_max", ess.max().item(), step)

    entropy = collector.get_weight_entropy()
    writer.add_scalar(f"{prefix}/weight_entropy_mean", entropy.mean().item(), step)

    variance = collector.get_particle_variance().mean(dim=-1)
    writer.add_scalar(f"{prefix}/particle_variance_mean", variance.mean().item(), step)

    pairwise = collector.get_pairwise_distances()
    writer.add_scalar(f"{prefix}/pairwise_distance_mean", pairwise.mean().item(), step)

    # Log health metrics
    health = collector.get_numerical_health()
    writer.add_scalar(f"{prefix}/has_nan", health["has_nan"].any().float().item(), step)
    writer.add_scalar(f"{prefix}/has_inf", health["has_inf"].any().float().item(), step)
    writer.add_scalar(f"{prefix}/max_norm", health["max_norm"].max().item(), step)

    # Log figures if matplotlib available
    if HAS_MATPLOTLIB:
        try:
            # ESS timeline
            fig, _ = visualizer.plot_ess_timeline()
            writer.add_figure(f"{prefix}/ess_timeline", fig, step)
            plt.close(fig)

            # Weight distribution
            fig, _ = visualizer.plot_weight_distribution(mode="heatmap")
            writer.add_figure(f"{prefix}/weight_distribution", fig, step)
            plt.close(fig)

        except Exception as e:
            # Don't fail training if visualization fails
            print(f"Warning: TensorBoard visualization failed: {e}")

    # Log histograms
    weights = collector.get_weights()
    writer.add_histogram(f"{prefix}/weights", weights[-1], step)

    particles = collector.get_particles()
    writer.add_histogram(f"{prefix}/particle_norms", particles[-1].norm(dim=-1), step)
