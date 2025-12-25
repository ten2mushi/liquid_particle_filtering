"""Export utilities for PFNCPS visualizations."""

import os
from typing import List, TYPE_CHECKING

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..core.base import PFVisualizer


def save_all_plots(
    visualizer: "PFVisualizer",
    output_dir: str,
    format: str = "png",
    dpi: int = 150,
) -> List[str]:
    """Save all relevant plots to a directory.

    Args:
        visualizer: PFVisualizer instance with collected data
        output_dir: Output directory path
        format: Image format ("png", "pdf", "svg")
        dpi: Resolution for raster formats

    Returns:
        List of saved file paths
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for saving plots")

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # Core plots to save
    plot_configs = [
        ("ess_timeline", visualizer.plot_ess_timeline, {}),
        ("weight_distribution", visualizer.plot_weight_distribution, {"mode": "heatmap"}),
        ("weight_entropy", visualizer.plot_weight_entropy, {}),
        ("particle_trajectories", visualizer.plot_particle_trajectories, {"style": "fan"}),
        ("numerical_health", visualizer.plot_numerical_health, {}),
        ("weighted_output", visualizer.plot_weighted_output, {}),
    ]

    for name, plot_fn, kwargs in plot_configs:
        try:
            result = plot_fn(**kwargs)
            if isinstance(result, tuple):
                fig = result[0]
            else:
                fig = result

            filepath = os.path.join(output_dir, f"{name}.{format}")
            fig.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(filepath)
        except Exception as e:
            print(f"Warning: Failed to save {name}: {e}")

    # Save dashboard
    try:
        fig = visualizer.dashboard("health")
        filepath = os.path.join(output_dir, f"dashboard_health.{format}")
        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(filepath)
    except Exception as e:
        print(f"Warning: Failed to save dashboard: {e}")

    return saved_files
