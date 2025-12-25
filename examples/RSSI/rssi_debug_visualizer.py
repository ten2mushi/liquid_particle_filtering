#!/usr/bin/env python3
"""RSSI Debug Visualizer with Spatial Heatmap Support.

Comprehensive debug visualization for SpatialPFNCP models, showing:
- Spatial probability heatmaps from particle distributions
- Particle uncertainty projections on the environment map
- Latent space analysis (PCA projections, clustering)
- Weight distribution and ESS tracking
- Prediction errors over time
- Polar (distance, bearing) vs ground truth comparisons

Usage:
    from rssi_debug_visualizer import RSSIDebugVisualizer, RSSIDebugConfig

    visualizer = RSSIDebugVisualizer(save_dir="./output", env_size=200.0)
    visualizer.start_episode(episode_id=1)

    for t in range(seq_len):
        step_data = visualizer.collect_step(
            step=t,
            inputs=x_t,
            targets=target_t,
            predictions=output,
            state=state,
            sensor_positions=sensor_pos,
            heatmap=heatmap_t,
        )
        visualizer.add_step(step_data)

    visualizer.save_episode()  # Creates animated GIF
    visualizer.save_summary()  # Creates static summary plots
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

import numpy as np
import torch
from torch import Tensor

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import Normalize
import matplotlib.cm as cm


@dataclass
class RSSIDebugConfig:
    """Configuration for RSSI debug visualization."""

    # Figure settings
    figsize: Tuple[int, int] = (24, 16)
    dpi: int = 150

    # Animation settings
    frame_duration_ms: int = 100
    fps: int = 10

    # Heatmap settings
    heatmap_cmap: str = "hot"
    heatmap_alpha: float = 0.8

    # Particle visualization
    max_particles_to_show: int = 32
    particle_alpha: float = 0.7
    particle_size_scale: float = 100.0  # Scale factor for weight-based sizing

    # Latent space visualization
    latent_dims_to_show: int = 6  # Number of latent dimensions to plot
    use_pca: bool = True  # Use PCA for 2D projection

    # Colors
    sensor_color: str = "#2196F3"  # Blue
    target_color: str = "#4CAF50"  # Green
    prediction_color: str = "#FF9800"  # Orange
    particle_cmap: str = "viridis"

    # Error thresholds for color coding
    dist_error_good_m: float = 10.0
    dist_error_warn_m: float = 30.0
    bear_error_good_rad: float = 0.3
    bear_error_warn_rad: float = 0.8


@dataclass
class StepData:
    """Data collected for a single timestep."""

    step: int
    inputs: np.ndarray  # (input_size,)
    targets: np.ndarray  # (2,) [dist_norm, bear_norm]
    predictions: np.ndarray  # (2,) [dist_norm, bear_norm]
    sensor_pos: np.ndarray  # (3,) [x_norm, y_norm, heading_norm]
    particles: Optional[np.ndarray] = None  # (K, hidden_size)
    log_weights: Optional[np.ndarray] = None  # (K,)
    heatmap: Optional[np.ndarray] = None  # (H, W)
    projected_positions: Optional[np.ndarray] = None  # (K, 2) polar
    projected_sigmas: Optional[np.ndarray] = None  # (K, 2)
    target_pos_abs: Optional[np.ndarray] = None  # (2,) absolute Cartesian


class RSSIDebugVisualizer:
    """Comprehensive debug visualizer for SpatialPFNCP RSSI localization.

    Generates multi-panel visualizations showing:
    1. Environment map with sensor, target, prediction, and uncertainty
    2. Spatial heatmap from particle distribution
    3. Particle state latent space (PCA projection)
    4. Weight distribution and ESS
    5. Prediction errors over time
    6. Polar coordinate comparison (distance and bearing)
    """

    def __init__(
        self,
        save_dir: str,
        env_size: float = 200.0,
        r_max: float = 150.0,
        n_particles: int = 32,
        hidden_size: int = 32,
        config: Optional[RSSIDebugConfig] = None,
        episode_prefix: str = "debug_episode",
        egocentric: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.env_size = env_size
        self.r_max = r_max
        self.n_particles = n_particles
        self.hidden_size = hidden_size
        self.config = config or RSSIDebugConfig()
        self.episode_prefix = episode_prefix
        self.egocentric = egocentric

        # Episode state
        self.episode_id: Optional[int] = None
        self.steps: List[StepData] = []

        # Accumulated metrics for plotting
        self.ess_history: List[float] = []
        self.dist_error_history: List[float] = []
        self.bear_error_history: List[float] = []
        self.pos_error_history: List[float] = []  # Cartesian error in meters
        self.weight_entropy_history: List[float] = []

        # For PCA projection caching
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None

    def start_episode(self, episode_id: int):
        """Start collecting data for a new episode."""
        self.episode_id = episode_id
        self.steps = []
        self.ess_history = []
        self.dist_error_history = []
        self.bear_error_history = []
        self.pos_error_history = []
        self.weight_entropy_history = []
        self._pca_components = None
        self._pca_mean = None

    def collect_step(
        self,
        step: int,
        inputs: Tensor,
        targets: Tensor,
        predictions: Tensor,
        state: Tuple[Tensor, Tensor],
        sensor_positions: Tensor,
        heatmap: Optional[Tensor] = None,
        projected_positions: Optional[Tensor] = None,
        projected_sigmas: Optional[Tensor] = None,
        target_pos_abs: Optional[Tensor] = None,
        batch_idx: int = 0,
    ) -> StepData:
        """Collect and preprocess data for a single timestep.

        Args:
            step: Current timestep index
            inputs: Input tensor [batch, input_size] or [input_size]
            targets: Target [batch, 2] or [2] (dist_norm, bear_norm)
            predictions: Predicted [batch, 2] or [2] (dist_norm, bear_norm)
            state: (particles, log_weights) tuple
            sensor_positions: [batch, 3] or [3] (x_norm, y_norm, heading_norm)
            heatmap: [batch, 1, H, W] or [H, W] spatial probability
            projected_positions: [batch, K, 2] particle polar positions
            projected_sigmas: [batch, K, 2] particle uncertainties
            target_pos_abs: [batch, 2] or [2] absolute Cartesian target
            batch_idx: Which batch element to visualize

        Returns:
            StepData containing preprocessed numpy arrays
        """

        def to_numpy(t: Tensor, squeeze_batch: bool = True) -> np.ndarray:
            if t is None:
                return None
            arr = t.detach().cpu().numpy()
            if squeeze_batch and arr.ndim > 1 and arr.shape[0] > batch_idx:
                arr = arr[batch_idx]
            return arr

        # Extract state
        particles, log_weights = state if state else (None, None)

        step_data = StepData(
            step=step,
            inputs=to_numpy(inputs),
            targets=to_numpy(targets),
            predictions=to_numpy(predictions),
            sensor_pos=to_numpy(sensor_positions),
            particles=to_numpy(particles) if particles is not None else None,
            log_weights=to_numpy(log_weights) if log_weights is not None else None,
            heatmap=to_numpy(heatmap.squeeze(1)) if heatmap is not None else None,
            projected_positions=to_numpy(projected_positions) if projected_positions is not None else None,
            projected_sigmas=to_numpy(projected_sigmas) if projected_sigmas is not None else None,
            target_pos_abs=to_numpy(target_pos_abs) if target_pos_abs is not None else None,
        )

        return step_data

    def add_step(self, step_data: StepData):
        """Add a step to the current episode and compute metrics."""
        self.steps.append(step_data)

        # Compute ESS
        if step_data.log_weights is not None:
            log_w = step_data.log_weights
            log_w = log_w - log_w.max()
            w = np.exp(log_w)
            w = w / w.sum()
            ess = 1.0 / (w ** 2).sum()
            self.ess_history.append(ess)

            # Weight entropy
            entropy = -np.sum(w * np.log(w + 1e-10))
            self.weight_entropy_history.append(entropy)
        else:
            self.ess_history.append(0)
            self.weight_entropy_history.append(0)

        # Compute errors
        pred = step_data.predictions
        target = step_data.targets

        # Distance error in meters
        dist_err_m = abs(pred[0] - target[0]) * self.r_max
        self.dist_error_history.append(dist_err_m)

        # Bearing error (circular)
        bear_diff = pred[1] - target[1]
        bear_diff = (bear_diff + 0.5) % 1.0 - 0.5  # Wrap to [-0.5, 0.5]
        bear_err_rad = abs(bear_diff) * 2 * np.pi
        self.bear_error_history.append(bear_err_rad)

        # Position error in Cartesian (if we have sensor position)
        if step_data.sensor_pos is not None:
            pos_err = self._compute_cartesian_error(step_data)
            self.pos_error_history.append(pos_err)
        else:
            self.pos_error_history.append(0)

    def _compute_cartesian_error(self, step_data: StepData) -> float:
        """Compute Cartesian position error in meters."""
        sensor = step_data.sensor_pos
        pred = step_data.predictions
        target = step_data.targets

        # Convert polar to Cartesian
        def polar_to_cart(dist_norm, bear_norm, sensor_heading_norm):
            dist_m = dist_norm * self.r_max
            bear_rad = bear_norm * 2 * np.pi - np.pi
            heading_rad = sensor_heading_norm * 2 * np.pi - np.pi
            world_bear = heading_rad + bear_rad
            dx = dist_m * np.cos(world_bear)
            dy = dist_m * np.sin(world_bear)
            return dx, dy

        pred_dx, pred_dy = polar_to_cart(pred[0], pred[1], sensor[2])
        gt_dx, gt_dy = polar_to_cart(target[0], target[1], sensor[2])

        error = np.sqrt((pred_dx - gt_dx) ** 2 + (pred_dy - gt_dy) ** 2)
        return error

    def _polar_to_cartesian_abs(
        self,
        dist_norm: float,
        bear_norm: float,
        sensor_pos: np.ndarray,
    ) -> Tuple[float, float]:
        """Convert polar (normalized) to absolute Cartesian coordinates."""
        dist_m = dist_norm * self.r_max
        bear_rad = bear_norm * 2 * np.pi - np.pi
        heading_rad = sensor_pos[2] * 2 * np.pi - np.pi
        world_bear = heading_rad + bear_rad

        # Sensor position in meters
        sensor_x = (sensor_pos[0] - 0.5) * self.env_size
        sensor_y = (sensor_pos[1] - 0.5) * self.env_size

        target_x = sensor_x + dist_m * np.cos(world_bear)
        target_y = sensor_y + dist_m * np.sin(world_bear)

        return target_x, target_y

    def _compute_pca(self, particles: np.ndarray) -> np.ndarray:
        """Compute 2D PCA projection of particles."""
        if particles is None or len(particles) == 0:
            return np.zeros((0, 2))

        # Flatten if needed
        if particles.ndim == 1:
            particles = particles.reshape(1, -1)

        # Center the data
        mean = particles.mean(axis=0)
        centered = particles - mean

        # Compute SVD for PCA
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Take first 2 components
            projection = centered @ Vt[:2].T

            # Cache for consistent projections across frames
            self._pca_mean = mean
            self._pca_components = Vt[:2]

            return projection
        except np.linalg.LinAlgError:
            return np.zeros((particles.shape[0], 2))

    def _render_frame(self, step_idx: int) -> plt.Figure:
        """Render a single frame for the animation."""
        step = self.steps[step_idx]
        cfg = self.config

        # Create figure with GridSpec layout
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Panel layout:
        # [0,0-1]: Environment map with positions and heatmap
        # [0,2]: Spatial heatmap (zoomed)
        # [0,3]: Particle latent space (PCA)
        # [1,0]: Weight distribution
        # [1,1]: ESS timeline
        # [1,2]: Distance error timeline
        # [1,3]: Bearing error timeline
        # [2,0-1]: Polar coordinate comparison
        # [2,2]: Latent dimension histograms
        # [2,3]: Summary metrics

        # === Panel 1: Environment Map ===
        ax_map = fig.add_subplot(gs[0, 0:2])
        self._plot_environment_map(ax_map, step)

        # === Panel 2: Spatial Heatmap (zoomed) ===
        ax_heat = fig.add_subplot(gs[0, 2])
        self._plot_spatial_heatmap(ax_heat, step)

        # === Panel 3: Particle Latent Space ===
        ax_latent = fig.add_subplot(gs[0, 3])
        self._plot_latent_space(ax_latent, step)

        # === Panel 4: Weight Distribution ===
        ax_weights = fig.add_subplot(gs[1, 0])
        self._plot_weight_distribution(ax_weights, step)

        # === Panel 5: ESS Timeline ===
        ax_ess = fig.add_subplot(gs[1, 1])
        self._plot_ess_timeline(ax_ess, step_idx)

        # === Panel 6: Distance Error Timeline ===
        ax_dist = fig.add_subplot(gs[1, 2])
        self._plot_error_timeline(ax_dist, step_idx, "distance")

        # === Panel 7: Bearing Error Timeline ===
        ax_bear = fig.add_subplot(gs[1, 3])
        self._plot_error_timeline(ax_bear, step_idx, "bearing")

        # === Panel 8: Polar Coordinate Comparison ===
        ax_polar = fig.add_subplot(gs[2, 0:2], polar=True)
        self._plot_polar_comparison(ax_polar, step)

        # === Panel 9: Latent Dimension Histograms ===
        ax_dims = fig.add_subplot(gs[2, 2])
        self._plot_latent_dimensions(ax_dims, step)

        # === Panel 10: Summary Metrics ===
        ax_summary = fig.add_subplot(gs[2, 3])
        self._plot_summary_metrics(ax_summary, step_idx)

        # Title
        fig.suptitle(
            f"Episode {self.episode_id} | Step {step.step + 1}/{len(self.steps)}",
            fontsize=16, fontweight="bold", y=0.98
        )

        return fig

    def _plot_environment_map(self, ax: plt.Axes, step: StepData):
        """Plot environment map with sensor, target, prediction, and particles.

        In egocentric mode, the heatmap is in sensor-centered coordinates and
        won't be overlaid on the world-frame map (shown separately in Panel 2).
        """
        cfg = self.config

        # Environment bounds in meters
        half_size = self.env_size / 2

        # Sensor position
        sensor_x = (step.sensor_pos[0] - 0.5) * self.env_size
        sensor_y = (step.sensor_pos[1] - 0.5) * self.env_size
        sensor_heading = step.sensor_pos[2] * 2 * np.pi - np.pi

        # Target position (from polar)
        target_x, target_y = self._polar_to_cartesian_abs(
            step.targets[0], step.targets[1], step.sensor_pos
        )

        # Prediction position (from polar)
        pred_x, pred_y = self._polar_to_cartesian_abs(
            step.predictions[0], step.predictions[1], step.sensor_pos
        )

        # Plot heatmap as background if available (only for world-frame mode)
        # In egocentric mode, heatmap is shown separately in Panel 2
        if step.heatmap is not None and not self.egocentric:
            # Heatmap coordinates: [-half_size, half_size]
            extent = [-half_size, half_size, -half_size, half_size]
            ax.imshow(
                step.heatmap,
                extent=extent,
                origin="lower",
                cmap=cfg.heatmap_cmap,
                alpha=cfg.heatmap_alpha,
                aspect="auto",
            )

        # Plot particles if projected positions available
        if step.projected_positions is not None and step.log_weights is not None:
            weights = np.exp(step.log_weights - step.log_weights.max())
            weights = weights / weights.sum()

            for i in range(min(len(step.projected_positions), cfg.max_particles_to_show)):
                px, py = self._polar_to_cartesian_abs(
                    step.projected_positions[i, 0],
                    step.projected_positions[i, 1],
                    step.sensor_pos,
                )
                size = weights[i] * cfg.particle_size_scale * self.n_particles
                alpha = min(0.3 + weights[i] * 2, 0.9)

                # Uncertainty ellipse if available
                if step.projected_sigmas is not None:
                    sigma_r = step.projected_sigmas[i, 0] * self.r_max
                    sigma_t = step.projected_sigmas[i, 1] * 2 * np.pi * step.projected_positions[i, 0] * self.r_max
                    sigma_avg = (sigma_r + sigma_t) / 2
                    circle = Circle(
                        (px, py), sigma_avg,
                        fill=False, color="gray", alpha=alpha * 0.5,
                        linewidth=0.5
                    )
                    ax.add_patch(circle)

                ax.scatter(px, py, s=size, c=[weights[i]], cmap=cfg.particle_cmap,
                          alpha=alpha, vmin=0, vmax=weights.max())

        # Plot sensor with heading arrow
        ax.scatter(sensor_x, sensor_y, s=200, c=cfg.sensor_color, marker="^",
                  zorder=10, label="Sensor", edgecolors="white", linewidth=2)

        # Heading arrow
        arrow_len = 15
        ax.annotate(
            "", xy=(sensor_x + arrow_len * np.cos(sensor_heading),
                   sensor_y + arrow_len * np.sin(sensor_heading)),
            xytext=(sensor_x, sensor_y),
            arrowprops=dict(arrowstyle="->", color=cfg.sensor_color, lw=2),
            zorder=11
        )

        # Plot target (ground truth)
        ax.scatter(target_x, target_y, s=150, c=cfg.target_color, marker="*",
                  zorder=12, label="Target (GT)", edgecolors="white", linewidth=1)

        # Plot prediction
        ax.scatter(pred_x, pred_y, s=100, c=cfg.prediction_color, marker="o",
                  zorder=11, label="Prediction", edgecolors="white", linewidth=1)

        # Detection range circle
        range_circle = Circle(
            (sensor_x, sensor_y), self.r_max,
            fill=False, color="gray", linestyle="--", alpha=0.5, linewidth=1
        )
        ax.add_patch(range_circle)

        # Connect sensor to target and prediction
        ax.plot([sensor_x, target_x], [sensor_y, target_y],
               color=cfg.target_color, linestyle="-", alpha=0.5, linewidth=1)
        ax.plot([sensor_x, pred_x], [sensor_y, pred_y],
               color=cfg.prediction_color, linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlim(-half_size, half_size)
        ax.set_ylim(-half_size, half_size)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Environment Map with Particle Distribution")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_spatial_heatmap(self, ax: plt.Axes, step: StepData):
        """Plot zoomed spatial heatmap.

        For egocentric mode: sensor at center, heading aligned with +X axis.
        Axes labeled as Forward (+X) and Left (+Y).
        """
        if step.heatmap is None:
            ax.text(0.5, 0.5, "No heatmap\navailable",
                   ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Spatial Probability")
            return

        H, W = step.heatmap.shape

        if self.egocentric:
            # Egocentric: sensor at center, r_max at edges
            # Grid goes from -r_max to +r_max in meters
            extent = [-self.r_max, self.r_max, -self.r_max, self.r_max]
        else:
            # World-frame: grid is [0, 1] normalized -> [-env/2, env/2] meters
            half_size = self.env_size / 2
            extent = [-half_size, half_size, -half_size, half_size]

        im = ax.imshow(
            step.heatmap,
            extent=extent,
            cmap=self.config.heatmap_cmap,
            origin="lower",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, label="Probability")

        # Mark max probability location
        max_idx = np.unravel_index(np.argmax(step.heatmap), step.heatmap.shape)
        # Convert grid indices to coordinate space
        max_x = extent[0] + (max_idx[1] + 0.5) / W * (extent[1] - extent[0])
        max_y = extent[2] + (max_idx[0] + 0.5) / H * (extent[3] - extent[2])
        ax.scatter(max_x, max_y, c="white", marker="x", s=50, linewidth=2)

        if self.egocentric:
            # Mark sensor at center
            ax.scatter(0, 0, c="cyan", marker="^", s=100, edgecolors="white",
                      linewidth=2, label="Sensor", zorder=10)

            # Add heading arrow (forward = +X)
            arrow_len = self.r_max * 0.3
            ax.annotate("", xy=(arrow_len, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color="cyan", lw=2), zorder=11)

            ax.set_xlabel("Forward → (+X, meters)")
            ax.set_ylabel("Left ↑ (+Y, meters)")
            ax.set_title(f"Egocentric Heatmap (max={step.heatmap.max():.4f})")
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.set_title(f"World-Frame Heatmap (max={step.heatmap.max():.4f})")

    def _plot_latent_space(self, ax: plt.Axes, step: StepData):
        """Plot 2D PCA projection of particle latent states."""
        if step.particles is None:
            ax.text(0.5, 0.5, "No particles\navailable",
                   ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Latent Space (PCA)")
            return

        # Compute PCA projection
        projection = self._compute_pca(step.particles)

        if step.log_weights is not None:
            weights = np.exp(step.log_weights - step.log_weights.max())
            weights = weights / weights.sum()
            sizes = weights * self.config.particle_size_scale * self.n_particles
            # colors = weights  # OLD: Color by weight
            colors = "blue"     # NEW: Fixed blue color
        else:
            sizes = 50
            colors = "blue"

        # Plot particles
        # Removed cmap and vmin since we use fixed color
        scatter = ax.scatter(
            projection[:, 0], projection[:, 1],
            s=sizes, c=colors, 
            alpha=self.config.particle_alpha
        )

        # Mark weighted mean
        if step.log_weights is not None:
            mean_proj = (projection * weights[:, np.newaxis]).sum(axis=0)
            # Changed marker to 'P' (plus/cross) for "red cross"
            ax.scatter(mean_proj[0], mean_proj[1], c="red", marker="P", s=150,
                      edgecolors="white", linewidth=2, label="Weighted Mean", zorder=10)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Particle Latent Space (PCA)")
        
        # Fixed scale [-1, 1]
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_weight_distribution(self, ax: plt.Axes, step: StepData):
        """Plot particle weight distribution."""
        if step.log_weights is None:
            ax.text(0.5, 0.5, "No weights\navailable",
                   ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Weight Distribution")
            return

        weights = np.exp(step.log_weights - step.log_weights.max())
        weights = weights / weights.sum()

        # Sort for visualization
        sorted_idx = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_idx]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(weights)))
        bars = ax.bar(range(len(weights)), sorted_weights, color=colors[sorted_idx])

        # Add uniform reference line
        uniform = 1.0 / len(weights)
        ax.axhline(y=uniform, color="red", linestyle="--", alpha=0.7, label=f"Uniform ({uniform:.3f})")

        ax.set_xlabel("Particle (sorted)")
        ax.set_ylabel("Weight")
        ax.set_title(f"Weight Distribution (max={weights.max():.3f})")
        ax.legend(fontsize=8)

    def _plot_ess_timeline(self, ax: plt.Axes, current_step: int):
        """Plot ESS over time."""
        steps = range(len(self.ess_history))

        ax.plot(steps, self.ess_history, "b-", linewidth=2, label="ESS")
        ax.axvline(x=current_step, color="red", linestyle="--", alpha=0.7, label="Current")

        # Threshold line
        threshold = self.n_particles * 0.5
        ax.axhline(y=threshold, color="orange", linestyle=":", alpha=0.7,
                  label=f"Resample threshold ({threshold:.0f})")

        ax.set_xlabel("Step")
        ax.set_ylabel("ESS")
        ax.set_title(f"Effective Sample Size ({self.ess_history[current_step]:.1f})")
        ax.set_ylim(0, self.n_particles * 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_error_timeline(self, ax: plt.Axes, current_step: int, error_type: str):
        """Plot error timeline for distance or bearing."""
        cfg = self.config

        if error_type == "distance":
            errors = self.dist_error_history
            ylabel = "Error (meters)"
            title = f"Distance Error ({errors[current_step]:.1f}m)"
            good_thresh = cfg.dist_error_good_m
            warn_thresh = cfg.dist_error_warn_m
        else:
            errors = self.bear_error_history
            ylabel = "Error (radians)"
            title = f"Bearing Error ({errors[current_step]:.2f}rad)"
            good_thresh = cfg.bear_error_good_rad
            warn_thresh = cfg.bear_error_warn_rad

        steps = range(len(errors))

        # Color based on current error
        current_err = errors[current_step]
        if current_err < good_thresh:
            color = "green"
        elif current_err < warn_thresh:
            color = "orange"
        else:
            color = "red"

        ax.plot(steps, errors, color=color, linewidth=2)
        ax.axvline(x=current_step, color="gray", linestyle="--", alpha=0.7)

        # Threshold bands
        ax.axhline(y=good_thresh, color="green", linestyle=":", alpha=0.5)
        ax.axhline(y=warn_thresh, color="orange", linestyle=":", alpha=0.5)

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _plot_polar_comparison(self, ax: plt.Axes, step: StepData):
        """Plot polar coordinate comparison between prediction and target."""
        # Ground truth
        gt_r = step.targets[0] * self.r_max
        gt_theta = step.targets[1] * 2 * np.pi - np.pi

        # Prediction
        pred_r = step.predictions[0] * self.r_max
        pred_theta = step.predictions[1] * 2 * np.pi - np.pi

        # Plot
        ax.scatter([gt_theta], [gt_r], s=150, c=self.config.target_color,
                  marker="*", label="Target", zorder=10)
        ax.scatter([pred_theta], [pred_r], s=100, c=self.config.prediction_color,
                  marker="o", label="Prediction", zorder=10)

        # Connect with arc
        ax.plot([gt_theta, pred_theta], [gt_r, pred_r],
               color="gray", linestyle="--", alpha=0.5)

        ax.set_ylim(0, self.r_max * 1.1)
        ax.set_title("Polar Coordinates (Sensor Frame)")
        ax.legend(loc="upper right", fontsize=8)

    def _plot_latent_dimensions(self, ax: plt.Axes, step: StepData):
        """Plot histograms of select latent dimensions."""
        if step.particles is None:
            ax.text(0.5, 0.5, "No particles",
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Latent Dimensions")
            return

        n_dims = min(self.config.latent_dims_to_show, step.particles.shape[1])

        for i in range(n_dims):
            values = step.particles[:, i]
            ax.hist(values, bins=15, alpha=0.5, label=f"D{i}")

        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_title(f"Latent Dimension Histograms (D0-D{n_dims-1})")
        ax.legend(fontsize=6, ncol=2)

    def _plot_summary_metrics(self, ax: plt.Axes, current_step: int):
        """Plot summary metrics panel."""
        ax.axis("off")

        cfg = self.config
        step = self.steps[current_step]

        # Current metrics
        ess = self.ess_history[current_step] if current_step < len(self.ess_history) else 0
        dist_err = self.dist_error_history[current_step] if current_step < len(self.dist_error_history) else 0
        bear_err = self.bear_error_history[current_step] if current_step < len(self.bear_error_history) else 0
        pos_err = self.pos_error_history[current_step] if current_step < len(self.pos_error_history) else 0

        # Running averages
        avg_dist = np.mean(self.dist_error_history[:current_step + 1]) if self.dist_error_history else 0
        avg_bear = np.mean(self.bear_error_history[:current_step + 1]) if self.bear_error_history else 0
        avg_ess = np.mean(self.ess_history[:current_step + 1]) if self.ess_history else 0

        # Color functions
        def dist_color(e):
            return "green" if e < cfg.dist_error_good_m else ("orange" if e < cfg.dist_error_warn_m else "red")

        def bear_color(e):
            return "green" if e < cfg.bear_error_good_rad else ("orange" if e < cfg.bear_error_warn_rad else "red")

        def ess_color(e):
            return "green" if e > self.n_particles * 0.5 else ("orange" if e > self.n_particles * 0.25 else "red")

        # Build text
        lines = [
            ("Current Step Metrics", "bold", "black"),
            ("─" * 25, "normal", "gray"),
            (f"Distance Error: {dist_err:.1f} m", "normal", dist_color(dist_err)),
            (f"Bearing Error: {bear_err:.2f} rad ({np.degrees(bear_err):.1f}°)", "normal", bear_color(bear_err)),
            (f"Position Error: {pos_err:.1f} m", "normal", dist_color(pos_err)),
            (f"ESS: {ess:.1f} / {self.n_particles}", "normal", ess_color(ess)),
            ("", "normal", "black"),
            ("Running Averages", "bold", "black"),
            ("─" * 25, "normal", "gray"),
            (f"Avg Distance Error: {avg_dist:.1f} m", "normal", dist_color(avg_dist)),
            (f"Avg Bearing Error: {avg_bear:.2f} rad", "normal", bear_color(avg_bear)),
            (f"Avg ESS: {avg_ess:.1f}", "normal", ess_color(avg_ess)),
            ("", "normal", "black"),
            ("Inputs (RSSI, actions)", "bold", "black"),
            ("─" * 25, "normal", "gray"),
            (f"RSSI: [{step.inputs[0]:.2f}, {step.inputs[1]:.2f}]", "normal", "black"),
            (f"Rot/Speed: [{step.inputs[2]:.2f}, {step.inputs[3]:.2f}]", "normal", "black"),
        ]

        y_pos = 0.95
        for text, weight, color in lines:
            ax.text(0.05, y_pos, text,
                   transform=ax.transAxes,
                   fontsize=9,
                   fontweight=weight,
                   color=color,
                   family="monospace",
                   verticalalignment="top")
            y_pos -= 0.058

        ax.set_title("Summary Metrics", fontsize=10, fontweight="bold")

    def save_episode(self) -> Path:
        """Save episode as animated GIF."""
        if not self.steps:
            raise ValueError("No steps collected. Call add_step() first.")

        frames = []
        for i in range(len(self.steps)):
            fig = self._render_frame(i)
            # Convert figure to image using buffer_rgba for reliable dimensions
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            # Convert RGBA to RGB
            image = image[:, :, :3]
            frames.append(image.copy())
            plt.close(fig)

        # Save as GIF using imageio if available, otherwise use PIL
        gif_path = self.save_dir / f"{self.episode_prefix}_{self.episode_id}.gif"

        try:
            import imageio
            imageio.mimsave(
                gif_path,
                frames,
                duration=self.config.frame_duration_ms / 1000.0,
                loop=0
            )
        except ImportError:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=self.config.frame_duration_ms,
                loop=0
            )

        print(f"  Saved: {gif_path}")
        return gif_path

    def save_summary(self) -> Path:
        """Save static summary plot for the episode."""
        if not self.steps:
            raise ValueError("No steps collected.")

        fig = plt.figure(figsize=(16, 10), dpi=self.config.dpi)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # ESS over episode
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.ess_history, "b-", linewidth=2)
        ax1.axhline(y=self.n_particles * 0.5, color="orange", linestyle="--", alpha=0.7)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("ESS")
        ax1.set_title(f"ESS (mean={np.mean(self.ess_history):.1f})")
        ax1.grid(True, alpha=0.3)

        # Distance error over episode
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.dist_error_history, "g-", linewidth=2)
        ax2.axhline(y=self.config.dist_error_good_m, color="green", linestyle=":", alpha=0.5)
        ax2.axhline(y=self.config.dist_error_warn_m, color="orange", linestyle=":", alpha=0.5)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Error (m)")
        ax2.set_title(f"Distance Error (mean={np.mean(self.dist_error_history):.1f}m)")
        ax2.grid(True, alpha=0.3)

        # Bearing error over episode
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.bear_error_history, "orange", linewidth=2)
        ax3.axhline(y=self.config.bear_error_good_rad, color="green", linestyle=":", alpha=0.5)
        ax3.axhline(y=self.config.bear_error_warn_rad, color="orange", linestyle=":", alpha=0.5)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Error (rad)")
        ax3.set_title(f"Bearing Error (mean={np.mean(self.bear_error_history):.2f}rad)")
        ax3.grid(True, alpha=0.3)

        # Weight entropy over episode
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.weight_entropy_history, "purple", linewidth=2)
        max_entropy = np.log(self.n_particles)
        ax4.axhline(y=max_entropy, color="gray", linestyle="--", alpha=0.7, label="Max entropy")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Entropy")
        ax4.set_title("Weight Entropy")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Position error over episode
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.pos_error_history, "red", linewidth=2)
        ax5.set_xlabel("Step")
        ax5.set_ylabel("Error (m)")
        ax5.set_title(f"Cartesian Position Error (mean={np.mean(self.pos_error_history):.1f}m)")
        ax5.grid(True, alpha=0.3)

        # Final heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        if self.steps[-1].heatmap is not None:
            im = ax6.imshow(
                self.steps[-1].heatmap,
                cmap=self.config.heatmap_cmap,
                origin="lower"
            )
            plt.colorbar(im, ax=ax6)
            ax6.set_title("Final Spatial Heatmap")
        else:
            ax6.text(0.5, 0.5, "No heatmap", ha="center", va="center", transform=ax6.transAxes)
            ax6.set_title("Final Spatial Heatmap")

        fig.suptitle(
            f"Episode {self.episode_id} Summary | {len(self.steps)} steps | "
            f"Final: dist={self.dist_error_history[-1]:.1f}m, bear={self.bear_error_history[-1]:.2f}rad",
            fontsize=14, fontweight="bold"
        )

        summary_path = self.save_dir / f"{self.episode_prefix}_{self.episode_id}_summary.png"
        plt.savefig(summary_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved: {summary_path}")
        return summary_path
