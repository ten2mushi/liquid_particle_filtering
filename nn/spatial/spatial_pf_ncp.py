"""Spatial PFNCP wrapper with heatmap generation.

This module provides a complete wrapper that combines PFNCP with spatial
projection and rendering to output both point estimates and probability heatmaps.
"""

from typing import Tuple, Optional, Union, List, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..wrappers import PFNCP
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel
from .projection_head import SpatialProjectionHead
from .soft_renderer import SoftSpatialRenderer


class SpatialPFNCP(nn.Module):
    """PFNCP with spatial heatmap output.

    Combines a PFNCP (Particle Filter Neural Circuit Policy) with a spatial
    projection head and soft renderer to produce both:
    1. Point estimates (distance, bearing) - from the NCP output
    2. Spatial probability heatmaps - from projected particles

    The spatial projection is learned end-to-end, allowing the model to
    discover interpretable spatial representations from latent particle states.

    Args:
        wiring: NCP wiring configuration
        input_size: Input dimension
        n_particles: Number of particles (configurable)
        map_size: Heatmap resolution (square)
        r_max: Maximum detection range in meters
        env_size: Environment size in meters (for world-frame mode)
        egocentric: If True, render heatmaps in sensor-centered frame (default)
        approach: PF approach ('state', 'param', 'dual', 'sde')
        cell_type: Cell type ('cfc' or 'ltc')
        mode: CfC mode
        noise_type: Type of noise injection
        noise_init: Initial noise scale
        alpha_mode: Soft resampling alpha mode
        alpha_init: Initial alpha value
        resample_threshold: ESS threshold for resampling
        observation_model: Model for p(y|h)
        sigma_min: Minimum spatial uncertainty
        sigma_max: Maximum spatial uncertainty
        projection_hidden: Hidden size for projection MLP

    Example (egocentric - default):
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=32, output_size=2)
        >>> model = SpatialPFNCP(
        ...     wiring=wiring,
        ...     input_size=5,
        ...     n_particles=8,
        ...     map_size=64,
        ...     r_max=150.0,
        ...     egocentric=True,  # Default: sensor-centered heatmaps
        ... )
        >>> x = torch.randn(4, 50, 5)
        >>> outputs, heatmaps, state = model(x)  # No sensor_positions needed!

    Example (world-frame):
        >>> model = SpatialPFNCP(..., egocentric=False, env_size=200.0)
        >>> sensor_pos = torch.rand(4, 50, 3)
        >>> outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)
    """

    def __init__(
        self,
        wiring,
        input_size: Optional[int] = None,
        n_particles: int = 8,
        # Spatial parameters
        map_size: int = 64,
        r_max: float = 150.0,
        env_size: float = 200.0,
        egocentric: bool = True,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        projection_hidden: int = 16,
        # PFNCP parameters
        approach: Literal["state", "param", "dual", "sde"] = "state",
        cell_type: Literal["cfc", "ltc"] = "cfc",
        mode: str = "default",
        noise_type: Union[str, NoiseType] = "time_scaled",
        noise_init: float = 0.1,
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Parameter-level specific
        tracked_params: Optional[List[str]] = None,
        param_evolution_noise: float = 0.01,
        # SDE specific
        diffusion_type: str = "learned",
        solver: str = "euler_maruyama",
        # Observation model
        observation_model: Optional[ObservationModel] = None,
        # Output options
        return_sequences: bool = True,
    ):
        super().__init__()

        # Build wiring if needed
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError("Wiring not built.")

        self.wiring = wiring
        self.n_particles = n_particles
        self.map_size = map_size
        self.r_max = r_max
        self.env_size = env_size
        self.egocentric = egocentric
        self.return_sequences = return_sequences

        # Store dimensions
        self.input_size = wiring.input_dim
        self.hidden_size = wiring.units
        self.output_size = wiring.output_dim

        # Create base PFNCP
        self.pfncp = PFNCP(
            wiring=wiring,
            input_size=None,  # Already built
            n_particles=n_particles,
            approach=approach,
            cell_type=cell_type,
            mode=mode,
            noise_type=noise_type,
            noise_init=noise_init,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
            tracked_params=tracked_params,
            param_evolution_noise=param_evolution_noise,
            diffusion_type=diffusion_type,
            solver=solver,
            return_sequences=return_sequences,
            return_state=True,
            observation_model=observation_model,
        )

        # Create spatial projection head
        self.projection_head = SpatialProjectionHead(
            hidden_size=wiring.units,
            r_max=r_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            mlp_hidden=projection_hidden,
        )

        # Create soft spatial renderer
        self.renderer = SoftSpatialRenderer(
            map_size=map_size,
            r_max=r_max,
            env_size=env_size,
            egocentric=egocentric,
        )

    @property
    def cell(self):
        """Access underlying cell for step-by-step processing."""
        return self.pfncp.cell

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple] = None,
        timespans: Optional[Tensor] = None,
        observations: Optional[Tensor] = None,
        sensor_positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tuple]:
        """Forward pass with spatial heatmap generation.

        Args:
            input: Input tensor [batch, seq_len, input_size]
            hx: Initial hidden state (particles, log_weights)
            timespans: Optional time deltas [batch, seq_len, 1]
            observations: Optional observations for weight updates
            sensor_positions: Sensor positions [batch, seq_len, 3] where
                            each position is (x_norm, y_norm, heading_norm).
                            Required for world-frame mode, optional for egocentric.

        Returns:
            outputs: Point estimates [batch, seq_len, output_size] or
                    [batch, output_size] if return_sequences=False
            heatmaps: Spatial probability [batch, seq_len, 1, H, W] or
                     [batch, 1, H, W] if return_sequences=False.
                     Always generated in egocentric mode.
                     None in world-frame mode if sensor_positions not provided.
            state: Final state (particles, log_weights)
        """
        batch_size, seq_len, _ = input.shape

        # Run through PFNCP sequence
        if self.return_sequences:
            # Process step by step to generate heatmaps at each timestep
            outputs, heatmaps, final_state = self._forward_sequence(
                input, hx, timespans, observations, sensor_positions
            )
        else:
            # Just run the full sequence and generate final heatmap
            outputs, final_state = self.pfncp(
                input, hx, timespans, observations
            )

            particles, log_weights = final_state

            if self.egocentric:
                # Egocentric: heatmaps always generated (no sensor_pos needed)
                heatmaps = self._generate_heatmap(particles, log_weights)
            elif sensor_positions is not None:
                # World-frame: need sensor position
                sensor_pos_last = sensor_positions[:, -1, :]
                heatmaps = self._generate_heatmap(
                    particles, log_weights, sensor_pos_last
                )
            else:
                heatmaps = None

        return outputs, heatmaps, final_state

    def _forward_sequence(
        self,
        input: Tensor,
        hx: Optional[Tuple],
        timespans: Optional[Tensor],
        observations: Optional[Tensor],
        sensor_positions: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Tuple]:
        """Process sequence step-by-step with heatmap generation.

        Args:
            input: [batch, seq_len, input_size]
            hx: Initial state
            timespans: [batch, seq_len, 1]
            observations: [batch, seq_len, obs_size]
            sensor_positions: [batch, seq_len, 3] (optional for egocentric)

        Returns:
            outputs: [batch, seq_len, output_size]
            heatmaps: [batch, seq_len, 1, H, W] or None
            final_state: (particles, log_weights)
        """
        batch_size, seq_len, _ = input.shape

        outputs = []
        # In egocentric mode, always generate heatmaps
        # In world-frame mode, only if sensor_positions provided
        generate_heatmaps = self.egocentric or (sensor_positions is not None)
        heatmaps = [] if generate_heatmaps else None
        state = hx

        for t in range(seq_len):
            # Extract timestep data
            x_t = input[:, t, :]

            ts_t = None
            if timespans is not None:
                if timespans.dim() == 3:
                    ts_t = timespans[:, t, :]
                elif timespans.dim() == 2:
                    ts_t = timespans[:, t:t+1]

            obs_t = None
            if observations is not None:
                obs_t = observations[:, t, :]

            # Forward through PFNCP cell
            output, state = self.pfncp.cell(
                x_t, state,
                timespans=ts_t,
                observation=obs_t,
            )
            outputs.append(output)

            # Generate heatmap
            if generate_heatmaps:
                particles, log_weights = state
                if self.egocentric:
                    # Egocentric: no sensor position needed
                    heatmap_t = self._generate_heatmap(particles, log_weights)
                else:
                    # World-frame: use sensor position
                    sensor_pos_t = sensor_positions[:, t, :]
                    heatmap_t = self._generate_heatmap(
                        particles, log_weights, sensor_pos_t
                    )
                heatmaps.append(heatmap_t)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, seq, output_size]

        if heatmaps is not None:
            heatmaps = torch.stack(heatmaps, dim=1)  # [batch, seq, 1, H, W]

        return outputs, heatmaps, state

    def _generate_heatmap(
        self,
        particles: Tensor,
        log_weights: Tensor,
        sensor_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate spatial heatmap from particles.

        Args:
            particles: [batch, K, hidden_size]
            log_weights: [batch, K]
            sensor_pos: [batch, 3] (x_norm, y_norm, heading_norm)
                       Optional for egocentric mode, required for world-frame.

        Returns:
            heatmap: [batch, 1, H, W]
        """
        # Project particles to spatial coordinates
        positions, sigmas, weights = self.projection_head(
            particles, log_weights
        )

        # Render to heatmap (sensor_pos only used in world-frame mode)
        heatmap = self.renderer(positions, sigmas, weights, sensor_pos)

        return heatmap

    def get_belief_state(
        self,
        particles: Tensor,
        log_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Get spatial belief state (mean position + uncertainty).

        Args:
            particles: [batch, K, hidden_size]
            log_weights: [batch, K]

        Returns:
            mean_position: [batch, 2] (r_norm, θ_norm)
            uncertainty: [batch, 2] (σ_r, σ_θ)
        """
        return self.projection_head.get_position_uncertainty(
            particles, log_weights
        )

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}, n_particles={self.n_particles}, "
            f"map_size={self.map_size}, r_max={self.r_max}, env_size={self.env_size}"
        )
