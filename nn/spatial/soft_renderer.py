"""Soft spatial renderer for differentiable heatmap generation.

This module renders particle distributions as differentiable spatial heatmaps
using a Gaussian mixture model representation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SoftSpatialRenderer(nn.Module):
    """Render particles as differentiable spatial heatmap.

    Converts weighted particles in polar coordinates to a 2D spatial probability
    distribution by rendering each particle as a Gaussian and summing them.

    The rendering is fully differentiable, allowing gradients to flow back
    through the heatmap to the particle positions and uncertainties.

    Supports two coordinate frames:
    - **World-frame** (egocentric=False): Heatmap in absolute world coordinates.
      Requires sensor_pos for coordinate transformation.
    - **Egocentric** (egocentric=True): Heatmap centered on sensor with sensor
      heading aligned to +X axis. Sensor is always at grid center.

    Args:
        map_size: Output heatmap resolution (square: H x W)
        r_max: Maximum range in meters
        env_size: Environment size in meters (for world-frame normalization)
        min_sigma: Minimum sigma for numerical stability
        egocentric: If True, render in sensor-centered frame (default: True)

    Example (egocentric):
        >>> renderer = SoftSpatialRenderer(map_size=64, r_max=150.0, egocentric=True)
        >>> positions = torch.rand(4, 8, 2)  # [batch, K, 2] polar (r_norm, θ_norm)
        >>> sigmas = torch.ones(4, 8, 2) * 0.1
        >>> weights = torch.ones(4, 8) / 8
        >>> heatmap = renderer(positions, sigmas, weights)  # sensor_pos not needed
        >>> heatmap.shape  # [4, 1, 64, 64]

    Example (world-frame):
        >>> renderer = SoftSpatialRenderer(map_size=64, r_max=150.0, egocentric=False)
        >>> sensor_pos = torch.rand(4, 3)  # [batch, 3] (x, y, heading) normalized
        >>> heatmap = renderer(positions, sigmas, weights, sensor_pos)
    """

    def __init__(
        self,
        map_size: int = 64,
        r_max: float = 150.0,
        env_size: float = 200.0,
        min_sigma: float = 0.01,
        egocentric: bool = True,
    ):
        super().__init__()

        self.map_size = map_size
        self.r_max = r_max
        self.env_size = env_size
        self.min_sigma = min_sigma
        self.egocentric = egocentric

        # Pre-compute grid coordinates in normalized space [-1, 1]
        coords = torch.linspace(-1, 1, map_size)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='xy')

        # Register as buffers (move to device with model)
        # Shape: [1, 1, H, W] for broadcasting with [B, K, 1, 1]
        self.register_buffer('grid_x', grid_x.view(1, 1, map_size, map_size))
        self.register_buffer('grid_y', grid_y.view(1, 1, map_size, map_size))

    def forward(
        self,
        positions: Tensor,
        sigmas: Tensor,
        weights: Tensor,
        sensor_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Render weighted Gaussian mixture to spatial heatmap.

        Args:
            positions: Polar positions [batch, K, 2] where
                      positions[..., 0] = r_norm ∈ [0, 1]
                      positions[..., 1] = θ_norm ∈ [0, 1]
            sigmas: Spatial uncertainties [batch, K, 2] (σ_r, σ_θ)
            weights: Normalized importance weights [batch, K]
            sensor_pos: Sensor position [batch, 3] where
                       sensor_pos[:, 0] = x_norm ∈ [0, 1]
                       sensor_pos[:, 1] = y_norm ∈ [0, 1]
                       sensor_pos[:, 2] = heading_norm ∈ [0, 1]
                       Required for world-frame mode, optional for egocentric.

        Returns:
            heatmap: Spatial probability distribution [batch, 1, H, W]
                    Values sum to 1 over spatial dimensions.
        """
        B, K, _ = positions.shape

        # Convert polar to Cartesian based on coordinate frame
        if self.egocentric:
            x, y = self._polar_to_cartesian_egocentric(positions)
            sigma_spatial = self._compute_spatial_sigma_egocentric(sigmas)
        else:
            if sensor_pos is None:
                raise ValueError("World-frame rendering requires sensor_pos")
            x, y = self._polar_to_cartesian(positions, sensor_pos)
            sigma_spatial = self._compute_spatial_sigma(sigmas)

        # Render each particle as a Gaussian
        gaussians = self._render_gaussians(x, y, sigma_spatial)  # [B, K, H, W]

        # Weight and sum over particles
        weights_4d = weights.view(B, K, 1, 1)  # [B, K, 1, 1]
        heatmap = (gaussians * weights_4d).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Normalize to sum to 1 (proper probability distribution)
        heatmap = self._normalize_heatmap(heatmap)

        return heatmap

    def _polar_to_cartesian(
        self,
        positions: Tensor,
        sensor_pos: Tensor,
    ) -> tuple:
        """Convert polar positions to Cartesian coordinates.

        Args:
            positions: [B, K, 2] polar (r_norm, θ_norm)
            sensor_pos: [B, 3] sensor (x_norm, y_norm, heading_norm)

        Returns:
            x_abs, y_abs: [B, K] absolute positions in [-1, 1]
        """
        B, K, _ = positions.shape

        # Extract polar components
        r_norm = positions[..., 0]      # [B, K] in [0, 1]
        theta_norm = positions[..., 1]  # [B, K] in [0, 1]

        # Convert to physical units
        r = r_norm * self.r_max  # meters

        # Convert theta from [0, 1] to [-π, π]
        theta = theta_norm * 2 * math.pi - math.pi  # radians (relative to sensor)

        # Get sensor heading: [0, 1] -> [-π, π]
        sensor_heading = sensor_pos[:, 2:3] * 2 * math.pi - math.pi  # [B, 1]

        # World bearing = sensor heading + relative bearing
        world_bearing = theta + sensor_heading  # [B, K]

        # Compute offset in world frame (meters)
        dx = r * torch.cos(world_bearing)  # [B, K]
        dy = r * torch.sin(world_bearing)  # [B, K]

        # Convert sensor position from [0, 1] to [-1, 1] for grid
        sensor_xy = (sensor_pos[:, :2] - 0.5) * 2  # [B, 2] in [-1, 1]

        # Scale offset to normalized coordinates
        # env_size/2 meters corresponds to 1 unit in [-1, 1] space
        scale = 2.0 / self.env_size  # meters to normalized

        x_abs = sensor_xy[:, 0:1] + dx * scale  # [B, K]
        y_abs = sensor_xy[:, 1:2] + dy * scale  # [B, K]

        return x_abs, y_abs

    def _polar_to_cartesian_egocentric(
        self,
        positions: Tensor,
    ) -> tuple:
        """Convert polar positions to egocentric Cartesian coordinates.

        Sensor is at grid center (0, 0), heading aligned with +X axis.
        Uses standard math convention: θ=0 → +X, θ=π/2 → +Y.

        Args:
            positions: [B, K, 2] polar (r_norm, θ_norm)

        Returns:
            x, y: [B, K] positions in [-1, 1] (sensor at origin)
        """
        # Extract polar components
        r_norm = positions[..., 0]      # [B, K] in [0, 1]
        theta_norm = positions[..., 1]  # [B, K] in [0, 1]

        # Convert to physical units
        r = r_norm * self.r_max  # meters

        # Convert theta from [0, 1] to [-π, π] (already sensor-relative)
        theta = theta_norm * 2 * math.pi - math.pi  # radians

        # Standard math convention: θ=0 → +X (forward), θ=π/2 → +Y (left)
        dx = r * torch.cos(theta)  # [B, K]
        dy = r * torch.sin(theta)  # [B, K]

        # Scale to [-1, 1] grid where r_max maps to grid edge (1.0)
        scale = 1.0 / self.r_max
        x = dx * scale  # [B, K]
        y = dy * scale  # [B, K]

        return x, y

    def _compute_spatial_sigma_egocentric(self, sigmas: Tensor) -> Tensor:
        """Convert polar sigmas to spatial sigma for egocentric rendering.

        For egocentric, sigma is directly in r_max-normalized space.

        Args:
            sigmas: [B, K, 2] (σ_r, σ_θ) in normalized polar units

        Returns:
            sigma_spatial: [B, K] sigma in grid-normalized units [-1, 1]
        """
        # Average the two sigma dimensions
        sigma_polar = sigmas.mean(dim=-1)  # [B, K]

        # In egocentric frame, r_max maps to 1.0 on the grid
        # So sigma in [0,1] polar space maps directly to grid space
        # sigma_polar is already normalized [0, 1]
        sigma_spatial = sigma_polar

        # Clamp for numerical stability
        sigma_spatial = sigma_spatial.clamp(min=self.min_sigma)

        return sigma_spatial

    def _compute_spatial_sigma(self, sigmas: Tensor) -> Tensor:
        """Convert polar sigmas to spatial sigma for Gaussian rendering.

        Args:
            sigmas: [B, K, 2] (σ_r, σ_θ) in normalized polar units

        Returns:
            sigma_spatial: [B, K] sigma in grid-normalized units
        """
        # Average the two sigma dimensions
        sigma_polar = sigmas.mean(dim=-1)  # [B, K]

        # Scale: polar sigma is in [0, 1] normalized space
        # Convert to spatial units: sigma * r_max gives meters
        # Then scale to [-1, 1] grid: * (2 / env_size)
        sigma_spatial = sigma_polar * self.r_max * (2.0 / self.env_size)

        # Clamp for numerical stability
        sigma_spatial = sigma_spatial.clamp(min=self.min_sigma)

        return sigma_spatial

    def _render_gaussians(
        self,
        x: Tensor,
        y: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        """Render K Gaussians on the grid.

        Args:
            x: X positions [B, K] in [-1, 1]
            y: Y positions [B, K] in [-1, 1]
            sigma: Spatial sigma [B, K] in grid units

        Returns:
            gaussians: [B, K, H, W] unnormalized Gaussian values
        """
        B, K = x.shape

        # Reshape for broadcasting: [B, K, 1, 1]
        x = x.view(B, K, 1, 1)
        y = y.view(B, K, 1, 1)
        sigma = sigma.view(B, K, 1, 1)

        # grid_x, grid_y: [1, 1, H, W]
        # Compute squared distance
        dist_sq = (self.grid_x - x) ** 2 + (self.grid_y - y) ** 2

        # Gaussian: exp(-d² / (2σ²))
        gaussians = torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))

        return gaussians  # [B, K, H, W]

    def _normalize_heatmap(self, heatmap: Tensor) -> Tensor:
        """Normalize heatmap to sum to 1.

        Args:
            heatmap: [B, 1, H, W] unnormalized

        Returns:
            heatmap: [B, 1, H, W] normalized probability distribution
        """
        # Sum over spatial dimensions
        total = heatmap.sum(dim=(-2, -1), keepdim=True)  # [B, 1, 1, 1]

        # Normalize (with epsilon for numerical stability)
        heatmap = heatmap / (total + 1e-8)

        return heatmap

    def render_single_gaussian(
        self,
        x: Tensor,
        y: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        """Render a single Gaussian for visualization/debugging.

        Args:
            x: X position [B] or scalar in [-1, 1]
            y: Y position [B] or scalar in [-1, 1]
            sigma: Sigma [B] or scalar

        Returns:
            heatmap: [B, 1, H, W] or [1, 1, H, W]
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            sigma = sigma.unsqueeze(0)

        B = x.shape[0]
        x = x.view(B, 1)
        y = y.view(B, 1)
        sigma = sigma.view(B, 1)

        gauss = self._render_gaussians(x, y, sigma)  # [B, 1, H, W]
        return self._normalize_heatmap(gauss)

    def extra_repr(self) -> str:
        return (
            f"map_size={self.map_size}, r_max={self.r_max}, "
            f"env_size={self.env_size}, egocentric={self.egocentric}"
        )
