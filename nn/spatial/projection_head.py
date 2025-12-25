"""Spatial projection head for mapping latent particles to polar coordinates.

This module provides the bridge between abstract latent particle states and
interpretable polar spatial coordinates with uncertainty estimates.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sigmoid_range(x: Tensor, low: float, high: float) -> Tensor:
    """Apply sigmoid and scale to range [low, high].

    Args:
        x: Input tensor
        low: Minimum output value
        high: Maximum output value

    Returns:
        Tensor scaled to [low, high]
    """
    return low + (high - low) * torch.sigmoid(x)


class SpatialProjectionHead(nn.Module):
    """Project latent particle states to polar coordinates with uncertainty.

    Maps each particle's latent hidden state h ∈ R^H to a spatial position
    in polar coordinates (r, θ) along with per-particle uncertainty estimates
    (σ_r, σ_θ).

    The projection is learned end-to-end, allowing the network to discover
    meaningful spatial representations from the latent particle states.

    Args:
        hidden_size: Dimension of input latent particle states
        r_max: Maximum range in meters (for context, not used in normalization)
        sigma_min: Minimum spatial uncertainty (normalized)
        sigma_max: Maximum spatial uncertainty (normalized)
        mlp_hidden: Hidden layer size for projection MLP

    Example:
        >>> head = SpatialProjectionHead(hidden_size=32)
        >>> particles = torch.randn(4, 8, 32)  # [batch, K, hidden]
        >>> log_weights = torch.zeros(4, 8)     # [batch, K]
        >>> positions, sigmas, weights = head(particles, log_weights)
        >>> positions.shape  # [4, 8, 2] (r_norm, θ_norm)
        >>> sigmas.shape     # [4, 8, 2] (σ_r, σ_θ)
    """

    def __init__(
        self,
        hidden_size: int,
        r_max: float = 150.0,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        mlp_hidden: int = 16,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.r_max = r_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # MLP: hidden_size -> 4 outputs [r, θ, log_σ_r, log_σ_θ]
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 4),
        )

        # Initialize for reasonable initial outputs
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable initial predictions."""
        # Initialize final layer near zero for bounded outputs near 0.5
        with torch.no_grad():
            # Get the final linear layer
            final_layer = self.projection[-1]
            # Small weights
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
            # Bias for sigma outputs (indices 2, 3) to start at mid-range
            final_layer.bias.zero_()

    def forward(
        self,
        particles: Tensor,
        log_weights: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Project particles to spatial coordinates with uncertainty.

        Args:
            particles: Latent particle states [batch, K, hidden_size]
            log_weights: Log importance weights [batch, K]

        Returns:
            positions: Polar positions [batch, K, 2] where
                       positions[..., 0] = r_norm ∈ [0, 1]
                       positions[..., 1] = θ_norm ∈ [0, 1]
            sigmas: Spatial uncertainties [batch, K, 2] where
                    sigmas[..., 0] = σ_r ∈ [sigma_min, sigma_max]
                    sigmas[..., 1] = σ_θ ∈ [sigma_min, sigma_max]
            weights: Normalized importance weights [batch, K]
        """
        batch, K, H = particles.shape

        # Project to [r, θ, log_σ_r, log_σ_θ]
        raw = self.projection(particles)  # [batch, K, 4]

        # Split and bound outputs
        r_norm = torch.sigmoid(raw[..., 0])      # [0, 1]
        theta_norm = torch.sigmoid(raw[..., 1])  # [0, 1]

        sigma_r = sigmoid_range(raw[..., 2], self.sigma_min, self.sigma_max)
        sigma_theta = sigmoid_range(raw[..., 3], self.sigma_min, self.sigma_max)

        # Stack into output tensors
        positions = torch.stack([r_norm, theta_norm], dim=-1)  # [batch, K, 2]
        sigmas = torch.stack([sigma_r, sigma_theta], dim=-1)   # [batch, K, 2]

        # Normalize log weights to get probabilities
        weights = F.softmax(log_weights, dim=-1)  # [batch, K]

        return positions, sigmas, weights

    def get_weighted_position(
        self,
        particles: Tensor,
        log_weights: Tensor,
    ) -> Tensor:
        """Get weighted mean position estimate.

        Args:
            particles: Latent particle states [batch, K, hidden_size]
            log_weights: Log importance weights [batch, K]

        Returns:
            mean_position: Weighted mean [batch, 2] (r_norm, θ_norm)
        """
        positions, _, weights = self.forward(particles, log_weights)

        # Weighted mean over particles
        # weights: [batch, K] -> [batch, K, 1]
        weights_expanded = weights.unsqueeze(-1)
        mean_position = (positions * weights_expanded).sum(dim=1)  # [batch, 2]

        return mean_position

    def get_position_uncertainty(
        self,
        particles: Tensor,
        log_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Get weighted mean position and combined uncertainty.

        Uncertainty combines:
        1. Per-particle predicted uncertainty (σ)
        2. Particle spread (epistemic uncertainty from particle diversity)

        Args:
            particles: Latent particle states [batch, K, hidden_size]
            log_weights: Log importance weights [batch, K]

        Returns:
            mean_position: Weighted mean [batch, 2]
            total_uncertainty: Combined uncertainty [batch, 2]
        """
        positions, sigmas, weights = self.forward(particles, log_weights)

        weights_expanded = weights.unsqueeze(-1)  # [batch, K, 1]

        # Weighted mean position
        mean_position = (positions * weights_expanded).sum(dim=1)  # [batch, 2]

        # Epistemic uncertainty: weighted variance of particle positions
        position_diff = positions - mean_position.unsqueeze(1)  # [batch, K, 2]
        epistemic_var = (weights_expanded * position_diff ** 2).sum(dim=1)  # [batch, 2]

        # Aleatoric uncertainty: weighted mean of predicted variances
        aleatoric_var = (weights_expanded * sigmas ** 2).sum(dim=1)  # [batch, 2]

        # Total uncertainty (sqrt of sum of variances)
        total_uncertainty = torch.sqrt(epistemic_var + aleatoric_var + 1e-8)

        return mean_position, total_uncertainty

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"sigma_range=[{self.sigma_min}, {self.sigma_max}]"
        )
