"""Soft resampling utilities for particle filters.

Implements differentiable soft resampling with three alpha modes:
- Fixed: Constant alpha value
- Adaptive: Alpha adjusts based on ESS
- Learnable: Alpha is a learned parameter
"""

from enum import Enum
from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
from torch import Tensor


class AlphaMode(Enum):
    """Mode for controlling the soft resampling mixing coefficient alpha."""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    LEARNABLE = "learnable"


def compute_proposal(
    weights: Tensor,
    alpha: Union[float, Tensor],
    n_particles: int,
) -> Tensor:
    """Compute the soft resampling proposal distribution.

    The proposal is: q(k) = alpha * w_k + (1 - alpha) / K

    Args:
        weights: Normalized particle weights [batch, K]
        alpha: Mixing coefficient (0 to 1)
        n_particles: Number of particles K

    Returns:
        proposal: Proposal distribution [batch, K]
    """
    uniform = 1.0 / n_particles
    return alpha * weights + (1.0 - alpha) * uniform


def soft_resample(
    particles: Tensor,
    log_weights: Tensor,
    alpha: Union[float, Tensor],
    return_indices: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Differentiable soft resampling.

    Implements the soft resampling scheme from PF-RNN:
    - Proposal: q(k) = alpha * w_k + (1 - alpha) / K
    - Sample ancestors from proposal
    - Correct weights via importance sampling

    Args:
        particles: Particle states [batch, K, hidden_size]
        log_weights: Log particle weights [batch, K]
        alpha: Mixing coefficient (higher = more deterministic)
        return_indices: If True, also return sampled ancestor indices

    Returns:
        resampled_particles: Resampled states [batch, K, hidden_size]
        new_log_weights: Corrected log weights [batch, K]
        ancestors: (optional) Ancestor indices [batch, K]
    """
    batch_size, n_particles, hidden_size = particles.shape
    device = particles.device

    # Convert log weights to weights
    weights = torch.exp(log_weights)

    # Compute proposal distribution
    proposal = compute_proposal(weights, alpha, n_particles)

    # Sample ancestor indices from proposal
    # Shape: [batch, K]
    ancestors = torch.multinomial(proposal, n_particles, replacement=True)

    # Gather resampled particles
    # Expand indices for gathering: [batch, K, hidden_size]
    ancestors_expanded = ancestors.unsqueeze(-1).expand(-1, -1, hidden_size)
    resampled_particles = torch.gather(particles, dim=1, index=ancestors_expanded)

    # Importance weight correction
    # w_new = w_old / q(ancestor)
    proposal_gathered = torch.gather(proposal, dim=1, index=ancestors)
    log_weights_gathered = torch.gather(log_weights, dim=1, index=ancestors)

    # Corrected weights in log space
    new_log_weights = log_weights_gathered - torch.log(proposal_gathered + 1e-10)

    # Normalize
    new_log_weights = new_log_weights - torch.logsumexp(new_log_weights, dim=1, keepdim=True)

    if return_indices:
        return resampled_particles, new_log_weights, ancestors
    return resampled_particles, new_log_weights


class SoftResampler(nn.Module):
    """Soft resampling module with configurable alpha mode.

    Supports three modes:
    - Fixed: Uses a constant alpha value
    - Adaptive: Adjusts alpha based on ESS ratio
    - Learnable: Alpha is a learned parameter

    Example:
        >>> resampler = SoftResampler(n_particles=32, alpha_mode='adaptive')
        >>> new_particles, new_weights = resampler(particles, log_weights)
    """

    def __init__(
        self,
        n_particles: int,
        alpha_mode: Union[str, AlphaMode] = AlphaMode.FIXED,
        alpha_init: float = 0.5,
        alpha_min: float = 0.2,
        alpha_max: float = 0.9,
        target_ess_ratio: float = 0.5,
        resample_threshold: float = 0.5,
    ):
        """Initialize the soft resampler.

        Args:
            n_particles: Number of particles K
            alpha_mode: One of 'fixed', 'adaptive', 'learnable'
            alpha_init: Initial alpha value (default 0.5)
            alpha_min: Minimum alpha for adaptive mode
            alpha_max: Maximum alpha for adaptive mode
            target_ess_ratio: Target ESS/K ratio for adaptive mode
            resample_threshold: Only resample if ESS < threshold * K
        """
        super().__init__()

        self.n_particles = n_particles
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.target_ess_ratio = target_ess_ratio
        self.resample_threshold = resample_threshold

        # Parse alpha mode
        if isinstance(alpha_mode, str):
            alpha_mode = AlphaMode(alpha_mode)
        self.alpha_mode = alpha_mode

        # Initialize alpha based on mode
        if alpha_mode == AlphaMode.LEARNABLE:
            # Use sigmoid parametrization to keep alpha in [0, 1]
            # inverse_sigmoid(alpha_init) = log(alpha / (1 - alpha))
            init_logit = math.log(alpha_init / (1.0 - alpha_init + 1e-8))
            self._alpha_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer("_alpha_fixed", torch.tensor(alpha_init))

    @property
    def alpha(self) -> Tensor:
        """Get current alpha value."""
        if self.alpha_mode == AlphaMode.LEARNABLE:
            return torch.sigmoid(self._alpha_logit)
        return self._alpha_fixed

    def compute_adaptive_alpha(self, ess: Tensor) -> Tensor:
        """Compute adaptive alpha based on ESS.

        When ESS is low, reduce alpha to increase exploration.
        When ESS is high, increase alpha for exploitation.

        Args:
            ess: Effective sample size [batch]

        Returns:
            alpha: Adapted alpha values [batch]
        """
        ess_ratio = ess / self.n_particles

        # Linear interpolation between alpha_min and alpha_max
        # based on how ESS compares to target
        alpha = torch.where(
            ess_ratio < self.target_ess_ratio,
            torch.full_like(ess_ratio, self.alpha_min),
            self.alpha_min + (self.alpha_max - self.alpha_min) *
            (ess_ratio - self.target_ess_ratio) / (1.0 - self.target_ess_ratio)
        )

        return torch.clamp(alpha, self.alpha_min, self.alpha_max)

    def should_resample(
        self,
        log_weights: Tensor,
        already_normalized: bool = False,
    ) -> Tensor:
        """Determine which samples need resampling based on ESS.

        Args:
            log_weights: Log particle weights [batch, K]
            already_normalized: If True, skip normalization in ESS computation

        Returns:
            mask: Boolean mask [batch] indicating which need resampling
        """
        from .weights import compute_ess
        ess = compute_ess(log_weights, already_normalized=already_normalized)
        return ess < (self.resample_threshold * self.n_particles)

    def forward(
        self,
        particles: Tensor,
        log_weights: Tensor,
        force_resample: bool = False,
        return_ess: bool = False,
        already_normalized: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Apply soft resampling.

        Args:
            particles: Particle states [batch, K, hidden_size]
            log_weights: Log particle weights [batch, K]
            force_resample: If True, always resample regardless of ESS
            return_ess: If True, also return ESS values
            already_normalized: If True, skip normalization in ESS (weights already normalized)

        Returns:
            particles: (Possibly resampled) particle states
            log_weights: (Possibly corrected) log weights
            ess: (optional) Effective sample size [batch]
        """
        from .weights import compute_ess

        batch_size = particles.shape[0]
        ess = compute_ess(log_weights, already_normalized=already_normalized)

        # Determine alpha based on mode
        if self.alpha_mode == AlphaMode.ADAPTIVE:
            alpha = self.compute_adaptive_alpha(ess)
            # Expand for broadcasting: [batch, 1]
            alpha = alpha.unsqueeze(-1)
        else:
            alpha = self.alpha

        # Determine which samples need resampling
        if force_resample:
            needs_resample = torch.ones(batch_size, dtype=torch.bool, device=particles.device)
        else:
            needs_resample = self.should_resample(log_weights, already_normalized=already_normalized)

        # If any need resampling, do it for all (simpler batching)
        # In practice, can be optimized to only resample needed samples
        if needs_resample.any():
            new_particles, new_log_weights = soft_resample(
                particles, log_weights, alpha
            )
        else:
            new_particles, new_log_weights = particles, log_weights

        if return_ess:
            return new_particles, new_log_weights, ess
        return new_particles, new_log_weights

    def extra_repr(self) -> str:
        return (
            f"n_particles={self.n_particles}, "
            f"alpha_mode={self.alpha_mode.value}, "
            f"resample_threshold={self.resample_threshold}"
        )
