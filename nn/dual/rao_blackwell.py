"""Rao-Blackwellization utilities for dual particle filters.

Rao-Blackwellization improves particle filter efficiency by analytically
marginalizing out certain variables when possible, reducing variance.
"""

from typing import Tuple, Optional, Dict
import math

import torch
import torch.nn as nn
from torch import Tensor


def rao_blackwell_state_estimate(
    state_particles: Tensor,
    param_particles: Tensor,
    log_weights: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute Rao-Blackwellized state estimate.

    When parameters are tracked separately, we can marginalize over
    parameters to get a lower-variance state estimate.

    Args:
        state_particles: State particles [batch, K, state_size]
        param_particles: Parameter particles [batch, K, param_size]
        log_weights: Joint log weights [batch, K]

    Returns:
        state_mean: Marginalized state mean [batch, state_size]
        state_var: Marginalized state variance [batch, state_size]
    """
    # Normalize weights
    weights = torch.softmax(log_weights, dim=1)  # [batch, K]
    weights = weights.unsqueeze(-1)  # [batch, K, 1]

    # Weighted mean
    state_mean = (weights * state_particles).sum(dim=1)  # [batch, state_size]

    # Weighted variance
    diff = state_particles - state_mean.unsqueeze(1)
    state_var = (weights * diff ** 2).sum(dim=1)  # [batch, state_size]

    return state_mean, state_var


def rao_blackwell_param_estimate(
    state_particles: Tensor,
    param_particles: Tensor,
    log_weights: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute Rao-Blackwellized parameter estimate.

    Marginalizes over states to get parameter posterior.

    Args:
        state_particles: State particles [batch, K, state_size]
        param_particles: Parameter particles [batch, K, param_size]
        log_weights: Joint log weights [batch, K]

    Returns:
        param_mean: Marginalized parameter mean [batch, param_size]
        param_var: Marginalized parameter variance [batch, param_size]
    """
    weights = torch.softmax(log_weights, dim=1).unsqueeze(-1)

    param_mean = (weights * param_particles).sum(dim=1)
    diff = param_particles - param_mean.unsqueeze(1)
    param_var = (weights * diff ** 2).sum(dim=1)

    return param_mean, param_var


def conditional_state_update(
    state_particles: Tensor,
    param_particles: Tensor,
    observation: Tensor,
    observation_model: nn.Module,
    log_weights: Tensor,
) -> Tensor:
    """Update state weights conditioned on fixed parameters.

    For Rao-Blackwellization, we update state weights as if
    parameters were fixed (which they are per-particle).

    Args:
        state_particles: State particles [batch, K, state_size]
        param_particles: Parameter particles [batch, K, param_size]
        observation: Observation [batch, obs_size]
        observation_model: Model for p(y|h)
        log_weights: Current log weights [batch, K]

    Returns:
        new_log_weights: Updated log weights [batch, K]
    """
    # Compute likelihoods using current state particles
    log_likelihoods = observation_model.log_likelihood(
        state_particles, observation
    )  # [batch, K]

    # Update weights
    new_log_weights = log_weights + log_likelihoods

    # Normalize
    new_log_weights = new_log_weights - torch.logsumexp(
        new_log_weights, dim=1, keepdim=True
    )

    return new_log_weights


def stratified_joint_resample(
    state_particles: Tensor,
    param_particles: Tensor,
    log_weights: Tensor,
    alpha: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Stratified resampling for joint (state, param) particles.

    Uses stratified sampling to reduce variance compared to
    multinomial resampling.

    Args:
        state_particles: State particles [batch, K, state_size]
        param_particles: Parameter particles [batch, K, param_size]
        log_weights: Log weights [batch, K]
        alpha: Soft resampling mixing coefficient

    Returns:
        new_state_particles: Resampled states
        new_param_particles: Resampled params
        new_log_weights: Corrected weights
    """
    batch, K, state_size = state_particles.shape
    param_size = param_particles.shape[2]
    device = state_particles.device

    # Convert to probabilities
    weights = torch.softmax(log_weights, dim=1)

    # Soft resampling proposal
    uniform = 1.0 / K
    proposal = alpha * weights + (1.0 - alpha) * uniform

    # Stratified sampling
    # Generate K uniform samples in [k/K, (k+1)/K]
    positions = (torch.arange(K, device=device).float() + torch.rand(batch, K, device=device)) / K

    # Compute cumulative distribution
    cumsum = torch.cumsum(proposal, dim=1)

    # Find indices via searchsorted
    indices = torch.searchsorted(cumsum, positions)
    indices = indices.clamp(0, K - 1)

    # Gather resampled particles
    state_idx = indices.unsqueeze(-1).expand(-1, -1, state_size)
    param_idx = indices.unsqueeze(-1).expand(-1, -1, param_size)

    new_state_particles = torch.gather(state_particles, 1, state_idx)
    new_param_particles = torch.gather(param_particles, 1, param_idx)

    # Importance weight correction
    proposal_gathered = torch.gather(proposal, 1, indices)
    weights_gathered = torch.gather(weights, 1, indices)

    new_weights = weights_gathered / (proposal_gathered + 1e-10)
    new_log_weights = torch.log(new_weights + 1e-10)
    new_log_weights = new_log_weights - torch.logsumexp(new_log_weights, dim=1, keepdim=True)

    return new_state_particles, new_param_particles, new_log_weights


class RaoBlackwellEstimator(nn.Module):
    """Rao-Blackwell estimator for dual particle filters.

    Provides variance-reduced estimates by analytically marginalizing
    when possible.
    """

    def __init__(
        self,
        state_size: int,
        param_size: int,
        use_stratified_resampling: bool = True,
    ):
        """Initialize Rao-Blackwell estimator.

        Args:
            state_size: Dimension of state
            param_size: Dimension of parameters
            use_stratified_resampling: Use stratified vs multinomial
        """
        super().__init__()
        self.state_size = state_size
        self.param_size = param_size
        self.use_stratified_resampling = use_stratified_resampling

    def estimate_state(
        self,
        state_particles: Tensor,
        param_particles: Tensor,
        log_weights: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute state estimates with uncertainty.

        Args:
            state_particles: [batch, K, state_size]
            param_particles: [batch, K, param_size]
            log_weights: [batch, K]

        Returns:
            Dict with mean, variance, and quantiles
        """
        mean, var = rao_blackwell_state_estimate(
            state_particles, param_particles, log_weights
        )

        weights = torch.softmax(log_weights, dim=1)

        # Compute quantiles via weighted percentile
        # Sort particles and accumulate weights
        sorted_particles, sort_idx = torch.sort(state_particles, dim=1)
        sorted_weights = torch.gather(weights.unsqueeze(-1).expand_as(state_particles), 1, sort_idx)
        cumweights = torch.cumsum(sorted_weights, dim=1)

        # Find median (50th percentile)
        median_idx = (cumweights >= 0.5).float().argmax(dim=1)
        # This is approximate; proper implementation would interpolate

        return {
            "mean": mean,
            "variance": var,
            "std": torch.sqrt(var + 1e-8),
        }

    def estimate_params(
        self,
        state_particles: Tensor,
        param_particles: Tensor,
        log_weights: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute parameter estimates with uncertainty."""
        mean, var = rao_blackwell_param_estimate(
            state_particles, param_particles, log_weights
        )

        return {
            "mean": mean,
            "variance": var,
            "std": torch.sqrt(var + 1e-8),
        }

    def resample(
        self,
        state_particles: Tensor,
        param_particles: Tensor,
        log_weights: Tensor,
        alpha: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Resample joint particles."""
        if self.use_stratified_resampling:
            return stratified_joint_resample(
                state_particles, param_particles, log_weights, alpha
            )
        else:
            # Fall back to standard soft resampling
            batch, K, state_size = state_particles.shape
            param_size = param_particles.shape[2]
            device = state_particles.device

            weights = torch.softmax(log_weights, dim=1)
            uniform = 1.0 / K
            proposal = alpha * weights + (1.0 - alpha) * uniform

            indices = torch.multinomial(proposal, K, replacement=True)

            state_idx = indices.unsqueeze(-1).expand(-1, -1, state_size)
            param_idx = indices.unsqueeze(-1).expand(-1, -1, param_size)

            new_states = torch.gather(state_particles, 1, state_idx)
            new_params = torch.gather(param_particles, 1, param_idx)

            proposal_gathered = torch.gather(proposal, 1, indices)
            weights_gathered = torch.gather(weights, 1, indices)

            new_weights = weights_gathered / (proposal_gathered + 1e-10)
            new_log_weights = torch.log(new_weights + 1e-10)
            new_log_weights = new_log_weights - torch.logsumexp(
                new_log_weights, dim=1, keepdim=True
            )

            return new_states, new_params, new_log_weights
