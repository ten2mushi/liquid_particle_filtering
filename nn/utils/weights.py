"""Log-space weight operations for numerical stability.

All weight operations are performed in log-space to prevent
underflow/overflow with long sequences and many particles.
"""

from typing import Optional
import math
import warnings

import torch
from torch import Tensor


def safe_logsumexp(
    log_weights: Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    """Numerically stable logsumexp operation.

    Equivalent to torch.logsumexp but with additional NaN/Inf checking.

    Args:
        log_weights: Log weights tensor
        dim: Dimension to reduce
        keepdim: Whether to keep the reduced dimension

    Returns:
        Result of logsumexp operation
    """
    result = torch.logsumexp(log_weights, dim=dim, keepdim=keepdim)

    # Check for numerical issues
    if torch.isnan(result).any() or torch.isinf(result).any():
        warnings.warn("NaN/Inf detected in logsumexp, returning zeros")
        result = torch.where(
            torch.isnan(result) | torch.isinf(result),
            torch.zeros_like(result),
            result
        )

    return result


def normalize_log_weights(
    log_weights: Tensor,
    dim: int = -1,
    max_log_weight: float = 50.0,
    min_log_weight: float = -50.0,
) -> Tensor:
    """Normalize log weights so that exp(log_weights).sum(dim) = 1.

    Args:
        log_weights: Unnormalized log weights [batch, K]
        dim: Dimension to normalize over
        max_log_weight: Maximum allowed log weight (for stability)
        min_log_weight: Minimum allowed log weight (for stability)

    Returns:
        Normalized log weights [batch, K]
    """
    # Clamp to prevent extreme values
    log_weights = torch.clamp(log_weights, min=min_log_weight, max=max_log_weight)

    # Normalize in log space
    log_normalizer = safe_logsumexp(log_weights, dim=dim, keepdim=True)
    normalized = log_weights - log_normalizer

    return normalized


def log_weight_update(
    log_weights: Tensor,
    log_likelihoods: Tensor,
    normalize: bool = True,
    max_log_weight: float = 50.0,
    min_log_weight: float = -50.0,
) -> Tensor:
    """Update particle weights with observation likelihood.

    log w_new = log w_old + log p(o | h)

    Args:
        log_weights: Current log weights [batch, K]
        log_likelihoods: Log observation likelihoods [batch, K]
        normalize: Whether to normalize after update
        max_log_weight: Maximum allowed log weight
        min_log_weight: Minimum allowed log weight

    Returns:
        Updated log weights [batch, K]
    """
    # Add log likelihood
    new_log_weights = log_weights + log_likelihoods

    # Clamp for numerical stability
    new_log_weights = torch.clamp(new_log_weights, min=min_log_weight, max=max_log_weight)

    # Normalize if requested
    if normalize:
        new_log_weights = normalize_log_weights(new_log_weights, dim=-1)

    return new_log_weights


def compute_ess(
    log_weights: Tensor,
    dim: int = -1,
    already_normalized: bool = False,
) -> Tensor:
    """Compute Effective Sample Size (ESS) from log weights.

    ESS = 1 / sum(w_i^2) where w_i are normalized weights.

    Higher ESS indicates more diverse particle weights.
    ESS = K means uniform weights (maximum diversity).
    ESS = 1 means one particle dominates (degeneracy).

    Args:
        log_weights: Log weights [batch, K]
        dim: Dimension containing particles
        already_normalized: If True, skip normalization (caller guarantees normalized)

    Returns:
        ess: Effective sample size [batch]
    """
    # Skip normalization if caller guarantees weights are normalized
    if already_normalized:
        log_weights_norm = log_weights
    else:
        log_weights_norm = normalize_log_weights(log_weights, dim=dim)

    # Compute ESS = 1 / sum(w_i^2) = 1 / sum(exp(2 * log_w_i))
    # In log space: log(ESS) = -log(sum(exp(2 * log_w)))
    log_sum_sq = safe_logsumexp(2.0 * log_weights_norm, dim=dim)
    ess = torch.exp(-log_sum_sq)

    return ess


def compute_entropy(log_weights: Tensor, dim: int = -1) -> Tensor:
    """Compute entropy of particle weight distribution.

    H = -sum(w_i * log(w_i))

    Higher entropy indicates more uniform distribution.

    Args:
        log_weights: Normalized log weights [batch, K]
        dim: Dimension containing particles

    Returns:
        entropy: Weight distribution entropy [batch]
    """
    # Ensure normalized
    log_weights_norm = normalize_log_weights(log_weights, dim=dim)
    weights = torch.exp(log_weights_norm)

    # H = -sum(w * log(w))
    entropy = -(weights * log_weights_norm).sum(dim=dim)

    return entropy


def init_uniform_log_weights(
    batch_size: int,
    n_particles: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Initialize uniform log weights.

    Creates log weights of -log(K) so that exp(log_weights).sum() = 1.

    Args:
        batch_size: Batch dimension
        n_particles: Number of particles K
        device: Target device
        dtype: Target dtype

    Returns:
        log_weights: Uniform log weights [batch, K]
    """
    log_weight = -math.log(n_particles)
    return torch.full(
        (batch_size, n_particles),
        log_weight,
        device=device,
        dtype=dtype,
    )


def weighted_mean(
    values: Tensor,
    log_weights: Tensor,
    dim: int = 1,
) -> Tensor:
    """Compute weighted mean of values using log weights.

    mean = sum(w_i * v_i)

    Args:
        values: Values tensor [batch, K, ...]
        log_weights: Normalized log weights [batch, K]
        dim: Particle dimension (default 1)

    Returns:
        weighted_mean: Mean over particles [batch, ...]
    """
    # Ensure weights are normalized
    log_weights_norm = normalize_log_weights(log_weights, dim=dim)
    weights = torch.exp(log_weights_norm)

    # Expand weights for broadcasting
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)

    # Weighted sum
    return (weights * values).sum(dim=dim)


def weighted_variance(
    values: Tensor,
    log_weights: Tensor,
    dim: int = 1,
) -> Tensor:
    """Compute weighted variance of values using log weights.

    var = sum(w_i * (v_i - mean)^2)

    Args:
        values: Values tensor [batch, K, ...]
        log_weights: Normalized log weights [batch, K]
        dim: Particle dimension (default 1)

    Returns:
        weighted_var: Variance over particles [batch, ...]
    """
    # Ensure weights are normalized
    log_weights_norm = normalize_log_weights(log_weights, dim=dim)
    weights = torch.exp(log_weights_norm)

    # Expand weights for broadcasting
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)

    # Compute mean
    mean = (weights * values).sum(dim=dim, keepdim=True)

    # Compute variance
    variance = (weights * (values - mean) ** 2).sum(dim=dim)

    return variance


def temperature_scaled_log_weights(
    log_weights: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Apply temperature scaling to log weights.

    Higher temperature -> more uniform distribution.
    Lower temperature -> more peaked distribution.

    Args:
        log_weights: Log weights [batch, K]
        temperature: Temperature parameter (> 0)

    Returns:
        Scaled and normalized log weights [batch, K]
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    scaled = log_weights / temperature
    return normalize_log_weights(scaled)
