"""Monitoring and diagnostic utilities for particle filters.

Provides tools for:
- Tracking particle health metrics (ESS, diversity)
- Detecting numerical issues (NaN, Inf)
- Logging and visualization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

import torch
from torch import Tensor


@dataclass
class ParticleMetrics:
    """Metrics for particle filter health assessment."""
    ess: float
    max_weight: float
    min_weight: float
    weight_entropy: float
    particle_variance: float
    avg_pairwise_distance: float
    max_particle_norm: float
    has_nan: bool
    has_inf: bool


def compute_particle_diversity(
    particles: Tensor,
    log_weights: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Compute diversity metrics for particle population.

    Args:
        particles: Particle states [batch, K, hidden_size]
        log_weights: Optional log weights [batch, K]

    Returns:
        Dict with diversity metrics
    """
    batch, K, H = particles.shape

    metrics = {}

    # 1. Variance across particles (per dimension, averaged)
    var_per_dim = particles.var(dim=1)  # [batch, H]
    metrics["variance_per_dim"] = var_per_dim.mean(dim=-1)  # [batch]

    # 2. Average pairwise distance
    # Compute distances between all pairs of particles
    # Using batched cdist
    dists = torch.cdist(particles, particles)  # [batch, K, K]
    # Mask diagonal (self-distances)
    mask = ~torch.eye(K, dtype=torch.bool, device=particles.device)
    mask = mask.unsqueeze(0).expand(batch, -1, -1)
    pairwise_dists = dists[mask].view(batch, -1)
    metrics["avg_pairwise_distance"] = pairwise_dists.mean(dim=-1)  # [batch]

    # 3. Effective dimension (participation ratio)
    # How many dimensions have significant variance
    dim_vars = particles.var(dim=1)  # [batch, H]
    total_var = dim_vars.sum(dim=-1)  # [batch]
    sum_sq_var = (dim_vars ** 2).sum(dim=-1)  # [batch]
    eff_dim = total_var ** 2 / (sum_sq_var + 1e-8)  # [batch]
    metrics["effective_dimension"] = eff_dim

    # 4. Relative spread (normalized by state magnitude)
    state_norm = particles.norm(dim=-1).mean(dim=-1)  # [batch]
    metrics["relative_spread"] = metrics["avg_pairwise_distance"] / (state_norm + 1e-8)

    # 5. Particle collapse detection
    # Ratio of variance to mean squared
    mean_state = particles.mean(dim=1)  # [batch, H]
    mean_sq = (mean_state ** 2).mean(dim=-1)  # [batch]
    var_mean = var_per_dim.mean(dim=-1)  # [batch]
    metrics["collapse_ratio"] = var_mean / (mean_sq + 1e-8)

    return metrics


def check_numerical_health(
    particles: Tensor,
    log_weights: Tensor,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Check numerical health of particle filter state.

    Args:
        particles: Particle states [batch, K, hidden_size]
        log_weights: Log weights [batch, K]
        thresholds: Optional dict of threshold values

    Returns:
        Dict with health checks and any warnings
    """
    thresholds = thresholds or {
        "max_particle_norm": 100.0,
        "min_ess_ratio": 0.1,
        "max_weight_ratio": 0.99,
    }

    health = {
        "healthy": True,
        "warnings": [],
        "metrics": {},
    }

    # Check for NaN
    has_nan_particles = torch.isnan(particles).any()
    has_nan_weights = torch.isnan(log_weights).any()
    if has_nan_particles or has_nan_weights:
        health["healthy"] = False
        health["warnings"].append("NaN detected")
    health["metrics"]["has_nan"] = has_nan_particles or has_nan_weights

    # Check for Inf
    has_inf_particles = torch.isinf(particles).any()
    has_inf_weights = torch.isinf(log_weights).any()
    if has_inf_particles or has_inf_weights:
        health["healthy"] = False
        health["warnings"].append("Inf detected")
    health["metrics"]["has_inf"] = has_inf_particles or has_inf_weights

    # Check particle norms
    max_norm = particles.norm(dim=-1).max().item()
    health["metrics"]["max_particle_norm"] = max_norm
    if max_norm > thresholds["max_particle_norm"]:
        health["warnings"].append(f"Large particle norm: {max_norm:.2f}")

    # Check ESS
    from .weights import compute_ess
    ess = compute_ess(log_weights)
    ess_ratio = ess.min().item() / particles.shape[1]
    health["metrics"]["min_ess_ratio"] = ess_ratio
    if ess_ratio < thresholds["min_ess_ratio"]:
        health["healthy"] = False
        health["warnings"].append(f"Low ESS ratio: {ess_ratio:.3f}")

    # Check weight degeneracy
    weights = torch.exp(log_weights)
    max_weight_ratio = weights.max(dim=-1).values.mean().item()
    health["metrics"]["max_weight_ratio"] = max_weight_ratio
    if max_weight_ratio > thresholds["max_weight_ratio"]:
        health["warnings"].append(f"Weight degeneracy: max_w={max_weight_ratio:.3f}")

    return health


class ParticleMonitor:
    """Monitor particle filter health during training/inference.

    Tracks metrics over time and provides diagnostics.

    Example:
        >>> monitor = ParticleMonitor()
        >>> for step in range(n_steps):
        ...     output, (particles, log_weights) = model(x)
        ...     monitor.log(particles, log_weights)
        >>> monitor.summary()
    """

    def __init__(
        self,
        log_interval: int = 100,
        warn_on_issues: bool = True,
    ):
        """Initialize monitor.

        Args:
            log_interval: Steps between detailed logging
            warn_on_issues: Whether to print warnings on issues
        """
        self.log_interval = log_interval
        self.warn_on_issues = warn_on_issues

        self.step = 0
        self.history: Dict[str, List[float]] = defaultdict(list)

    def log(
        self,
        particles: Tensor,
        log_weights: Tensor,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> ParticleMetrics:
        """Log metrics for current step.

        Args:
            particles: Particle states [batch, K, hidden_size]
            log_weights: Log weights [batch, K]
            extra_metrics: Optional additional metrics to log

        Returns:
            ParticleMetrics for current step
        """
        from .weights import compute_ess, compute_entropy

        # Compute metrics
        ess = compute_ess(log_weights).mean().item()
        weights = torch.exp(log_weights)
        max_weight = weights.max().item()
        min_weight = weights.min().item()
        entropy = compute_entropy(log_weights).mean().item()

        diversity = compute_particle_diversity(particles)
        particle_var = diversity["variance_per_dim"].mean().item()
        avg_dist = diversity["avg_pairwise_distance"].mean().item()

        max_norm = particles.norm(dim=-1).max().item()
        has_nan = torch.isnan(particles).any().item() or torch.isnan(log_weights).any().item()
        has_inf = torch.isinf(particles).any().item() or torch.isinf(log_weights).any().item()

        metrics = ParticleMetrics(
            ess=ess,
            max_weight=max_weight,
            min_weight=min_weight,
            weight_entropy=entropy,
            particle_variance=particle_var,
            avg_pairwise_distance=avg_dist,
            max_particle_norm=max_norm,
            has_nan=has_nan,
            has_inf=has_inf,
        )

        # Store in history
        self.history["ess"].append(ess)
        self.history["max_weight"].append(max_weight)
        self.history["weight_entropy"].append(entropy)
        self.history["particle_variance"].append(particle_var)
        self.history["max_particle_norm"].append(max_norm)

        if extra_metrics:
            for k, v in extra_metrics.items():
                self.history[k].append(v)

        # Check for issues and warn
        if self.warn_on_issues:
            if has_nan:
                print(f"[Step {self.step}] WARNING: NaN detected!")
            if has_inf:
                print(f"[Step {self.step}] WARNING: Inf detected!")
            if ess < 2:
                print(f"[Step {self.step}] WARNING: Low ESS ({ess:.2f})")

        # Periodic logging
        if self.step % self.log_interval == 0 and self.step > 0:
            self._print_summary()

        self.step += 1
        return metrics

    def _print_summary(self):
        """Print summary of recent metrics."""
        recent = min(self.log_interval, len(self.history["ess"]))
        if recent == 0:
            return

        print(f"\n=== Step {self.step} ===")
        print(f"ESS: {self.history['ess'][-1]:.2f} "
              f"(avg: {sum(self.history['ess'][-recent:]) / recent:.2f})")
        print(f"Max Weight: {self.history['max_weight'][-1]:.4f}")
        print(f"Particle Variance: {self.history['particle_variance'][-1]:.4f}")

    def summary(self) -> Dict[str, float]:
        """Get summary statistics over all logged steps.

        Returns:
            Dict with summary statistics
        """
        if not self.history["ess"]:
            return {}

        summary = {}
        for key, values in self.history.items():
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_last"] = values[-1]

        return summary

    def reset(self):
        """Reset monitor state."""
        self.step = 0
        self.history = defaultdict(list)

    def get_history(self, key: str) -> List[float]:
        """Get history for a specific metric.

        Args:
            key: Metric name

        Returns:
            List of values over time
        """
        return self.history.get(key, [])
