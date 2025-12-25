"""Base data collector for PFNCPS visualization.

Provides memory-bounded collection of particle filter data
with support for downsampling long sequences.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import math

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..core.base import ArchitectureInfo


@dataclass
class CollectedStep:
    """Data collected at a single timestep.

    Attributes:
        timestep: Step index in the sequence
        particles: Particle states [K, hidden_size] (single batch element)
        log_weights: Log weights [K]
        ess: Effective sample size (scalar)
        outputs: Model outputs [K, output_size] or [output_size]
        observation: Ground truth observation if provided
        log_likelihoods: Observation log-likelihoods [K] if computed
        resampled: Whether resampling occurred at this step
        extra: Architecture-specific additional data
    """
    timestep: int
    particles: Tensor
    log_weights: Tensor
    ess: float
    outputs: Optional[Tensor] = None
    observation: Optional[Tensor] = None
    log_likelihoods: Optional[Tensor] = None
    resampled: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def lttb_downsample(data: List[CollectedStep], target_size: int) -> List[CollectedStep]:
    """Largest-Triangle-Three-Buckets downsampling algorithm.

    Preserves visually important points while reducing data size.
    Uses ESS as the primary signal for point selection.

    Args:
        data: List of collected steps
        target_size: Target number of points

    Returns:
        Downsampled list of collected steps
    """
    n = len(data)
    if n <= target_size:
        return data

    # Always keep first and last points
    sampled = [data[0]]

    # Calculate bucket size
    bucket_size = (n - 2) / (target_size - 2)

    # Use ESS as y-values for point selection
    y_values = [step.ess for step in data]

    a = 0  # Initially the first point is selected

    for i in range(target_size - 2):
        # Calculate bucket boundaries
        bucket_start = int(math.floor((i + 1) * bucket_size)) + 1
        bucket_end = int(math.floor((i + 2) * bucket_size)) + 1
        bucket_end = min(bucket_end, n - 1)

        # Calculate average for next bucket (for point selection)
        next_bucket_start = bucket_end
        next_bucket_end = int(math.floor((i + 3) * bucket_size)) + 1
        next_bucket_end = min(next_bucket_end, n)

        avg_x = 0
        avg_y = 0
        count = next_bucket_end - next_bucket_start
        if count > 0:
            for j in range(next_bucket_start, next_bucket_end):
                avg_x += data[j].timestep
                avg_y += y_values[j]
            avg_x /= count
            avg_y /= count
        else:
            avg_x = data[-1].timestep
            avg_y = y_values[-1]

        # Find point in current bucket that creates largest triangle
        max_area = -1
        max_area_point = bucket_start

        point_a_x = data[a].timestep
        point_a_y = y_values[a]

        for j in range(bucket_start, bucket_end):
            # Calculate triangle area using cross product
            area = abs(
                (point_a_x - avg_x) * (y_values[j] - point_a_y) -
                (point_a_x - data[j].timestep) * (avg_y - point_a_y)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = j

        sampled.append(data[max_area_point])
        a = max_area_point

    # Always include the last point
    sampled.append(data[-1])

    return sampled


def uniform_downsample(data: List[CollectedStep], target_size: int) -> List[CollectedStep]:
    """Uniform downsampling (every n-th point).

    Args:
        data: List of collected steps
        target_size: Target number of points

    Returns:
        Downsampled list of collected steps
    """
    n = len(data)
    if n <= target_size:
        return data

    # Always include first and last
    indices = [0]

    step = (n - 1) / (target_size - 1)
    for i in range(1, target_size - 1):
        indices.append(int(round(i * step)))

    indices.append(n - 1)

    return [data[i] for i in indices]


class BaseDataCollector:
    """Base class for collecting particle filter data.

    Handles:
    - Memory-bounded storage with downsampling
    - Lazy computation of derived metrics
    - Batch element selection
    - Device/dtype handling

    Attributes:
        max_history: Maximum number of timesteps to store
        downsample_strategy: Downsampling algorithm ("lttb" or "uniform")
        batch_idx: Which batch element to track
        arch_info: Architecture information
        history: List of collected timesteps
    """

    def __init__(
        self,
        max_history: int = 10000,
        downsample_strategy: str = "lttb",
        batch_idx: int = 0,
        arch_info: Optional["ArchitectureInfo"] = None,
    ):
        """Initialize collector.

        Args:
            max_history: Maximum timesteps to store
            downsample_strategy: "lttb" or "uniform"
            batch_idx: Which batch element to track
            arch_info: Architecture information
        """
        self.max_history = max_history
        self.downsample_strategy = downsample_strategy
        self.batch_idx = batch_idx
        self.arch_info = arch_info

        self.history: List[CollectedStep] = []
        self._step_counter = 0

        # Cache for computed tensors
        self._cache: Dict[str, Tensor] = {}

    def __len__(self) -> int:
        """Return number of collected steps."""
        return len(self.history)

    def log_step(
        self,
        particles: Tensor,
        log_weights: Tensor,
        outputs: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        **extra,
    ) -> CollectedStep:
        """Log a single timestep.

        Args:
            particles: Particle states [batch, K, hidden_size]
            log_weights: Log weights [batch, K]
            outputs: Model outputs (optional)
            observation: Ground truth observation (optional)
            **extra: Additional data to store

        Returns:
            CollectedStep for this timestep
        """
        # Clear cache when new data arrives
        self._cache.clear()

        # Extract single batch element and move to CPU
        batch_idx = min(self.batch_idx, particles.shape[0] - 1)

        particles_cpu = particles[batch_idx].detach().cpu()
        log_weights_cpu = log_weights[batch_idx].detach().cpu()

        # Compute ESS
        ess = self._compute_ess(log_weights_cpu)

        # Extract outputs
        outputs_cpu = None
        if outputs is not None:
            if outputs.dim() >= 2:
                outputs_cpu = outputs[batch_idx].detach().cpu()
            else:
                outputs_cpu = outputs.detach().cpu()

        # Extract observation
        obs_cpu = None
        if observation is not None:
            if observation.dim() >= 2:
                obs_cpu = observation[batch_idx].detach().cpu()
            else:
                obs_cpu = observation.detach().cpu()

        # Process extra data
        extra_cpu = {}
        for key, value in extra.items():
            if isinstance(value, Tensor):
                if value.dim() >= 2 and value.shape[0] > batch_idx:
                    extra_cpu[key] = value[batch_idx].detach().cpu()
                else:
                    extra_cpu[key] = value.detach().cpu()
            else:
                extra_cpu[key] = value

        # Create step
        step = CollectedStep(
            timestep=self._step_counter,
            particles=particles_cpu,
            log_weights=log_weights_cpu,
            ess=ess,
            outputs=outputs_cpu,
            observation=obs_cpu,
            extra=extra_cpu,
        )

        self.history.append(step)
        self._step_counter += 1

        # Downsample if needed
        self._maybe_downsample()

        return step

    def _compute_ess(self, log_weights: Tensor) -> float:
        """Compute effective sample size from log weights.

        Args:
            log_weights: Normalized log weights [K]

        Returns:
            ESS value (scalar)
        """
        # Normalize weights
        log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
        # ESS = 1 / sum(w_i^2)
        log_sum_sq = torch.logsumexp(2.0 * log_weights, dim=-1)
        ess = torch.exp(-log_sum_sq).item()
        return ess

    def _maybe_downsample(self) -> None:
        """Downsample history if it exceeds max_history."""
        if len(self.history) > self.max_history:
            target_size = int(self.max_history * 0.9)  # Keep 90% to avoid frequent downsampling

            if self.downsample_strategy == "lttb":
                self.history = lttb_downsample(self.history, target_size)
            else:
                self.history = uniform_downsample(self.history, target_size)

    def reset(self) -> None:
        """Clear all collected data."""
        self.history.clear()
        self._step_counter = 0
        self._cache.clear()

    # =========================================================================
    # Data Access Methods
    # =========================================================================

    def get_particles(self) -> Tensor:
        """Get stacked particle states.

        Returns:
            Tensor of shape [time, K, hidden_size]
        """
        if "particles" not in self._cache:
            self._cache["particles"] = torch.stack([s.particles for s in self.history])
        return self._cache["particles"]

    def get_log_weights(self) -> Tensor:
        """Get stacked log weights.

        Returns:
            Tensor of shape [time, K]
        """
        if "log_weights" not in self._cache:
            self._cache["log_weights"] = torch.stack([s.log_weights for s in self.history])
        return self._cache["log_weights"]

    def get_weights(self) -> Tensor:
        """Get stacked normalized weights (not log).

        Returns:
            Tensor of shape [time, K]
        """
        if "weights" not in self._cache:
            log_weights = self.get_log_weights()
            # Normalize per timestep
            log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
            self._cache["weights"] = torch.exp(log_weights)
        return self._cache["weights"]

    def get_ess(self) -> Tensor:
        """Get ESS timeline.

        Returns:
            Tensor of shape [time]
        """
        if "ess" not in self._cache:
            self._cache["ess"] = torch.tensor([s.ess for s in self.history])
        return self._cache["ess"]

    def get_timesteps(self) -> Tensor:
        """Get timestep indices.

        Returns:
            Tensor of shape [time]
        """
        if "timesteps" not in self._cache:
            self._cache["timesteps"] = torch.tensor([s.timestep for s in self.history])
        return self._cache["timesteps"]

    def get_outputs(self) -> Optional[Tensor]:
        """Get stacked outputs if available.

        Returns:
            Tensor of shape [time, ...] or None
        """
        if not self.history or self.history[0].outputs is None:
            return None

        if "outputs" not in self._cache:
            self._cache["outputs"] = torch.stack([s.outputs for s in self.history if s.outputs is not None])
        return self._cache["outputs"]

    def get_observations(self) -> Optional[Tensor]:
        """Get stacked observations if available.

        Returns:
            Tensor of shape [time, ...] or None
        """
        if not self.history or self.history[0].observation is None:
            return None

        if "observations" not in self._cache:
            obs = [s.observation for s in self.history if s.observation is not None]
            if obs:
                self._cache["observations"] = torch.stack(obs)
            else:
                return None
        return self._cache["observations"]

    def get_resampling_events(self) -> List[int]:
        """Get timesteps where resampling occurred.

        Returns:
            List of timestep indices
        """
        return [s.timestep for s in self.history if s.resampled]

    def get_extra(self, key: str) -> Optional[Tensor]:
        """Get extra data by key.

        Args:
            key: Extra data key

        Returns:
            Stacked tensor if key exists, None otherwise
        """
        cache_key = f"extra_{key}"
        if cache_key not in self._cache:
            values = [s.extra.get(key) for s in self.history if key in s.extra]
            if values and isinstance(values[0], Tensor):
                self._cache[cache_key] = torch.stack(values)
            elif values:
                self._cache[cache_key] = values
            else:
                return None
        return self._cache[cache_key]

    # =========================================================================
    # Computed Metrics
    # =========================================================================

    def get_weight_entropy(self) -> Tensor:
        """Compute weight entropy over time.

        Returns:
            Tensor of shape [time]
        """
        if "weight_entropy" not in self._cache:
            weights = self.get_weights()
            log_weights = self.get_log_weights()
            # Normalize per timestep
            log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
            entropy = -(weights * log_weights).sum(dim=-1)
            self._cache["weight_entropy"] = entropy
        return self._cache["weight_entropy"]

    def get_particle_variance(self) -> Tensor:
        """Compute per-dimension variance across particles.

        Returns:
            Tensor of shape [time, hidden_size]
        """
        if "particle_variance" not in self._cache:
            particles = self.get_particles()
            self._cache["particle_variance"] = particles.var(dim=1)
        return self._cache["particle_variance"]

    def get_weighted_mean(self) -> Tensor:
        """Compute weighted mean of particles over time.

        Returns:
            Tensor of shape [time, hidden_size]
        """
        if "weighted_mean" not in self._cache:
            particles = self.get_particles()  # [time, K, H]
            weights = self.get_weights()  # [time, K]
            weighted_mean = (weights.unsqueeze(-1) * particles).sum(dim=1)
            self._cache["weighted_mean"] = weighted_mean
        return self._cache["weighted_mean"]

    def get_weighted_variance(self) -> Tensor:
        """Compute weighted variance of particles over time.

        Returns:
            Tensor of shape [time, hidden_size]
        """
        if "weighted_variance" not in self._cache:
            particles = self.get_particles()  # [time, K, H]
            weights = self.get_weights()  # [time, K]
            mean = self.get_weighted_mean()  # [time, H]
            diff_sq = (particles - mean.unsqueeze(1)) ** 2
            weighted_var = (weights.unsqueeze(-1) * diff_sq).sum(dim=1)
            self._cache["weighted_variance"] = weighted_var
        return self._cache["weighted_variance"]

    def get_pairwise_distances(self) -> Tensor:
        """Compute average pairwise distance between particles.

        Returns:
            Tensor of shape [time]
        """
        if "pairwise_distances" not in self._cache:
            particles = self.get_particles()  # [time, K, H]
            time_steps, K, H = particles.shape

            # Compute pairwise distances for each timestep
            distances = []
            for t in range(time_steps):
                dists = torch.cdist(particles[t:t+1], particles[t:t+1]).squeeze(0)
                # Get upper triangle (excluding diagonal)
                mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
                avg_dist = dists[mask].mean()
                distances.append(avg_dist)

            self._cache["pairwise_distances"] = torch.tensor(distances)
        return self._cache["pairwise_distances"]

    def get_numerical_health(self) -> Dict[str, Tensor]:
        """Check numerical health over time.

        Returns:
            Dict with health indicators
        """
        particles = self.get_particles()
        log_weights = self.get_log_weights()

        has_nan = torch.isnan(particles).any(dim=(1, 2)) | torch.isnan(log_weights).any(dim=1)
        has_inf = torch.isinf(particles).any(dim=(1, 2)) | torch.isinf(log_weights).any(dim=1)
        max_norm = particles.norm(dim=-1).max(dim=-1).values

        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "max_norm": max_norm,
        }
