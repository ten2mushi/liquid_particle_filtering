"""State-level particle filter data collector."""

from typing import Optional

import torch
from torch import Tensor

from .base_collector import BaseDataCollector, CollectedStep


class StateCollector(BaseDataCollector):
    """Collector for state-level particle filters (Approach A).

    Extends base collector with state-level specific data:
    - Noise injection magnitude
    - State-dependent noise values
    - Collapse detection metrics
    """

    def log_step(
        self,
        particles: Tensor,
        log_weights: Tensor,
        outputs: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        noise_scale: Optional[Tensor] = None,
        pre_noise_particles: Optional[Tensor] = None,
        **extra,
    ) -> CollectedStep:
        """Log a single timestep with state-level specific data.

        Args:
            particles: Particle states [batch, K, hidden_size]
            log_weights: Log weights [batch, K]
            outputs: Model outputs (optional)
            observation: Ground truth observation (optional)
            noise_scale: Noise injection scale (optional)
            pre_noise_particles: Particles before noise injection (optional)
            **extra: Additional data to store

        Returns:
            CollectedStep for this timestep
        """
        # Add state-level specific data to extra
        if noise_scale is not None:
            extra["noise_scale"] = noise_scale
        if pre_noise_particles is not None:
            extra["pre_noise_particles"] = pre_noise_particles

        return super().log_step(
            particles=particles,
            log_weights=log_weights,
            outputs=outputs,
            observation=observation,
            **extra,
        )

    def get_noise_scale(self) -> Optional[Tensor]:
        """Get noise injection scale over time.

        Returns:
            Tensor of shape [time] or [time, hidden_size], or None
        """
        return self.get_extra("noise_scale")

    def get_pre_noise_particles(self) -> Optional[Tensor]:
        """Get particles before noise injection.

        Returns:
            Tensor of shape [time, K, hidden_size] or None
        """
        return self.get_extra("pre_noise_particles")

    def get_noise_magnitude(self) -> Optional[Tensor]:
        """Compute actual noise magnitude injected.

        Returns:
            Tensor of shape [time] or None
        """
        pre_noise = self.get_pre_noise_particles()
        if pre_noise is None:
            return None

        if "noise_magnitude" not in self._cache:
            particles = self.get_particles()
            # Compute L2 norm of noise per particle, average over particles
            noise = particles - pre_noise
            noise_norm = noise.norm(dim=-1).mean(dim=-1)  # [time]
            self._cache["noise_magnitude"] = noise_norm

        return self._cache["noise_magnitude"]

    def get_collapse_ratio(self) -> Tensor:
        """Compute collapse ratio over time.

        Collapse ratio = variance / (mean^2 + eps)
        Lower values indicate particle collapse.

        Returns:
            Tensor of shape [time]
        """
        if "collapse_ratio" not in self._cache:
            particles = self.get_particles()
            mean_state = particles.mean(dim=1)  # [time, H]
            var_mean = particles.var(dim=1).mean(dim=-1)  # [time]
            mean_sq = (mean_state ** 2).mean(dim=-1)  # [time]
            self._cache["collapse_ratio"] = var_mean / (mean_sq + 1e-8)

        return self._cache["collapse_ratio"]
