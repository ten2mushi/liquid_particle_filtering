"""Dual particle filter data collector."""

from typing import Dict, Optional

import torch
from torch import Tensor

from .base_collector import BaseDataCollector, CollectedStep


class DualCollector(BaseDataCollector):
    """Collector for dual particle filters (Approach C).

    Extends base collector with dual-specific data:
    - Joint state-parameter particles
    - Rao-Blackwell variance reduction metrics
    - State-parameter correlation
    """

    def log_step(
        self,
        particles: Tensor,
        log_weights: Tensor,
        outputs: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        param_particles: Optional[Tensor] = None,
        rao_blackwell_variance_before: Optional[Tensor] = None,
        rao_blackwell_variance_after: Optional[Tensor] = None,
        **extra,
    ) -> CollectedStep:
        """Log a single timestep with dual-specific data.

        Args:
            particles: State particles [batch, K, hidden_size]
            log_weights: Log weights [batch, K]
            outputs: Model outputs (optional)
            observation: Ground truth observation (optional)
            param_particles: Parameter particles [batch, K, total_params]
            rao_blackwell_variance_before: Variance before RB [batch, hidden_size]
            rao_blackwell_variance_after: Variance after RB [batch, hidden_size]
            **extra: Additional data to store

        Returns:
            CollectedStep for this timestep
        """
        # Add dual-specific data to extra
        if param_particles is not None:
            extra["param_particles"] = param_particles
        if rao_blackwell_variance_before is not None:
            extra["rb_var_before"] = rao_blackwell_variance_before
        if rao_blackwell_variance_after is not None:
            extra["rb_var_after"] = rao_blackwell_variance_after

        return super().log_step(
            particles=particles,
            log_weights=log_weights,
            outputs=outputs,
            observation=observation,
            **extra,
        )

    def get_param_particles(self) -> Optional[Tensor]:
        """Get parameter particle values over time.

        Returns:
            Tensor of shape [time, K, total_params] or None
        """
        return self.get_extra("param_particles")

    def get_joint_particles(self) -> Optional[Tensor]:
        """Get concatenated state-parameter particles.

        Returns:
            Tensor of shape [time, K, hidden_size + total_params] or None
        """
        param_particles = self.get_param_particles()
        if param_particles is None:
            return None

        if "joint_particles" not in self._cache:
            state_particles = self.get_particles()
            self._cache["joint_particles"] = torch.cat(
                [state_particles, param_particles], dim=-1
            )

        return self._cache["joint_particles"]

    def get_rao_blackwell_variance(self) -> Optional[Dict[str, Tensor]]:
        """Get Rao-Blackwell variance reduction metrics.

        Returns:
            Dict with:
                - before: [time, hidden_size] variance before RB
                - after: [time, hidden_size] variance after RB
                - reduction: [time] variance reduction ratio
        """
        before = self.get_extra("rb_var_before")
        after = self.get_extra("rb_var_after")

        if before is None or after is None:
            return None

        if "rao_blackwell_variance" not in self._cache:
            # Compute reduction ratio
            before_mean = before.mean(dim=-1)
            after_mean = after.mean(dim=-1)
            reduction = 1.0 - (after_mean / (before_mean + 1e-8))

            self._cache["rao_blackwell_variance"] = {
                "before": before,
                "after": after,
                "reduction": reduction,
            }

        return self._cache["rao_blackwell_variance"]

    def get_state_param_correlation(self) -> Optional[Tensor]:
        """Compute correlation between state and parameter dimensions.

        Returns:
            Tensor of shape [time, hidden_size, total_params] or None
        """
        param_particles = self.get_param_particles()
        if param_particles is None:
            return None

        if "state_param_correlation" not in self._cache:
            state_particles = self.get_particles()  # [time, K, H]
            time_steps, K, H = state_particles.shape
            P = param_particles.shape[-1]

            correlations = []
            for t in range(time_steps):
                # Center both
                state_centered = state_particles[t] - state_particles[t].mean(dim=0)
                param_centered = param_particles[t] - param_particles[t].mean(dim=0)

                # Compute cross-covariance
                cross_cov = (state_centered.T @ param_centered) / (K - 1)

                # Normalize to correlation
                state_std = state_particles[t].std(dim=0) + 1e-8
                param_std = param_particles[t].std(dim=0) + 1e-8
                corr = cross_cov / (state_std.unsqueeze(1) * param_std.unsqueeze(0))

                correlations.append(corr)

            self._cache["state_param_correlation"] = torch.stack(correlations)

        return self._cache["state_param_correlation"]

    def get_marginal_state_stats(self) -> Dict[str, Tensor]:
        """Compute marginalized state posterior statistics.

        Returns:
            Dict with mean, std, quantiles
        """
        if "marginal_state_stats" not in self._cache:
            particles = self.get_particles()
            weights = self.get_weights()

            mean = (weights.unsqueeze(-1) * particles).sum(dim=1)
            diff_sq = (particles - mean.unsqueeze(1)) ** 2
            variance = (weights.unsqueeze(-1) * diff_sq).sum(dim=1)
            std = torch.sqrt(variance + 1e-8)

            quantiles = torch.quantile(
                particles, torch.tensor([0.25, 0.5, 0.75]), dim=1
            ).permute(1, 0, 2)

            self._cache["marginal_state_stats"] = {
                "mean": mean,
                "std": std,
                "quantiles": quantiles,
            }

        return self._cache["marginal_state_stats"]
