"""Parameter-level particle filter data collector."""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from .base_collector import BaseDataCollector, CollectedStep


class ParamCollector(BaseDataCollector):
    """Collector for parameter-level particle filters (Approach B).

    Extends base collector with parameter-level specific data:
    - Parameter particle values
    - Per-parameter posterior distributions
    - Parameter correlation matrices
    """

    def log_step(
        self,
        particles: Tensor,
        log_weights: Tensor,
        outputs: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        param_particles: Optional[Tensor] = None,
        shared_state: Optional[Tensor] = None,
        param_names: Optional[List[str]] = None,
        **extra,
    ) -> CollectedStep:
        """Log a single timestep with parameter-level specific data.

        Args:
            particles: State particles [batch, K, hidden_size]
            log_weights: Log weights [batch, K]
            outputs: Model outputs (optional)
            observation: Ground truth observation (optional)
            param_particles: Parameter particles [batch, K, total_params]
            shared_state: Shared hidden state [batch, hidden_size]
            param_names: Names of tracked parameters
            **extra: Additional data to store

        Returns:
            CollectedStep for this timestep
        """
        # Add param-level specific data to extra
        if param_particles is not None:
            extra["param_particles"] = param_particles
        if shared_state is not None:
            extra["shared_state"] = shared_state
        if param_names is not None:
            extra["param_names"] = param_names

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

    def get_shared_state(self) -> Optional[Tensor]:
        """Get shared state values over time.

        Returns:
            Tensor of shape [time, hidden_size] or None
        """
        return self.get_extra("shared_state")

    def get_param_names(self) -> Optional[List[str]]:
        """Get tracked parameter names.

        Returns:
            List of parameter names or None
        """
        if self.arch_info and self.arch_info.tracked_params:
            return self.arch_info.tracked_params

        # Try to get from first step's extra
        if self.history and "param_names" in self.history[0].extra:
            return self.history[0].extra["param_names"]

        return None

    def get_param_posterior_stats(self) -> Optional[Dict[str, Tensor]]:
        """Compute posterior statistics for tracked parameters.

        Returns:
            Dict with:
                - mean: [time, total_params]
                - std: [time, total_params]
                - quantiles: [time, 3, total_params] (25%, 50%, 75%)
        """
        param_particles = self.get_param_particles()
        if param_particles is None:
            return None

        if "param_posterior_stats" not in self._cache:
            weights = self.get_weights()  # [time, K]

            # Weighted mean
            mean = (weights.unsqueeze(-1) * param_particles).sum(dim=1)

            # Weighted std
            diff_sq = (param_particles - mean.unsqueeze(1)) ** 2
            variance = (weights.unsqueeze(-1) * diff_sq).sum(dim=1)
            std = torch.sqrt(variance + 1e-8)

            # Quantiles (unweighted for simplicity)
            quantiles = torch.quantile(
                param_particles, torch.tensor([0.25, 0.5, 0.75]), dim=1
            ).permute(1, 0, 2)  # [time, 3, params]

            self._cache["param_posterior_stats"] = {
                "mean": mean,
                "std": std,
                "quantiles": quantiles,
            }

        return self._cache["param_posterior_stats"]

    def get_param_correlation(self) -> Optional[Tensor]:
        """Compute correlation matrix between parameters over time.

        Returns:
            Tensor of shape [time, params, params] or None
        """
        param_particles = self.get_param_particles()
        if param_particles is None:
            return None

        if "param_correlation" not in self._cache:
            time_steps, K, P = param_particles.shape
            correlations = []

            for t in range(time_steps):
                # Center the data
                centered = param_particles[t] - param_particles[t].mean(dim=0)
                # Compute covariance
                cov = (centered.T @ centered) / (K - 1)
                # Compute correlation
                std = torch.sqrt(torch.diag(cov) + 1e-8)
                corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
                correlations.append(corr)

            self._cache["param_correlation"] = torch.stack(correlations)

        return self._cache["param_correlation"]
