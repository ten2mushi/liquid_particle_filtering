"""SDE particle filter data collector."""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from .base_collector import BaseDataCollector, CollectedStep


class SDECollector(BaseDataCollector):
    """Collector for SDE particle filters (Approach D).

    Extends base collector with SDE-specific data:
    - Diffusion coefficient g(h) values
    - Per-unfold state trajectories
    - Drift/diffusion ratio
    - State clamping events
    """

    def log_step(
        self,
        particles: Tensor,
        log_weights: Tensor,
        outputs: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        diffusion_values: Optional[Tensor] = None,
        per_unfold_states: Optional[Tensor] = None,
        drift_values: Optional[Tensor] = None,
        clamped_mask: Optional[Tensor] = None,
        **extra,
    ) -> CollectedStep:
        """Log a single timestep with SDE-specific data.

        Args:
            particles: Particle states [batch, K, hidden_size]
            log_weights: Log weights [batch, K]
            outputs: Model outputs (optional)
            observation: Ground truth observation (optional)
            diffusion_values: Diffusion g(h) [batch, unfolds, hidden_size] or [batch, hidden_size]
            per_unfold_states: States at each unfold [batch, unfolds, K, hidden_size]
            drift_values: Drift f(h) [batch, hidden_size]
            clamped_mask: Boolean mask where clamping occurred [batch, K, hidden_size]
            **extra: Additional data to store

        Returns:
            CollectedStep for this timestep
        """
        # Add SDE-specific data to extra
        if diffusion_values is not None:
            extra["diffusion"] = diffusion_values
        if per_unfold_states is not None:
            extra["unfold_states"] = per_unfold_states
        if drift_values is not None:
            extra["drift"] = drift_values
        if clamped_mask is not None:
            extra["clamped"] = clamped_mask

        return super().log_step(
            particles=particles,
            log_weights=log_weights,
            outputs=outputs,
            observation=observation,
            **extra,
        )

    def get_diffusion(self) -> Optional[Tensor]:
        """Get diffusion coefficient values over time.

        Returns:
            Tensor of shape [time, unfolds, hidden_size] or [time, hidden_size]
        """
        return self.get_extra("diffusion")

    def get_unfold_states(self) -> Optional[Tensor]:
        """Get per-unfold state trajectories.

        Returns:
            Tensor of shape [time, unfolds, K, hidden_size] or None
        """
        return self.get_extra("unfold_states")

    def get_drift(self) -> Optional[Tensor]:
        """Get drift values over time.

        Returns:
            Tensor of shape [time, hidden_size] or None
        """
        return self.get_extra("drift")

    def get_clamping_events(self) -> Optional[Dict[str, Tensor]]:
        """Get state clamping events over time.

        Returns:
            Dict with:
                - mask: [time, K, hidden_size] boolean mask
                - count: [time] number of clamping events
                - fraction: [time] fraction of states clamped
        """
        clamped = self.get_extra("clamped")
        if clamped is None:
            return None

        if "clamping_events" not in self._cache:
            # Count clamping events
            count = clamped.sum(dim=(1, 2)).float()
            total = clamped.shape[1] * clamped.shape[2]
            fraction = count / total

            self._cache["clamping_events"] = {
                "mask": clamped,
                "count": count,
                "fraction": fraction,
            }

        return self._cache["clamping_events"]

    def get_drift_diffusion_ratio(self) -> Optional[Tensor]:
        """Compute ratio of drift magnitude to diffusion magnitude.

        Higher ratio means drift dominates (more deterministic).
        Lower ratio means diffusion dominates (more stochastic).

        Returns:
            Tensor of shape [time] or None
        """
        drift = self.get_drift()
        diffusion = self.get_diffusion()

        if drift is None or diffusion is None:
            return None

        if "drift_diffusion_ratio" not in self._cache:
            # Handle different diffusion shapes
            if diffusion.dim() == 3:
                # [time, unfolds, H] -> average over unfolds
                diff_mag = diffusion.norm(dim=-1).mean(dim=-1)
            else:
                # [time, H]
                diff_mag = diffusion.norm(dim=-1)

            drift_mag = drift.norm(dim=-1)
            ratio = drift_mag / (diff_mag + 1e-8)

            self._cache["drift_diffusion_ratio"] = ratio

        return self._cache["drift_diffusion_ratio"]

    def get_unfold_convergence(self) -> Optional[Tensor]:
        """Compute state change per unfold step.

        Returns:
            Tensor of shape [time, unfolds-1] showing state delta per unfold
        """
        unfold_states = self.get_unfold_states()
        if unfold_states is None:
            return None

        if "unfold_convergence" not in self._cache:
            # Compute state change between consecutive unfolds
            # unfold_states: [time, unfolds, K, H]
            deltas = unfold_states[:, 1:] - unfold_states[:, :-1]
            # Average delta magnitude over particles and dimensions
            convergence = deltas.norm(dim=-1).mean(dim=-1)  # [time, unfolds-1]

            self._cache["unfold_convergence"] = convergence

        return self._cache["unfold_convergence"]

    def get_euler_maruyama_stability(self) -> Optional[Dict[str, Tensor]]:
        """Compute Euler-Maruyama stability metrics.

        For stable integration, we want dt * |f(h)| << 1 and sqrt(dt) * |g(h)| << 1

        Returns:
            Dict with:
                - drift_term: [time] estimated dt * |f|
                - diffusion_term: [time] estimated sqrt(dt) * |g|
                - ratio: [time] drift_term / diffusion_term
        """
        drift = self.get_drift()
        diffusion = self.get_diffusion()

        if drift is None or diffusion is None:
            return None

        if "euler_maruyama_stability" not in self._cache:
            # Estimate dt from unfold convergence or assume 1.0
            dt = 1.0 / 6.0  # Default assumption: 6 unfolds per timestep

            drift_mag = drift.norm(dim=-1)

            if diffusion.dim() == 3:
                diff_mag = diffusion.norm(dim=-1).mean(dim=-1)
            else:
                diff_mag = diffusion.norm(dim=-1)

            drift_term = dt * drift_mag
            diffusion_term = (dt ** 0.5) * diff_mag
            ratio = drift_term / (diffusion_term + 1e-8)

            self._cache["euler_maruyama_stability"] = {
                "drift_term": drift_term,
                "diffusion_term": diffusion_term,
                "ratio": ratio,
            }

        return self._cache["euler_maruyama_stability"]
