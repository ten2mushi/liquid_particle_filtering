"""Abstract base class for SDE particle filter cells (Approach D).

SDE particle filters integrate stochastic differential equations with
noise injected at each integration step, rather than post-hoc as in
state-level particle filters.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict
import math

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import (
    SoftResampler,
    AlphaMode,
    normalize_log_weights,
    log_weight_update,
    compute_ess,
)
from ..observation import ObservationModel
from .diffusion import DiffusionCoefficient, DiffusionType, create_diffusion
from .solvers import SDEIntegrator


class SDEPFCell(nn.Module, ABC):
    """Abstract base class for SDE particle filter cells.

    SDE particle filters (Approach D) integrate stochastic differential
    equations where noise is injected at each ODE unfold step. This is
    more physically meaningful for continuous-time dynamics.

    dh = f(h, x) dt + g(h) dW

    Key differences from state-level PF:
    - Noise is integral to the dynamics, not added post-hoc
    - Natural fit for continuous-time models like LTC
    - Noise scales correctly with time discretization
    - More principled for SDE-based uncertainty modeling

    Note: SDE approach is designed specifically for LTC-style cells
    with ODE unfolding, as they naturally discretize continuous dynamics.

    Subclasses must implement:
    - _create_ltc_params: Create LTC-specific parameters
    - _compute_drift: Compute drift term f(h, x) for one step
    """

    def __init__(
        self,
        hidden_size: int,
        n_particles: int = 32,
        # SDE configuration
        diffusion_type: Union[str, DiffusionType] = "learned",
        sigma_init: float = 0.1,
        n_unfolds: int = 6,
        solver: str = "euler_maruyama",
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize SDE particle filter cell.

        Args:
            hidden_size: Dimension of hidden state
            n_particles: Number of particles K
            diffusion_type: Type of diffusion coefficient
            sigma_init: Initial diffusion magnitude
            n_unfolds: Number of integration steps per time step
            solver: SDE solver type
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_particles = n_particles
        self.n_unfolds = n_unfolds

        # Create diffusion coefficient
        self.diffusion = create_diffusion(
            diffusion_type=diffusion_type,
            hidden_size=hidden_size,
            sigma_init=sigma_init,
        )

        # Create SDE integrator
        self.integrator = SDEIntegrator(
            hidden_size=hidden_size,
            diffusion=self.diffusion,
            solver=solver,
            n_steps=n_unfolds,
        )

        # Create soft resampler
        self.resampler = SoftResampler(
            n_particles=n_particles,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
        )

        # Observation model
        self.observation_model = observation_model

        # Create LTC-specific parameters
        self._create_ltc_params()

    @abstractmethod
    def _create_ltc_params(self):
        """Create LTC-specific parameters. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _compute_drift(
        self,
        state: Tensor,
        inputs: Tensor,
        elapsed_time: Tensor,
    ) -> Tensor:
        """Compute drift term f(h, x) for SDE.

        Args:
            state: Current state [batch, hidden_size]
            inputs: Input features [batch, input_size]
            elapsed_time: Time step [batch, 1]

        Returns:
            drift: Drift term [batch, hidden_size]
        """
        pass

    @abstractmethod
    def _map_outputs(self, state: Tensor) -> Tensor:
        """Map state to output."""
        pass

    def _propagate_particles(
        self,
        input: Tensor,
        particles: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate all particles through SDE dynamics.

        Args:
            input: Input tensor [batch, input_size]
            particles: Particle states [batch, K, hidden_size]
            timespans: Time deltas [batch, 1] or scalar

        Returns:
            outputs: Cell outputs [batch, K, output_size]
            new_particles: Propagated particles [batch, K, hidden_size]
        """
        batch, K, H = particles.shape
        device = particles.device
        dtype = particles.dtype

        # Default elapsed time
        if timespans is None:
            elapsed_time = torch.ones(batch, 1, device=device, dtype=dtype)
        elif timespans.dim() == 0:
            elapsed_time = timespans.expand(batch, 1)
        elif timespans.dim() == 1:
            elapsed_time = timespans.unsqueeze(-1)
        else:
            elapsed_time = timespans

        # Expand input for all particles
        input_expanded = input.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)
        elapsed_expanded = elapsed_time.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)

        # Flatten particles
        particles_flat = particles.reshape(batch * K, H)

        # Create drift function that uses expanded inputs
        def drift_fn(state):
            return self._compute_drift(state, input_expanded, elapsed_expanded)

        # Integrate SDE
        new_particles_flat = self.integrator(
            state=particles_flat,
            drift_fn=drift_fn,
            elapsed_time=elapsed_expanded,
            inputs=input_expanded,
        )

        # Reshape back
        new_particles = new_particles_flat.reshape(batch, K, H)

        # Compute outputs
        outputs_flat = self._map_outputs(new_particles_flat)
        output_size = outputs_flat.shape[-1]
        outputs = outputs_flat.reshape(batch, K, output_size)

        return outputs, new_particles

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize particles and log weights."""
        particles = torch.randn(
            batch_size, self.n_particles, self.hidden_size,
            device=device, dtype=dtype,
        ) * 0.1

        log_weights = torch.full(
            (batch_size, self.n_particles),
            -math.log(self.n_particles),
            device=device, dtype=dtype,
        )

        return particles, log_weights

    def set_observation_model(self, observation_model: ObservationModel):
        """Set the observation model for weight updates."""
        self.observation_model = observation_model

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        timespans: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        return_all_particles: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of SDE particle filter cell.

        Args:
            input: Input tensor [batch, input_size]
            hx: Tuple of (particles, log_weights)
            timespans: Time deltas [batch, 1] or scalar
            observation: Optional observation for weight update
            return_all_particles: If True, return all particle outputs

        Returns:
            output: Weighted average output or all outputs
            (particles, log_weights): Updated particle state
        """
        batch_size = input.shape[0]
        device = input.device
        dtype = input.dtype

        # Initialize if needed
        if hx is None:
            particles, log_weights = self.init_hidden(batch_size, device, dtype)
        else:
            particles, log_weights = hx

        # 1. Propagate particles through SDE (noise is integral)
        outputs, new_particles = self._propagate_particles(input, particles, timespans)

        # 2. Update weights if observation provided
        weights_normalized = False
        if observation is not None and self.observation_model is not None:
            log_likelihoods = self.observation_model.log_likelihood(
                new_particles, observation
            )
            log_weights = log_weight_update(log_weights, log_likelihoods)
            weights_normalized = True  # log_weight_update normalizes by default

        # 3. Resample if needed (skip ESS normalization if already normalized)
        new_particles, log_weights = self.resampler(
            new_particles, log_weights, already_normalized=weights_normalized
        )

        # 4. Compute output
        if return_all_particles:
            output = outputs
        else:
            weights = torch.exp(normalize_log_weights(log_weights))
            weights = weights.unsqueeze(-1)
            output = (weights * outputs).sum(dim=1)

        return output, (new_particles, log_weights)

    def get_diffusion_scale(self, particles: Tensor) -> Tensor:
        """Get current diffusion scale for monitoring.

        Args:
            particles: Particle states [batch, K, hidden_size]

        Returns:
            scale: Mean diffusion scale [batch]
        """
        batch, K, H = particles.shape
        particles_flat = particles.reshape(batch * K, H)
        g = self.diffusion(particles_flat)
        g = g.reshape(batch, K, -1)
        return g.mean(dim=(1, 2))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, n_particles={self.n_particles}, "
            f"n_unfolds={self.n_unfolds}"
        )
