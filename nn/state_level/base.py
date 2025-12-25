"""Abstract base class for state-level particle filter cells.

State-level particle filters maintain K particles over the hidden state,
with each particle representing a hypothesis about the current state.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict, Any
import math

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import (
    SoftResampler,
    AlphaMode,
    NoiseInjector,
    NoiseType,
    ConstantNoise,
    TimeScaledNoise,
    LearnedNoise,
    StateDependentNoise,
    normalize_log_weights,
    log_weight_update,
    compute_ess,
)
from ..observation import ObservationModel


class StateLevelPFCell(nn.Module, ABC):
    """Abstract base class for state-level particle filter cells.

    State-level particle filters maintain K particles over the hidden state h.
    At each timestep:
    1. Propagate each particle through the deterministic dynamics
    2. Inject noise to explore state space
    3. Update weights based on observation likelihood
    4. Resample if effective sample size is too low

    Subclasses must implement:
    - _create_base_cell: Create the underlying NCP cell
    - _propagate_particles: Run particles through dynamics

    Attributes:
        n_particles: Number of particles K
        hidden_size: Dimension of hidden state
        resampler: SoftResampler for particle resampling
        noise_injector: NoiseInjector for state perturbation
        observation_model: Optional model for computing likelihoods
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_particles: int = 32,
        # Noise configuration
        noise_type: Union[str, NoiseType] = "time_scaled",
        noise_init: float = 0.1,
        noise_learnable: bool = True,
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model (optional - can be set later)
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize state-level particle filter cell.

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            n_particles: Number of particles K
            noise_type: Type of noise injection
            noise_init: Initial noise scale
            noise_learnable: Whether noise parameters are learnable
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for computing p(y|h)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_particles = n_particles

        # Create noise injector
        self.noise_injector = self._create_noise_injector(
            hidden_size=hidden_size,
            noise_type=noise_type,
            noise_init=noise_init,
            learnable=noise_learnable,
        )

        # Create soft resampler
        self.resampler = SoftResampler(
            n_particles=n_particles,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
        )

        # Observation model (can be set later)
        self.observation_model = observation_model

        # Create the underlying base cell
        self._create_base_cell()

    def _create_noise_injector(
        self,
        hidden_size: int,
        noise_type: Union[str, NoiseType],
        noise_init: float,
        learnable: bool,
    ) -> NoiseInjector:
        """Create appropriate noise injector based on type."""
        if isinstance(noise_type, str):
            noise_type = NoiseType(noise_type)

        if noise_type == NoiseType.CONSTANT:
            return ConstantNoise(hidden_size, noise_init)
        elif noise_type == NoiseType.TIME_SCALED:
            return TimeScaledNoise(hidden_size, noise_init)
        elif noise_type == NoiseType.LEARNED:
            return LearnedNoise(hidden_size, noise_init, time_scaled=True)
        elif noise_type == NoiseType.STATE_DEPENDENT:
            return StateDependentNoise(hidden_size, noise_init=noise_init)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    @abstractmethod
    def _create_base_cell(self):
        """Create the underlying NCP cell. Must set self.base_cell."""
        pass

    @abstractmethod
    def _propagate_single(
        self,
        input: Tensor,
        state: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate a single state through the base cell.

        Args:
            input: Input tensor [batch, input_size]
            state: Hidden state [batch, hidden_size]
            timespans: Optional time deltas [batch, 1]

        Returns:
            output: Cell output [batch, output_size]
            new_state: New hidden state [batch, hidden_size]
        """
        pass

    def _propagate_particles(
        self,
        input: Tensor,
        particles: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate all particles through dynamics.

        Default implementation processes particles in parallel by
        reshaping to [batch * K, ...].

        Args:
            input: Input tensor [batch, input_size]
            particles: Particle states [batch, K, hidden_size]
            timespans: Optional time deltas [batch, 1]

        Returns:
            outputs: Cell outputs [batch, K, output_size]
            new_particles: Propagated particles [batch, K, hidden_size]
        """
        batch, K, H = particles.shape

        # Expand input for all particles: [batch, input_size] -> [batch * K, input_size]
        input_expanded = input.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)

        # Expand timespans if provided
        if timespans is not None:
            if timespans.dim() == 2:
                ts_expanded = timespans.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)
            else:
                ts_expanded = timespans
        else:
            ts_expanded = None

        # Flatten particles: [batch, K, H] -> [batch * K, H]
        particles_flat = particles.reshape(batch * K, H)

        # Propagate through base cell
        output_flat, new_state_flat = self._propagate_single(
            input_expanded, particles_flat, ts_expanded
        )

        # Reshape back
        output_size = output_flat.shape[-1]
        outputs = output_flat.reshape(batch, K, output_size)
        new_particles = new_state_flat.reshape(batch, K, H)

        return outputs, new_particles

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize particles and log weights.

        Args:
            batch_size: Batch dimension
            device: Target device
            dtype: Target dtype

        Returns:
            particles: Initial particles [batch, K, hidden_size]
            log_weights: Initial log weights [batch, K]
        """
        # Initialize particles near zero with small variance
        particles = torch.randn(
            batch_size, self.n_particles, self.hidden_size,
            device=device, dtype=dtype,
        ) * 0.1

        # Uniform log weights: log(1/K)
        log_weights = torch.full(
            (batch_size, self.n_particles),
            -math.log(self.n_particles),
            device=device, dtype=dtype,
        )

        return particles, log_weights

    def set_observation_model(self, observation_model: ObservationModel):
        """Set the observation model for weight updates.

        Args:
            observation_model: Model for computing p(y|h)
        """
        self.observation_model = observation_model

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        timespans: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        return_all_particles: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of particle filter cell.

        Args:
            input: Input tensor [batch, input_size]
            hx: Tuple of (particles, log_weights)
                - particles: [batch, K, hidden_size]
                - log_weights: [batch, K]
            timespans: Optional time deltas [batch, 1] or scalar
            observation: Optional observation for weight update [batch, obs_size]
            return_all_particles: If True, return outputs for all particles

        Returns:
            output: Weighted average output [batch, output_size] or
                   all outputs [batch, K, output_size] if return_all_particles
            (particles, log_weights): Updated particle state
        """
        batch_size = input.shape[0]
        device = input.device
        dtype = input.dtype

        # Initialize hidden state if needed
        if hx is None:
            particles, log_weights = self.init_hidden(batch_size, device, dtype)
        else:
            particles, log_weights = hx

        # 1. Propagate particles through dynamics
        outputs, new_particles = self._propagate_particles(
            input, particles, timespans
        )

        # 2. Inject noise
        new_particles = self.noise_injector(new_particles, timespans)

        # 3. Update weights if observation provided
        # Track whether weights are normalized to avoid redundant normalization
        weights_normalized = False
        if observation is not None and self.observation_model is not None:
            log_likelihoods = self.observation_model.log_likelihood(
                new_particles, observation
            )
            log_weights = log_weight_update(log_weights, log_likelihoods)
            weights_normalized = True  # log_weight_update normalizes by default

        # 4. Resample if needed (skip ESS normalization if already normalized)
        new_particles, log_weights = self.resampler(
            new_particles, log_weights, already_normalized=weights_normalized
        )

        # 5. Compute output
        if return_all_particles:
            output = outputs
        else:
            # Weighted average of outputs
            weights = torch.exp(normalize_log_weights(log_weights))
            weights = weights.unsqueeze(-1)  # [batch, K, 1]
            output = (weights * outputs).sum(dim=1)  # [batch, output_size]

        return output, (new_particles, log_weights)

    def get_particle_statistics(
        self,
        particles: Tensor,
        log_weights: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute statistics about particle population.

        Args:
            particles: Particle states [batch, K, hidden_size]
            log_weights: Log weights [batch, K]

        Returns:
            Dict with statistics (mean, variance, ESS, etc.)
        """
        weights = torch.exp(normalize_log_weights(log_weights))
        weights = weights.unsqueeze(-1)  # [batch, K, 1]

        # Weighted mean
        mean = (weights * particles).sum(dim=1)  # [batch, hidden_size]

        # Weighted variance
        diff = particles - mean.unsqueeze(1)
        variance = (weights * diff ** 2).sum(dim=1)  # [batch, hidden_size]

        # ESS
        ess = compute_ess(log_weights)

        return {
            "mean": mean,
            "variance": variance,
            "ess": ess,
            "max_weight": weights.max(dim=1).values.squeeze(-1),
        }

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"n_particles={self.n_particles}"
        )
