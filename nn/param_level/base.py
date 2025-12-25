"""Abstract base class for parameter-level particle filter cells.

Parameter-level particle filters maintain K particles over model parameters,
allowing uncertainty estimation over the model's learned representations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict, List
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
from .param_registry import ParameterRegistry


class ParamLevelPFCell(nn.Module, ABC):
    """Abstract base class for parameter-level particle filter cells.

    Parameter-level particle filters (Approach B) maintain K particles
    over model parameters rather than hidden states. This enables:
    - Uncertainty over learned representations
    - Online adaptation via parameter evolution
    - Ensemble-like predictions with principled weighting

    Key differences from state-level PF:
    - Particles are parameter vectors, not hidden states
    - Single hidden state, multiple parameter hypotheses
    - Useful when parameters should adapt during inference

    Subclasses must implement:
    - _create_base_cell: Create the underlying NCP cell
    - _get_trackable_params: Return which params to track
    - _forward_with_params: Run forward with specific params
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_particles: int = 8,
        tracked_params: Optional[List[str]] = None,
        param_evolution_noise: float = 0.01,
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize parameter-level particle filter cell.

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            n_particles: Number of particles K (typically smaller than state-level)
            tracked_params: List of parameter names to track (None = all)
            param_evolution_noise: Std of noise for parameter evolution
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_particles = n_particles
        self.param_evolution_noise = param_evolution_noise
        self._tracked_param_names = tracked_params

        # Parameter registry (populated by subclass)
        self.param_registry = ParameterRegistry()

        # Create soft resampler
        self.resampler = SoftResampler(
            n_particles=n_particles,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
        )

        # Observation model
        self.observation_model = observation_model

        # Create base cell and register parameters
        self._create_base_cell()
        self._register_tracked_params()

    @abstractmethod
    def _create_base_cell(self):
        """Create the underlying NCP cell. Must set self.base_cell."""
        pass

    def _register_tracked_params(self):
        """Register parameters for tracking based on tracked_params list."""
        trackable = self._get_trackable_params()

        # If no specific params requested, track all trackable
        if self._tracked_param_names is None:
            self._tracked_param_names = list(trackable.keys())

        # Register each tracked parameter
        for name in self._tracked_param_names:
            if name not in trackable:
                raise ValueError(
                    f"Parameter '{name}' not trackable. "
                    f"Available: {list(trackable.keys())}"
                )
            param = trackable[name]
            self.param_registry.register_group(
                name=name,
                shape=tuple(param.shape),
                evolution_noise=self.param_evolution_noise,
            )

        self.param_registry.freeze()

    @abstractmethod
    def _get_trackable_params(self) -> Dict[str, Tensor]:
        """Return dict of parameters that can be tracked.

        Returns:
            Dict mapping parameter names to their tensors
        """
        pass

    @abstractmethod
    def _forward_with_params(
        self,
        input: Tensor,
        state: Tensor,
        params: Dict[str, Tensor],
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass using specific parameter values.

        Args:
            input: Input tensor [batch * K, input_size]
            state: Hidden state [batch * K, hidden_size]
            params: Dict of parameter tensors (already expanded for particles)
            timespans: Optional time deltas

        Returns:
            output: Cell output [batch * K, output_size]
            new_state: New hidden state [batch * K, hidden_size]
        """
        pass

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Initialize hidden state, parameter particles, and log weights.

        Args:
            batch_size: Batch dimension
            device: Target device
            dtype: Target dtype

        Returns:
            state: Initial hidden state [batch, hidden_size]
            param_particles: Parameter particles [batch, K, n_params]
            log_weights: Initial log weights [batch, K]
        """
        # Single hidden state (shared across particles)
        state = torch.zeros(
            batch_size, self.hidden_size,
            device=device, dtype=dtype,
        )

        # Initialize parameter particles from current params
        base_params = self._get_trackable_params()
        param_particles = self.param_registry.init_particles(
            batch_size=batch_size,
            n_particles=self.n_particles,
            base_params=base_params,
            device=device,
            dtype=dtype,
        )

        # Uniform log weights
        log_weights = torch.full(
            (batch_size, self.n_particles),
            -math.log(self.n_particles),
            device=device, dtype=dtype,
        )

        return state, param_particles, log_weights

    def set_observation_model(self, observation_model: ObservationModel):
        """Set the observation model for weight updates."""
        self.observation_model = observation_model

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        timespans: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        return_all_particles: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward pass of parameter-level particle filter cell.

        Args:
            input: Input tensor [batch, input_size]
            hx: Tuple of (state, param_particles, log_weights)
                - state: Hidden state [batch, hidden_size]
                - param_particles: [batch, K, n_params]
                - log_weights: [batch, K]
            timespans: Optional time deltas
            observation: Optional observation for weight update
            return_all_particles: If True, return all particle outputs

        Returns:
            output: Weighted average output or all outputs
            (state, param_particles, log_weights): Updated state
        """
        batch_size = input.shape[0]
        device = input.device
        dtype = input.dtype

        # Initialize if needed
        if hx is None:
            state, param_particles, log_weights = self.init_hidden(
                batch_size, device, dtype
            )
        else:
            state, param_particles, log_weights = hx

        K = self.n_particles

        # 1. Evolve parameter particles (add noise)
        param_particles = self.param_registry.evolve_particles(
            param_particles, timespans
        )

        # 2. Run forward pass for each particle
        # Expand state for all particles: [batch, hidden] -> [batch * K, hidden]
        state_expanded = state.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)
        input_expanded = input.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)

        if timespans is not None:
            if timespans.dim() == 0:
                ts_expanded = timespans
            else:
                ts_expanded = timespans.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)
        else:
            ts_expanded = None

        # Convert flattened particles to dict and expand for batch
        params_dict = {}
        for name in self.param_registry.group_names:
            param_values = self.param_registry.extract_group(param_particles, name)
            # [batch, K, *shape] -> [batch * K, *shape]
            shape = param_values.shape[2:]
            params_dict[name] = param_values.reshape(batch_size * K, *shape)

        # Forward with particle-specific parameters
        outputs_flat, new_state_flat = self._forward_with_params(
            input_expanded, state_expanded, params_dict, ts_expanded
        )

        # Reshape outputs
        output_size = outputs_flat.shape[-1]
        outputs = outputs_flat.reshape(batch_size, K, output_size)

        # For state: aggregate across particles (weighted mean)
        new_states = new_state_flat.reshape(batch_size, K, self.hidden_size)

        # 3. Update weights if observation provided
        weights_normalized = False
        if observation is not None and self.observation_model is not None:
            # Expand states to [batch, K, hidden] for observation model
            log_likelihoods = self.observation_model.log_likelihood(
                new_states, observation
            )
            log_weights = log_weight_update(log_weights, log_likelihoods)
            weights_normalized = True  # log_weight_update normalizes by default

        # 4. Resample parameter particles if needed
        # Need to handle both param_particles and states during resampling
        combined = torch.cat([
            param_particles,
            new_states,
        ], dim=-1)

        combined_resampled, log_weights = self.resampler(
            combined, log_weights, already_normalized=weights_normalized
        )

        # Split back
        param_particles = combined_resampled[:, :, :self.param_registry.total_params]
        new_states = combined_resampled[:, :, self.param_registry.total_params:]

        # Aggregate state (weighted mean)
        weights = torch.exp(normalize_log_weights(log_weights))
        new_state = (weights.unsqueeze(-1) * new_states).sum(dim=1)

        # 5. Compute output
        if return_all_particles:
            output = outputs
        else:
            weights = torch.exp(normalize_log_weights(log_weights))
            output = (weights.unsqueeze(-1) * outputs).sum(dim=1)

        return output, (new_state, param_particles, log_weights)

    def get_parameter_statistics(
        self,
        param_particles: Tensor,
        log_weights: Tensor,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Compute statistics about parameter particles.

        Args:
            param_particles: Parameter particles [batch, K, n_params]
            log_weights: Log weights [batch, K]

        Returns:
            Dict mapping param names to their statistics
        """
        weights = torch.exp(normalize_log_weights(log_weights))
        stats = {}

        for name in self.param_registry.group_names:
            params = self.param_registry.extract_group(param_particles, name)
            # [batch, K, *shape]

            # Flatten shape for statistics
            batch, K = params.shape[:2]
            params_flat = params.reshape(batch, K, -1)
            weights_expanded = weights.unsqueeze(-1)

            mean = (weights_expanded * params_flat).sum(dim=1)
            variance = (weights_expanded * (params_flat - mean.unsqueeze(1)) ** 2).sum(dim=1)

            stats[name] = {
                "mean": mean,
                "variance": variance,
                "std": torch.sqrt(variance),
            }

        stats["_ess"] = compute_ess(log_weights)

        return stats

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"n_particles={self.n_particles}, tracked_params={self._tracked_param_names}"
        )
