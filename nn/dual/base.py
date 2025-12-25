"""Abstract base class for dual particle filter cells (Approach C).

Dual particle filters maintain joint (state, parameter) particles,
combining the benefits of both state-level and parameter-level approaches.
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
from ..param_level.param_registry import ParameterRegistry
from .rao_blackwell import RaoBlackwellEstimator


class DualPFCell(nn.Module, ABC):
    """Abstract base class for dual particle filter cells.

    Dual particle filters (Approach C) maintain K particles over
    joint (state, parameter) pairs. This enables simultaneous
    uncertainty estimation over both hidden states and model parameters.

    Advantages:
    - Captures state-parameter correlations
    - More complete uncertainty quantification
    - Can handle non-stationary systems

    Considerations:
    - Higher computational cost than single-level PF
    - Requires careful balancing of state vs param noise
    - Joint resampling maintains correlations

    Subclasses must implement:
    - _create_base_cell: Create the underlying NCP cell
    - _get_trackable_params: Return trackable parameters
    - _forward_with_params: Run forward with specific params
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_particles: int = 16,
        tracked_params: Optional[List[str]] = None,
        # State noise configuration
        state_noise_type: Union[str, NoiseType] = "time_scaled",
        state_noise_init: float = 0.1,
        # Parameter noise configuration
        param_evolution_noise: float = 0.01,
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Advanced options
        use_rao_blackwell: bool = True,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize dual particle filter cell.

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            n_particles: Number of particles K (shared for state+param)
            tracked_params: Parameter names to track
            state_noise_type: Type of state noise injection
            state_noise_init: Initial state noise scale
            param_evolution_noise: Std of parameter evolution noise
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            use_rao_blackwell: Use Rao-Blackwellization for estimates
            observation_model: Model for p(y|h)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_particles = n_particles
        self._tracked_param_names = tracked_params
        self.param_evolution_noise = param_evolution_noise
        self.use_rao_blackwell = use_rao_blackwell

        # Create state noise injector
        self.state_noise = self._create_noise_injector(
            hidden_size=hidden_size,
            noise_type=state_noise_type,
            noise_init=state_noise_init,
        )

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

        # Rao-Blackwell estimator (created after param registry is frozen)
        if use_rao_blackwell:
            self.rb_estimator = RaoBlackwellEstimator(
                state_size=hidden_size,
                param_size=self.param_registry.total_params,
            )

    def _create_noise_injector(
        self,
        hidden_size: int,
        noise_type: Union[str, NoiseType],
        noise_init: float,
    ) -> NoiseInjector:
        """Create appropriate noise injector."""
        if isinstance(noise_type, str):
            noise_type = NoiseType(noise_type)

        if noise_type == NoiseType.CONSTANT:
            return ConstantNoise(hidden_size, noise_init)
        elif noise_type == NoiseType.TIME_SCALED:
            return TimeScaledNoise(hidden_size, noise_init)
        elif noise_type == NoiseType.LEARNED:
            return LearnedNoise(hidden_size, noise_init)
        elif noise_type == NoiseType.STATE_DEPENDENT:
            return StateDependentNoise(hidden_size, noise_init=noise_init)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    @abstractmethod
    def _create_base_cell(self):
        """Create the underlying NCP cell."""
        pass

    def _register_tracked_params(self):
        """Register parameters for tracking."""
        trackable = self._get_trackable_params()

        if self._tracked_param_names is None:
            # Default: track output-related params
            self._tracked_param_names = [
                name for name in trackable.keys()
                if 'output' in name.lower() or 'ff' in name.lower()
            ][:4]  # Limit to avoid huge param dimension

        for name in self._tracked_param_names:
            if name not in trackable:
                raise ValueError(
                    f"Parameter '{name}' not trackable. Available: {list(trackable.keys())}"
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
        """Return dict of parameters that can be tracked."""
        pass

    @abstractmethod
    def _forward_with_params(
        self,
        input: Tensor,
        state: Tensor,
        params: Dict[str, Tensor],
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass using specific parameter values."""
        pass

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Initialize joint (state, param) particles and log weights.

        Args:
            batch_size: Batch dimension
            device: Target device
            dtype: Target dtype

        Returns:
            state_particles: State particles [batch, K, hidden_size]
            param_particles: Parameter particles [batch, K, n_params]
            log_weights: Log weights [batch, K]
        """
        K = self.n_particles

        # Initialize state particles
        state_particles = torch.randn(
            batch_size, K, self.hidden_size,
            device=device, dtype=dtype,
        ) * 0.1

        # Initialize parameter particles
        base_params = self._get_trackable_params()
        param_particles = self.param_registry.init_particles(
            batch_size=batch_size,
            n_particles=K,
            base_params=base_params,
            device=device,
            dtype=dtype,
        )

        # Uniform log weights
        log_weights = torch.full(
            (batch_size, K),
            -math.log(K),
            device=device, dtype=dtype,
        )

        return state_particles, param_particles, log_weights

    def set_observation_model(self, observation_model: ObservationModel):
        """Set the observation model."""
        self.observation_model = observation_model

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        timespans: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        return_all_particles: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward pass of dual particle filter cell.

        Args:
            input: Input tensor [batch, input_size]
            hx: Tuple of (state_particles, param_particles, log_weights)
            timespans: Optional time deltas
            observation: Optional observation for weight update
            return_all_particles: Return all particle outputs

        Returns:
            output: Weighted average output or all outputs
            (state_particles, param_particles, log_weights): Updated state
        """
        batch_size = input.shape[0]
        device = input.device
        dtype = input.dtype
        K = self.n_particles

        # Initialize if needed
        if hx is None:
            state_particles, param_particles, log_weights = self.init_hidden(
                batch_size, device, dtype
            )
        else:
            state_particles, param_particles, log_weights = hx

        # 1. Evolve parameter particles (add noise)
        param_particles = self.param_registry.evolve_particles(
            param_particles, timespans
        )

        # 2. Propagate state particles with particle-specific parameters
        input_expanded = input.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)
        state_flat = state_particles.reshape(batch_size * K, self.hidden_size)

        if timespans is not None:
            if timespans.dim() == 0:
                ts_expanded = timespans
            else:
                ts_expanded = timespans.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)
        else:
            ts_expanded = None

        # Prepare particle-specific parameters
        params_dict = {}
        for name in self.param_registry.group_names:
            param_values = self.param_registry.extract_group(param_particles, name)
            shape = param_values.shape[2:]
            params_dict[name] = param_values.reshape(batch_size * K, *shape)

        # Forward with particle params
        outputs_flat, new_state_flat = self._forward_with_params(
            input_expanded, state_flat, params_dict, ts_expanded
        )

        # Reshape
        output_size = outputs_flat.shape[-1]
        outputs = outputs_flat.reshape(batch_size, K, output_size)
        new_state_particles = new_state_flat.reshape(batch_size, K, self.hidden_size)

        # 3. Inject state noise
        new_state_particles = self.state_noise(new_state_particles, timespans)

        # 4. Update weights if observation provided
        weights_normalized = False
        if observation is not None and self.observation_model is not None:
            log_likelihoods = self.observation_model.log_likelihood(
                new_state_particles, observation
            )
            log_weights = log_weight_update(log_weights, log_likelihoods)
            weights_normalized = True  # log_weight_update normalizes by default

        # 5. Joint resampling of (state, param) pairs
        # Skip redundant normalization if weights already normalized
        ess = compute_ess(log_weights, already_normalized=weights_normalized)
        if (ess < self.resampler.resample_threshold * K).any():
            if self.use_rao_blackwell:
                new_state_particles, param_particles, log_weights = \
                    self.rb_estimator.resample(
                        new_state_particles, param_particles, log_weights,
                        alpha=self.resampler.alpha.item() if hasattr(self.resampler.alpha, 'item') else self.resampler.alpha
                    )
            else:
                # Standard joint resampling
                combined = torch.cat([
                    new_state_particles,
                    param_particles,
                ], dim=-1)
                combined, log_weights = self.resampler(
                    combined, log_weights, force_resample=True, already_normalized=weights_normalized
                )
                new_state_particles = combined[:, :, :self.hidden_size]
                param_particles = combined[:, :, self.hidden_size:]

        # 6. Compute output
        if return_all_particles:
            output = outputs
        else:
            weights = torch.exp(normalize_log_weights(log_weights))
            output = (weights.unsqueeze(-1) * outputs).sum(dim=1)

        return output, (new_state_particles, param_particles, log_weights)

    def get_statistics(
        self,
        state_particles: Tensor,
        param_particles: Tensor,
        log_weights: Tensor,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Get comprehensive statistics about particle population.

        Args:
            state_particles: [batch, K, hidden_size]
            param_particles: [batch, K, n_params]
            log_weights: [batch, K]

        Returns:
            Dict with state and parameter statistics
        """
        stats = {}

        if self.use_rao_blackwell:
            stats["state"] = self.rb_estimator.estimate_state(
                state_particles, param_particles, log_weights
            )
            stats["params"] = self.rb_estimator.estimate_params(
                state_particles, param_particles, log_weights
            )
        else:
            weights = torch.exp(normalize_log_weights(log_weights)).unsqueeze(-1)

            state_mean = (weights * state_particles).sum(dim=1)
            state_var = (weights * (state_particles - state_mean.unsqueeze(1)) ** 2).sum(dim=1)

            param_mean = (weights * param_particles).sum(dim=1)
            param_var = (weights * (param_particles - param_mean.unsqueeze(1)) ** 2).sum(dim=1)

            stats["state"] = {"mean": state_mean, "variance": state_var}
            stats["params"] = {"mean": param_mean, "variance": param_var}

        stats["ess"] = compute_ess(log_weights)

        return stats

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"n_particles={self.n_particles}, tracked_params={self._tracked_param_names}"
        )
