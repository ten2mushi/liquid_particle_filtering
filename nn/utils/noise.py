"""Noise injection utilities for particle filters.

Supports four noise types:
- Constant: Fixed noise scale per dimension
- Time-scaled: Noise scaled by sqrt(dt) for variable timesteps
- Learned: Learnable noise scale per dimension
- State-dependent: MLP-based noise scale depending on current state
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union
import math

import torch
import torch.nn as nn
from torch import Tensor


class NoiseType(Enum):
    """Types of noise injection."""
    CONSTANT = "constant"
    TIME_SCALED = "time_scaled"
    LEARNED = "learned"
    STATE_DEPENDENT = "state_dependent"


class NoiseInjector(nn.Module, ABC):
    """Abstract base class for noise injection."""

    @abstractmethod
    def forward(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        """Inject noise into states.

        Args:
            states: Particle states [batch, K, hidden_size]
            timespans: Optional time deltas [batch, 1] or scalar

        Returns:
            noisy_states: States with added noise [batch, K, hidden_size]
        """
        pass

    @abstractmethod
    def get_noise_scale(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        """Get the noise scale (for monitoring).

        Args:
            states: Particle states [batch, K, hidden_size]
            timespans: Optional time deltas

        Returns:
            scale: Noise scale [hidden_size] or [batch, K, hidden_size]
        """
        pass


class ConstantNoise(NoiseInjector):
    """Constant noise scale per dimension.

    noise = scale * epsilon, where epsilon ~ N(0, I)
    """

    def __init__(
        self,
        hidden_size: int,
        noise_init: float = 0.1,
    ):
        """Initialize constant noise injector.

        Args:
            hidden_size: Dimension of hidden state
            noise_init: Initial noise scale
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer(
            "noise_scale",
            torch.full((hidden_size,), noise_init)
        )

    def forward(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        noise = self.noise_scale * torch.randn_like(states)
        return states + noise

    def get_noise_scale(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        return self.noise_scale

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, scale={self.noise_scale.mean().item():.4f}"


class TimeScaledNoise(NoiseInjector):
    """Noise scaled by sqrt(dt) for SDE consistency.

    noise = scale * sqrt(dt) * epsilon

    This is consistent with Euler-Maruyama discretization of SDEs.
    """

    def __init__(
        self,
        hidden_size: int,
        noise_init: float = 0.1,
        default_dt: float = 1.0,
    ):
        """Initialize time-scaled noise injector.

        Args:
            hidden_size: Dimension of hidden state
            noise_init: Initial noise scale (per unit time)
            default_dt: Default time step if timespans not provided
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.default_dt = default_dt
        self.register_buffer(
            "noise_scale",
            torch.full((hidden_size,), noise_init)
        )

    def forward(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        # Get time step
        if timespans is not None:
            if timespans.dim() == 2:
                # [batch, 1] -> [batch, 1, 1] for broadcasting
                dt = timespans.unsqueeze(-1)
            else:
                dt = timespans
        else:
            dt = self.default_dt

        # Scale by sqrt(dt)
        sqrt_dt = torch.sqrt(torch.as_tensor(dt, device=states.device, dtype=states.dtype))
        scaled_noise = self.noise_scale * sqrt_dt

        noise = scaled_noise * torch.randn_like(states)
        return states + noise

    def get_noise_scale(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        if timespans is not None:
            sqrt_dt = torch.sqrt(torch.as_tensor(timespans, device=states.device, dtype=states.dtype))
            return self.noise_scale * sqrt_dt.mean()
        return self.noise_scale * math.sqrt(self.default_dt)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, default_dt={self.default_dt}"


class LearnedNoise(NoiseInjector):
    """Learnable noise scale per dimension.

    noise = exp(log_scale) * epsilon

    The log scale is learned during training.
    """

    def __init__(
        self,
        hidden_size: int,
        noise_init: float = 0.1,
        time_scaled: bool = True,
        min_scale: float = 1e-6,
        max_scale: float = 10.0,
    ):
        """Initialize learned noise injector.

        Args:
            hidden_size: Dimension of hidden state
            noise_init: Initial noise scale
            time_scaled: Whether to scale by sqrt(dt)
            min_scale: Minimum noise scale (for stability)
            max_scale: Maximum noise scale (for stability)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.time_scaled = time_scaled
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Use log parameterization for positive scale
        log_init = math.log(max(noise_init, min_scale))
        self.log_noise_scale = nn.Parameter(
            torch.full((hidden_size,), log_init)
        )

    @property
    def noise_scale(self) -> Tensor:
        """Get current noise scale (clamped)."""
        scale = torch.exp(self.log_noise_scale)
        return torch.clamp(scale, self.min_scale, self.max_scale)

    def forward(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        scale = self.noise_scale

        # Optional time scaling
        if self.time_scaled and timespans is not None:
            if timespans.dim() == 2:
                dt = timespans.unsqueeze(-1)
            else:
                dt = timespans
            sqrt_dt = torch.sqrt(torch.as_tensor(dt, device=states.device, dtype=states.dtype))
            scale = scale * sqrt_dt

        noise = scale * torch.randn_like(states)
        return states + noise

    def get_noise_scale(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        scale = self.noise_scale
        if self.time_scaled and timespans is not None:
            sqrt_dt = torch.sqrt(torch.as_tensor(timespans, device=states.device, dtype=states.dtype))
            scale = scale * sqrt_dt.mean()
        return scale

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, time_scaled={self.time_scaled}"


class StateDependentNoise(NoiseInjector):
    """State-dependent noise scale via MLP.

    scale(h) = MLP(h)
    noise = scale(h) * epsilon

    The noise scale depends on the current state, allowing
    the model to learn heteroscedastic uncertainty.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
        noise_init: float = 0.1,
        time_scaled: bool = True,
        min_scale: float = 1e-6,
        max_scale: float = 10.0,
        activation: str = "tanh",
    ):
        """Initialize state-dependent noise injector.

        Args:
            hidden_size: Dimension of hidden state
            mlp_hidden: Hidden units in noise prediction MLP
            mlp_layers: Number of layers in MLP
            noise_init: Target initial noise scale
            time_scaled: Whether to scale by sqrt(dt)
            min_scale: Minimum noise scale
            max_scale: Maximum noise scale
            activation: Activation function ('tanh', 'relu', 'gelu')
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.time_scaled = time_scaled
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Activation function
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        act_fn = activations.get(activation, nn.Tanh)

        # Build MLP
        layers = []
        in_dim = hidden_size
        for i in range(mlp_layers - 1):
            layers.extend([
                nn.Linear(in_dim, mlp_hidden),
                act_fn(),
            ])
            in_dim = mlp_hidden

        # Final layer outputs log scale
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.Softplus())  # Ensure positive output

        self.noise_mlp = nn.Sequential(*layers)

        # Initialize to output approximately noise_init
        self._init_weights(noise_init)

    def _init_weights(self, target_scale: float):
        """Initialize weights so initial output is approximately target_scale."""
        with torch.no_grad():
            # Initialize last layer bias to achieve target scale
            last_linear = None
            for module in reversed(list(self.noise_mlp.modules())):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            if last_linear is not None:
                # Softplus(x) ≈ x for x > 2, so set bias ≈ target_scale
                last_linear.bias.fill_(target_scale)
                last_linear.weight.zero_()

    def forward(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        batch, K, H = states.shape

        # Flatten for MLP
        states_flat = states.reshape(-1, H)
        scale_flat = self.noise_mlp(states_flat)
        scale = scale_flat.reshape(batch, K, H)

        # Clamp scale
        scale = torch.clamp(scale, self.min_scale, self.max_scale)

        # Optional time scaling
        if self.time_scaled and timespans is not None:
            if timespans.dim() == 2:
                dt = timespans.unsqueeze(-1)
            else:
                dt = timespans
            sqrt_dt = torch.sqrt(torch.as_tensor(dt, device=states.device, dtype=states.dtype))
            scale = scale * sqrt_dt

        noise = scale * torch.randn_like(states)
        return states + noise

    def get_noise_scale(
        self,
        states: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        batch, K, H = states.shape

        # Flatten for MLP
        states_flat = states.reshape(-1, H)
        scale_flat = self.noise_mlp(states_flat)
        scale = scale_flat.reshape(batch, K, H)

        # Clamp and optionally time-scale
        scale = torch.clamp(scale, self.min_scale, self.max_scale)
        if self.time_scaled and timespans is not None:
            sqrt_dt = torch.sqrt(torch.as_tensor(timespans, device=states.device, dtype=states.dtype))
            scale = scale * sqrt_dt.mean()

        return scale

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, time_scaled={self.time_scaled}"


def create_noise_injector(
    noise_type: Union[str, NoiseType],
    hidden_size: int,
    **kwargs,
) -> NoiseInjector:
    """Factory function to create noise injector.

    Args:
        noise_type: Type of noise injection
        hidden_size: Dimension of hidden state
        **kwargs: Additional arguments for specific noise type

    Returns:
        NoiseInjector instance
    """
    if isinstance(noise_type, str):
        noise_type = NoiseType(noise_type)

    if noise_type == NoiseType.CONSTANT:
        return ConstantNoise(hidden_size, **kwargs)
    elif noise_type == NoiseType.TIME_SCALED:
        return TimeScaledNoise(hidden_size, **kwargs)
    elif noise_type == NoiseType.LEARNED:
        return LearnedNoise(hidden_size, **kwargs)
    elif noise_type == NoiseType.STATE_DEPENDENT:
        return StateDependentNoise(hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
