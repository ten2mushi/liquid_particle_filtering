"""Diffusion coefficient models for SDE particle filters.

Diffusion coefficients define the volatility/noise term in SDEs:
    dX = f(X, t) dt + g(X, t) dW

where g is the diffusion coefficient.
"""

from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
import math

import torch
import torch.nn as nn
from torch import Tensor


class DiffusionType(Enum):
    """Types of diffusion coefficients."""
    CONSTANT = "constant"
    STATE_DEPENDENT = "state_dependent"
    LEARNED = "learned"
    DIAGONAL = "diagonal"


class DiffusionCoefficient(nn.Module, ABC):
    """Abstract base class for diffusion coefficients.

    Computes g(X, t) for the SDE dX = f dt + g dW.
    The output shape determines how noise is applied:
    - [hidden_size]: Diagonal diffusion (independent noise per dimension)
    - [hidden_size, hidden_size]: Full diffusion matrix
    """

    @abstractmethod
    def forward(
        self,
        state: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute diffusion coefficient.

        Args:
            state: Current state [batch, hidden_size]
            t: Optional time [batch, 1] or scalar

        Returns:
            g: Diffusion coefficient [batch, hidden_size] or [batch, hidden_size, hidden_size]
        """
        pass

    @property
    @abstractmethod
    def is_diagonal(self) -> bool:
        """Whether this is a diagonal diffusion (returns [batch, hidden_size])."""
        pass


class ConstantDiffusion(DiffusionCoefficient):
    """Constant diffusion coefficient.

    g(X, t) = sigma (constant)

    Simplest form, assumes homogeneous noise across all states.
    """

    def __init__(
        self,
        hidden_size: int,
        sigma: float = 0.1,
        per_dimension: bool = True,
    ):
        """Initialize constant diffusion.

        Args:
            hidden_size: Dimension of state
            sigma: Diffusion constant (or initial value if per_dimension)
            per_dimension: If True, use different sigma per dimension
        """
        super().__init__()
        self.hidden_size = hidden_size
        self._per_dimension = per_dimension

        if per_dimension:
            self.register_buffer(
                "sigma",
                torch.full((hidden_size,), sigma)
            )
        else:
            self.register_buffer(
                "sigma",
                torch.tensor(sigma)
            )

    def forward(
        self,
        state: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Return constant diffusion."""
        batch = state.shape[0]

        if self._per_dimension:
            # [hidden_size] -> [batch, hidden_size]
            return self.sigma.unsqueeze(0).expand(batch, -1)
        else:
            # scalar -> [batch, hidden_size]
            return self.sigma.expand(batch, self.hidden_size)

    @property
    def is_diagonal(self) -> bool:
        return True

    def extra_repr(self) -> str:
        mean_sigma = self.sigma.mean().item()
        return f"hidden_size={self.hidden_size}, sigma={mean_sigma:.4f}"


class LearnedDiffusion(DiffusionCoefficient):
    """Learnable constant diffusion coefficient.

    g(X, t) = exp(log_sigma) where log_sigma is learned

    Uses log parameterization to ensure positivity.
    """

    def __init__(
        self,
        hidden_size: int,
        sigma_init: float = 0.1,
        min_sigma: float = 1e-6,
        max_sigma: float = 10.0,
    ):
        """Initialize learned diffusion.

        Args:
            hidden_size: Dimension of state
            sigma_init: Initial diffusion value
            min_sigma: Minimum allowed sigma
            max_sigma: Maximum allowed sigma
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        log_init = math.log(max(sigma_init, min_sigma))
        self.log_sigma = nn.Parameter(
            torch.full((hidden_size,), log_init)
        )

    @property
    def sigma(self) -> Tensor:
        """Get current sigma values."""
        return torch.exp(self.log_sigma).clamp(self.min_sigma, self.max_sigma)

    def forward(
        self,
        state: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Return learned diffusion."""
        batch = state.shape[0]
        return self.sigma.unsqueeze(0).expand(batch, -1)

    @property
    def is_diagonal(self) -> bool:
        return True

    def extra_repr(self) -> str:
        mean_sigma = self.sigma.mean().item()
        return f"hidden_size={self.hidden_size}, sigma={mean_sigma:.4f}"


class StateDependentDiffusion(DiffusionCoefficient):
    """State-dependent diffusion via MLP.

    g(X, t) = MLP(X)

    Allows heteroscedastic noise that varies with state.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
        sigma_init: float = 0.1,
        min_sigma: float = 1e-6,
        max_sigma: float = 10.0,
        activation: str = "tanh",
    ):
        """Initialize state-dependent diffusion.

        Args:
            hidden_size: Dimension of state
            mlp_hidden: Hidden units in MLP
            mlp_layers: Number of MLP layers
            sigma_init: Target initial diffusion
            min_sigma: Minimum sigma
            max_sigma: Maximum sigma
            activation: Activation function
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Activation
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

        # Output layer with Softplus for positivity
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.Softplus())

        self.mlp = nn.Sequential(*layers)

        # Initialize to output approximately sigma_init
        self._init_weights(sigma_init)

    def _init_weights(self, target_sigma: float):
        """Initialize to output target_sigma."""
        with torch.no_grad():
            for module in reversed(list(self.mlp.modules())):
                if isinstance(module, nn.Linear):
                    module.bias.fill_(target_sigma)
                    module.weight.zero_()
                    break

    def forward(
        self,
        state: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute state-dependent diffusion."""
        sigma = self.mlp(state)
        return sigma.clamp(self.min_sigma, self.max_sigma)

    @property
    def is_diagonal(self) -> bool:
        return True

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}"


class TimeVaryingDiffusion(DiffusionCoefficient):
    """Time-varying diffusion coefficient.

    g(X, t) = sigma * sqrt(t) or sigma * f(t)

    Useful for SDEs where noise should scale with elapsed time.
    """

    def __init__(
        self,
        hidden_size: int,
        sigma_init: float = 0.1,
        time_scaling: str = "sqrt",
        learnable: bool = True,
    ):
        """Initialize time-varying diffusion.

        Args:
            hidden_size: Dimension of state
            sigma_init: Base diffusion value
            time_scaling: How to scale with time ('sqrt', 'linear', 'none')
            learnable: Whether sigma is learnable
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.time_scaling = time_scaling

        if learnable:
            self.log_sigma = nn.Parameter(
                torch.full((hidden_size,), math.log(sigma_init))
            )
        else:
            self.register_buffer(
                "log_sigma",
                torch.full((hidden_size,), math.log(sigma_init))
            )

    @property
    def sigma(self) -> Tensor:
        return torch.exp(self.log_sigma)

    def forward(
        self,
        state: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute time-varying diffusion."""
        batch = state.shape[0]
        sigma = self.sigma.unsqueeze(0).expand(batch, -1)

        if t is not None:
            if t.dim() == 0:
                t = t.expand(batch, 1)
            elif t.dim() == 1:
                t = t.unsqueeze(-1)

            if self.time_scaling == "sqrt":
                sigma = sigma * torch.sqrt(t.clamp(min=1e-8))
            elif self.time_scaling == "linear":
                sigma = sigma * t

        return sigma

    @property
    def is_diagonal(self) -> bool:
        return True


def create_diffusion(
    diffusion_type: str,
    hidden_size: int,
    **kwargs,
) -> DiffusionCoefficient:
    """Factory function to create diffusion coefficient.

    Args:
        diffusion_type: Type of diffusion
        hidden_size: Dimension of state
        **kwargs: Additional arguments for specific type

    Returns:
        DiffusionCoefficient instance
    """
    if isinstance(diffusion_type, str):
        diffusion_type = DiffusionType(diffusion_type)

    if diffusion_type == DiffusionType.CONSTANT:
        # Map sigma_init to sigma for ConstantDiffusion API compatibility
        sigma = kwargs.pop('sigma_init', kwargs.pop('sigma', 0.1))
        return ConstantDiffusion(hidden_size, sigma=sigma, **kwargs)
    elif diffusion_type == DiffusionType.LEARNED:
        return LearnedDiffusion(hidden_size, **kwargs)
    elif diffusion_type == DiffusionType.STATE_DEPENDENT:
        return StateDependentDiffusion(hidden_size, **kwargs)
    elif diffusion_type == DiffusionType.DIAGONAL:
        return LearnedDiffusion(hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")
