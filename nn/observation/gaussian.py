"""Gaussian observation model for particle filters.

Implements observation model where hidden states are mapped to
observation space via a linear transformation with Gaussian noise.
"""

from typing import Optional, Union
import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import ObservationModel


class GaussianObservationModel(ObservationModel):
    """Gaussian observation model with linear mapping.

    Maps hidden states to observation space via:
        y = Wh + b + epsilon, where epsilon ~ N(0, sigma^2 I)

    The log-likelihood is:
        log p(y | h) = -0.5 * ||y - (Wh + b)||^2 / sigma^2 - const

    Supports:
    - Diagonal covariance (per-dimension noise)
    - Scalar covariance (shared noise)
    - Learnable or fixed noise parameters
    """

    def __init__(
        self,
        hidden_size: int,
        obs_size: int,
        obs_noise_std: Union[float, Tensor] = 1.0,
        learnable_noise: bool = False,
        diagonal_covariance: bool = True,
        min_noise_std: float = 1e-4,
    ):
        """Initialize Gaussian observation model.

        Args:
            hidden_size: Dimension of hidden states
            obs_size: Dimension of observations
            obs_noise_std: Observation noise std (scalar or per-dim)
            learnable_noise: If True, noise std is learnable
            diagonal_covariance: If True, use per-dim noise; else scalar
            min_noise_std: Minimum noise std (for stability)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.diagonal_covariance = diagonal_covariance
        self.min_noise_std = min_noise_std

        # Linear projection from hidden to observation space
        self.projection = nn.Linear(hidden_size, obs_size)

        # Initialize noise parameters
        if diagonal_covariance:
            noise_shape = (obs_size,)
        else:
            noise_shape = ()

        if isinstance(obs_noise_std, Tensor):
            log_noise = torch.log(obs_noise_std.clamp(min=min_noise_std))
        else:
            log_noise = torch.full(noise_shape, math.log(max(obs_noise_std, min_noise_std)))

        if learnable_noise:
            self.log_noise_std = nn.Parameter(log_noise)
        else:
            self.register_buffer("log_noise_std", log_noise)

    @property
    def noise_std(self) -> Tensor:
        """Get current noise standard deviation."""
        return torch.exp(self.log_noise_std).clamp(min=self.min_noise_std)

    @property
    def noise_var(self) -> Tensor:
        """Get current noise variance."""
        return self.noise_std ** 2

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute Gaussian log-likelihood.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observations [batch, obs_size]

        Returns:
            log_likelihoods: [batch, K]
        """
        # Project states to observation space
        # nn.Linear broadcasts over batch dimensions, so no reshape needed
        predictions = self.projection(states)  # [batch, K, obs_size]

        # Residuals (expand observations for broadcasting)
        residuals = observations.unsqueeze(1) - predictions  # [batch, K, obs_size]

        # Compute log-likelihood
        var = self.noise_var

        if self.diagonal_covariance:
            # Per-dimension variance: sum over dimensions
            # log p = -0.5 * sum_d [(y_d - mu_d)^2 / var_d + log(2*pi*var_d)]
            sq_mahal = (residuals ** 2 / var).sum(dim=-1)  # [batch, K]
            log_det = torch.log(2 * math.pi * var).sum()
            log_lik = -0.5 * (sq_mahal + log_det)
        else:
            # Scalar variance
            sq_norm = (residuals ** 2).sum(dim=-1)  # [batch, K]
            log_lik = -0.5 * sq_norm / var - 0.5 * self.obs_size * torch.log(
                2 * math.pi * var
            )

        return log_lik

    def predict(self, states: Tensor, **kwargs) -> Tensor:
        """Predict mean observations from states.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            predictions: Mean predictions [batch, K, obs_size]
        """
        # nn.Linear broadcasts over batch dimensions, so no reshape needed
        return self.projection(states)

    def sample(self, states: Tensor, **kwargs) -> Tensor:
        """Sample observations from the Gaussian model.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            samples: Sampled observations [batch, K, obs_size]
        """
        predictions = self.predict(states)
        noise = self.noise_std * torch.randn_like(predictions)
        return predictions + noise

    def extra_repr(self) -> str:
        noise_str = f"{self.noise_std.mean().item():.4f}"
        if self.diagonal_covariance:
            noise_str += " (diagonal)"
        return (
            f"hidden_size={self.hidden_size}, obs_size={self.obs_size}, "
            f"noise_std={noise_str}"
        )


class HeteroscedasticGaussianObservationModel(ObservationModel):
    """Gaussian observation model with state-dependent noise.

    The noise variance is predicted from the hidden state, allowing
    the model to express different levels of uncertainty in different
    regions of state space.

    y ~ N(mu(h), diag(sigma(h)^2))

    where both mu and sigma are learned functions of h.
    """

    def __init__(
        self,
        hidden_size: int,
        obs_size: int,
        noise_hidden_size: Optional[int] = None,
        min_noise_std: float = 1e-4,
        max_noise_std: float = 10.0,
        init_noise_std: float = 1.0,
    ):
        """Initialize heteroscedastic Gaussian observation model.

        Args:
            hidden_size: Dimension of hidden states
            obs_size: Dimension of observations
            noise_hidden_size: Hidden size for noise MLP (default: hidden_size // 2)
            min_noise_std: Minimum noise std
            max_noise_std: Maximum noise std
            init_noise_std: Initial noise std
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std

        if noise_hidden_size is None:
            noise_hidden_size = max(hidden_size // 2, 16)

        # Mean prediction network
        self.mean_net = nn.Linear(hidden_size, obs_size)

        # Noise prediction network (outputs log std)
        self.noise_net = nn.Sequential(
            nn.Linear(hidden_size, noise_hidden_size),
            nn.Tanh(),
            nn.Linear(noise_hidden_size, obs_size),
        )

        # Initialize noise network to output init_noise_std
        self._init_noise_net(init_noise_std)

    def _init_noise_net(self, target_std: float):
        """Initialize noise network to output target std."""
        with torch.no_grad():
            # Set final layer bias to achieve target std through softplus
            target_log_std = math.log(target_std)
            self.noise_net[-1].bias.fill_(target_log_std)
            self.noise_net[-1].weight.zero_()

    def get_noise_std(self, states: Tensor) -> Tensor:
        """Compute state-dependent noise std.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            noise_std: Per-state noise std [batch, K, obs_size]
        """
        # nn.Sequential with Linear layers broadcasts over batch dimensions
        log_std = self.noise_net(states)  # [batch, K, obs_size]

        # Apply softplus and clamp for stability
        noise_std = torch.nn.functional.softplus(log_std)
        noise_std = noise_std.clamp(self.min_noise_std, self.max_noise_std)

        return noise_std

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute heteroscedastic Gaussian log-likelihood.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observations [batch, obs_size]

        Returns:
            log_likelihoods: [batch, K]
        """
        batch, K, H = states.shape

        # Get predictions and state-dependent noise
        predictions = self.predict(states)  # [batch, K, obs_size]
        noise_std = self.get_noise_std(states)  # [batch, K, obs_size]
        noise_var = noise_std ** 2

        # Expand observations
        obs_expanded = observations.unsqueeze(1)  # [batch, 1, obs_size]

        # Residuals
        residuals = obs_expanded - predictions  # [batch, K, obs_size]

        # Log-likelihood with per-sample variance
        sq_mahal = (residuals ** 2 / noise_var).sum(dim=-1)  # [batch, K]
        log_det = torch.log(2 * math.pi * noise_var).sum(dim=-1)  # [batch, K]
        log_lik = -0.5 * (sq_mahal + log_det)

        return log_lik

    def predict(self, states: Tensor, **kwargs) -> Tensor:
        """Predict mean observations from states.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            predictions: Mean predictions [batch, K, obs_size]
        """
        # nn.Linear broadcasts over batch dimensions, so no reshape needed
        return self.mean_net(states)

    def sample(self, states: Tensor, **kwargs) -> Tensor:
        """Sample observations from the heteroscedastic model.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            samples: Sampled observations [batch, K, obs_size]
        """
        predictions = self.predict(states)
        noise_std = self.get_noise_std(states)
        noise = noise_std * torch.randn_like(predictions)
        return predictions + noise

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, obs_size={self.obs_size}"
