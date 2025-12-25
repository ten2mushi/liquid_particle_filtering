"""Abstract base class for observation models.

Observation models compute the likelihood of observations given particle states,
enabling weight updates in particle filter inference.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor


class ObservationModel(nn.Module, ABC):
    """Abstract base class for observation models.

    Observation models define p(y | h) - the probability of observing y
    given hidden state h. They are used to compute importance weights
    for particle filtering.

    All implementations must provide:
    - log_likelihood: Compute log p(y | h) for each particle

    Example:
        >>> obs_model = GaussianObservationModel(hidden_size=64, obs_size=10)
        >>> # particles: [batch, K, hidden_size]
        >>> # observations: [batch, obs_size]
        >>> log_liks = obs_model.log_likelihood(particles, observations)
        >>> # log_liks: [batch, K]
    """

    @abstractmethod
    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute log-likelihood of observations given states.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observed values [batch, obs_size]
            **kwargs: Additional model-specific arguments

        Returns:
            log_likelihoods: Log p(obs | state) for each particle [batch, K]
        """
        pass

    def forward(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Forward pass computes log-likelihoods.

        This is an alias for log_likelihood for nn.Module compatibility.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observed values [batch, obs_size]
            **kwargs: Additional arguments

        Returns:
            log_likelihoods: [batch, K]
        """
        return self.log_likelihood(states, observations, **kwargs)

    def predict(
        self,
        states: Tensor,
        **kwargs,
    ) -> Tensor:
        """Predict observations from states (optional).

        Not all observation models support prediction. Subclasses
        can override this to enable generative sampling.

        Args:
            states: Particle states [batch, K, hidden_size]
            **kwargs: Additional arguments

        Returns:
            predictions: Predicted observations [batch, K, obs_size]
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support prediction"
        )

    def sample(
        self,
        states: Tensor,
        **kwargs,
    ) -> Tensor:
        """Sample observations from the model (optional).

        Not all observation models support sampling. Subclasses
        can override this for generative use cases.

        Args:
            states: Particle states [batch, K, hidden_size]
            **kwargs: Additional arguments

        Returns:
            samples: Sampled observations [batch, K, obs_size]
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sampling"
        )


class IdentityObservationModel(ObservationModel):
    """Observation model that assumes observations equal hidden states.

    Useful when the observation is a direct (noisy) measurement of
    the hidden state, with Gaussian noise.

    log p(y | h) = -0.5 * ||y - h||^2 / sigma^2 - const

    The hidden state and observation must have the same dimension.
    """

    def __init__(
        self,
        obs_noise_std: float = 1.0,
        learnable_noise: bool = False,
    ):
        """Initialize identity observation model.

        Args:
            obs_noise_std: Observation noise standard deviation
            learnable_noise: If True, noise std is a learnable parameter
        """
        super().__init__()

        if learnable_noise:
            self.log_noise_std = nn.Parameter(
                torch.tensor(obs_noise_std).log()
            )
        else:
            self.register_buffer(
                "log_noise_std",
                torch.tensor(obs_noise_std).log()
            )

    @property
    def noise_std(self) -> Tensor:
        """Get current noise standard deviation."""
        return torch.exp(self.log_noise_std)

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute Gaussian log-likelihood.

        Args:
            states: Particle states [batch, K, state_size]
            observations: Observations [batch, state_size]

        Returns:
            log_likelihoods: [batch, K]
        """
        batch, K, state_size = states.shape

        # Expand observations for broadcasting
        obs_expanded = observations.unsqueeze(1)  # [batch, 1, state_size]

        # Squared error
        sq_error = ((states - obs_expanded) ** 2).sum(dim=-1)  # [batch, K]

        # Gaussian log-likelihood
        var = self.noise_std ** 2
        log_lik = -0.5 * sq_error / var - 0.5 * state_size * torch.log(
            2 * torch.pi * var
        )

        return log_lik

    def predict(self, states: Tensor, **kwargs) -> Tensor:
        """Predict observations (identity mapping)."""
        return states

    def sample(self, states: Tensor, **kwargs) -> Tensor:
        """Sample observations with Gaussian noise."""
        noise = self.noise_std * torch.randn_like(states)
        return states + noise

    def extra_repr(self) -> str:
        return f"noise_std={self.noise_std.item():.4f}"
