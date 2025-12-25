"""Learned MLP observation model for particle filters.

Provides a flexible MLP-based observation model that learns to compute
log-likelihoods directly from state-observation pairs.
"""

from typing import Optional, List, Union
import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import ObservationModel


class LearnedMLPObservationModel(ObservationModel):
    """MLP-based observation model that learns log-likelihood directly.

    Instead of assuming a parametric form for p(y | h), this model
    learns to predict log p(y | h) directly using an MLP that takes
    concatenated [h, y] as input.

    This is useful when the observation model is complex or unknown.

    Note: This model outputs unnormalized log-likelihoods. For valid
    probability interpretation, ensure proper regularization during training.
    """

    def __init__(
        self,
        hidden_size: int,
        obs_size: int,
        mlp_hidden_sizes: Optional[List[int]] = None,
        activation: str = "tanh",
        dropout: float = 0.0,
        output_scale: float = 1.0,
    ):
        """Initialize learned MLP observation model.

        Args:
            hidden_size: Dimension of hidden states
            obs_size: Dimension of observations
            mlp_hidden_sizes: List of hidden layer sizes (default: [64, 32])
            activation: Activation function ('tanh', 'relu', 'gelu')
            dropout: Dropout probability
            output_scale: Scale factor for output log-likelihoods
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.obs_size = obs_size
        self.output_scale = output_scale

        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [64, 32]

        # Activation function
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        act_fn = activations.get(activation, nn.Tanh)

        # Build MLP: input is [state, observation] concatenated
        input_size = hidden_size + obs_size
        layers = []

        in_dim = input_size
        for hidden_dim in mlp_hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer: single log-likelihood value
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize output layer for reasonable initial log-likelihoods
        self._init_output_layer()

    def _init_output_layer(self):
        """Initialize output layer for stable initial log-likelihoods."""
        with torch.no_grad():
            # Find the last linear layer
            for module in reversed(list(self.mlp.modules())):
                if isinstance(module, nn.Linear):
                    # Initialize to output values near 0 (log-lik = 0 means lik = 1)
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
                    break

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute learned log-likelihood.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observations [batch, obs_size]

        Returns:
            log_likelihoods: [batch, K]
        """
        batch, K, H = states.shape

        # Expand observations to match particles
        obs_expanded = observations.unsqueeze(1).expand(-1, K, -1)  # [batch, K, obs_size]

        # Concatenate state and observation
        combined = torch.cat([states, obs_expanded], dim=-1)  # [batch, K, H + obs_size]

        # Flatten for MLP
        combined_flat = combined.reshape(-1, H + self.obs_size)

        # Forward through MLP
        log_lik_flat = self.mlp(combined_flat)  # [batch * K, 1]
        log_lik = log_lik_flat.reshape(batch, K)

        # Apply output scaling
        log_lik = log_lik * self.output_scale

        return log_lik

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, obs_size={self.obs_size}"


class EnergyBasedObservationModel(ObservationModel):
    """Energy-based observation model.

    Models log p(y | h) = -E(h, y) where E is an energy function.
    The energy function is learned via an MLP.

    This formulation is more principled than directly predicting
    log-likelihoods, as the energy can be interpreted as a
    compatibility score.
    """

    def __init__(
        self,
        hidden_size: int,
        obs_size: int,
        energy_hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        temperature: float = 1.0,
        learnable_temperature: bool = False,
    ):
        """Initialize energy-based observation model.

        Args:
            hidden_size: Dimension of hidden states
            obs_size: Dimension of observations
            energy_hidden_sizes: Hidden sizes for energy network
            activation: Activation function
            temperature: Temperature for energy scaling
            learnable_temperature: If True, temperature is learnable
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.obs_size = obs_size

        if energy_hidden_sizes is None:
            energy_hidden_sizes = [128, 64]

        # Activation
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        act_fn = activations.get(activation, nn.ReLU)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_size, energy_hidden_sizes[0]),
            act_fn(),
        )

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_size, energy_hidden_sizes[0]),
            act_fn(),
        )

        # Joint energy network (takes concatenated encodings)
        layers = []
        in_dim = 2 * energy_hidden_sizes[0]  # Concatenated encodings
        for hidden_dim in energy_hidden_sizes[1:]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.energy_net = nn.Sequential(*layers)

        # Temperature
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(temperature).log()
            )

    @property
    def temperature(self) -> Tensor:
        """Get current temperature."""
        return torch.exp(self.log_temperature)

    def energy(self, states: Tensor, observations: Tensor) -> Tensor:
        """Compute energy E(h, y).

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observations [batch, obs_size]

        Returns:
            energy: Energy values [batch, K]
        """
        batch, K, H = states.shape

        # Encode states
        states_flat = states.reshape(-1, H)
        state_enc_flat = self.state_encoder(states_flat)  # [batch * K, enc_dim]
        state_enc = state_enc_flat.reshape(batch, K, -1)

        # Encode observations (shared across particles)
        obs_enc = self.obs_encoder(observations)  # [batch, enc_dim]
        obs_enc_expanded = obs_enc.unsqueeze(1).expand(-1, K, -1)  # [batch, K, enc_dim]

        # Concatenate and compute energy
        combined = torch.cat([state_enc, obs_enc_expanded], dim=-1)
        combined_flat = combined.reshape(-1, combined.shape[-1])
        energy_flat = self.energy_net(combined_flat)
        energy = energy_flat.reshape(batch, K)

        return energy

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute log-likelihood as negative scaled energy.

        log p(y | h) = -E(h, y) / T

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observations [batch, obs_size]

        Returns:
            log_likelihoods: [batch, K]
        """
        energy = self.energy(states, observations)
        return -energy / self.temperature

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, obs_size={self.obs_size}, "
            f"temperature={self.temperature.item():.4f}"
        )


class AttentionObservationModel(ObservationModel):
    """Attention-based observation model.

    Uses cross-attention between state and observation to compute
    a compatibility score that serves as log-likelihood.

    Useful when observations have structure (e.g., sequences) that
    should attend to different aspects of the hidden state.
    """

    def __init__(
        self,
        hidden_size: int,
        obs_size: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        """Initialize attention observation model.

        Args:
            hidden_size: Dimension of hidden states
            obs_size: Dimension of observations
            n_heads: Number of attention heads
            dropout: Attention dropout
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.obs_size = obs_size

        # Project state and observation to same dimension
        self.state_proj = nn.Linear(hidden_size, hidden_size)
        self.obs_proj = nn.Linear(obs_size, hidden_size)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection to log-likelihood
        self.output_proj = nn.Linear(hidden_size, 1)

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute attention-based log-likelihood.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Observations [batch, obs_size]

        Returns:
            log_likelihoods: [batch, K]
        """
        batch, K, H = states.shape

        # Project states and observations
        # States: [batch, K, hidden_size]
        state_proj = self.state_proj(states)

        # Observations: [batch, obs_size] -> [batch, 1, hidden_size]
        obs_proj = self.obs_proj(observations).unsqueeze(1)

        # For each particle, compute attention from obs query to state key/value
        # Reshape for batch attention: [batch * K, 1, hidden_size] for query
        # Need to expand obs_proj to match particles
        obs_proj_expanded = obs_proj.expand(-1, K, -1)  # [batch, K, hidden_size]

        # Flatten for attention
        # Query: observation encoding, Key/Value: state encoding
        # We want each particle to independently attend
        state_flat = state_proj.reshape(batch * K, 1, H)  # [batch*K, 1, H]
        obs_flat = obs_proj_expanded.reshape(batch * K, 1, H)  # [batch*K, 1, H]

        # Cross-attention: obs attends to state
        attn_out, _ = self.attention(
            query=obs_flat,
            key=state_flat,
            value=state_flat,
        )  # [batch*K, 1, H]

        # Project to log-likelihood
        log_lik_flat = self.output_proj(attn_out.squeeze(1))  # [batch*K, 1]
        log_lik = log_lik_flat.reshape(batch, K)

        return log_lik

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, obs_size={self.obs_size}"
