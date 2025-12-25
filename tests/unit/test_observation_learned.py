"""Unit tests for learned observation models (nn/observation/learned.py).

Tests for:
- LearnedMLPObservationModel
- EnergyBasedObservationModel
- AttentionObservationModel
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.observation.learned import (
    LearnedMLPObservationModel,
    EnergyBasedObservationModel,
    AttentionObservationModel,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def obs_size() -> int:
    """Observation dimension."""
    return 10


@pytest.fixture
def random_observations(batch_size: int, obs_size: int) -> Tensor:
    """Random observation tensor [batch, obs_size]."""
    return torch.randn(batch_size, obs_size)


# =============================================================================
# Tests for LearnedMLPObservationModel
# =============================================================================

class TestLearnedMLPObservationModel:
    """Tests for the LearnedMLPObservationModel class."""

    def test_output_shape(self, batch_size, n_particles, hidden_size, obs_size):
        """Log-likelihood has shape [batch, K]."""
        model = LearnedMLPObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_initial_output_near_zero(self, batch_size, n_particles, hidden_size, obs_size):
        """Initial log-likelihood near 0 (likelihood ~ 1) due to zero init."""
        model = LearnedMLPObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        # With zero-initialized output layer, should be near 0
        assert torch.allclose(log_lik, torch.zeros_like(log_lik), atol=0.1)

    def test_custom_hidden_sizes(self, batch_size, n_particles, hidden_size, obs_size):
        """Custom MLP hidden sizes work."""
        model = LearnedMLPObservationModel(
            hidden_size, obs_size, mlp_hidden_sizes=[128, 64, 32]
        )
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_different_activations(self, batch_size, n_particles, hidden_size, obs_size):
        """Different activation functions work."""
        for activation in ["tanh", "relu", "gelu", "silu"]:
            model = LearnedMLPObservationModel(
                hidden_size, obs_size, activation=activation
            )
            states = torch.randn(batch_size, n_particles, hidden_size)
            obs = torch.randn(batch_size, obs_size)

            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (batch_size, n_particles)

    def test_output_scale(self, batch_size, n_particles, hidden_size, obs_size):
        """Output scale affects log-likelihood magnitude."""
        model_scale1 = LearnedMLPObservationModel(
            hidden_size, obs_size, output_scale=1.0
        )
        model_scale2 = LearnedMLPObservationModel(
            hidden_size, obs_size, output_scale=2.0
        )

        # Copy MLP weights
        model_scale2.mlp.load_state_dict(model_scale1.mlp.state_dict())

        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik1 = model_scale1.log_likelihood(states, obs)
        log_lik2 = model_scale2.log_likelihood(states, obs)

        # Scale 2 should have 2x the magnitude (approximately)
        assert torch.allclose(log_lik2, log_lik1 * 2, atol=1e-5)

    def test_dropout(self, batch_size, n_particles, hidden_size, obs_size):
        """Dropout affects training but not eval."""
        model = LearnedMLPObservationModel(hidden_size, obs_size, dropout=0.5)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        model.eval()
        log_lik_eval1 = model.log_likelihood(states, obs)
        log_lik_eval2 = model.log_likelihood(states, obs)

        # Eval mode should be deterministic
        assert torch.allclose(log_lik_eval1, log_lik_eval2)

    def test_gradient_flow(self, hidden_size, obs_size):
        """Gradients flow through MLP."""
        model = LearnedMLPObservationModel(hidden_size, obs_size)
        states = torch.randn(4, 8, hidden_size, requires_grad=True)
        obs = torch.randn(4, obs_size)

        log_lik = model.log_likelihood(states, obs)
        loss = -log_lik.mean()
        loss.backward()

        assert states.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, obs_size):
        """No NaN or Inf in outputs."""
        model = LearnedMLPObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()


# =============================================================================
# Tests for EnergyBasedObservationModel
# =============================================================================

class TestEnergyBasedObservationModel:
    """Tests for the EnergyBasedObservationModel class."""

    def test_output_shape(self, batch_size, n_particles, hidden_size, obs_size):
        """Log-likelihood has shape [batch, K]."""
        model = EnergyBasedObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_energy_shape(self, batch_size, n_particles, hidden_size, obs_size):
        """Energy has shape [batch, K]."""
        model = EnergyBasedObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        energy = model.energy(states, obs)
        assert energy.shape == (batch_size, n_particles)

    def test_log_likelihood_is_negative_energy(self, batch_size, n_particles, hidden_size, obs_size):
        """Log-likelihood = -energy / temperature."""
        model = EnergyBasedObservationModel(hidden_size, obs_size, temperature=2.0)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        energy = model.energy(states, obs)
        log_lik = model.log_likelihood(states, obs)

        assert torch.allclose(log_lik, -energy / model.temperature, atol=1e-5)

    def test_temperature_property(self, hidden_size, obs_size):
        """Temperature property returns correct value."""
        model = EnergyBasedObservationModel(hidden_size, obs_size, temperature=2.5)
        assert torch.isclose(model.temperature, torch.tensor(2.5))

    def test_learnable_temperature(self, hidden_size, obs_size):
        """Learnable temperature has gradient."""
        model = EnergyBasedObservationModel(
            hidden_size, obs_size, temperature=1.0, learnable_temperature=True
        )
        states = torch.randn(4, 8, hidden_size)
        obs = torch.randn(4, obs_size)

        log_lik = model.log_likelihood(states, obs)
        loss = -log_lik.mean()
        loss.backward()

        assert model.log_temperature.grad is not None

    def test_temperature_effect(self, batch_size, n_particles, hidden_size, obs_size):
        """Higher temperature = smaller magnitude log-likelihoods."""
        model_low_temp = EnergyBasedObservationModel(
            hidden_size, obs_size, temperature=0.5
        )
        model_high_temp = EnergyBasedObservationModel(
            hidden_size, obs_size, temperature=5.0
        )

        # Copy weights
        model_high_temp.state_encoder.load_state_dict(model_low_temp.state_encoder.state_dict())
        model_high_temp.obs_encoder.load_state_dict(model_low_temp.obs_encoder.state_dict())
        model_high_temp.energy_net.load_state_dict(model_low_temp.energy_net.state_dict())

        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik_low = model_low_temp.log_likelihood(states, obs)
        log_lik_high = model_high_temp.log_likelihood(states, obs)

        # High temp should have smaller magnitude
        assert log_lik_low.abs().mean() > log_lik_high.abs().mean()

    def test_custom_hidden_sizes(self, batch_size, n_particles, hidden_size, obs_size):
        """Custom energy hidden sizes work."""
        model = EnergyBasedObservationModel(
            hidden_size, obs_size, energy_hidden_sizes=[256, 128, 64]
        )
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_different_activations(self, batch_size, n_particles, hidden_size, obs_size):
        """Different activation functions work."""
        for activation in ["tanh", "relu", "gelu"]:
            model = EnergyBasedObservationModel(
                hidden_size, obs_size, activation=activation
            )
            states = torch.randn(batch_size, n_particles, hidden_size)
            obs = torch.randn(batch_size, obs_size)

            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (batch_size, n_particles)

    def test_gradient_flow(self, hidden_size, obs_size):
        """Gradients flow through energy network."""
        model = EnergyBasedObservationModel(hidden_size, obs_size)
        states = torch.randn(4, 8, hidden_size, requires_grad=True)
        obs = torch.randn(4, obs_size)

        log_lik = model.log_likelihood(states, obs)
        loss = -log_lik.mean()
        loss.backward()

        assert states.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, obs_size):
        """No NaN or Inf in outputs."""
        model = EnergyBasedObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        energy = model.energy(states, obs)
        log_lik = model.log_likelihood(states, obs)

        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()
        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()


# =============================================================================
# Tests for AttentionObservationModel
# =============================================================================

class TestAttentionObservationModel:
    """Tests for the AttentionObservationModel class."""

    def test_output_shape(self, batch_size, n_particles, hidden_size, obs_size):
        """Log-likelihood has shape [batch, K]."""
        model = AttentionObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_different_n_heads(self, batch_size, n_particles, obs_size):
        """Different number of attention heads work."""
        # Hidden size must be divisible by n_heads
        for n_heads in [1, 2, 4, 8]:
            hidden_size = 64  # Divisible by all
            model = AttentionObservationModel(hidden_size, obs_size, n_heads=n_heads)
            states = torch.randn(batch_size, n_particles, hidden_size)
            obs = torch.randn(batch_size, obs_size)

            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (batch_size, n_particles)

    def test_dropout(self, batch_size, n_particles, hidden_size, obs_size):
        """Dropout affects training but not eval."""
        model = AttentionObservationModel(hidden_size, obs_size, dropout=0.5)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        model.eval()
        log_lik_eval1 = model.log_likelihood(states, obs)
        log_lik_eval2 = model.log_likelihood(states, obs)

        # Eval mode should be deterministic
        assert torch.allclose(log_lik_eval1, log_lik_eval2)

    def test_gradient_flow(self, hidden_size, obs_size):
        """Gradients flow through attention mechanism."""
        model = AttentionObservationModel(hidden_size, obs_size)
        states = torch.randn(4, 8, hidden_size, requires_grad=True)
        obs = torch.randn(4, obs_size)

        log_lik = model.log_likelihood(states, obs)
        loss = -log_lik.mean()
        loss.backward()

        assert states.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_obs_proj_dimension(self, batch_size, n_particles, hidden_size, obs_size):
        """Observation is projected to hidden_size."""
        model = AttentionObservationModel(hidden_size, obs_size)

        # Check projection dimensions
        assert model.obs_proj.in_features == obs_size
        assert model.obs_proj.out_features == hidden_size

    def test_state_proj_dimension(self, batch_size, n_particles, hidden_size, obs_size):
        """State projection maintains dimension."""
        model = AttentionObservationModel(hidden_size, obs_size)

        assert model.state_proj.in_features == hidden_size
        assert model.state_proj.out_features == hidden_size

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, obs_size):
        """No NaN or Inf in outputs."""
        model = AttentionObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestLearnedObservationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self, batch_size, hidden_size, obs_size):
        """K=1 works for all models."""
        models = [
            LearnedMLPObservationModel(hidden_size, obs_size),
            EnergyBasedObservationModel(hidden_size, obs_size),
            AttentionObservationModel(hidden_size, obs_size),
        ]
        states = torch.randn(batch_size, 1, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        for model in models:
            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (batch_size, 1)

    def test_single_batch(self, n_particles, hidden_size, obs_size):
        """Batch size 1 works for all models."""
        models = [
            LearnedMLPObservationModel(hidden_size, obs_size),
            EnergyBasedObservationModel(hidden_size, obs_size),
            AttentionObservationModel(hidden_size, obs_size),
        ]
        states = torch.randn(1, n_particles, hidden_size)
        obs = torch.randn(1, obs_size)

        for model in models:
            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (1, n_particles)

    def test_single_obs_dim(self, batch_size, n_particles, hidden_size):
        """obs_size=1 works for all models."""
        obs_size = 1
        models = [
            LearnedMLPObservationModel(hidden_size, obs_size),
            EnergyBasedObservationModel(hidden_size, obs_size),
            AttentionObservationModel(hidden_size, obs_size),
        ]
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        for model in models:
            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (batch_size, n_particles)

    def test_large_obs_dim(self, batch_size, n_particles, hidden_size):
        """Large observation dimension works."""
        obs_size = 256
        models = [
            LearnedMLPObservationModel(hidden_size, obs_size),
            EnergyBasedObservationModel(hidden_size, obs_size),
            AttentionObservationModel(hidden_size, obs_size),
        ]
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        for model in models:
            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (batch_size, n_particles)

    def test_extreme_state_values(self, batch_size, n_particles, hidden_size, obs_size):
        """Extreme state values handled gracefully."""
        models = [
            LearnedMLPObservationModel(hidden_size, obs_size),
            EnergyBasedObservationModel(hidden_size, obs_size),
            AttentionObservationModel(hidden_size, obs_size),
        ]
        states = torch.randn(batch_size, n_particles, hidden_size) * 100
        obs = torch.randn(batch_size, obs_size)

        for model in models:
            log_lik = model.log_likelihood(states, obs)
            assert not torch.isnan(log_lik).any()

    def test_extreme_obs_values(self, batch_size, n_particles, hidden_size, obs_size):
        """Extreme observation values handled gracefully."""
        models = [
            LearnedMLPObservationModel(hidden_size, obs_size),
            EnergyBasedObservationModel(hidden_size, obs_size),
            AttentionObservationModel(hidden_size, obs_size),
        ]
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size) * 100

        for model in models:
            log_lik = model.log_likelihood(states, obs)
            assert not torch.isnan(log_lik).any()

    def test_empty_hidden_layers(self, batch_size, n_particles, hidden_size, obs_size):
        """Empty hidden layers list works for LearnedMLPObservationModel."""
        model = LearnedMLPObservationModel(hidden_size, obs_size, mlp_hidden_sizes=[])
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_single_head_attention(self, batch_size, n_particles, hidden_size, obs_size):
        """Single attention head works."""
        model = AttentionObservationModel(hidden_size, obs_size, n_heads=1)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_matching_obs_state_dims(self, batch_size, n_particles):
        """Same observation and hidden dimensions work."""
        dim = 64
        models = [
            LearnedMLPObservationModel(dim, dim),
            EnergyBasedObservationModel(dim, dim),
            AttentionObservationModel(dim, dim),
        ]
        states = torch.randn(batch_size, n_particles, dim)
        obs = torch.randn(batch_size, dim)

        for model in models:
            log_lik = model.log_likelihood(states, obs)
            assert log_lik.shape == (batch_size, n_particles)
