"""Unit tests for Gaussian observation models (nn/observation/gaussian.py).

Tests for:
- GaussianObservationModel
- HeteroscedasticGaussianObservationModel
"""

import math
import pytest
import torch
from torch import Tensor

from pfncps.nn.observation.gaussian import (
    GaussianObservationModel,
    HeteroscedasticGaussianObservationModel,
)


# =============================================================================
# Tests for GaussianObservationModel
# =============================================================================

class TestGaussianObservationModel:
    """Tests for the GaussianObservationModel class."""

    # === Shape Tests ===

    def test_log_likelihood_output_shape(self, batch_size, n_particles, hidden_size):
        """Output [batch, K] for input [batch, K, hidden]."""
        obs_size = 10
        model = GaussianObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_predict_output_shape(self, batch_size, n_particles, hidden_size):
        """Predict output [batch, K, obs_size]."""
        obs_size = 10
        model = GaussianObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)

        predictions = model.predict(states)
        assert predictions.shape == (batch_size, n_particles, obs_size)

    def test_sample_output_shape(self, batch_size, n_particles, hidden_size):
        """Sample output [batch, K, obs_size]."""
        obs_size = 10
        model = GaussianObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)

        samples = model.sample(states)
        assert samples.shape == (batch_size, n_particles, obs_size)

    # === Mathematical Properties ===

    def test_log_likelihood_is_negative_or_small(self, hidden_size_fixed, obs_size_fixed):
        """Log likelihood should typically be negative or small positive."""
        model = GaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 16, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        # Gaussian log-likelihood can be slightly positive for well-matched predictions
        # but should not be extremely large
        assert torch.all(log_lik < 100)

    def test_closer_prediction_higher_likelihood(self, hidden_size_fixed, obs_size_fixed):
        """Predictions closer to obs have higher log-likelihood."""
        model = GaussianObservationModel(hidden_size_fixed, obs_size_fixed, obs_noise_std=0.1)
        states = torch.randn(1, 10, hidden_size_fixed)

        with torch.no_grad():
            predictions = model.predict(states)[:, 0, :]  # [1, obs_size]

        # Perfect match should have higher likelihood than random
        log_lik_perfect = model.log_likelihood(states, predictions)[:, 0]
        log_lik_random = model.log_likelihood(states, torch.randn(1, obs_size_fixed))[:, 0]

        assert log_lik_perfect > log_lik_random

    def test_lower_noise_sharper_likelihood(self, hidden_size_fixed, obs_size_fixed):
        """Lower noise std -> sharper (more peaked) likelihood."""
        states = torch.randn(4, 16, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed)

        model_low = GaussianObservationModel(hidden_size_fixed, obs_size_fixed, obs_noise_std=0.1)
        model_high = GaussianObservationModel(hidden_size_fixed, obs_size_fixed, obs_noise_std=1.0)

        log_lik_low = model_low.log_likelihood(states, obs)
        log_lik_high = model_high.log_likelihood(states, obs)

        # Low noise: larger range of log-likelihoods (more discriminating)
        range_low = log_lik_low.max() - log_lik_low.min()
        range_high = log_lik_high.max() - log_lik_high.min()

        assert range_low > range_high

    # === Covariance Types ===

    def test_diagonal_covariance_per_dimension(self, hidden_size_fixed, obs_size_fixed):
        """With diagonal_covariance=True, each dim has its own variance."""
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed,
            obs_noise_std=1.0, diagonal_covariance=True
        )
        # noise_std should have shape (obs_size,)
        assert model.noise_std.shape == (obs_size_fixed,)

    def test_scalar_covariance_shared(self, hidden_size_fixed, obs_size_fixed):
        """With diagonal_covariance=False, single shared variance."""
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed,
            obs_noise_std=1.0, diagonal_covariance=False
        )
        # noise_std should be a scalar
        assert model.noise_std.shape == ()

    # === Learnable Parameters ===

    def test_learnable_noise_is_parameter(self, hidden_size_fixed, obs_size_fixed):
        """Noise std is learnable when specified."""
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed, learnable_noise=True
        )
        param_names = [name for name, _ in model.named_parameters()]
        assert "log_noise_std" in param_names

    def test_fixed_noise_is_buffer(self, hidden_size_fixed, obs_size_fixed):
        """Noise std is buffer when not learnable."""
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed, learnable_noise=False
        )
        param_names = [name for name, _ in model.named_parameters()]
        assert "log_noise_std" not in param_names
        buffer_names = [name for name, _ in model.named_buffers()]
        assert "log_noise_std" in buffer_names

    def test_learnable_noise_gradient(self, hidden_size_fixed, obs_size_fixed):
        """Gradient flows to learnable noise."""
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed, learnable_noise=True
        )
        states = torch.randn(4, 8, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        (-log_lik.mean()).backward()

        assert model.log_noise_std.grad is not None

    def test_projection_gradient(self, hidden_size_fixed, obs_size_fixed):
        """Gradient flows through linear projection."""
        model = GaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 8, hidden_size_fixed, requires_grad=True)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        (-log_lik.mean()).backward()

        assert model.projection.weight.grad is not None

    # === Sampling ===

    def test_sample_mean_near_prediction(self, hidden_size_fixed, obs_size_fixed):
        """Samples have mean approximately equal to prediction."""
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed, obs_noise_std=0.1
        )
        states = torch.randn(100, 32, hidden_size_fixed)

        # Average over many samples
        n_samples = 100
        samples = []
        with torch.no_grad():
            predictions = model.predict(states)
            for _ in range(n_samples):
                samples.append(model.sample(states))

        sample_mean = torch.stack(samples).mean(dim=0)
        assert torch.allclose(sample_mean, predictions, atol=0.1)

    def test_sample_variance_matches_noise(self, hidden_size_fixed, obs_size_fixed):
        """Sample variance approximately equals noise variance."""
        noise_std = 0.5
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed, obs_noise_std=noise_std
        )
        states = torch.randn(1, 1, hidden_size_fixed)

        # Many samples from same state
        samples = []
        with torch.no_grad():
            for _ in range(1000):
                samples.append(model.sample(states)[0, 0, :])

        sample_var = torch.stack(samples).var(dim=0).mean()
        expected_var = noise_std ** 2
        assert abs(sample_var - expected_var) < 0.1

    # === No NaN/Inf ===

    def test_no_nan_inf_in_log_likelihood(self, hidden_size_fixed, obs_size_fixed):
        """Log likelihood should not contain NaN or Inf."""
        model = GaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 16, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()


# =============================================================================
# Tests for HeteroscedasticGaussianObservationModel
# =============================================================================

class TestHeteroscedasticGaussianObservationModel:
    """Tests for the HeteroscedasticGaussianObservationModel class."""

    def test_noise_varies_with_state(self, hidden_size_fixed, obs_size_fixed):
        """Different states should produce different noise scales after MLP weights are non-zero."""
        model = HeteroscedasticGaussianObservationModel(hidden_size_fixed, obs_size_fixed)

        # Re-initialize weights with non-zero values so state-dependence works
        # (default init sets last layer weights to zero for consistent initial output)
        for module in model.noise_net.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)

        states1 = torch.zeros(4, 16, hidden_size_fixed)
        states2 = torch.ones(4, 16, hidden_size_fixed) * 5

        noise1 = model.get_noise_std(states1)
        noise2 = model.get_noise_std(states2)

        # Should not be identical with non-zero MLP weights
        assert not torch.allclose(noise1, noise2)

    def test_noise_always_positive(self, hidden_size_fixed, obs_size_fixed):
        """get_noise_std() always > 0."""
        model = HeteroscedasticGaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 16, hidden_size_fixed) * 100  # Extreme values

        noise_std = model.get_noise_std(states)
        assert torch.all(noise_std > 0)

    def test_noise_clamped_to_range(self, hidden_size_fixed, obs_size_fixed):
        """Noise std in [min_noise_std, max_noise_std]."""
        model = HeteroscedasticGaussianObservationModel(
            hidden_size_fixed, obs_size_fixed,
            min_noise_std=0.01, max_noise_std=5.0
        )
        states = torch.randn(4, 16, hidden_size_fixed) * 100

        noise = model.get_noise_std(states)
        assert torch.all(noise >= 0.01)
        assert torch.all(noise <= 5.0)

    def test_log_likelihood_shape(self, batch_size, n_particles, hidden_size):
        """Log likelihood has shape [batch, K]."""
        obs_size = 10
        model = HeteroscedasticGaussianObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)
        obs = torch.randn(batch_size, obs_size)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (batch_size, n_particles)

    def test_predict_shape(self, batch_size, n_particles, hidden_size):
        """Predictions have shape [batch, K, obs_size]."""
        obs_size = 10
        model = HeteroscedasticGaussianObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)

        predictions = model.predict(states)
        assert predictions.shape == (batch_size, n_particles, obs_size)

    def test_sample_shape(self, batch_size, n_particles, hidden_size):
        """Samples have shape [batch, K, obs_size]."""
        obs_size = 10
        model = HeteroscedasticGaussianObservationModel(hidden_size, obs_size)
        states = torch.randn(batch_size, n_particles, hidden_size)

        samples = model.sample(states)
        assert samples.shape == (batch_size, n_particles, obs_size)

    def test_gradient_to_noise_network(self, hidden_size_fixed, obs_size_fixed):
        """Backprop updates noise prediction network."""
        model = HeteroscedasticGaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 8, hidden_size_fixed, requires_grad=True)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        (-log_lik.mean()).backward()

        # Check noise net has gradients
        has_grad = False
        for name, param in model.noise_net.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_gradient_to_mean_network(self, hidden_size_fixed, obs_size_fixed):
        """Backprop updates mean prediction network."""
        model = HeteroscedasticGaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 8, hidden_size_fixed, requires_grad=True)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        (-log_lik.mean()).backward()

        assert model.mean_net.weight.grad is not None

    def test_no_nan_inf(self, hidden_size_fixed, obs_size_fixed):
        """Log likelihood should not contain NaN or Inf."""
        model = HeteroscedasticGaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 16, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestGaussianObservationEdgeCases:
    """Test edge cases for Gaussian observation models."""

    def test_single_particle(self, hidden_size_fixed, obs_size_fixed):
        """K=1 should work."""
        model = GaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 1, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (4, 1)

    def test_single_batch(self, hidden_size_fixed, obs_size_fixed):
        """Batch size 1 should work."""
        model = GaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(1, 16, hidden_size_fixed)
        obs = torch.randn(1, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (1, 16)

    def test_large_observations(self, hidden_size_fixed, obs_size_fixed):
        """Large observation values should work."""
        model = GaussianObservationModel(hidden_size_fixed, obs_size_fixed)
        states = torch.randn(4, 16, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed) * 100

        log_lik = model.log_likelihood(states, obs)
        assert not torch.isnan(log_lik).any()

    def test_min_noise_std_respected(self, hidden_size_fixed, obs_size_fixed):
        """Minimum noise std is respected."""
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed,
            obs_noise_std=1e-10, min_noise_std=0.001
        )
        assert torch.all(model.noise_std >= 0.001)

    def test_tensor_noise_std_init(self, hidden_size_fixed, obs_size_fixed):
        """Can initialize with tensor noise std."""
        noise_std = torch.rand(obs_size_fixed) * 0.5 + 0.1
        model = GaussianObservationModel(
            hidden_size_fixed, obs_size_fixed,
            obs_noise_std=noise_std, diagonal_covariance=True
        )
        assert model.noise_std.shape == (obs_size_fixed,)

    def test_heteroscedastic_custom_hidden(self, hidden_size_fixed, obs_size_fixed):
        """Heteroscedastic model with custom noise hidden size."""
        model = HeteroscedasticGaussianObservationModel(
            hidden_size_fixed, obs_size_fixed, noise_hidden_size=64
        )
        states = torch.randn(4, 16, hidden_size_fixed)
        obs = torch.randn(4, obs_size_fixed)

        log_lik = model.log_likelihood(states, obs)
        assert log_lik.shape == (4, 16)
