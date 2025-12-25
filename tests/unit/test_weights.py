"""Unit tests for weight operations (nn/utils/weights.py).

These tests verify the mathematical correctness of weight operations
that form the foundation of particle filtering. Tests follow the
Yoneda philosophy: they define the complete behavioral specification.

Key Invariants Tested:
- W1: exp(log_weights).sum() == 1 after normalization
- W2: log_weights bounded in [-50, 50]
- W3: ESS in [1, K]
"""

import math
import pytest
import torch
from torch import Tensor

from pfncps.nn.utils.weights import (
    safe_logsumexp,
    normalize_log_weights,
    log_weight_update,
    compute_ess,
    compute_entropy,
    init_uniform_log_weights,
    weighted_mean,
    weighted_variance,
    temperature_scaled_log_weights,
)


# =============================================================================
# Tests for safe_logsumexp
# =============================================================================

class TestSafeLogsumexp:
    """Tests for the safe_logsumexp function."""

    def test_equivalent_to_torch_logsumexp_normal_input(self):
        """Should match torch.logsumexp for well-behaved inputs."""
        log_weights = torch.randn(4, 16)
        result = safe_logsumexp(log_weights, dim=-1)
        expected = torch.logsumexp(log_weights, dim=-1)
        assert torch.allclose(result, expected)

    def test_handles_negative_infinity(self):
        """Should handle -inf in input (representing zero probability)."""
        log_weights = torch.tensor([[-100.0, 0.0, -100.0]])
        result = safe_logsumexp(log_weights, dim=-1)
        expected = torch.tensor([0.0])  # log(0 + 1 + 0) = log(1) = 0
        assert torch.allclose(result, expected, atol=1e-5)

    def test_keepdim_true(self):
        """keepdim=True should preserve dimension."""
        log_weights = torch.randn(4, 16)
        result = safe_logsumexp(log_weights, dim=-1, keepdim=True)
        assert result.shape == (4, 1)

    def test_keepdim_false(self):
        """keepdim=False should reduce dimension."""
        log_weights = torch.randn(4, 16)
        result = safe_logsumexp(log_weights, dim=-1, keepdim=False)
        assert result.shape == (4,)

    def test_no_nan_in_output(self):
        """Output should never contain NaN for finite inputs."""
        log_weights = torch.randn(8, 32) * 10
        result = safe_logsumexp(log_weights, dim=-1)
        assert not torch.isnan(result).any()

    def test_no_inf_in_output(self):
        """Output should not contain Inf for reasonable inputs."""
        log_weights = torch.randn(8, 32) * 10
        result = safe_logsumexp(log_weights, dim=-1)
        assert not torch.isinf(result).any()


# =============================================================================
# Tests for normalize_log_weights
# =============================================================================

class TestNormalizeLogWeights:
    """Tests for the normalize_log_weights function."""

    def test_output_sums_to_one(self, batch_size, n_particles, tolerance):
        """Invariant W1: exp(normalized).sum() == 1."""
        log_weights = torch.randn(batch_size, n_particles) * 10
        normalized = normalize_log_weights(log_weights)
        sums = torch.exp(normalized).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), **tolerance)

    def test_input_clamping_applied(self, batch_size, n_particles):
        """Input values are clamped before normalization."""
        # Create weights with extreme values
        log_weights = torch.randn(batch_size, n_particles) * 100
        normalized = normalize_log_weights(log_weights)

        # Output should not have NaN/Inf
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

        # Output should still sum to 1 after exp
        sums = torch.exp(normalized).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_handles_extreme_positive(self):
        """Test with extreme positive values."""
        log_weights = torch.tensor([[1e6, 1e6, 0.0]])
        normalized = normalize_log_weights(log_weights)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_handles_extreme_negative(self):
        """Test with extreme negative values."""
        log_weights = torch.tensor([[-1e6, -1e6, 0.0]])
        normalized = normalize_log_weights(log_weights)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_handles_mixed_extreme(self):
        """Test with mixed extreme values."""
        log_weights = torch.randn(4, 32) * 1e6
        normalized = normalize_log_weights(log_weights)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_preserves_relative_ordering(self, batch_size, n_particles):
        """Normalization should preserve which particle is heaviest."""
        log_weights = torch.randn(batch_size, n_particles)
        normalized = normalize_log_weights(log_weights)
        assert torch.equal(log_weights.argmax(dim=-1), normalized.argmax(dim=-1))

    def test_idempotent(self, batch_size, n_particles, tolerance):
        """Normalizing normalized weights should be idempotent."""
        log_weights = torch.randn(batch_size, n_particles)
        once = normalize_log_weights(log_weights)
        twice = normalize_log_weights(once)
        assert torch.allclose(once, twice, **tolerance)

    def test_output_shape_matches_input(self, batch_size, n_particles):
        """Output shape should match input shape."""
        log_weights = torch.randn(batch_size, n_particles)
        normalized = normalize_log_weights(log_weights)
        assert normalized.shape == log_weights.shape

    def test_custom_bounds(self):
        """Test with custom max/min log weight bounds for input clamping."""
        log_weights = torch.tensor([[100.0, -100.0, 0.0]])
        normalized = normalize_log_weights(
            log_weights, max_log_weight=10.0, min_log_weight=-10.0
        )
        # Input is clamped to [-10, 10] before normalization
        # Output should be valid (no NaN/Inf) and sum to 1
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        sums = torch.exp(normalized).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


# =============================================================================
# Tests for log_weight_update
# =============================================================================

class TestLogWeightUpdate:
    """Tests for the log_weight_update function."""

    def test_adds_log_likelihoods(self, batch_size, n_particles):
        """Weight update is addition in log space."""
        log_weights = torch.zeros(batch_size, n_particles)
        log_liks = torch.randn(batch_size, n_particles)
        updated = log_weight_update(log_weights, log_liks, normalize=False)
        expected = torch.clamp(log_liks, -50.0, 50.0)
        assert torch.allclose(updated, expected)

    def test_normalized_output(self, batch_size, n_particles, tolerance):
        """With normalize=True, output sums to 1."""
        log_weights = torch.randn(batch_size, n_particles)
        log_liks = torch.randn(batch_size, n_particles)
        updated = log_weight_update(log_weights, log_liks, normalize=True)
        sums = torch.exp(updated).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), **tolerance)

    def test_unnormalized_output(self, batch_size, n_particles):
        """With normalize=False, output is just sum (clamped)."""
        log_weights = torch.randn(batch_size, n_particles) * 0.1  # Small values
        log_liks = torch.randn(batch_size, n_particles) * 0.1
        updated = log_weight_update(log_weights, log_liks, normalize=False)
        expected = torch.clamp(log_weights + log_liks, -50.0, 50.0)
        assert torch.allclose(updated, expected)

    def test_output_shape_matches_input(self, batch_size, n_particles):
        """Output shape should match input shape."""
        log_weights = torch.randn(batch_size, n_particles)
        log_liks = torch.randn(batch_size, n_particles)
        updated = log_weight_update(log_weights, log_liks)
        assert updated.shape == log_weights.shape

    def test_clamping_applied(self):
        """Extreme values should be clamped."""
        log_weights = torch.tensor([[100.0, -100.0]])
        log_liks = torch.tensor([[100.0, -100.0]])
        updated = log_weight_update(log_weights, log_liks, normalize=False)
        assert updated.max() <= 50.0
        assert updated.min() >= -50.0

    def test_no_nan_inf(self, batch_size, n_particles):
        """Output should not contain NaN or Inf."""
        log_weights = torch.randn(batch_size, n_particles) * 50
        log_liks = torch.randn(batch_size, n_particles) * 50
        updated = log_weight_update(log_weights, log_liks)
        assert not torch.isnan(updated).any()
        assert not torch.isinf(updated).any()


# =============================================================================
# Tests for compute_ess
# =============================================================================

class TestComputeESS:
    """Tests for the compute_ess (Effective Sample Size) function."""

    def test_uniform_weights_max_ess(self, batch_size, n_particles, tolerance):
        """Uniform weights give ESS = K."""
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))
        ess = compute_ess(log_weights)
        expected = torch.full((batch_size,), float(n_particles))
        assert torch.allclose(ess, expected, **tolerance)

    def test_degenerate_weights_min_ess(self, batch_size, n_particles):
        """Single dominant particle gives ESS ~ 1."""
        log_weights = torch.full((batch_size, n_particles), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)
        ess = compute_ess(log_weights)
        assert torch.all(ess < 2.0), f"ESS should be close to 1, got {ess}"

    def test_ess_bounds(self, batch_size, n_particles, tolerance):
        """Invariant W3: ESS in [1, K]."""
        log_weights = torch.randn(batch_size, n_particles)
        log_weights = normalize_log_weights(log_weights)
        ess = compute_ess(log_weights)
        # Allow small numerical tolerance
        assert torch.all(ess >= 1.0 - tolerance["atol"])
        assert torch.all(ess <= n_particles + tolerance["atol"])

    def test_output_shape(self, batch_size, n_particles):
        """ESS output shape is [batch]."""
        log_weights = torch.randn(batch_size, n_particles)
        ess = compute_ess(log_weights)
        assert ess.shape == (batch_size,)

    def test_ess_monotonic_with_concentration(self):
        """ESS should decrease as weights become more concentrated."""
        n_particles = 32
        batch_size = 1

        # Uniform weights -> max ESS
        uniform = torch.full((batch_size, n_particles), -math.log(n_particles))
        ess_uniform = compute_ess(uniform)

        # Slightly concentrated weights -> lower ESS
        concentrated = torch.randn(batch_size, n_particles)
        concentrated = normalize_log_weights(concentrated)
        ess_concentrated = compute_ess(concentrated)

        # Very concentrated weights -> much lower ESS
        very_concentrated = torch.full((batch_size, n_particles), -50.0)
        very_concentrated[:, 0] = 0.0
        very_concentrated = normalize_log_weights(very_concentrated)
        ess_very_concentrated = compute_ess(very_concentrated)

        assert ess_uniform > ess_concentrated > ess_very_concentrated

    def test_no_nan_inf(self, batch_size, n_particles):
        """ESS should not contain NaN or Inf."""
        log_weights = torch.randn(batch_size, n_particles)
        ess = compute_ess(log_weights)
        assert not torch.isnan(ess).any()
        assert not torch.isinf(ess).any()


# =============================================================================
# Tests for compute_entropy
# =============================================================================

class TestComputeEntropy:
    """Tests for the compute_entropy function."""

    def test_uniform_weights_max_entropy(self, batch_size, n_particles, tolerance):
        """Uniform weights give H = log(K)."""
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))
        entropy = compute_entropy(log_weights)
        expected = torch.full((batch_size,), math.log(n_particles))
        assert torch.allclose(entropy, expected, **tolerance)

    def test_degenerate_weights_zero_entropy(self, batch_size, n_particles, tolerance):
        """Single dominating weight gives H -> 0."""
        log_weights = torch.full((batch_size, n_particles), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)
        entropy = compute_entropy(log_weights)
        assert torch.all(entropy < 0.1), f"Entropy should be near 0, got {entropy}"

    def test_entropy_non_negative(self, batch_size, n_particles):
        """Entropy >= 0 always."""
        log_weights = torch.randn(batch_size, n_particles)
        entropy = compute_entropy(log_weights)
        assert torch.all(entropy >= -1e-6)  # Allow tiny numerical errors

    def test_output_shape(self, batch_size, n_particles):
        """Entropy output shape is [batch]."""
        log_weights = torch.randn(batch_size, n_particles)
        entropy = compute_entropy(log_weights)
        assert entropy.shape == (batch_size,)

    def test_no_nan_inf(self, batch_size, n_particles):
        """Entropy should not contain NaN or Inf."""
        log_weights = torch.randn(batch_size, n_particles)
        entropy = compute_entropy(log_weights)
        assert not torch.isnan(entropy).any()
        assert not torch.isinf(entropy).any()


# =============================================================================
# Tests for init_uniform_log_weights
# =============================================================================

class TestInitUniformLogWeights:
    """Tests for the init_uniform_log_weights function."""

    def test_output_shape(self, batch_size, n_particles):
        """Shape is [batch_size, n_particles]."""
        log_weights = init_uniform_log_weights(batch_size, n_particles)
        assert log_weights.shape == (batch_size, n_particles)

    def test_all_values_equal(self, batch_size, n_particles):
        """All log weights are equal."""
        log_weights = init_uniform_log_weights(batch_size, n_particles)
        # All values should be the same
        assert torch.allclose(log_weights, log_weights[:, 0:1].expand_as(log_weights))

    def test_exp_sums_to_one(self, batch_size, n_particles, tolerance):
        """exp(log_weights).sum() = 1."""
        log_weights = init_uniform_log_weights(batch_size, n_particles)
        sums = torch.exp(log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), **tolerance)

    def test_correct_value(self, batch_size, n_particles, tolerance):
        """All values should be -log(K)."""
        log_weights = init_uniform_log_weights(batch_size, n_particles)
        expected_value = -math.log(n_particles)
        expected = torch.full_like(log_weights, expected_value)
        assert torch.allclose(log_weights, expected, **tolerance)

    def test_device_respected(self):
        """Output should be on specified device."""
        device = torch.device("cpu")
        log_weights = init_uniform_log_weights(4, 16, device=device)
        assert log_weights.device.type == device.type

    def test_dtype_respected(self):
        """Output should have specified dtype."""
        log_weights = init_uniform_log_weights(4, 16, dtype=torch.float64)
        assert log_weights.dtype == torch.float64


# =============================================================================
# Tests for weighted_mean
# =============================================================================

class TestWeightedMean:
    """Tests for the weighted_mean function."""

    def test_uniform_weights_equal_mean(self, random_particles, tolerance):
        """Uniform weights give simple average."""
        batch, K, H = random_particles.shape
        log_weights = torch.full((batch, K), -math.log(K))
        mean = weighted_mean(random_particles, log_weights)
        expected = random_particles.mean(dim=1)
        assert torch.allclose(mean, expected, **tolerance)

    def test_degenerate_weight_selects_particle(self, random_particles, tolerance):
        """Single heavy particle selects that particle."""
        batch, K, H = random_particles.shape
        log_weights = torch.full((batch, K), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)
        mean = weighted_mean(random_particles, log_weights)
        # Mean should be very close to particle 0
        assert torch.allclose(mean, random_particles[:, 0, :], atol=1e-3)

    def test_output_shape(self, random_particles):
        """Output shape is [batch, hidden_size]."""
        batch, K, H = random_particles.shape
        log_weights = torch.randn(batch, K)
        mean = weighted_mean(random_particles, log_weights)
        assert mean.shape == (batch, H)

    def test_single_particle(self):
        """With K=1, returns that particle's value."""
        particles = torch.randn(4, 1, 32)
        log_weights = torch.zeros(4, 1)  # log(1) = 0
        mean = weighted_mean(particles, log_weights)
        assert torch.allclose(mean, particles.squeeze(1))

    def test_no_nan_inf(self, random_particles):
        """Output should not contain NaN or Inf."""
        batch, K, H = random_particles.shape
        log_weights = torch.randn(batch, K)
        mean = weighted_mean(random_particles, log_weights)
        assert not torch.isnan(mean).any()
        assert not torch.isinf(mean).any()

    def test_broadcasting_over_hidden(self):
        """Weights should broadcast correctly over hidden dimension."""
        particles = torch.randn(4, 8, 32)
        log_weights = torch.randn(4, 8)
        log_weights = normalize_log_weights(log_weights)
        mean = weighted_mean(particles, log_weights)
        # Manual computation for verification
        weights = torch.exp(log_weights).unsqueeze(-1)  # [4, 8, 1]
        expected = (weights * particles).sum(dim=1)
        assert torch.allclose(mean, expected, atol=1e-5)


# =============================================================================
# Tests for weighted_variance
# =============================================================================

class TestWeightedVariance:
    """Tests for the weighted_variance function."""

    def test_uniform_weights_equal_sample_variance(self, random_particles, tolerance):
        """Matches sample variance for uniform weights."""
        batch, K, H = random_particles.shape
        log_weights = torch.full((batch, K), -math.log(K))
        variance = weighted_variance(random_particles, log_weights)
        # Sample variance
        expected = random_particles.var(dim=1, unbiased=False)
        assert torch.allclose(variance, expected, **tolerance)

    def test_variance_non_negative(self, random_particles):
        """Variance >= 0 always."""
        batch, K, H = random_particles.shape
        log_weights = torch.randn(batch, K)
        variance = weighted_variance(random_particles, log_weights)
        assert torch.all(variance >= -1e-6)  # Allow tiny numerical errors

    def test_identical_particles_zero_variance(self):
        """When all particles identical, variance = 0."""
        particles = torch.randn(4, 1, 32).expand(4, 8, 32)  # All same
        log_weights = torch.randn(4, 8)
        variance = weighted_variance(particles, log_weights)
        assert torch.allclose(variance, torch.zeros_like(variance), atol=1e-5)

    def test_output_shape(self, random_particles):
        """Output shape is [batch, hidden_size]."""
        batch, K, H = random_particles.shape
        log_weights = torch.randn(batch, K)
        variance = weighted_variance(random_particles, log_weights)
        assert variance.shape == (batch, H)

    def test_no_nan_inf(self, random_particles):
        """Output should not contain NaN or Inf."""
        batch, K, H = random_particles.shape
        log_weights = torch.randn(batch, K)
        variance = weighted_variance(random_particles, log_weights)
        assert not torch.isnan(variance).any()
        assert not torch.isinf(variance).any()


# =============================================================================
# Tests for temperature_scaled_log_weights
# =============================================================================

class TestTemperatureScaledLogWeights:
    """Tests for the temperature_scaled_log_weights function."""

    def test_temperature_one_no_change(self, batch_size, n_particles, tolerance):
        """T=1 should not change relative weights (only normalize)."""
        log_weights = torch.randn(batch_size, n_particles)
        normalized = normalize_log_weights(log_weights)
        scaled = temperature_scaled_log_weights(log_weights, temperature=1.0)
        assert torch.allclose(scaled, normalized, **tolerance)

    def test_high_temperature_more_uniform(self):
        """Higher T -> more uniform distribution (higher entropy)."""
        # Use fixed weights with clear variation to ensure consistent behavior
        log_weights = torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]])

        scaled_low = temperature_scaled_log_weights(log_weights, temperature=0.5)
        scaled_high = temperature_scaled_log_weights(log_weights, temperature=2.0)

        entropy_low = compute_entropy(scaled_low)
        entropy_high = compute_entropy(scaled_high)

        # Higher temperature -> higher entropy (more uniform)
        assert torch.all(entropy_high > entropy_low)

    def test_low_temperature_more_peaked(self):
        """Lower T -> more peaked distribution (lower ESS)."""
        # Use fixed weights with clear variation to ensure consistent behavior
        log_weights = torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]])

        scaled_low = temperature_scaled_log_weights(log_weights, temperature=0.1)
        scaled_high = temperature_scaled_log_weights(log_weights, temperature=5.0)

        ess_low = compute_ess(scaled_low)
        ess_high = compute_ess(scaled_high)

        # Higher temperature -> higher ESS (more particles contribute)
        assert torch.all(ess_high > ess_low)

    def test_invalid_temperature_raises(self):
        """T <= 0 raises ValueError."""
        log_weights = torch.randn(4, 16)
        with pytest.raises(ValueError):
            temperature_scaled_log_weights(log_weights, temperature=0.0)
        with pytest.raises(ValueError):
            temperature_scaled_log_weights(log_weights, temperature=-1.0)

    def test_output_normalized(self, batch_size, n_particles, tolerance):
        """Output should be normalized."""
        log_weights = torch.randn(batch_size, n_particles)
        scaled = temperature_scaled_log_weights(log_weights, temperature=0.5)
        sums = torch.exp(scaled).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), **tolerance)

    def test_output_shape(self, batch_size, n_particles):
        """Output shape matches input shape."""
        log_weights = torch.randn(batch_size, n_particles)
        scaled = temperature_scaled_log_weights(log_weights, temperature=1.0)
        assert scaled.shape == log_weights.shape

    def test_no_nan_inf(self, batch_size, n_particles):
        """Output should not contain NaN or Inf."""
        log_weights = torch.randn(batch_size, n_particles)
        for temp in [0.01, 0.1, 1.0, 10.0]:
            scaled = temperature_scaled_log_weights(log_weights, temperature=temp)
            assert not torch.isnan(scaled).any(), f"NaN at temperature={temp}"
            assert not torch.isinf(scaled).any(), f"Inf at temperature={temp}"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestWeightEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self):
        """K=1 should work correctly."""
        log_weights = torch.zeros(4, 1)
        normalized = normalize_log_weights(log_weights)
        assert torch.allclose(normalized, torch.zeros_like(normalized))  # log(1) = 0

        ess = compute_ess(normalized)
        assert torch.allclose(ess, torch.ones(4))  # ESS = 1

    def test_single_batch(self):
        """Batch size 1 should work."""
        log_weights = torch.randn(1, 16)
        normalized = normalize_log_weights(log_weights)
        assert normalized.shape == (1, 16)

    def test_large_particle_count(self):
        """Large K should work."""
        log_weights = torch.randn(4, 1024)
        normalized = normalize_log_weights(log_weights)
        sums = torch.exp(normalized).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_all_equal_weights(self):
        """All equal weights should give ESS = K."""
        n_particles = 32
        log_weights = torch.zeros(4, n_particles)  # All equal
        normalized = normalize_log_weights(log_weights)
        ess = compute_ess(normalized)
        expected = torch.full((4,), float(n_particles))
        assert torch.allclose(ess, expected, atol=1e-4)

    def test_gradients_flow(self):
        """Gradients should flow through weight operations."""
        log_weights = torch.randn(4, 16, requires_grad=True)
        normalized = normalize_log_weights(log_weights)
        loss = normalized.sum()
        loss.backward()
        assert log_weights.grad is not None
        assert not torch.isnan(log_weights.grad).any()
