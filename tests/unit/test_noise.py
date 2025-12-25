"""Unit tests for noise injection (nn/utils/noise.py).

Tests for all noise injection types:
- ConstantNoise
- TimeScaledNoise
- LearnedNoise
- StateDependentNoise
"""

import math
import pytest
import torch
from torch import Tensor

from pfncps.nn.utils.noise import (
    NoiseType,
    NoiseInjector,
    ConstantNoise,
    TimeScaledNoise,
    LearnedNoise,
    StateDependentNoise,
    create_noise_injector,
)


# =============================================================================
# Tests for ConstantNoise
# =============================================================================

class TestConstantNoise:
    """Tests for ConstantNoise injector."""

    def test_output_shape_equals_input_shape(self, random_particles_fixed):
        """Output shape [batch, K, hidden] equals input."""
        batch, K, H = random_particles_fixed.shape
        injector = ConstantNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert noisy.shape == random_particles_fixed.shape

    def test_mean_approximately_preserved(self, hidden_size_fixed):
        """Adding zero-mean noise preserves mean (in expectation)."""
        injector = ConstantNoise(hidden_size_fixed, noise_init=0.1)
        particles = torch.randn(100, 32, hidden_size_fixed)

        # Average over many samples
        n_samples = 100
        original_mean = particles.mean()
        noisy_means = []
        for _ in range(n_samples):
            noisy = injector(particles.clone())
            noisy_means.append(noisy.mean())

        avg_noisy_mean = sum(noisy_means) / n_samples
        assert abs(avg_noisy_mean - original_mean) < 0.1

    def test_variance_increases(self, hidden_size_fixed):
        """Noise increases variance by scale^2."""
        injector = ConstantNoise(hidden_size_fixed, noise_init=0.5)
        particles = torch.zeros(100, 32, hidden_size_fixed)  # Zero variance

        noisy = injector(particles)
        # Variance should be approximately noise_scale^2
        var = noisy.var()
        expected_var = 0.5 ** 2
        assert abs(var - expected_var) < 0.1

    def test_different_outputs_each_call(self, random_particles_fixed):
        """Two calls produce different results (stochastic)."""
        batch, K, H = random_particles_fixed.shape
        injector = ConstantNoise(H, noise_init=0.1)

        noisy1 = injector(random_particles_fixed)
        noisy2 = injector(random_particles_fixed)

        assert not torch.allclose(noisy1, noisy2)

    def test_reproducible_with_seed(self, random_particles_fixed):
        """Same seed produces same noise."""
        batch, K, H = random_particles_fixed.shape
        injector = ConstantNoise(H, noise_init=0.1)

        torch.manual_seed(42)
        noisy1 = injector(random_particles_fixed.clone())

        torch.manual_seed(42)
        noisy2 = injector(random_particles_fixed.clone())

        assert torch.allclose(noisy1, noisy2)

    def test_timespans_ignored(self, random_particles_fixed):
        """Constant noise doesn't use timespans."""
        batch, K, H = random_particles_fixed.shape
        injector = ConstantNoise(H, noise_init=0.1)

        torch.manual_seed(42)
        noisy1 = injector(random_particles_fixed.clone(), timespans=None)

        torch.manual_seed(42)
        noisy2 = injector(random_particles_fixed.clone(), timespans=torch.tensor([1.0]))

        assert torch.allclose(noisy1, noisy2)

    def test_no_nan_inf(self, random_particles_fixed):
        """Output should not contain NaN or Inf."""
        batch, K, H = random_particles_fixed.shape
        injector = ConstantNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()

    def test_get_noise_scale(self, hidden_size_fixed):
        """get_noise_scale returns the scale."""
        injector = ConstantNoise(hidden_size_fixed, noise_init=0.2)
        states = torch.randn(4, 16, hidden_size_fixed)
        scale = injector.get_noise_scale(states)
        assert torch.allclose(scale, torch.full((hidden_size_fixed,), 0.2))


# =============================================================================
# Tests for TimeScaledNoise
# =============================================================================

class TestTimeScaledNoise:
    """Tests for TimeScaledNoise injector."""

    def test_noise_scales_with_sqrt_dt(self, hidden_size_fixed):
        """Noise variance proportional to dt."""
        injector = TimeScaledNoise(hidden_size_fixed, noise_init=1.0)
        particles = torch.zeros(100, 32, hidden_size_fixed)

        # Small dt
        dt_small = torch.tensor([[0.1]])
        torch.manual_seed(42)
        noisy_small = injector(particles.clone(), timespans=dt_small)
        var_small = noisy_small.var()

        # Large dt (4x)
        dt_large = torch.tensor([[0.4]])
        torch.manual_seed(42)
        noisy_large = injector(particles.clone(), timespans=dt_large)
        var_large = noisy_large.var()

        # Variance ratio should be ~4 (since var ~ dt)
        ratio = var_large / var_small
        assert 3.0 < ratio < 5.0, f"Expected ratio ~4, got {ratio}"

    def test_default_dt_used_when_none(self, random_particles_fixed):
        """Uses default_dt when timespans not provided."""
        batch, K, H = random_particles_fixed.shape
        injector = TimeScaledNoise(H, noise_init=0.1, default_dt=1.0)

        torch.manual_seed(42)
        noisy1 = injector(random_particles_fixed.clone(), timespans=None)

        torch.manual_seed(42)
        noisy2 = injector(random_particles_fixed.clone(), timespans=torch.tensor([[1.0]]))

        assert torch.allclose(noisy1, noisy2, atol=1e-5)

    def test_respects_batch_timespans(self, hidden_size_fixed):
        """Handles [batch, 1] timespans."""
        injector = TimeScaledNoise(hidden_size_fixed, noise_init=0.1)
        particles = torch.randn(4, 16, hidden_size_fixed)
        dt = torch.rand(4, 1)

        noisy = injector(particles, timespans=dt)
        assert noisy.shape == particles.shape

    def test_output_shape(self, random_particles_fixed):
        """Output shape matches input."""
        batch, K, H = random_particles_fixed.shape
        injector = TimeScaledNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert noisy.shape == random_particles_fixed.shape

    def test_no_nan_inf(self, random_particles_fixed):
        """Output should not contain NaN or Inf."""
        batch, K, H = random_particles_fixed.shape
        injector = TimeScaledNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()


# =============================================================================
# Tests for LearnedNoise
# =============================================================================

class TestLearnedNoise:
    """Tests for LearnedNoise injector."""

    def test_log_scale_is_parameter(self, hidden_size_fixed):
        """log_noise_scale is nn.Parameter."""
        injector = LearnedNoise(hidden_size_fixed, noise_init=0.1)
        param_names = [name for name, _ in injector.named_parameters()]
        assert "log_noise_scale" in param_names

    def test_noise_scale_always_positive(self, hidden_size_fixed):
        """Exp of log scale is always positive."""
        injector = LearnedNoise(hidden_size_fixed, noise_init=0.1)
        # Even with extreme log values
        injector.log_noise_scale.data.fill_(-100)
        assert torch.all(injector.noise_scale > 0)

    def test_clamping_applied(self, hidden_size_fixed):
        """Scale clamped to [min_scale, max_scale]."""
        injector = LearnedNoise(
            hidden_size_fixed, noise_init=0.1,
            min_scale=0.01, max_scale=1.0
        )

        # Try to set very large scale
        injector.log_noise_scale.data.fill_(100)
        assert torch.all(injector.noise_scale <= 1.0)

        # Try to set very small scale
        injector.log_noise_scale.data.fill_(-100)
        assert torch.all(injector.noise_scale >= 0.01)

    def test_gradient_flows_to_scale(self, random_particles_fixed):
        """Backprop updates log_noise_scale."""
        batch, K, H = random_particles_fixed.shape
        injector = LearnedNoise(H, noise_init=0.1)

        particles = random_particles_fixed.clone().requires_grad_(True)
        noisy = injector(particles)
        loss = noisy.sum()
        loss.backward()

        assert injector.log_noise_scale.grad is not None

    def test_time_scaled_option(self, hidden_size_fixed):
        """time_scaled=True multiplies by sqrt(dt)."""
        injector_scaled = LearnedNoise(hidden_size_fixed, noise_init=1.0, time_scaled=True)
        injector_unscaled = LearnedNoise(hidden_size_fixed, noise_init=1.0, time_scaled=False)

        # Set same scale
        injector_unscaled.log_noise_scale.data = injector_scaled.log_noise_scale.data.clone()

        particles = torch.zeros(100, 32, hidden_size_fixed)
        dt = torch.tensor([[4.0]])  # sqrt(4) = 2

        torch.manual_seed(42)
        noisy_scaled = injector_scaled(particles.clone(), timespans=dt)
        var_scaled = noisy_scaled.var()

        torch.manual_seed(42)
        noisy_unscaled = injector_unscaled(particles.clone(), timespans=dt)
        var_unscaled = noisy_unscaled.var()

        # Scaled variance should be ~4x (dt factor)
        ratio = var_scaled / var_unscaled
        assert 3.0 < ratio < 5.0

    def test_output_shape(self, random_particles_fixed):
        """Output shape matches input."""
        batch, K, H = random_particles_fixed.shape
        injector = LearnedNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert noisy.shape == random_particles_fixed.shape

    def test_no_nan_inf(self, random_particles_fixed):
        """Output should not contain NaN or Inf."""
        batch, K, H = random_particles_fixed.shape
        injector = LearnedNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()


# =============================================================================
# Tests for StateDependentNoise
# =============================================================================

class TestStateDependentNoise:
    """Tests for StateDependentNoise injector."""

    def test_noise_varies_with_state(self, hidden_size_fixed):
        """Different states get different noise scales after MLP weights are non-zero."""
        injector = StateDependentNoise(hidden_size_fixed, noise_init=0.1)

        # Re-initialize weights with non-zero values so state-dependence works
        for module in injector.noise_mlp.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)

        states1 = torch.zeros(4, 16, hidden_size_fixed)
        states2 = torch.ones(4, 16, hidden_size_fixed) * 5

        scale1 = injector.get_noise_scale(states1)
        scale2 = injector.get_noise_scale(states2)

        # Should not be identical with non-zero MLP weights
        assert not torch.allclose(scale1, scale2)

    def test_mlp_processes_each_particle(self, random_particles_fixed):
        """MLP applied to each [batch*K, hidden] input."""
        batch, K, H = random_particles_fixed.shape
        injector = StateDependentNoise(H, noise_init=0.1)

        scale = injector.get_noise_scale(random_particles_fixed)
        assert scale.shape == (batch, K, H)

    def test_softplus_ensures_positive(self, random_particles_fixed):
        """Output of MLP is always positive."""
        batch, K, H = random_particles_fixed.shape
        injector = StateDependentNoise(H, noise_init=0.1)

        scale = injector.get_noise_scale(random_particles_fixed)
        assert torch.all(scale > 0)

    def test_gradient_flows_through_mlp(self, random_particles_fixed):
        """Backprop updates MLP parameters."""
        batch, K, H = random_particles_fixed.shape
        injector = StateDependentNoise(H, noise_init=0.1)

        particles = random_particles_fixed.clone().requires_grad_(True)
        noisy = injector(particles)
        loss = noisy.sum()
        loss.backward()

        # Check some MLP params have gradients
        has_grad = False
        for name, param in injector.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_clamping_applied(self, hidden_size_fixed):
        """Scale clamped to [min_scale, max_scale]."""
        injector = StateDependentNoise(
            hidden_size_fixed, noise_init=0.1,
            min_scale=0.01, max_scale=1.0
        )
        particles = torch.randn(4, 16, hidden_size_fixed) * 100

        scale = injector.get_noise_scale(particles)
        assert torch.all(scale >= 0.01)
        assert torch.all(scale <= 1.0)

    def test_output_shape(self, random_particles_fixed):
        """Output shape matches input."""
        batch, K, H = random_particles_fixed.shape
        injector = StateDependentNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert noisy.shape == random_particles_fixed.shape

    def test_no_nan_inf(self, random_particles_fixed):
        """Output should not contain NaN or Inf."""
        batch, K, H = random_particles_fixed.shape
        injector = StateDependentNoise(H, noise_init=0.1)
        noisy = injector(random_particles_fixed)
        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()

    def test_different_activations(self, hidden_size_fixed):
        """Test different activation functions."""
        for activation in ["tanh", "relu", "gelu"]:
            injector = StateDependentNoise(
                hidden_size_fixed, noise_init=0.1, activation=activation
            )
            particles = torch.randn(4, 16, hidden_size_fixed)
            noisy = injector(particles)
            assert noisy.shape == particles.shape
            assert not torch.isnan(noisy).any()


# =============================================================================
# Tests for create_noise_injector Factory
# =============================================================================

class TestCreateNoiseInjector:
    """Tests for the create_noise_injector factory function."""

    def test_creates_constant_noise(self, hidden_size_fixed):
        """'constant' creates ConstantNoise."""
        injector = create_noise_injector("constant", hidden_size_fixed)
        assert isinstance(injector, ConstantNoise)

    def test_creates_time_scaled_noise(self, hidden_size_fixed):
        """'time_scaled' creates TimeScaledNoise."""
        injector = create_noise_injector("time_scaled", hidden_size_fixed)
        assert isinstance(injector, TimeScaledNoise)

    def test_creates_learned_noise(self, hidden_size_fixed):
        """'learned' creates LearnedNoise."""
        injector = create_noise_injector("learned", hidden_size_fixed)
        assert isinstance(injector, LearnedNoise)

    def test_creates_state_dependent_noise(self, hidden_size_fixed):
        """'state_dependent' creates StateDependentNoise."""
        injector = create_noise_injector("state_dependent", hidden_size_fixed)
        assert isinstance(injector, StateDependentNoise)

    def test_accepts_enum(self, hidden_size_fixed):
        """Accepts NoiseType enum."""
        injector = create_noise_injector(NoiseType.CONSTANT, hidden_size_fixed)
        assert isinstance(injector, ConstantNoise)

    def test_unknown_type_raises(self, hidden_size_fixed):
        """Unknown noise type raises ValueError."""
        with pytest.raises(ValueError):
            create_noise_injector("unknown_type", hidden_size_fixed)

    def test_passes_kwargs(self, hidden_size_fixed):
        """Extra kwargs are passed to constructor."""
        injector = create_noise_injector("constant", hidden_size_fixed, noise_init=0.5)
        scale = injector.get_noise_scale(torch.randn(1, 1, hidden_size_fixed))
        assert torch.allclose(scale.mean(), torch.tensor(0.5))


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestNoiseEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_noise_init(self, hidden_size_fixed):
        """Very small noise_init should work."""
        injector = ConstantNoise(hidden_size_fixed, noise_init=1e-10)
        particles = torch.randn(4, 16, hidden_size_fixed)
        noisy = injector(particles)
        assert not torch.isnan(noisy).any()
        # Particles should be nearly unchanged
        assert torch.allclose(noisy, particles, atol=1e-8)

    def test_large_noise_init(self, hidden_size_fixed):
        """Large noise_init should work."""
        injector = ConstantNoise(hidden_size_fixed, noise_init=10.0)
        particles = torch.randn(4, 16, hidden_size_fixed)
        noisy = injector(particles)
        assert not torch.isnan(noisy).any()

    def test_single_particle(self, hidden_size_fixed):
        """K=1 should work."""
        injector = ConstantNoise(hidden_size_fixed, noise_init=0.1)
        particles = torch.randn(4, 1, hidden_size_fixed)
        noisy = injector(particles)
        assert noisy.shape == (4, 1, hidden_size_fixed)

    def test_zero_timespan(self, hidden_size_fixed):
        """Zero timespan in time-scaled noise."""
        injector = TimeScaledNoise(hidden_size_fixed, noise_init=0.1)
        particles = torch.randn(4, 16, hidden_size_fixed)
        dt = torch.tensor([[0.0]])

        noisy = injector(particles, timespans=dt)
        # With dt=0, no noise should be added
        assert torch.allclose(noisy, particles, atol=1e-6)

    def test_inherited_from_base_class(self, hidden_size_fixed):
        """All injectors inherit from NoiseInjector."""
        for noise_type in ["constant", "time_scaled", "learned", "state_dependent"]:
            injector = create_noise_injector(noise_type, hidden_size_fixed)
            assert isinstance(injector, NoiseInjector)
