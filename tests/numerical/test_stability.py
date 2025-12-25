"""Numerical stability tests.

Tests for edge cases, extreme values, and numerical robustness
across the pfncps library.
"""

import math
import pytest
import torch
from torch import Tensor

from pfncps.nn.utils.weights import (
    normalize_log_weights,
    compute_ess,
    weighted_mean,
    weighted_variance,
    log_weight_update,
)
from pfncps.nn.utils.resampling import soft_resample, SoftResampler
from pfncps.nn.utils.noise import (
    ConstantNoise,
    TimeScaledNoise,
    LearnedNoise,
    StateDependentNoise,
)
from pfncps.nn.state_level import PFCfCCell, PFLTCCell
from pfncps.nn.param_level import ParamPFCfCCell
from pfncps.nn.sde import SDELTCCell
from pfncps.wirings import FullyConnected


# =============================================================================
# Extreme Value Tests for Weights
# =============================================================================

class TestExtremeLogWeights:
    """Tests for numerical stability with extreme log weight values."""

    def test_very_large_positive(self):
        """Very large positive log weights handled without NaN/Inf."""
        log_weights = torch.full((4, 32), 1e6)
        normalized = normalize_log_weights(log_weights)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        # Should be uniform after normalization
        sums = torch.exp(normalized).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-4)

    def test_very_large_negative(self):
        """Very large negative log weights handled without NaN/Inf."""
        log_weights = torch.full((4, 32), -1e6)
        normalized = normalize_log_weights(log_weights)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        sums = torch.exp(normalized).sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-4)

    def test_mixed_extreme_values(self):
        """Mixed extreme positive and negative values handled."""
        log_weights = torch.randn(4, 32) * 1e6
        normalized = normalize_log_weights(log_weights)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_single_extreme_high(self):
        """Single extremely high value dominates but no overflow."""
        log_weights = torch.full((4, 32), -100.0)
        log_weights[:, 0] = 1e6
        normalized = normalize_log_weights(log_weights)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        # First particle should have nearly all weight
        assert torch.all(torch.exp(normalized[:, 0]) > 0.99)

    def test_all_very_small(self):
        """All very small (near-zero probability) handled."""
        log_weights = torch.full((4, 32), -1e10)
        normalized = normalize_log_weights(log_weights)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_ess_with_extreme_weights(self):
        """ESS computation stable with extreme weights."""
        # Nearly degenerate case
        log_weights = torch.full((4, 32), -1000.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        ess = compute_ess(log_weights)

        assert not torch.isnan(ess).any()
        assert not torch.isinf(ess).any()
        assert torch.all(ess >= 1.0 - 1e-4)
        assert torch.all(ess <= 32 + 1e-4)

    def test_weighted_mean_extreme_weights(self):
        """Weighted mean stable with extreme weights."""
        particles = torch.randn(4, 32, 64)
        log_weights = torch.randn(4, 32) * 100
        log_weights = normalize_log_weights(log_weights)

        mean = weighted_mean(particles, log_weights)

        assert not torch.isnan(mean).any()
        assert not torch.isinf(mean).any()
        assert mean.shape == (4, 64)

    def test_weighted_variance_extreme_weights(self):
        """Weighted variance stable with extreme weights."""
        particles = torch.randn(4, 32, 64)
        log_weights = torch.randn(4, 32) * 100
        log_weights = normalize_log_weights(log_weights)

        variance = weighted_variance(particles, log_weights)

        assert not torch.isnan(variance).any()
        assert not torch.isinf(variance).any()
        assert torch.all(variance >= 0)


# =============================================================================
# Long Sequence Stability Tests
# =============================================================================

class TestLongSequenceStability:
    """Tests for numerical stability over long sequences."""

    def test_state_level_cell_long_sequence(self):
        """State-level PF cell remains stable over long sequences."""
        cell = PFCfCCell(20, 64, n_particles=32)
        seq_len = 200

        state = None
        for t in range(seq_len):
            x = torch.randn(4, 20)
            output, state = cell(x, state)

            assert not torch.isnan(output).any(), f"NaN output at step {t}"
            assert not torch.isinf(output).any(), f"Inf output at step {t}"

            particles, log_weights = state
            assert not torch.isnan(particles).any(), f"NaN particles at step {t}"
            assert not torch.isnan(log_weights).any(), f"NaN weights at step {t}"
            assert not torch.isinf(particles).any(), f"Inf particles at step {t}"
            assert not torch.isinf(log_weights).any(), f"Inf weights at step {t}"

    def test_param_level_cell_long_sequence(self):
        """Parameter-level PF cell remains stable over long sequences."""
        cell = ParamPFCfCCell(20, 64, n_particles=8)
        seq_len = 100

        state = None
        for t in range(seq_len):
            x = torch.randn(4, 20)
            output, state = cell(x, state)

            assert not torch.isnan(output).any(), f"NaN output at step {t}"
            assert not torch.isinf(output).any(), f"Inf output at step {t}"

    def test_sde_cell_long_sequence(self):
        """SDE cell remains stable over long sequences."""
        wiring = FullyConnected(units=64, output_dim=10)
        cell = SDELTCCell(wiring=wiring, in_features=20, n_particles=16)
        seq_len = 100

        state = None
        for t in range(seq_len):
            x = torch.randn(4, 20)
            output, state = cell(x, state)

            assert not torch.isnan(output).any(), f"NaN output at step {t}"
            assert not torch.isinf(output).any(), f"Inf output at step {t}"

    def test_weights_remain_normalized_long_sequence(self):
        """Weights remain properly normalized over long sequences."""
        cell = PFCfCCell(20, 64, n_particles=32)
        seq_len = 100

        state = None
        for t in range(seq_len):
            x = torch.randn(4, 20)
            _, state = cell(x, state)

            _, log_weights = state
            sums = torch.exp(log_weights).sum(dim=-1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-3), \
                f"Weights not normalized at step {t}: sum={sums}"

    def test_ess_remains_valid_long_sequence(self):
        """ESS remains in valid range over long sequences."""
        cell = PFCfCCell(20, 64, n_particles=32)
        seq_len = 100

        state = None
        ess_values = []
        for t in range(seq_len):
            x = torch.randn(4, 20)
            _, state = cell(x, state)

            _, log_weights = state
            ess = compute_ess(log_weights)
            ess_values.append(ess.mean().item())

            assert torch.all(ess >= 1.0 - 1e-3), f"ESS below 1 at step {t}"
            assert torch.all(ess <= 32 + 1e-3), f"ESS above K at step {t}"


# =============================================================================
# Timespan Edge Cases
# =============================================================================

class TestTimespanEdgeCases:
    """Tests for timespan edge cases."""

    def test_zero_timespan(self):
        """Zero timespan handled gracefully."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(4, 20)
        ts = torch.tensor(0.0)

        output, state = cell(x, timespans=ts)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_small_timespan(self):
        """Very small timespan handled."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(4, 20)
        ts = torch.tensor(1e-10)

        output, _ = cell(x, timespans=ts)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_large_timespan(self):
        """Very large timespan handled."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(4, 20)
        ts = torch.tensor(1000.0)

        output, _ = cell(x, timespans=ts)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_variable_timespans(self):
        """Variable timespans per batch item handled."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(4, 20)
        ts = torch.tensor([[0.01], [0.1], [1.0], [10.0]])

        output, _ = cell(x, timespans=ts)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sde_zero_timespan(self):
        """SDE cell handles zero timespan."""
        wiring = FullyConnected(units=64, output_dim=10)
        cell = SDELTCCell(wiring=wiring, in_features=20, n_particles=16)
        x = torch.randn(4, 20)
        ts = torch.tensor(0.0)

        output, _ = cell(x, timespans=ts)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Noise Edge Cases
# =============================================================================

class TestNoiseEdgeCases:
    """Tests for noise injection edge cases."""

    def test_very_small_constant_noise(self):
        """Very small constant noise doesn't cause issues."""
        noise = ConstantNoise(64, noise_init=1e-10)
        states = torch.randn(4, 32, 64)

        noisy = noise(states)

        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()
        # Should be almost identical
        assert torch.allclose(noisy, states, atol=1e-6)

    def test_very_large_constant_noise(self):
        """Very large constant noise handled."""
        noise = ConstantNoise(64, noise_init=100.0)
        states = torch.randn(4, 32, 64)

        noisy = noise(states)

        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()

    def test_learned_noise_near_bounds(self):
        """Learned noise near min/max bounds stable."""
        noise = LearnedNoise(64, noise_init=1e-8, min_scale=1e-8, max_scale=10.0)
        states = torch.randn(4, 32, 64)

        noisy = noise(states)

        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()

    def test_time_scaled_noise_zero_dt(self):
        """Time-scaled noise with zero dt handled."""
        noise = TimeScaledNoise(64)
        states = torch.randn(4, 32, 64)
        dt = torch.tensor(0.0)

        noisy = noise(states, timespans=dt)

        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()

    def test_time_scaled_noise_large_dt(self):
        """Time-scaled noise with large dt handled."""
        noise = TimeScaledNoise(64)
        states = torch.randn(4, 32, 64)
        dt = torch.tensor(1000.0)

        noisy = noise(states, timespans=dt)

        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()

    def test_state_dependent_noise_extreme_states(self):
        """State-dependent noise with extreme states handled."""
        noise = StateDependentNoise(64, min_scale=0.01, max_scale=1.0)
        states = torch.randn(4, 32, 64) * 1000  # Extreme values

        noisy = noise(states)

        assert not torch.isnan(noisy).any()
        assert not torch.isinf(noisy).any()


# =============================================================================
# Resampling Edge Cases
# =============================================================================

class TestResamplingEdgeCases:
    """Tests for resampling edge cases."""

    def test_soft_resample_degenerate_weights(self):
        """Soft resampling with degenerate weights."""
        particles = torch.randn(4, 32, 64)
        log_weights = torch.full((4, 32), -1000.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        new_p, new_w = soft_resample(particles, log_weights, alpha=0.5)

        assert not torch.isnan(new_p).any()
        assert not torch.isnan(new_w).any()
        assert not torch.isinf(new_p).any()
        assert not torch.isinf(new_w).any()

    def test_soft_resample_uniform_weights(self):
        """Soft resampling with perfectly uniform weights."""
        particles = torch.randn(4, 32, 64)
        log_weights = torch.full((4, 32), -math.log(32))

        new_p, new_w = soft_resample(particles, log_weights, alpha=0.5)

        assert not torch.isnan(new_p).any()
        assert not torch.isnan(new_w).any()

    def test_soft_resample_alpha_zero(self):
        """Soft resampling with alpha=0 (uniform proposal)."""
        particles = torch.randn(4, 32, 64)
        log_weights = torch.randn(4, 32)
        log_weights = normalize_log_weights(log_weights)

        new_p, new_w = soft_resample(particles, log_weights, alpha=0.0)

        assert not torch.isnan(new_p).any()
        assert not torch.isnan(new_w).any()

    def test_soft_resample_alpha_one(self):
        """Soft resampling with alpha=1 (original proposal)."""
        particles = torch.randn(4, 32, 64)
        log_weights = torch.randn(4, 32)
        log_weights = normalize_log_weights(log_weights)

        new_p, new_w = soft_resample(particles, log_weights, alpha=1.0)

        assert not torch.isnan(new_p).any()
        assert not torch.isnan(new_w).any()

    def test_single_particle_resampling(self):
        """Resampling with single particle (K=1)."""
        particles = torch.randn(4, 1, 64)
        log_weights = torch.zeros(4, 1)

        new_p, new_w = soft_resample(particles, log_weights, alpha=0.5)

        assert new_p.shape == particles.shape
        assert torch.allclose(torch.exp(new_w), torch.ones(4, 1))


# =============================================================================
# Input Edge Cases
# =============================================================================

class TestInputEdgeCases:
    """Tests for various input edge cases."""

    def test_zero_input(self):
        """Cell handles zero input."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.zeros(4, 20)

        output, _ = cell(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_input_values(self):
        """Cell handles large input values."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(4, 20) * 100

        output, _ = cell(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_small_input_values(self):
        """Cell handles very small input values."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(4, 20) * 1e-10

        output, _ = cell(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mixed_extreme_inputs(self):
        """Cell handles inputs with mixed extreme values."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(4, 20)
        x[:, :5] = 1e6
        x[:, 5:10] = -1e6
        x[:, 10:15] = 1e-10

        output, state = cell(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Particle Count Edge Cases
# =============================================================================

class TestParticleCountEdgeCases:
    """Tests for edge cases with particle counts."""

    def test_single_particle(self):
        """K=1 works correctly."""
        cell = PFCfCCell(20, 64, n_particles=1)
        x = torch.randn(4, 20)

        output, (particles, log_weights) = cell(x)

        assert output.shape == (4, 64)
        assert particles.shape == (4, 1, 64)
        assert log_weights.shape == (4, 1)
        assert torch.allclose(torch.exp(log_weights), torch.ones(4, 1))

    def test_two_particles(self):
        """K=2 (minimum for meaningful PF) works."""
        cell = PFCfCCell(20, 64, n_particles=2)
        x = torch.randn(4, 20)

        output, (particles, log_weights) = cell(x)

        assert output.shape == (4, 64)
        assert particles.shape == (4, 2, 64)

    def test_large_particle_count(self):
        """Large particle count works."""
        cell = PFCfCCell(20, 64, n_particles=256)
        x = torch.randn(4, 20)

        output, (particles, log_weights) = cell(x)

        assert output.shape == (4, 64)
        assert particles.shape == (4, 256, 64)
        assert not torch.isnan(output).any()


# =============================================================================
# Batch Size Edge Cases
# =============================================================================

class TestBatchSizeEdgeCases:
    """Tests for edge cases with batch sizes."""

    def test_single_batch(self):
        """Batch size 1 works."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x = torch.randn(1, 20)

        output, (particles, log_weights) = cell(x)

        assert output.shape == (1, 64)
        assert particles.shape == (1, 32, 64)

    def test_large_batch(self):
        """Large batch size works."""
        cell = PFCfCCell(20, 64, n_particles=16)
        x = torch.randn(128, 20)

        output, (particles, log_weights) = cell(x)

        assert output.shape == (128, 64)
        assert not torch.isnan(output).any()
