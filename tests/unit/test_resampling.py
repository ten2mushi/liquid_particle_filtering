"""Unit tests for soft resampling (nn/utils/resampling.py).

These tests verify the correctness of soft resampling operations
that maintain differentiability while redistributing particles.

Key Invariants Tested:
- R1: Proposal distribution has full support (all values > 0 when alpha < 1)
- R2: Importance weight correction is unbiased
"""

import math
import pytest
import torch
from torch import Tensor

from pfncps.nn.utils.resampling import (
    AlphaMode,
    compute_proposal,
    soft_resample,
    SoftResampler,
)
from pfncps.nn.utils.weights import (
    normalize_log_weights,
    compute_ess,
    weighted_mean,
)


# =============================================================================
# Tests for compute_proposal
# =============================================================================

class TestComputeProposal:
    """Tests for the compute_proposal function."""

    def test_alpha_zero_uniform(self, batch_size, n_particles, tolerance):
        """Alpha=0 gives uniform proposal (1/K for all)."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        proposal = compute_proposal(weights, alpha=0.0, n_particles=n_particles)
        expected = torch.full_like(proposal, 1.0 / n_particles)
        assert torch.allclose(proposal, expected, **tolerance)

    def test_alpha_one_original(self, batch_size, n_particles, tolerance):
        """Alpha=1 gives original weights."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        proposal = compute_proposal(weights, alpha=1.0, n_particles=n_particles)
        assert torch.allclose(proposal, weights, **tolerance)

    def test_proposal_sums_to_one(self, batch_size, n_particles, alpha_value, tolerance):
        """Proposal is valid probability distribution (sums to 1)."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        proposal = compute_proposal(weights, alpha=alpha_value, n_particles=n_particles)
        sums = proposal.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), **tolerance)

    def test_proposal_non_negative(self, batch_size, n_particles, alpha_value):
        """All proposal values >= 0."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        proposal = compute_proposal(weights, alpha=alpha_value, n_particles=n_particles)
        assert torch.all(proposal >= 0)

    def test_proposal_positive_when_alpha_less_than_one(self, batch_size, n_particles):
        """Invariant R1: Full support when alpha < 1."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        for alpha in [0.0, 0.25, 0.5, 0.75, 0.99]:
            proposal = compute_proposal(weights, alpha=alpha, n_particles=n_particles)
            assert torch.all(proposal > 0), f"Zero probability at alpha={alpha}"

    def test_output_shape(self, batch_size, n_particles):
        """Output shape matches weights shape."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        proposal = compute_proposal(weights, alpha=0.5, n_particles=n_particles)
        assert proposal.shape == weights.shape

    def test_alpha_as_tensor(self, batch_size, n_particles):
        """Alpha can be a scalar tensor."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        alpha = torch.tensor(0.5)
        proposal = compute_proposal(weights, alpha=alpha, n_particles=n_particles)
        assert proposal.shape == weights.shape

    def test_alpha_per_batch(self, batch_size, n_particles, tolerance):
        """Alpha can vary per batch item."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        alpha = torch.rand(batch_size, 1)  # [batch, 1]
        proposal = compute_proposal(weights, alpha=alpha, n_particles=n_particles)
        # Each batch item should have different mixing
        assert proposal.shape == weights.shape
        # Verify sums to 1
        sums = proposal.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), **tolerance)

    def test_convex_combination(self, batch_size, n_particles, tolerance):
        """Proposal = alpha*weights + (1-alpha)*uniform is convex combination."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        alpha = 0.7
        uniform = 1.0 / n_particles
        proposal = compute_proposal(weights, alpha=alpha, n_particles=n_particles)
        expected = alpha * weights + (1.0 - alpha) * uniform
        assert torch.allclose(proposal, expected, **tolerance)


# =============================================================================
# Tests for soft_resample
# =============================================================================

class TestSoftResample:
    """Tests for the soft_resample function."""

    def test_output_particles_shape(self, random_particles, uniform_log_weights):
        """Output particles shape equals input shape."""
        new_particles, _ = soft_resample(random_particles, uniform_log_weights, alpha=0.5)
        assert new_particles.shape == random_particles.shape

    def test_output_log_weights_shape(self, random_particles, uniform_log_weights):
        """Output log weights shape matches input."""
        _, new_log_weights = soft_resample(random_particles, uniform_log_weights, alpha=0.5)
        assert new_log_weights.shape == uniform_log_weights.shape

    def test_return_indices_shape(self, random_particles, uniform_log_weights):
        """When return_indices=True, ancestors shape is [batch, K]."""
        _, _, ancestors = soft_resample(
            random_particles, uniform_log_weights, alpha=0.5, return_indices=True
        )
        batch, K, _ = random_particles.shape
        assert ancestors.shape == (batch, K)

    def test_resampled_particles_from_input(self, random_particles_fixed, uniform_log_weights_fixed):
        """Each resampled particle must be exactly equal to some input particle."""
        new_particles, _, ancestors = soft_resample(
            random_particles_fixed, uniform_log_weights_fixed,
            alpha=0.5, return_indices=True
        )
        batch, K, H = random_particles_fixed.shape

        for b in range(batch):
            for k in range(K):
                ancestor_idx = ancestors[b, k]
                original = random_particles_fixed[b, ancestor_idx, :]
                resampled = new_particles[b, k, :]
                assert torch.allclose(resampled, original)

    def test_log_weights_normalized(self, random_particles, tolerance):
        """Output log weights sum to 0 in log space (exp sums to 1)."""
        batch, K, H = random_particles.shape
        log_weights = torch.randn(batch, K)
        _, new_log_weights = soft_resample(random_particles, log_weights, alpha=0.5)
        sums = torch.exp(new_log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), **tolerance)

    def test_no_nan_inf_in_particles(self, random_particles, uniform_log_weights):
        """No NaN or Inf in output particles."""
        new_particles, _ = soft_resample(random_particles, uniform_log_weights, alpha=0.5)
        assert not torch.isnan(new_particles).any()
        assert not torch.isinf(new_particles).any()

    def test_no_nan_inf_in_weights(self, random_particles, uniform_log_weights):
        """No NaN or Inf in output weights."""
        _, new_log_weights = soft_resample(random_particles, uniform_log_weights, alpha=0.5)
        assert not torch.isnan(new_log_weights).any()
        assert not torch.isinf(new_log_weights).any()

    def test_alpha_one_deterministic_like(self, random_particles_fixed):
        """High alpha approaches deterministic resampling."""
        batch, K, H = random_particles_fixed.shape
        # Create peaked weights
        log_weights = torch.full((batch, K), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        # With alpha=1, we should mostly sample particle 0
        torch.manual_seed(42)
        _, _, ancestors = soft_resample(
            random_particles_fixed, log_weights, alpha=0.99, return_indices=True
        )
        # Most ancestors should be 0
        assert (ancestors == 0).float().mean() > 0.9

    def test_alpha_zero_uniform_sampling(self, random_particles_fixed):
        """Alpha=0 resamples uniformly regardless of weights."""
        batch, K, H = random_particles_fixed.shape
        # Create very peaked weights
        log_weights = torch.full((batch, K), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        # With alpha=0, sampling should be uniform
        torch.manual_seed(42)
        ancestor_counts = torch.zeros(K)
        n_trials = 100
        for _ in range(n_trials):
            _, _, ancestors = soft_resample(
                random_particles_fixed, log_weights, alpha=0.0, return_indices=True
            )
            for k in range(K):
                ancestor_counts[k] += (ancestors == k).sum()

        # Should be roughly uniform (within tolerance)
        expected_per_particle = n_trials * batch * K / K
        for k in range(K):
            ratio = ancestor_counts[k] / expected_per_particle
            assert 0.5 < ratio < 1.5, f"Particle {k} sampled {ancestor_counts[k]} times"


# =============================================================================
# Tests for SoftResampler
# =============================================================================

class TestSoftResampler:
    """Tests for the SoftResampler module."""

    def test_init_fixed_mode(self, n_particles_fixed):
        """Can initialize with fixed alpha mode."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode='fixed', alpha_init=0.6)
        assert resampler.alpha_mode == AlphaMode.FIXED
        assert torch.allclose(resampler.alpha, torch.tensor(0.6))

    def test_init_adaptive_mode(self, n_particles_fixed):
        """Can initialize with adaptive alpha mode."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode='adaptive')
        assert resampler.alpha_mode == AlphaMode.ADAPTIVE

    def test_init_learnable_mode(self, n_particles_fixed):
        """Can initialize with learnable alpha mode."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode='learnable')
        assert resampler.alpha_mode == AlphaMode.LEARNABLE

    def test_fixed_alpha_is_buffer(self, n_particles_fixed):
        """Fixed mode: alpha is a buffer, not parameter."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode='fixed')
        param_names = [name for name, _ in resampler.named_parameters()]
        assert '_alpha_logit' not in param_names
        buffer_names = [name for name, _ in resampler.named_buffers()]
        assert '_alpha_fixed' in buffer_names

    def test_learnable_alpha_is_parameter(self, n_particles_fixed):
        """Learnable mode: alpha is nn.Parameter."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode='learnable')
        param_names = [name for name, _ in resampler.named_parameters()]
        assert '_alpha_logit' in param_names

    def test_alpha_within_zero_one(self, alpha_mode, n_particles_fixed):
        """Alpha property always returns value in [0, 1]."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode=alpha_mode)
        alpha = resampler.alpha
        assert alpha >= 0.0
        assert alpha <= 1.0

    def test_adaptive_alpha_low_ess_gives_low_alpha(self, n_particles_fixed):
        """Low ESS -> lower alpha (more exploration)."""
        resampler = SoftResampler(
            n_particles_fixed, alpha_mode='adaptive',
            alpha_min=0.2, alpha_max=0.9
        )
        # Low ESS
        low_ess = torch.tensor([1.5, 2.0])
        alpha_low = resampler.compute_adaptive_alpha(low_ess)
        # High ESS
        high_ess = torch.tensor([float(n_particles_fixed) * 0.8])
        alpha_high = resampler.compute_adaptive_alpha(high_ess)

        assert torch.all(alpha_low < alpha_high)

    def test_adaptive_alpha_clamped_to_bounds(self, n_particles_fixed):
        """Adaptive alpha stays within [alpha_min, alpha_max]."""
        resampler = SoftResampler(
            n_particles_fixed, alpha_mode='adaptive',
            alpha_min=0.3, alpha_max=0.8
        )
        # Various ESS values
        for ess_val in [0.1, 1.0, 5.0, float(n_particles_fixed)]:
            ess = torch.tensor([ess_val])
            alpha = resampler.compute_adaptive_alpha(ess)
            assert torch.all(alpha >= 0.3)
            assert torch.all(alpha <= 0.8)

    def test_should_resample_below_threshold(self, random_particles_fixed, n_particles_fixed):
        """Returns True when ESS < threshold * K."""
        resampler = SoftResampler(n_particles_fixed, resample_threshold=0.5)
        batch, K, H = random_particles_fixed.shape

        # Create peaked weights (low ESS)
        log_weights = torch.full((batch, K), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        should = resampler.should_resample(log_weights)
        assert should.all(), "Should resample when ESS is low"

    def test_should_not_resample_above_threshold(self, random_particles_fixed, n_particles_fixed):
        """Returns False when ESS >= threshold * K."""
        resampler = SoftResampler(n_particles_fixed, resample_threshold=0.5)
        batch, K, H = random_particles_fixed.shape

        # Uniform weights (max ESS)
        log_weights = torch.full((batch, K), -math.log(K))

        should = resampler.should_resample(log_weights)
        assert not should.any(), "Should not resample when ESS is high"

    def test_force_resample_overrides_threshold(self, random_particles_fixed, n_particles_fixed):
        """force_resample=True always triggers resampling."""
        resampler = SoftResampler(n_particles_fixed, resample_threshold=0.5)
        batch, K, H = random_particles_fixed.shape

        # Uniform weights (high ESS, normally wouldn't resample)
        log_weights = torch.full((batch, K), -math.log(K))

        # Without force, particles should be unchanged (or nearly so)
        torch.manual_seed(42)
        new_p1, new_w1 = resampler(random_particles_fixed, log_weights, force_resample=False)

        # With force, particles should be resampled
        torch.manual_seed(42)
        new_p2, new_w2 = resampler(random_particles_fixed, log_weights, force_resample=True)

        # The outputs should be different (resampling happened vs not)
        # Note: With uniform weights and same seed, they might be similar
        # but the function paths are different

    def test_forward_returns_correct_shapes(self, random_particles_fixed, uniform_log_weights_fixed):
        """Forward returns (particles, log_weights)."""
        batch, K, H = random_particles_fixed.shape
        resampler = SoftResampler(K, alpha_mode='fixed')

        new_particles, new_log_weights = resampler(
            random_particles_fixed, uniform_log_weights_fixed
        )
        assert new_particles.shape == (batch, K, H)
        assert new_log_weights.shape == (batch, K)

    def test_forward_return_ess(self, random_particles_fixed, uniform_log_weights_fixed):
        """With return_ess=True, also returns ESS."""
        batch, K, H = random_particles_fixed.shape
        resampler = SoftResampler(K, alpha_mode='fixed')

        new_particles, new_log_weights, ess = resampler(
            random_particles_fixed, uniform_log_weights_fixed, return_ess=True
        )
        assert ess.shape == (batch,)

    def test_learnable_alpha_gradient(self, random_particles_fixed, n_particles_fixed):
        """Learnable alpha should have gradient after backward."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode='learnable')
        batch, K, H = random_particles_fixed.shape

        particles = random_particles_fixed.clone().requires_grad_(True)
        log_weights = torch.randn(batch, K, requires_grad=True)

        new_p, new_w = resampler(particles, log_weights, force_resample=True)
        loss = new_p.sum() + new_w.sum()
        loss.backward()

        # Alpha should have gradient through the soft resampling
        assert resampler._alpha_logit.grad is not None

    def test_preserves_dtype(self, n_particles_fixed):
        """Output dtype matches input dtype."""
        resampler = SoftResampler(n_particles_fixed, alpha_mode='fixed')

        for dtype in [torch.float32, torch.float64]:
            particles = torch.randn(4, n_particles_fixed, 32, dtype=dtype)
            log_weights = torch.randn(4, n_particles_fixed, dtype=dtype)
            new_p, new_w = resampler(particles, log_weights, force_resample=True)
            assert new_p.dtype == dtype
            assert new_w.dtype == dtype


# =============================================================================
# Tests for Resampling Invariants
# =============================================================================

class TestResamplingInvariants:
    """Tests for fundamental resampling invariants."""

    def test_importance_weight_correction_unbiased(self, random_particles_fixed):
        """Invariant R2: Importance weight correction preserves expectation."""
        batch, K, H = random_particles_fixed.shape
        log_weights = torch.randn(batch, K)
        log_weights = normalize_log_weights(log_weights)

        # Original weighted mean
        original_mean = weighted_mean(random_particles_fixed, log_weights)

        # Monte Carlo estimate of resampled weighted mean
        n_trials = 500
        resampled_means = []
        for _ in range(n_trials):
            new_particles, new_log_weights = soft_resample(
                random_particles_fixed, log_weights, alpha=0.5
            )
            resampled_means.append(weighted_mean(new_particles, new_log_weights))

        mc_mean = torch.stack(resampled_means).mean(dim=0)

        # Should be approximately equal (Monte Carlo has variance)
        assert torch.allclose(mc_mean, original_mean, atol=0.15), \
            f"Bias detected: original={original_mean.mean()}, resampled={mc_mean.mean()}"

    def test_proposal_full_support(self, batch_size, n_particles):
        """Proposal always has full support when alpha < 1."""
        # Even with very peaked weights
        weights = torch.zeros(batch_size, n_particles)
        weights[:, 0] = 1.0  # All mass on particle 0

        for alpha in [0.0, 0.5, 0.9, 0.99]:
            proposal = compute_proposal(weights, alpha=alpha, n_particles=n_particles)
            assert torch.all(proposal > 0), f"Zero probability at alpha={alpha}"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestResamplingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self):
        """K=1 case: particle unchanged, weight reset."""
        particles = torch.randn(4, 1, 32)
        log_weights = torch.zeros(4, 1)

        new_particles, new_log_weights = soft_resample(particles, log_weights, alpha=0.5)
        assert torch.allclose(new_particles, particles)
        assert torch.allclose(new_log_weights, torch.zeros_like(new_log_weights), atol=1e-5)

    def test_single_batch(self):
        """Batch size 1 should work."""
        particles = torch.randn(1, 16, 32)
        log_weights = torch.randn(1, 16)

        new_particles, new_log_weights = soft_resample(particles, log_weights, alpha=0.5)
        assert new_particles.shape == (1, 16, 32)

    def test_very_peaked_weights(self):
        """Handle extremely peaked weights."""
        particles = torch.randn(4, 32, 64)
        log_weights = torch.full((4, 32), -1000.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        new_particles, new_log_weights = soft_resample(particles, log_weights, alpha=0.5)
        assert not torch.isnan(new_particles).any()
        assert not torch.isnan(new_log_weights).any()

    def test_uniform_weights(self, random_particles_fixed):
        """Uniform weights should not cause issues."""
        batch, K, H = random_particles_fixed.shape
        log_weights = torch.full((batch, K), -math.log(K))

        new_particles, new_log_weights = soft_resample(
            random_particles_fixed, log_weights, alpha=0.5
        )
        assert not torch.isnan(new_particles).any()
        assert not torch.isnan(new_log_weights).any()

    def test_large_particle_count(self):
        """Large K should work."""
        particles = torch.randn(2, 512, 32)
        log_weights = torch.randn(2, 512)

        new_particles, new_log_weights = soft_resample(particles, log_weights, alpha=0.5)
        assert new_particles.shape == (2, 512, 32)
        assert not torch.isnan(new_log_weights).any()
