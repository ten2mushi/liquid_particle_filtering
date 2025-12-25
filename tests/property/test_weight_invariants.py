"""Property-based tests for weight and resampling invariants.

Uses Hypothesis to test mathematical invariants that must always hold
regardless of input values.
"""

import math
import pytest
import torch
from hypothesis import given, strategies as st, settings, assume
from torch import Tensor

from pfncps.nn.utils.weights import (
    normalize_log_weights,
    compute_ess,
    weighted_mean,
    weighted_variance,
    log_weight_update,
)
from pfncps.nn.utils.resampling import (
    compute_proposal,
    soft_resample,
    SoftResampler,
)


# =============================================================================
# Property Tests for Weight Normalization
# =============================================================================

class TestWeightNormalizationInvariants:
    """Property-based tests for weight normalization invariants."""

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_normalization_sums_to_one(self, batch_size, n_particles):
        """Invariant W1: exp(normalized).sum() == 1 for all inputs."""
        log_weights = torch.randn(batch_size, n_particles) * 10
        normalized = normalize_log_weights(log_weights)

        sums = torch.exp(normalized).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
        scale=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_normalization_bounded(self, batch_size, n_particles, scale):
        """Invariant W2: Normalized log weights are valid (no NaN/Inf, max <= 0)."""
        log_weights = torch.randn(batch_size, n_particles) * scale
        normalized = normalize_log_weights(log_weights)

        # Output should be numerically stable (no NaN/Inf)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        # Log probabilities should be <= 0
        assert normalized.max() <= 0.0

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_normalization_preserves_ordering(self, batch_size, n_particles):
        """Normalization preserves relative ordering of weights."""
        log_weights = torch.randn(batch_size, n_particles)
        normalized = normalize_log_weights(log_weights)

        # Argmax should be same
        assert torch.equal(
            log_weights.argmax(dim=-1),
            normalized.argmax(dim=-1)
        )

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_normalization_no_nan_inf(self, batch_size, n_particles):
        """Normalization never produces NaN or Inf."""
        log_weights = torch.randn(batch_size, n_particles) * 50
        normalized = normalize_log_weights(log_weights)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()


# =============================================================================
# Property Tests for ESS Computation
# =============================================================================

class TestESSInvariants:
    """Property-based tests for ESS invariants."""

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 64),
    )
    @settings(max_examples=50, deadline=None)
    def test_ess_in_valid_range(self, batch_size, n_particles):
        """Invariant W3: ESS always in [1, K]."""
        log_weights = torch.randn(batch_size, n_particles)
        log_weights = normalize_log_weights(log_weights)

        ess = compute_ess(log_weights)

        assert torch.all(ess >= 1.0 - 1e-4)
        assert torch.all(ess <= n_particles + 1e-4)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 64),
    )
    @settings(max_examples=50, deadline=None)
    def test_uniform_weights_max_ess(self, batch_size, n_particles):
        """Uniform weights give ESS = K."""
        log_weights = torch.full(
            (batch_size, n_particles), -math.log(n_particles)
        )

        ess = compute_ess(log_weights)

        assert torch.allclose(
            ess, torch.full((batch_size,), float(n_particles)), atol=1e-4
        )

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_ess_no_nan_inf(self, batch_size, n_particles):
        """ESS never produces NaN or Inf."""
        log_weights = torch.randn(batch_size, n_particles) * 20
        log_weights = normalize_log_weights(log_weights)

        ess = compute_ess(log_weights)

        assert not torch.isnan(ess).any()
        assert not torch.isinf(ess).any()


# =============================================================================
# Property Tests for Proposal Distribution
# =============================================================================

class TestProposalInvariants:
    """Property-based tests for proposal distribution invariants."""

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
        alpha=st.floats(0.0, 1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_proposal_is_valid_distribution(self, batch_size, n_particles, alpha):
        """Invariant R1: Proposal sums to 1 and is non-negative."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        proposal = compute_proposal(weights, alpha, n_particles)

        # Sums to 1
        sums = proposal.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

        # Non-negative
        assert torch.all(proposal >= 0)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_proposal_full_support_low_alpha(self, batch_size, n_particles):
        """With alpha < 1, proposal has full support (all values > 0)."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        alpha = 0.5

        proposal = compute_proposal(weights, alpha, n_particles)

        assert torch.all(proposal > 0)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_proposal_alpha_zero_is_uniform(self, batch_size, n_particles):
        """Alpha=0 gives uniform proposal."""
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        proposal = compute_proposal(weights, alpha=0.0, n_particles=n_particles)

        expected = torch.full_like(proposal, 1.0 / n_particles)
        assert torch.allclose(proposal, expected, atol=1e-5)


# =============================================================================
# Property Tests for Soft Resampling
# =============================================================================

class TestSoftResamplingInvariants:
    """Property-based tests for soft resampling invariants."""

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
        hidden_size=st.integers(4, 64),
    )
    @settings(max_examples=30, deadline=None)
    def test_resampling_preserves_shape(self, batch_size, n_particles, hidden_size):
        """Resampling preserves tensor shapes."""
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.randn(batch_size, n_particles)

        new_particles, new_log_weights = soft_resample(
            particles, log_weights, alpha=0.5
        )

        assert new_particles.shape == particles.shape
        assert new_log_weights.shape == log_weights.shape

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
        hidden_size=st.integers(4, 64),
    )
    @settings(max_examples=30, deadline=None)
    def test_resampling_output_weights_normalized(self, batch_size, n_particles, hidden_size):
        """Output weights from resampling are normalized."""
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.randn(batch_size, n_particles)

        _, new_log_weights = soft_resample(particles, log_weights, alpha=0.5)

        sums = torch.exp(new_log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


# =============================================================================
# Property Tests for Weighted Statistics
# =============================================================================

class TestWeightedStatisticsInvariants:
    """Property-based tests for weighted mean/variance."""

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
        hidden_size=st.integers(4, 64),
    )
    @settings(max_examples=30, deadline=None)
    def test_weighted_mean_shape(self, batch_size, n_particles, hidden_size):
        """Weighted mean has correct output shape."""
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        mean = weighted_mean(particles, log_weights)

        assert mean.shape == (batch_size, hidden_size)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
        hidden_size=st.integers(4, 64),
    )
    @settings(max_examples=30, deadline=None)
    def test_weighted_variance_non_negative(self, batch_size, n_particles, hidden_size):
        """Weighted variance is always non-negative."""
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        variance = weighted_variance(particles, log_weights)

        assert torch.all(variance >= 0)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
        hidden_size=st.integers(4, 64),
    )
    @settings(max_examples=30, deadline=None)
    def test_uniform_weights_simple_mean(self, batch_size, n_particles, hidden_size):
        """With uniform weights, weighted mean equals simple mean."""
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        w_mean = weighted_mean(particles, log_weights)
        s_mean = particles.mean(dim=1)

        assert torch.allclose(w_mean, s_mean, atol=1e-5)
