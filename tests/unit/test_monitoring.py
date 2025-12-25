"""Unit tests for monitoring utilities (nn/utils/monitoring.py).

Tests for:
- compute_particle_diversity
- check_numerical_health
- ParticleMonitor
- ParticleMetrics
"""

import math
import pytest
import torch
from torch import Tensor

from pfncps.nn.utils.monitoring import (
    ParticleMetrics,
    compute_particle_diversity,
    check_numerical_health,
    ParticleMonitor,
)
from pfncps.nn.utils.weights import normalize_log_weights


# =============================================================================
# Tests for compute_particle_diversity
# =============================================================================

class TestComputeParticleDiversity:
    """Tests for the compute_particle_diversity function."""

    def test_variance_per_dim_computed(self, random_particles_fixed):
        """Computes variance across particles per dimension."""
        metrics = compute_particle_diversity(random_particles_fixed)
        assert "variance_per_dim" in metrics
        assert metrics["variance_per_dim"].shape == (random_particles_fixed.shape[0],)

    def test_pairwise_distance_computed(self, random_particles_fixed):
        """Average pairwise distance is computed."""
        metrics = compute_particle_diversity(random_particles_fixed)
        assert "avg_pairwise_distance" in metrics
        batch = random_particles_fixed.shape[0]
        assert metrics["avg_pairwise_distance"].shape == (batch,)

    def test_effective_dimension_computed(self, random_particles_fixed):
        """Effective dimension is computed."""
        metrics = compute_particle_diversity(random_particles_fixed)
        assert "effective_dimension" in metrics

    def test_relative_spread_computed(self, random_particles_fixed):
        """Relative spread is computed."""
        metrics = compute_particle_diversity(random_particles_fixed)
        assert "relative_spread" in metrics

    def test_collapse_ratio_computed(self, random_particles_fixed):
        """Collapse ratio is computed."""
        metrics = compute_particle_diversity(random_particles_fixed)
        assert "collapse_ratio" in metrics

    def test_variance_non_negative(self, random_particles_fixed):
        """Variance >= 0."""
        metrics = compute_particle_diversity(random_particles_fixed)
        assert torch.all(metrics["variance_per_dim"] >= 0)

    def test_pairwise_distance_non_negative(self, random_particles_fixed):
        """Pairwise distance >= 0."""
        metrics = compute_particle_diversity(random_particles_fixed)
        assert torch.all(metrics["avg_pairwise_distance"] >= 0)

    def test_identical_particles_zero_variance(self):
        """When all particles are identical, variance = 0."""
        # All particles are the same
        single = torch.randn(4, 1, 32)
        particles = single.expand(4, 16, 32).clone()  # Clone to detach

        metrics = compute_particle_diversity(particles)
        assert torch.allclose(metrics["variance_per_dim"], torch.zeros(4), atol=1e-6)

    def test_identical_particles_zero_distance(self):
        """When all particles identical, avg distance = 0."""
        single = torch.randn(4, 1, 32)
        particles = single.expand(4, 16, 32).clone()

        metrics = compute_particle_diversity(particles)
        assert torch.allclose(metrics["avg_pairwise_distance"], torch.zeros(4), atol=1e-6)

    def test_effective_dimension_in_valid_range(self, random_particles_fixed):
        """Effective dimension in [1, hidden_size]."""
        batch, K, H = random_particles_fixed.shape
        metrics = compute_particle_diversity(random_particles_fixed)
        # Effective dimension should be positive and bounded by hidden_size
        assert torch.all(metrics["effective_dimension"] > 0)
        assert torch.all(metrics["effective_dimension"] <= H + 1)

    def test_no_nan_inf(self, random_particles_fixed):
        """No NaN or Inf in metrics."""
        metrics = compute_particle_diversity(random_particles_fixed)
        for key, value in metrics.items():
            assert not torch.isnan(value).any(), f"NaN in {key}"
            assert not torch.isinf(value).any(), f"Inf in {key}"


# =============================================================================
# Tests for check_numerical_health
# =============================================================================

class TestCheckNumericalHealth:
    """Tests for the check_numerical_health function."""

    def test_detects_nan_in_particles(self):
        """Flags NaN in particles."""
        particles = torch.randn(4, 16, 32)
        particles[0, 0, 0] = float("nan")
        log_weights = torch.randn(4, 16)

        health = check_numerical_health(particles, log_weights)
        assert not health["healthy"]
        assert "NaN detected" in health["warnings"]
        assert health["metrics"]["has_nan"]

    def test_detects_nan_in_weights(self):
        """Flags NaN in log_weights."""
        particles = torch.randn(4, 16, 32)
        log_weights = torch.randn(4, 16)
        log_weights[0, 0] = float("nan")

        health = check_numerical_health(particles, log_weights)
        assert not health["healthy"]
        assert "NaN detected" in health["warnings"]

    def test_detects_inf_in_particles(self):
        """Flags Inf in particles."""
        particles = torch.randn(4, 16, 32)
        particles[0, 0, 0] = float("inf")
        log_weights = torch.randn(4, 16)

        health = check_numerical_health(particles, log_weights)
        assert not health["healthy"]
        assert "Inf detected" in health["warnings"]
        assert health["metrics"]["has_inf"]

    def test_detects_large_particle_norm(self):
        """Warns when particle norm exceeds threshold."""
        particles = torch.randn(4, 16, 32) * 200  # Large norms
        log_weights = normalize_log_weights(torch.randn(4, 16))

        health = check_numerical_health(particles, log_weights)
        # Should have warning about large norm
        assert any("Large particle norm" in w for w in health["warnings"])

    def test_detects_low_ess(self):
        """Marks unhealthy when ESS too low."""
        particles = torch.randn(4, 16, 32)
        # Peaked weights (low ESS)
        log_weights = torch.full((4, 16), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        health = check_numerical_health(particles, log_weights)
        assert not health["healthy"]
        assert any("Low ESS" in w for w in health["warnings"])

    def test_detects_weight_degeneracy(self):
        """Warns when max weight ratio too high."""
        particles = torch.randn(4, 16, 32)
        # Very peaked weights
        log_weights = torch.full((4, 16), -1000.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        health = check_numerical_health(particles, log_weights)
        assert any("Weight degeneracy" in w for w in health["warnings"])

    def test_healthy_particles_pass(self, random_particles_fixed, uniform_log_weights_fixed):
        """Well-behaved particles pass all checks."""
        health = check_numerical_health(random_particles_fixed, uniform_log_weights_fixed)
        assert health["healthy"], f"Warnings: {health['warnings']}"
        assert len(health["warnings"]) == 0

    def test_custom_thresholds(self, random_particles_fixed, uniform_log_weights_fixed):
        """Custom thresholds are respected."""
        thresholds = {
            "max_particle_norm": 0.001,  # Very low
            "min_ess_ratio": 0.99,  # Very high
            "max_weight_ratio": 0.01,  # Very low
        }
        health = check_numerical_health(
            random_particles_fixed, uniform_log_weights_fixed, thresholds=thresholds
        )
        # Should fail with these extreme thresholds
        assert len(health["warnings"]) > 0

    def test_metrics_returned(self, random_particles_fixed, uniform_log_weights_fixed):
        """Health check returns metrics dict."""
        health = check_numerical_health(random_particles_fixed, uniform_log_weights_fixed)
        assert "metrics" in health
        assert "has_nan" in health["metrics"]
        assert "has_inf" in health["metrics"]
        assert "max_particle_norm" in health["metrics"]
        assert "min_ess_ratio" in health["metrics"]


# =============================================================================
# Tests for ParticleMonitor
# =============================================================================

class TestParticleMonitor:
    """Tests for the ParticleMonitor class."""

    def test_log_accumulates_history(self, random_particles_fixed, uniform_log_weights_fixed):
        """Each log() call adds to history."""
        monitor = ParticleMonitor(warn_on_issues=False)

        for _ in range(5):
            monitor.log(random_particles_fixed, uniform_log_weights_fixed)

        assert len(monitor.history["ess"]) == 5
        assert monitor.step == 5

    def test_returns_particle_metrics(self, random_particles_fixed, uniform_log_weights_fixed):
        """log() returns ParticleMetrics dataclass."""
        monitor = ParticleMonitor(warn_on_issues=False)
        metrics = monitor.log(random_particles_fixed, uniform_log_weights_fixed)

        assert isinstance(metrics, ParticleMetrics)
        assert hasattr(metrics, "ess")
        assert hasattr(metrics, "max_weight")
        assert hasattr(metrics, "has_nan")

    def test_summary_statistics(self, random_particles_fixed, uniform_log_weights_fixed):
        """summary() returns mean/min/max/last for each metric."""
        monitor = ParticleMonitor(warn_on_issues=False)

        for _ in range(10):
            monitor.log(random_particles_fixed, uniform_log_weights_fixed)

        summary = monitor.summary()

        assert "ess_mean" in summary
        assert "ess_min" in summary
        assert "ess_max" in summary
        assert "ess_last" in summary

    def test_reset_clears_history(self, random_particles_fixed, uniform_log_weights_fixed):
        """reset() clears step counter and history."""
        monitor = ParticleMonitor(warn_on_issues=False)

        monitor.log(random_particles_fixed, uniform_log_weights_fixed)
        assert monitor.step == 1

        monitor.reset()
        assert monitor.step == 0
        assert len(monitor.history["ess"]) == 0

    def test_get_history_returns_correct_list(self, random_particles_fixed, uniform_log_weights_fixed):
        """get_history(key) returns that metric's history."""
        monitor = ParticleMonitor(warn_on_issues=False)

        for _ in range(5):
            monitor.log(random_particles_fixed, uniform_log_weights_fixed)

        ess_history = monitor.get_history("ess")
        assert len(ess_history) == 5

    def test_get_history_unknown_key(self):
        """get_history() with unknown key returns empty list."""
        monitor = ParticleMonitor(warn_on_issues=False)
        history = monitor.get_history("nonexistent_key")
        assert history == []

    def test_extra_metrics_logged(self, random_particles_fixed, uniform_log_weights_fixed):
        """Extra metrics are logged when provided."""
        monitor = ParticleMonitor(warn_on_issues=False)
        monitor.log(
            random_particles_fixed, uniform_log_weights_fixed,
            extra_metrics={"custom_metric": 1.5}
        )

        assert "custom_metric" in monitor.history
        assert monitor.history["custom_metric"][0] == 1.5

    def test_empty_summary(self):
        """summary() returns empty dict when no logs."""
        monitor = ParticleMonitor(warn_on_issues=False)
        summary = monitor.summary()
        assert summary == {}

    def test_metrics_values_reasonable(self, random_particles_fixed, uniform_log_weights_fixed):
        """Logged metric values are reasonable."""
        batch, K, H = random_particles_fixed.shape
        monitor = ParticleMonitor(warn_on_issues=False)
        metrics = monitor.log(random_particles_fixed, uniform_log_weights_fixed)

        # ESS should be close to K for uniform weights
        assert abs(metrics.ess - K) < 0.5

        # No NaN/Inf for good inputs
        assert not metrics.has_nan
        assert not metrics.has_inf


# =============================================================================
# Tests for ParticleMetrics Dataclass
# =============================================================================

class TestParticleMetrics:
    """Tests for the ParticleMetrics dataclass."""

    def test_dataclass_fields(self):
        """ParticleMetrics has expected fields."""
        metrics = ParticleMetrics(
            ess=16.0,
            max_weight=0.1,
            min_weight=0.01,
            weight_entropy=2.5,
            particle_variance=0.5,
            avg_pairwise_distance=1.0,
            max_particle_norm=5.0,
            has_nan=False,
            has_inf=False,
        )

        assert metrics.ess == 16.0
        assert metrics.max_weight == 0.1
        assert metrics.has_nan is False


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestMonitoringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self):
        """K=1 should work (though metrics may be NaN due to no degrees of freedom)."""
        particles = torch.randn(4, 1, 32)
        log_weights = torch.zeros(4, 1)

        # Should not raise an error
        metrics = compute_particle_diversity(particles)
        # With K=1, variance is NaN (no degrees of freedom for sample variance)
        # Just verify the function runs without error and returns expected keys
        assert "variance_per_dim" in metrics
        assert metrics["variance_per_dim"].shape == (4,)

    def test_single_batch(self):
        """Batch size 1 should work."""
        particles = torch.randn(1, 16, 32)
        log_weights = torch.randn(1, 16)

        metrics = compute_particle_diversity(particles)
        assert metrics["variance_per_dim"].shape == (1,)

    def test_large_particle_count(self):
        """Large K should work."""
        particles = torch.randn(2, 512, 32)
        log_weights = normalize_log_weights(torch.randn(2, 512))

        health = check_numerical_health(particles, log_weights)
        assert "metrics" in health

    def test_monitor_with_nan_input(self):
        """Monitor handles NaN input gracefully."""
        monitor = ParticleMonitor(warn_on_issues=False)
        particles = torch.randn(4, 16, 32)
        particles[0, 0, 0] = float("nan")
        log_weights = torch.randn(4, 16)

        metrics = monitor.log(particles, log_weights)
        assert metrics.has_nan
