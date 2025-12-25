"""Integration tests for dual particle filter cells.

Tests for:
- DualPFCfCCell
- DualPFLTCCell
- DualPFWiredCfCCell
- RaoBlackwellEstimator
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.dual import (
    DualPFCell,
    DualPFCfCCell,
    DualPFLTCCell,
    DualPFWiredCfCCell,
    RaoBlackwellEstimator,
    rao_blackwell_state_estimate,
    rao_blackwell_param_estimate,
)
from pfncps.nn.observation import GaussianObservationModel
from pfncps.nn.utils.weights import normalize_log_weights, compute_ess
from pfncps.wirings import FullyConnected


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def input_size() -> int:
    return 20


@pytest.fixture
def obs_size() -> int:
    return 10


@pytest.fixture
def dual_ltc_wiring(hidden_size, input_size):
    """Create wiring for DualPFLTCCell tests."""
    wiring = FullyConnected(units=hidden_size, output_dim=min(hidden_size, 10))
    wiring.build(input_size)
    return wiring


@pytest.fixture
def dual_wired_cfc_wiring(input_size):
    """Create wiring for DualPFWiredCfCCell tests."""
    hidden_size = 32
    wiring = FullyConnected(units=hidden_size, output_dim=10)
    wiring.build(input_size)
    return wiring


# =============================================================================
# Tests for DualPFCfCCell
# =============================================================================

class TestDualPFCfCCell:
    """Integration tests for DualPFCfCCell."""

    def test_forward_output_shape(self, batch_size, hidden_size, input_size):
        """Forward produces correct output shape."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, (state_particles, param_particles, log_weights) = cell(x)

        assert output.shape == (batch_size, hidden_size)
        assert state_particles.shape == (batch_size, n_particles, hidden_size)
        assert param_particles.shape[0] == batch_size
        assert param_particles.shape[1] == n_particles
        assert log_weights.shape == (batch_size, n_particles)

    def test_forward_without_state(self, batch_size, hidden_size, input_size):
        """Forward works without initial state."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, state = cell(x, hx=None)
        assert output.shape == (batch_size, hidden_size)

    def test_forward_with_state(self, batch_size, hidden_size, input_size):
        """Forward works with provided state."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        # Get initial state
        hx = cell.init_hidden(batch_size)

        output, state = cell(x, hx=hx)
        assert output.shape == (batch_size, hidden_size)

    def test_sequence_processing(self, batch_size, hidden_size, input_size):
        """Cell processes sequences correctly."""
        n_particles = 16
        seq_len = 10
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        state_particles, param_particles, log_weights = state
        # Weights should be valid
        sums = torch.exp(log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_joint_state_param_particles(self, batch_size, hidden_size, input_size):
        """Both state and parameter particles are tracked."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        _, (state_particles, param_particles, _) = cell(x)

        # State particles should have hidden_size dimension
        assert state_particles.shape[-1] == hidden_size

        # Param particles should have parameter count
        assert param_particles.shape[-1] == cell.param_registry.total_params

    def test_with_observation_model(self, batch_size, hidden_size, input_size, obs_size):
        """Observation model updates weights."""
        n_particles = 16
        obs_model = GaussianObservationModel(hidden_size, obs_size)
        cell = DualPFCfCCell(input_size, hidden_size, n_particles, observation_model=obs_model)

        x = torch.randn(batch_size, input_size)
        obs = torch.randn(batch_size, obs_size)

        output, (_, _, log_weights) = cell(x, observation=obs)

        ess = compute_ess(log_weights)
        assert torch.all(ess >= 1.0 - 1e-4)
        assert torch.all(ess <= n_particles + 1e-4)

    def test_return_all_particles(self, batch_size, hidden_size, input_size):
        """return_all_particles returns all outputs."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x, return_all_particles=True)
        assert output.shape == (batch_size, n_particles, hidden_size)

    def test_gradient_flow(self, hidden_size, input_size):
        """Gradients flow through cell."""
        n_particles = 8
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(4, input_size, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_no_nan_inf(self, batch_size, hidden_size, input_size):
        """No NaN or Inf in outputs."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)

        state = None
        for _ in range(10):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Tests for DualPFLTCCell
# =============================================================================

class TestDualPFLTCCell:
    """Integration tests for DualPFLTCCell."""

    def test_forward_output_shape(self, batch_size, hidden_size, input_size, dual_ltc_wiring):
        """Forward produces correct output shape."""
        n_particles = 16
        cell = DualPFLTCCell(wiring=dual_ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, (state_particles, param_particles, log_weights) = cell(x)

        assert output.shape[0] == batch_size
        assert state_particles.shape == (batch_size, n_particles, hidden_size)

    def test_sequence_processing(self, batch_size, hidden_size, input_size, dual_ltc_wiring):
        """Cell processes sequences correctly."""
        n_particles = 16
        seq_len = 10
        cell = DualPFLTCCell(wiring=dual_ltc_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, hidden_size, input_size, dual_ltc_wiring):
        """No NaN or Inf in outputs."""
        n_particles = 16
        cell = DualPFLTCCell(wiring=dual_ltc_wiring, n_particles=n_particles)

        state = None
        for _ in range(10):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Tests for DualPFWiredCfCCell
# =============================================================================

class TestDualPFWiredCfCCell:
    """Integration tests for DualPFWiredCfCCell."""

    def test_forward_output_shape(self, batch_size, input_size, dual_wired_cfc_wiring):
        """Forward produces correct output shape."""
        n_particles = 16
        cell = DualPFWiredCfCCell(wiring=dual_wired_cfc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x)
        assert output.shape[0] == batch_size

    def test_sequence_processing(self, batch_size, input_size, dual_wired_cfc_wiring):
        """Cell processes sequences correctly."""
        n_particles = 8
        seq_len = 10
        cell = DualPFWiredCfCCell(wiring=dual_wired_cfc_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, input_size, dual_wired_cfc_wiring):
        """No NaN or Inf in outputs."""
        n_particles = 8
        cell = DualPFWiredCfCCell(wiring=dual_wired_cfc_wiring, n_particles=n_particles)

        state = None
        for _ in range(5):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Tests for Rao-Blackwell Estimator
# =============================================================================

class TestRaoBlackwellEstimator:
    """Tests for RaoBlackwellEstimator utilities."""

    def test_rao_blackwell_state_estimate_shape(self, batch_size, n_particles, hidden_size):
        """State estimate has correct shape."""
        state_particles = torch.randn(batch_size, n_particles, hidden_size)
        param_particles = torch.randn(batch_size, n_particles, 50)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        mean, var = rao_blackwell_state_estimate(state_particles, param_particles, log_weights)

        assert mean.shape == (batch_size, hidden_size)
        assert var.shape == (batch_size, hidden_size)

    def test_rao_blackwell_state_estimate_values(self, batch_size, n_particles, hidden_size):
        """State estimate produces valid statistics."""
        state_particles = torch.randn(batch_size, n_particles, hidden_size)
        param_particles = torch.randn(batch_size, n_particles, 50)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        mean, var = rao_blackwell_state_estimate(state_particles, param_particles, log_weights)

        # Variance should be non-negative
        assert torch.all(var >= 0)

    def test_rao_blackwell_param_estimate_shape(self, batch_size, n_particles):
        """Param estimate has correct shape."""
        n_params = 100
        state_particles = torch.randn(batch_size, n_particles, 32)
        param_particles = torch.randn(batch_size, n_particles, n_params)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        mean, var = rao_blackwell_param_estimate(state_particles, param_particles, log_weights)

        assert mean.shape == (batch_size, n_params)
        assert var.shape == (batch_size, n_params)

    def test_uniform_weights_equals_simple_mean(self, batch_size, n_particles, hidden_size):
        """With uniform weights, weighted mean equals simple mean."""
        state_particles = torch.randn(batch_size, n_particles, hidden_size)
        param_particles = torch.randn(batch_size, n_particles, 50)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        mean, _ = rao_blackwell_state_estimate(state_particles, param_particles, log_weights)
        simple_mean = state_particles.mean(dim=1)

        assert torch.allclose(mean, simple_mean, atol=1e-5)

    def test_peaked_weights_selects_particle(self, batch_size, n_particles, hidden_size):
        """Peaked weights approximate selecting single particle."""
        state_particles = torch.randn(batch_size, n_particles, hidden_size)
        param_particles = torch.randn(batch_size, n_particles, 50)

        # Create peaked weights
        log_weights = torch.full((batch_size, n_particles), -100.0)
        log_weights[:, 0] = 0.0
        log_weights = normalize_log_weights(log_weights)

        mean, var = rao_blackwell_state_estimate(state_particles, param_particles, log_weights)

        # Should be close to first particle
        assert torch.allclose(mean, state_particles[:, 0, :], atol=1e-4)

        # Variance should be near zero
        assert torch.allclose(var, torch.zeros_like(var), atol=1e-4)


class TestRaoBlackwellEstimatorClass:
    """Tests for the RaoBlackwellEstimator class."""

    def test_estimator_initialization(self, hidden_size):
        """Estimator initializes correctly."""
        param_size = 50
        estimator = RaoBlackwellEstimator(hidden_size, param_size)
        assert estimator.state_size == hidden_size
        assert estimator.param_size == param_size

    def test_estimator_compute_estimates(self, batch_size, n_particles, hidden_size):
        """Estimator computes valid estimates."""
        param_size = 50
        estimator = RaoBlackwellEstimator(hidden_size, param_size)

        state_particles = torch.randn(batch_size, n_particles, hidden_size)
        param_particles = torch.randn(batch_size, n_particles, param_size)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        state_estimates = estimator.estimate_state(state_particles, param_particles, log_weights)
        param_estimates = estimator.estimate_params(state_particles, param_particles, log_weights)

        assert "mean" in state_estimates
        assert "variance" in state_estimates
        assert "mean" in param_estimates
        assert "variance" in param_estimates


# =============================================================================
# Common Tests for All Dual Cells
# =============================================================================

class TestDualCellsCommon:
    """Tests that apply to DualPFCfCCell (simple API cells)."""

    @pytest.fixture(params=["DualPFCfCCell"])
    def cell_class(self, request):
        """Parameterized cell class."""
        return {
            "DualPFCfCCell": DualPFCfCCell,
        }[request.param]

    def test_is_subclass(self, cell_class, input_size, hidden_size):
        """All cells are subclasses of DualPFCell."""
        n_particles = 8
        cell = cell_class(input_size, hidden_size, n_particles)
        assert isinstance(cell, DualPFCell)

    def test_has_required_attributes(self, cell_class, input_size, hidden_size):
        """All cells have required attributes."""
        n_particles = 8
        cell = cell_class(input_size, hidden_size, n_particles)

        assert hasattr(cell, "input_size")
        assert hasattr(cell, "hidden_size")
        assert hasattr(cell, "n_particles")
        assert hasattr(cell, "param_registry")
        assert hasattr(cell, "resampler")

    def test_init_hidden(self, cell_class, batch_size, input_size, hidden_size):
        """init_hidden produces valid initial state."""
        n_particles = 8
        cell = cell_class(input_size, hidden_size, n_particles)

        state_p, param_p, log_w = cell.init_hidden(batch_size)

        assert state_p.shape == (batch_size, n_particles, hidden_size)
        # Uniform weights
        assert torch.allclose(
            torch.exp(log_w).sum(dim=-1),
            torch.ones(batch_size),
            atol=1e-4
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestDualCellsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self, batch_size, hidden_size, input_size):
        """K=1 works."""
        cell = DualPFCfCCell(input_size, hidden_size, n_particles=1)
        x = torch.randn(batch_size, input_size)

        output, (state_p, param_p, weights) = cell(x)
        assert output.shape == (batch_size, hidden_size)
        assert state_p.shape[1] == 1

    def test_single_batch(self, hidden_size, input_size):
        """Batch size 1 works."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(1, input_size)

        output, _ = cell(x)
        assert output.shape == (1, hidden_size)

    def test_long_sequence(self, hidden_size, input_size):
        """Long sequence doesn't cause numerical issues."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        batch_size = 4
        seq_len = 50

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        state_p, param_p, log_weights = state
        assert not torch.isnan(state_p).any()
        assert not torch.isnan(param_p).any()
        assert not torch.isnan(log_weights).any()

    def test_correlations_preserved(self, batch_size, hidden_size, input_size):
        """State-param correlations are preserved through resampling."""
        n_particles = 16
        cell = DualPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        # Multiple steps to trigger resampling
        state = None
        for _ in range(20):
            output, state = cell(x, state)

        state_p, param_p, log_weights = state

        # Both should have same first two dimensions
        assert state_p.shape[:2] == param_p.shape[:2]
        assert log_weights.shape == (batch_size, n_particles)
