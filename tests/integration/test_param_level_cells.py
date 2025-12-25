"""Integration tests for parameter-level particle filter cells.

Tests for:
- ParamPFCfCCell
- ParamPFLTCCell
- ParamPFWiredCfCCell
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.param_level import (
    ParamLevelPFCell,
    ParamPFCfCCell,
    ParamPFLTCCell,
    ParamPFWiredCfCCell,
    ParameterRegistry,
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
def n_particles_small() -> int:
    """Smaller particle count for param-level PF."""
    return 8


@pytest.fixture
def param_ltc_wiring(hidden_size, input_size):
    """Create wiring for ParamPFLTCCell tests."""
    wiring = FullyConnected(units=hidden_size, output_dim=min(hidden_size, 10))
    wiring.build(input_size)
    return wiring


@pytest.fixture
def param_wired_cfc_wiring(input_size):
    """Create wiring for ParamPFWiredCfCCell tests."""
    hidden_size = 32
    wiring = FullyConnected(units=hidden_size, output_dim=10)
    wiring.build(input_size)
    return wiring


# =============================================================================
# Tests for ParamPFCfCCell
# =============================================================================

class TestParamPFCfCCell:
    """Integration tests for ParamPFCfCCell."""

    def test_forward_output_shape(self, batch_size, hidden_size, input_size):
        """Forward produces correct output shape."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, (state, param_particles, log_weights) = cell(x)

        assert output.shape == (batch_size, hidden_size)
        assert state.shape == (batch_size, hidden_size)
        assert param_particles.shape[0] == batch_size
        assert param_particles.shape[1] == n_particles
        assert log_weights.shape == (batch_size, n_particles)

    def test_forward_without_state(self, batch_size, hidden_size, input_size):
        """Forward works without initial state."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, state = cell(x, hx=None)
        assert output.shape == (batch_size, hidden_size)

    def test_forward_with_state(self, batch_size, hidden_size, input_size):
        """Forward works with provided state."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        # Get initial state
        hx = cell.init_hidden(batch_size)

        output, (new_state, new_params, new_weights) = cell(x, hx=hx)
        assert output.shape == (batch_size, hidden_size)

    def test_sequence_processing(self, batch_size, hidden_size, input_size):
        """Cell processes sequences correctly."""
        n_particles = 8
        seq_len = 10
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        _, param_particles, log_weights = state
        # Weights should be valid
        sums = torch.exp(log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_param_registry_populated(self, hidden_size, input_size):
        """Parameter registry is populated with tracked params."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)

        assert cell.param_registry.n_groups > 0
        assert cell.param_registry.total_params > 0
        assert cell.param_registry.is_frozen

    def test_parameter_evolution(self, batch_size, hidden_size, input_size):
        """Parameter particles evolve over time."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        _, (_, params1, _) = cell(x)
        _, (_, params2, _) = cell(x)

        # Parameters should have evolved (due to noise)
        assert not torch.allclose(params1, params2)

    def test_with_observation_model(self, batch_size, hidden_size, input_size, obs_size):
        """Observation model updates weights."""
        n_particles = 8
        obs_model = GaussianObservationModel(hidden_size, obs_size)
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles, observation_model=obs_model)

        x = torch.randn(batch_size, input_size)
        obs = torch.randn(batch_size, obs_size)

        output, (state, params, log_weights) = cell(x, observation=obs)

        ess = compute_ess(log_weights)
        assert torch.all(ess >= 1.0 - 1e-4)
        assert torch.all(ess <= n_particles + 1e-4)

    def test_get_parameter_statistics(self, batch_size, hidden_size, input_size):
        """get_parameter_statistics returns valid stats."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        _, (_, param_particles, log_weights) = cell(x)
        stats = cell.get_parameter_statistics(param_particles, log_weights)

        assert "_ess" in stats
        for name in cell.param_registry.group_names:
            assert name in stats
            assert "mean" in stats[name]
            assert "variance" in stats[name]

    def test_return_all_particles(self, batch_size, hidden_size, input_size):
        """return_all_particles returns all outputs."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x, return_all_particles=True)
        assert output.shape == (batch_size, n_particles, hidden_size)

    def test_gradient_flow(self, hidden_size, input_size):
        """Gradients flow through cell."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(4, input_size, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_no_nan_inf(self, batch_size, hidden_size, input_size):
        """No NaN or Inf in outputs."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)

        state = None
        for _ in range(10):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Tests for ParamPFLTCCell
# =============================================================================

class TestParamPFLTCCell:
    """Integration tests for ParamPFLTCCell."""

    def test_forward_output_shape(self, batch_size, hidden_size, input_size, param_ltc_wiring):
        """Forward produces correct output shape."""
        n_particles = 8
        cell = ParamPFLTCCell(wiring=param_ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, (state, param_particles, log_weights) = cell(x)

        # Output size is motor_size from wiring
        assert output.shape[0] == batch_size
        assert state.shape == (batch_size, hidden_size)

    def test_sequence_processing(self, batch_size, hidden_size, input_size, param_ltc_wiring):
        """Cell processes sequences correctly."""
        n_particles = 8
        seq_len = 10
        cell = ParamPFLTCCell(wiring=param_ltc_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, hidden_size, input_size, param_ltc_wiring):
        """No NaN or Inf in outputs."""
        n_particles = 8
        cell = ParamPFLTCCell(wiring=param_ltc_wiring, n_particles=n_particles)

        state = None
        for _ in range(10):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Tests for ParamPFWiredCfCCell
# =============================================================================

class TestParamPFWiredCfCCell:
    """Integration tests for ParamPFWiredCfCCell."""

    def test_forward_output_shape(self, batch_size, input_size, param_wired_cfc_wiring):
        """Forward produces correct output shape."""
        n_particles = 8
        cell = ParamPFWiredCfCCell(wiring=param_wired_cfc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x)
        assert output.shape[0] == batch_size

    def test_sequence_processing(self, batch_size, input_size, param_wired_cfc_wiring):
        """Cell processes sequences correctly."""
        n_particles = 8
        seq_len = 10
        cell = ParamPFWiredCfCCell(wiring=param_wired_cfc_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, input_size, param_wired_cfc_wiring):
        """No NaN or Inf in outputs."""
        n_particles = 8
        cell = ParamPFWiredCfCCell(wiring=param_wired_cfc_wiring, n_particles=n_particles)

        state = None
        for _ in range(5):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Common Tests for All Param-Level Cells
# =============================================================================

class TestParamLevelCellsCommon:
    """Tests that apply to all param-level cells."""

    @pytest.fixture(params=["ParamPFCfCCell"])
    def cell_class(self, request):
        """Parameterized cell class - only CfC uses direct init."""
        return {
            "ParamPFCfCCell": ParamPFCfCCell,
        }[request.param]

    def test_is_subclass(self, cell_class, input_size, hidden_size):
        """All cells are subclasses of ParamLevelPFCell."""
        n_particles = 8
        cell = cell_class(input_size, hidden_size, n_particles)
        assert isinstance(cell, ParamLevelPFCell)

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

        state, param_particles, log_weights = cell.init_hidden(batch_size)

        assert state.shape == (batch_size, hidden_size)
        assert param_particles.shape == (batch_size, n_particles, cell.param_registry.total_params)
        # Uniform weights
        assert torch.allclose(
            torch.exp(log_weights).sum(dim=-1),
            torch.ones(batch_size),
            atol=1e-4
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestParamLevelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self, batch_size, hidden_size, input_size):
        """K=1 works."""
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles=1)
        x = torch.randn(batch_size, input_size)

        output, (state, params, weights) = cell(x)
        assert output.shape == (batch_size, hidden_size)
        assert params.shape[1] == 1

    def test_single_batch(self, hidden_size, input_size):
        """Batch size 1 works."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(1, input_size)

        output, _ = cell(x)
        assert output.shape == (1, hidden_size)

    def test_long_sequence(self, hidden_size, input_size):
        """Long sequence doesn't cause numerical issues."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        batch_size = 4
        seq_len = 50

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        _, param_particles, log_weights = state
        assert not torch.isnan(param_particles).any()
        assert not torch.isnan(log_weights).any()

    def test_with_timespans(self, batch_size, hidden_size, input_size):
        """Timespans affect computation."""
        n_particles = 8
        cell = ParamPFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        ts1 = torch.tensor(0.1)
        ts2 = torch.tensor(1.0)

        output1, _ = cell(x, timespans=ts1)
        output2, _ = cell(x, timespans=ts2)

        # Outputs should differ
        assert not torch.allclose(output1, output2)
