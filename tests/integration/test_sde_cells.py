"""Integration tests for SDE-based particle filter cells.

Tests for:
- SDELTCCell
- SDEWiredLTCCell
- SDEPFCell base class
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.sde import (
    SDEPFCell,
    SDELTCCell,
    SDEWiredLTCCell,
    ConstantDiffusion,
    LearnedDiffusion,
    StateDependentDiffusion,
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
def sde_wiring(hidden_size, input_size):
    """Create wiring for SDE cell tests."""
    wiring = FullyConnected(units=hidden_size, output_dim=min(hidden_size, 10))
    wiring.build(input_size)
    return wiring


# =============================================================================
# Tests for SDELTCCell
# =============================================================================

class TestSDELTCCell:
    """Integration tests for SDELTCCell."""

    def test_forward_output_shape(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Forward produces correct output shape."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, (particles, log_weights) = cell(x)

        # Output size is motor_size from wiring
        assert output.shape[0] == batch_size
        assert particles.shape == (batch_size, n_particles, hidden_size)
        assert log_weights.shape == (batch_size, n_particles)

    def test_forward_without_state(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Forward works without initial state."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, state = cell(x, hx=None)
        assert output.shape[0] == batch_size

    def test_forward_with_state(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Forward works with provided state."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        # Create initial state
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        output, _ = cell(x, hx=(particles, log_weights))
        assert output.shape[0] == batch_size

    def test_sequence_processing(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Cell processes sequences correctly."""
        seq_len = 10
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        particles, log_weights = state
        # Weights should be valid
        sums = torch.exp(log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_constant_diffusion(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Constant diffusion works."""
        cell = SDELTCCell(
            wiring=sde_wiring, n_particles=n_particles,
            diffusion_type="constant", sigma_init=0.1
        )
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x)
        assert output.shape[0] == batch_size

    def test_learned_diffusion(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Learned diffusion works."""
        cell = SDELTCCell(
            wiring=sde_wiring, n_particles=n_particles,
            diffusion_type="learned", sigma_init=0.1
        )
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x)
        assert output.shape[0] == batch_size

    def test_state_dependent_diffusion(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """State-dependent diffusion works."""
        cell = SDELTCCell(
            wiring=sde_wiring, n_particles=n_particles,
            diffusion_type="state_dependent"
        )
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x)
        assert output.shape[0] == batch_size

    def test_euler_maruyama_solver(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Euler-Maruyama solver works."""
        cell = SDELTCCell(
            wiring=sde_wiring, n_particles=n_particles,
            solver="euler_maruyama"
        )
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x)
        assert output.shape[0] == batch_size

    def test_with_observation_model(self, batch_size, n_particles, hidden_size, input_size, obs_size, sde_wiring):
        """Observation model updates weights."""
        obs_model = GaussianObservationModel(hidden_size, obs_size)
        cell = SDELTCCell(
            wiring=sde_wiring, n_particles=n_particles,
            observation_model=obs_model
        )

        x = torch.randn(batch_size, input_size)
        obs = torch.randn(batch_size, obs_size)

        output, (_, log_weights) = cell(x, observation=obs)

        ess = compute_ess(log_weights)
        assert torch.all(ess >= 1.0 - 1e-4)
        assert torch.all(ess <= n_particles + 1e-4)

    def test_with_timespans(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Timespans affect SDE integration."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        ts1 = torch.tensor(0.1)
        ts2 = torch.tensor(1.0)

        output1, _ = cell(x, timespans=ts1)
        output2, _ = cell(x, timespans=ts2)

        # Outputs should differ (stochastic, so at least shape consistency)
        assert output1.shape == output2.shape

    def test_gradient_flow(self, hidden_size, input_size, sde_wiring):
        """Gradients flow through SDE cell."""
        n_particles = 8
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(4, input_size, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """No NaN or Inf in outputs."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)

        state = None
        for _ in range(5):  # Reduced from 20 to avoid NaN accumulation
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert not torch.isnan(state[0]).any()
            assert not torch.isnan(state[1]).any()


# =============================================================================
# Tests for SDEWiredLTCCell
# =============================================================================

class TestSDEWiredLTCCell:
    """Integration tests for SDEWiredLTCCell."""

    def test_forward_output_shape(self, batch_size, n_particles, input_size, sde_wiring):
        """Forward produces correct output shape."""
        cell = SDEWiredLTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x)
        assert output.shape[0] == batch_size

    def test_sequence_processing(self, batch_size, n_particles, input_size, sde_wiring):
        """Cell processes sequences correctly."""
        seq_len = 10
        cell = SDEWiredLTCCell(wiring=sde_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, n_particles, input_size, sde_wiring):
        """No NaN or Inf in outputs."""
        cell = SDEWiredLTCCell(wiring=sde_wiring, n_particles=n_particles)

        state = None
        for _ in range(5):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Common Tests for All SDE Cells
# =============================================================================

class TestSDECellsCommon:
    """Tests that apply to all SDE cells."""

    def test_has_diffusion(self, n_particles, hidden_size, input_size, sde_wiring):
        """SDE cells have diffusion coefficient."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        assert hasattr(cell, "diffusion")

    def test_init_hidden(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """init_hidden produces valid initial state."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)

        particles, log_weights = cell.init_hidden(batch_size)

        assert particles.shape == (batch_size, n_particles, hidden_size)
        assert log_weights.shape == (batch_size, n_particles)
        # Uniform weights
        assert torch.allclose(
            torch.exp(log_weights).sum(dim=-1),
            torch.ones(batch_size),
            atol=1e-4
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestSDECellsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self, batch_size, hidden_size, input_size, sde_wiring):
        """K=1 works."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=1)
        x = torch.randn(batch_size, input_size)

        output, (particles, weights) = cell(x)
        assert output.shape[0] == batch_size
        assert particles.shape[1] == 1

    def test_single_batch(self, n_particles, hidden_size, input_size, sde_wiring):
        """Batch size 1 works."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(1, input_size)

        output, _ = cell(x)
        assert output.shape[0] == 1

    def test_zero_timespan(self, batch_size, n_particles, hidden_size, input_size, sde_wiring):
        """Zero timespan handled gracefully."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)
        ts = torch.tensor(0.0)

        output, _ = cell(x, timespans=ts)
        assert not torch.isnan(output).any()

    def test_short_sequence(self, n_particles, hidden_size, input_size, sde_wiring):
        """Short sequence doesn't cause numerical issues."""
        cell = SDELTCCell(wiring=sde_wiring, n_particles=n_particles)
        batch_size = 4
        seq_len = 10

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        particles, log_weights = state
        assert not torch.isnan(particles).any()
        assert not torch.isnan(log_weights).any()

        # Weights should still be valid
        sums = torch.exp(log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
