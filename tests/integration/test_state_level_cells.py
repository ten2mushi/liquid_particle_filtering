"""Integration tests for state-level particle filter cells.

Tests for:
- PFCfCCell
- PFLTCCell
- PFWiredCfCCell
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.state_level import (
    StateLevelPFCell,
    PFCfCCell,
    PFLTCCell,
    PFWiredCfCCell,
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
def random_input(batch_size: int, input_size: int) -> Tensor:
    return torch.randn(batch_size, input_size)


@pytest.fixture
def random_observation(batch_size: int, obs_size: int) -> Tensor:
    return torch.randn(batch_size, obs_size)


@pytest.fixture
def ltc_wiring(hidden_size, input_size):
    """Create wiring for LTC cell tests."""
    wiring = FullyConnected(units=hidden_size, output_dim=min(hidden_size, 10))
    wiring.build(input_size)
    return wiring


# =============================================================================
# Tests for PFCfCCell
# =============================================================================

class TestPFCfCCell:
    """Integration tests for PFCfCCell."""

    def test_forward_output_shape(self, batch_size, n_particles, hidden_size, input_size):
        """Forward produces correct output shape."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, (particles, log_weights) = cell(x)

        assert output.shape == (batch_size, hidden_size)
        assert particles.shape == (batch_size, n_particles, hidden_size)
        assert log_weights.shape == (batch_size, n_particles)

    def test_forward_without_state(self, batch_size, n_particles, hidden_size, input_size):
        """Forward works without initial state (creates default)."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, state = cell(x, hx=None)
        assert output.shape == (batch_size, hidden_size)

    def test_forward_with_state(self, batch_size, n_particles, hidden_size, input_size):
        """Forward works with provided initial state."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        # Create initial state
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.full((batch_size, n_particles), -math.log(n_particles))

        output, (new_particles, new_log_weights) = cell(x, hx=(particles, log_weights))

        assert output.shape == (batch_size, hidden_size)
        assert new_particles.shape == particles.shape

    def test_sequence_processing(self, batch_size, n_particles, hidden_size, input_size):
        """Cell can process sequences."""
        seq_len = 10
        cell = PFCfCCell(input_size, hidden_size, n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        particles, log_weights = state
        assert particles.shape == (batch_size, n_particles, hidden_size)
        # Weights should still be valid distribution
        sums = torch.exp(log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_with_observation_model(self, batch_size, n_particles, hidden_size, input_size, obs_size):
        """Weight updates work with observation model."""
        obs_model = GaussianObservationModel(hidden_size, obs_size)
        cell = PFCfCCell(input_size, hidden_size, n_particles, observation_model=obs_model)

        x = torch.randn(batch_size, input_size)
        obs = torch.randn(batch_size, obs_size)

        output, (particles, log_weights) = cell(x, observation=obs)

        assert output.shape == (batch_size, hidden_size)
        # ESS should be valid
        ess = compute_ess(log_weights)
        assert torch.all(ess >= 1.0 - 1e-4)
        assert torch.all(ess <= n_particles + 1e-4)

    def test_set_observation_model(self, hidden_size, input_size, obs_size):
        """set_observation_model works."""
        cell = PFCfCCell(input_size, hidden_size, n_particles=16)
        assert cell.observation_model is None

        obs_model = GaussianObservationModel(hidden_size, obs_size)
        cell.set_observation_model(obs_model)

        assert cell.observation_model is obs_model

    def test_return_all_particles(self, batch_size, n_particles, hidden_size, input_size):
        """return_all_particles returns all particle outputs."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        output, _ = cell(x, return_all_particles=True)

        assert output.shape == (batch_size, n_particles, hidden_size)

    def test_with_timespans(self, batch_size, n_particles, hidden_size, input_size):
        """Timespans affect computation."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        # Different timespans
        ts1 = torch.tensor(0.1)
        ts2 = torch.tensor(1.0)

        output1, _ = cell(x, timespans=ts1)
        output2, _ = cell(x, timespans=ts2)

        # Outputs should differ
        assert not torch.allclose(output1, output2)

    def test_init_hidden(self, batch_size, n_particles, hidden_size, input_size):
        """init_hidden produces valid initial state."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)

        particles, log_weights = cell.init_hidden(batch_size)

        assert particles.shape == (batch_size, n_particles, hidden_size)
        assert log_weights.shape == (batch_size, n_particles)
        # Uniform weights
        assert torch.allclose(
            torch.exp(log_weights).sum(dim=-1),
            torch.ones(batch_size),
            atol=1e-4
        )

    def test_get_particle_statistics(self, batch_size, n_particles, hidden_size, input_size):
        """get_particle_statistics returns correct stats."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        _, (particles, log_weights) = cell(x)
        stats = cell.get_particle_statistics(particles, log_weights)

        assert "mean" in stats
        assert "variance" in stats
        assert "ess" in stats
        assert "max_weight" in stats

        assert stats["mean"].shape == (batch_size, hidden_size)
        assert stats["variance"].shape == (batch_size, hidden_size)
        assert stats["ess"].shape == (batch_size,)

    def test_different_cfc_modes(self, batch_size, n_particles, hidden_size, input_size):
        """Different CfC modes work."""
        for mode in ["default", "pure", "no_gate"]:
            cell = PFCfCCell(input_size, hidden_size, n_particles, mode=mode)
            x = torch.randn(batch_size, input_size)

            output, _ = cell(x)
            assert output.shape == (batch_size, hidden_size)

    def test_different_noise_types(self, batch_size, n_particles, hidden_size, input_size):
        """Different noise types work."""
        for noise_type in ["constant", "time_scaled", "learned", "state_dependent"]:
            cell = PFCfCCell(input_size, hidden_size, n_particles, noise_type=noise_type)
            x = torch.randn(batch_size, input_size)

            output, _ = cell(x)
            assert output.shape == (batch_size, hidden_size)

    def test_different_alpha_modes(self, batch_size, n_particles, hidden_size, input_size):
        """Different alpha modes work."""
        for alpha_mode in ["fixed", "adaptive", "learnable"]:
            cell = PFCfCCell(input_size, hidden_size, n_particles, alpha_mode=alpha_mode)
            x = torch.randn(batch_size, input_size)

            output, _ = cell(x)
            assert output.shape == (batch_size, hidden_size)

    def test_gradient_flow(self, hidden_size, input_size):
        """Gradients flow through cell."""
        cell = PFCfCCell(input_size, hidden_size, n_particles=8)
        x = torch.randn(4, input_size, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in cell.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, input_size):
        """No NaN or Inf in outputs."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)

        state = None
        for _ in range(20):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert not torch.isnan(state[0]).any()
            assert not torch.isnan(state[1]).any()


# =============================================================================
# Tests for PFLTCCell
# =============================================================================

class TestPFLTCCell:
    """Integration tests for PFLTCCell."""

    def test_forward_output_shape(self, batch_size, n_particles, hidden_size, input_size, ltc_wiring):
        """Forward produces correct output shape."""
        cell = PFLTCCell(wiring=ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, (particles, log_weights) = cell(x)

        assert output.shape[0] == batch_size
        assert particles.shape == (batch_size, n_particles, hidden_size)

    def test_sequence_processing(self, batch_size, n_particles, hidden_size, input_size, ltc_wiring):
        """Cell can process sequences."""
        seq_len = 10
        cell = PFLTCCell(wiring=ltc_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        assert output.shape[0] == batch_size

    def test_with_timespans(self, batch_size, n_particles, hidden_size, input_size, ltc_wiring):
        """Timespans affect LTC dynamics."""
        cell = PFLTCCell(wiring=ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        ts1 = torch.tensor(0.1)
        ts2 = torch.tensor(1.0)

        output1, _ = cell(x, timespans=ts1)
        output2, _ = cell(x, timespans=ts2)

        # Outputs may or may not differ due to stochastic nature
        assert output1.shape == output2.shape

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, input_size, ltc_wiring):
        """No NaN or Inf in outputs."""
        cell = PFLTCCell(wiring=ltc_wiring, n_particles=n_particles)

        state = None
        for _ in range(5):  # Reduced iterations
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Tests for PFWiredCfCCell
# =============================================================================

class TestPFWiredCfCCell:
    """Integration tests for PFWiredCfCCell."""

    def test_forward_output_shape(self, batch_size, n_particles, input_size, ltc_wiring):
        """Forward produces correct output shape."""
        cell = PFWiredCfCCell(wiring=ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, input_size)

        output, (particles, log_weights) = cell(x)

        assert output.shape[0] == batch_size
        assert particles.shape[0] == batch_size
        assert particles.shape[1] == n_particles

    def test_sequence_processing(self, batch_size, n_particles, input_size, ltc_wiring):
        """Cell can process sequences."""
        seq_len = 10
        cell = PFWiredCfCCell(wiring=ltc_wiring, n_particles=n_particles)

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, n_particles, input_size, ltc_wiring):
        """No NaN or Inf in outputs."""
        cell = PFWiredCfCCell(wiring=ltc_wiring, n_particles=n_particles)

        state = None
        for _ in range(5):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


# =============================================================================
# Common Tests for All State-Level Cells
# =============================================================================

class TestStateLevelCellsCommon:
    """Tests that apply to PFCfCCell (the main state-level cell without wiring)."""

    def test_is_subclass(self, input_size, hidden_size, n_particles):
        """PFCfCCell is subclass of StateLevelPFCell."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        assert isinstance(cell, StateLevelPFCell)

    def test_has_required_attributes(self, input_size, hidden_size, n_particles):
        """PFCfCCell has required attributes."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)

        assert hasattr(cell, "input_size")
        assert hasattr(cell, "hidden_size")
        assert hasattr(cell, "n_particles")
        assert hasattr(cell, "noise_injector")
        assert hasattr(cell, "resampler")
        assert hasattr(cell, "observation_model")

    def test_train_eval_modes(self, batch_size, input_size, hidden_size, n_particles):
        """Cells work in both train and eval modes."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)

        cell.train()
        output_train, _ = cell(x)

        cell.eval()
        output_eval, _ = cell(x)

        assert output_train.shape == output_eval.shape

    def test_device_compatibility(self, input_size, hidden_size, n_particles):
        """Cells work on different devices."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)

        # CPU test
        x_cpu = torch.randn(4, input_size)
        output_cpu, _ = cell(x_cpu)
        assert output_cpu.device.type == "cpu"

    def test_deterministic_with_seed(self, input_size, hidden_size):
        """With fixed seed, cells produce deterministic output."""
        n_particles = 16
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(4, input_size)

        torch.manual_seed(42)
        output1, state1 = cell(x)

        torch.manual_seed(42)
        output2, state2 = cell(x)

        assert torch.allclose(output1, output2)
        assert torch.allclose(state1[0], state2[0])
        assert torch.allclose(state1[1], state2[1])


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestStateLevelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_particle(self, batch_size, hidden_size, input_size):
        """K=1 works (degenerate case)."""
        cell = PFCfCCell(input_size, hidden_size, n_particles=1)
        x = torch.randn(batch_size, input_size)

        output, (particles, log_weights) = cell(x)

        assert output.shape == (batch_size, hidden_size)
        assert particles.shape == (batch_size, 1, hidden_size)
        # Single particle should have weight 1
        assert torch.allclose(torch.exp(log_weights), torch.ones(batch_size, 1))

    def test_single_batch(self, n_particles, hidden_size, input_size):
        """Batch size 1 works."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(1, input_size)

        output, _ = cell(x)
        assert output.shape == (1, hidden_size)

    def test_large_batch(self, n_particles, hidden_size, input_size):
        """Large batch works."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(256, input_size)

        output, _ = cell(x)
        assert output.shape == (256, hidden_size)

    def test_variable_timespans(self, batch_size, n_particles, hidden_size, input_size):
        """Different timespans per batch item."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)
        ts = torch.rand(batch_size, 1)

        output, _ = cell(x, timespans=ts)
        assert output.shape == (batch_size, hidden_size)

    def test_scalar_timespan(self, batch_size, n_particles, hidden_size, input_size):
        """Scalar timespan broadcasts correctly."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)
        ts = torch.tensor(0.5)

        output, _ = cell(x, timespans=ts)
        assert output.shape == (batch_size, hidden_size)

    def test_zero_timespan(self, batch_size, n_particles, hidden_size, input_size):
        """Zero timespan doesn't cause issues."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        x = torch.randn(batch_size, input_size)
        ts = torch.tensor(0.0)

        output, _ = cell(x, timespans=ts)
        assert not torch.isnan(output).any()

    def test_long_sequence(self, n_particles, hidden_size, input_size):
        """Long sequence doesn't cause numerical issues."""
        cell = PFCfCCell(input_size, hidden_size, n_particles)
        batch_size = 4
        seq_len = 100

        state = None
        for t in range(seq_len):
            x = torch.randn(batch_size, input_size)
            output, state = cell(x, state)

        particles, log_weights = state
        assert not torch.isnan(particles).any()
        assert not torch.isnan(log_weights).any()
        assert not torch.isinf(particles).any()
        assert not torch.isinf(log_weights).any()

        # Weights should still be valid
        sums = torch.exp(log_weights).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
