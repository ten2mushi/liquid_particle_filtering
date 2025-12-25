"""Integration tests for sequence wrappers.

Tests for:
- PFCfC (doesn't require wiring)
- PFLTC (requires wiring)
- PFNCP (requires wiring)
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.wrappers import PFCfC, PFLTC, PFNCP
from pfncps.nn.observation import GaussianObservationModel
from pfncps.wirings import FullyConnected


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def input_size() -> int:
    return 20


@pytest.fixture
def output_size() -> int:
    return 10


@pytest.fixture
def seq_len() -> int:
    return 25


@pytest.fixture
def random_sequence(batch_size: int, seq_len: int, input_size: int) -> Tensor:
    return torch.randn(batch_size, seq_len, input_size)


@pytest.fixture
def ltc_wiring(hidden_size, input_size):
    """Create wiring for PFLTC/PFNCP tests."""
    wiring = FullyConnected(units=hidden_size, output_dim=min(hidden_size, 10))
    wiring.build(input_size)
    return wiring


# =============================================================================
# Tests for PFCfC Wrapper
# =============================================================================

class TestPFCfCWrapper:
    """Integration tests for PFCfC sequence wrapper."""

    def test_forward_output_shape(self, batch_size, seq_len, hidden_size, input_size):
        """Forward produces correct output shape."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)

        output, state = model(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_return_sequences_true(self, batch_size, seq_len, hidden_size, input_size):
        """return_sequences=True returns all timestep outputs."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles, return_sequences=True)
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_return_sequences_false(self, batch_size, seq_len, hidden_size, input_size):
        """return_sequences=False returns only final output."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles, return_sequences=False)
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)

        assert output.shape == (batch_size, hidden_size)

    def test_state_approach(self, batch_size, seq_len, hidden_size, input_size):
        """State approach works."""
        model = PFCfC(input_size, hidden_size, n_particles=16, approach="state")
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_with_timespans(self, batch_size, seq_len, hidden_size, input_size):
        """Timespans are passed through wrapper."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)
        ts = torch.rand(batch_size, seq_len, 1)

        output, _ = model(x, timespans=ts)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_stateful_processing(self, batch_size, seq_len, hidden_size, input_size):
        """Stateful processing maintains state between calls."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)

        x1 = torch.randn(batch_size, seq_len // 2, input_size)
        x2 = torch.randn(batch_size, seq_len // 2, input_size)

        output1, state = model(x1)
        output2, _ = model(x2, hx=state)

        # Both should produce valid outputs
        assert output1.shape == (batch_size, seq_len // 2, hidden_size)
        assert output2.shape == (batch_size, seq_len // 2, hidden_size)

    def test_gradient_flow(self, hidden_size, input_size):
        """Gradients flow through wrapper."""
        n_particles = 8
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(4, 10, input_size, requires_grad=True)

        output, _ = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_no_nan_inf(self, batch_size, seq_len, hidden_size, input_size):
        """No NaN or Inf in outputs."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Tests for PFLTC Wrapper
# =============================================================================

class TestPFLTCWrapper:
    """Integration tests for PFLTC sequence wrapper."""

    def test_forward_output_shape(self, batch_size, seq_len, hidden_size, input_size, ltc_wiring):
        """Forward produces correct output shape."""
        n_particles = 16
        model = PFLTC(wiring=ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)

        output, state = model(x)

        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len

    def test_return_sequences_false(self, batch_size, seq_len, hidden_size, input_size, ltc_wiring):
        """return_sequences=False returns only final output."""
        n_particles = 16
        model = PFLTC(wiring=ltc_wiring, n_particles=n_particles, return_sequences=False)
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)

        assert output.shape[0] == batch_size

    def test_state_approach(self, batch_size, seq_len, hidden_size, input_size, ltc_wiring):
        """State approach works."""
        model = PFLTC(wiring=ltc_wiring, n_particles=16, approach="state")
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)
        assert output.shape[0] == batch_size

    def test_sde_approach(self, batch_size, seq_len, hidden_size, input_size, ltc_wiring):
        """SDE approach (D) works through LTC wrapper."""
        n_particles = 16
        model = PFLTC(wiring=ltc_wiring, n_particles=n_particles, approach="sde")
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, seq_len, hidden_size, input_size, ltc_wiring):
        """No NaN or Inf in outputs."""
        n_particles = 16
        model = PFLTC(wiring=ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, 5, input_size)  # Shorter sequence

        output, _ = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Tests for PFNCP Wrapper
# =============================================================================

class TestPFNCPWrapper:
    """Integration tests for PFNCP sequence wrapper."""

    def test_forward_output_shape(self, batch_size, seq_len, input_size, ltc_wiring):
        """Forward produces correct output shape."""
        n_particles = 16
        model = PFNCP(wiring=ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)

        output, state = model(x)

        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len

    def test_return_sequences_false(self, batch_size, seq_len, input_size, ltc_wiring):
        """return_sequences=False returns only final output."""
        n_particles = 16
        model = PFNCP(wiring=ltc_wiring, n_particles=n_particles, return_sequences=False)
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)

        assert output.shape[0] == batch_size

    def test_no_nan_inf(self, batch_size, seq_len, input_size, ltc_wiring):
        """No NaN or Inf in outputs."""
        n_particles = 16
        model = PFNCP(wiring=ltc_wiring, n_particles=n_particles)
        x = torch.randn(batch_size, 5, input_size)

        output, _ = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# =============================================================================
# Common Tests for Wrappers
# =============================================================================

class TestWrappersCommon:
    """Tests that apply to PFCfC wrapper (the one without wiring)."""

    def test_is_module(self, input_size, hidden_size):
        """PFCfC is nn.Module."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        assert isinstance(model, nn.Module)

    def test_train_eval_modes(self, batch_size, seq_len, input_size, hidden_size):
        """Wrappers work in both train and eval modes."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)

        model.train()
        output_train, _ = model(x)

        model.eval()
        output_eval, _ = model(x)

        assert output_train.shape == output_eval.shape

    def test_batch_first(self, batch_size, seq_len, input_size, hidden_size):
        """Default is batch_first=True."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)

        output, _ = model(x)
        assert output.shape[0] == batch_size


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestWrapperEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_timestep(self, batch_size, hidden_size, input_size):
        """Single timestep works."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, 1, input_size)

        output, _ = model(x)
        assert output.shape == (batch_size, 1, hidden_size)

    def test_single_batch(self, seq_len, hidden_size, input_size):
        """Batch size 1 works."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(1, seq_len, input_size)

        output, _ = model(x)
        assert output.shape == (1, seq_len, hidden_size)

    def test_long_sequence(self, batch_size, hidden_size, input_size):
        """Long sequence works."""
        n_particles = 16
        seq_len = 50
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)

        output, state = model(x)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    def test_variable_timespans(self, batch_size, seq_len, hidden_size, input_size):
        """Different timespans per step work."""
        n_particles = 16
        model = PFCfC(input_size, hidden_size, n_particles=n_particles)
        x = torch.randn(batch_size, seq_len, input_size)
        ts = torch.rand(batch_size, seq_len, 1)

        output, _ = model(x, timespans=ts)
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_observations_sequence(self, batch_size, seq_len, hidden_size, input_size):
        """Observations sequence is handled."""
        obs_size = 10
        n_particles = 16
        obs_model = GaussianObservationModel(hidden_size, obs_size)
        model = PFCfC(
            input_size, hidden_size, n_particles=n_particles,
            observation_model=obs_model
        )

        x = torch.randn(batch_size, seq_len, input_size)
        obs = torch.randn(batch_size, seq_len, obs_size)

        output, _ = model(x, observations=obs)
        assert output.shape == (batch_size, seq_len, hidden_size)
