"""Unit tests for diffusion coefficients (nn/sde/diffusion.py).

Tests for:
- ConstantDiffusion
- LearnedDiffusion
- StateDependentDiffusion
- TimeVaryingDiffusion
- create_diffusion factory function
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.sde.diffusion import (
    DiffusionType,
    DiffusionCoefficient,
    ConstantDiffusion,
    LearnedDiffusion,
    StateDependentDiffusion,
    TimeVaryingDiffusion,
    create_diffusion,
)


# =============================================================================
# Tests for ConstantDiffusion
# =============================================================================

class TestConstantDiffusion:
    """Tests for the ConstantDiffusion class."""

    def test_output_shape(self, batch_size, hidden_size):
        """Output has shape [batch, hidden_size]."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert output.shape == (batch_size, hidden_size)

    def test_constant_value(self, batch_size, hidden_size):
        """Output is constant regardless of state."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.5, per_dimension=False)
        state1 = torch.randn(batch_size, hidden_size)
        state2 = torch.randn(batch_size, hidden_size) * 100

        output1 = diffusion(state1)
        output2 = diffusion(state2)

        assert torch.allclose(output1, output2)
        assert torch.allclose(output1, torch.full_like(output1, 0.5))

    def test_per_dimension_sigma(self, batch_size, hidden_size):
        """Per-dimension sigma creates vector of values."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.3, per_dimension=True)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert output.shape == (batch_size, hidden_size)
        assert torch.allclose(output, torch.full_like(output, 0.3))

    def test_is_diagonal(self, hidden_size):
        """ConstantDiffusion is diagonal."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        assert diffusion.is_diagonal

    def test_ignores_time(self, batch_size, hidden_size):
        """Time parameter is ignored."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        state = torch.randn(batch_size, hidden_size)
        t = torch.randn(batch_size, 1)

        output_no_t = diffusion(state)
        output_with_t = diffusion(state, t)

        assert torch.allclose(output_no_t, output_with_t)

    def test_positive_output(self, batch_size, hidden_size):
        """Output is positive."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert torch.all(output > 0)


# =============================================================================
# Tests for LearnedDiffusion
# =============================================================================

class TestLearnedDiffusion:
    """Tests for the LearnedDiffusion class."""

    def test_output_shape(self, batch_size, hidden_size):
        """Output has shape [batch, hidden_size]."""
        diffusion = LearnedDiffusion(hidden_size, sigma_init=0.1)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert output.shape == (batch_size, hidden_size)

    def test_initial_sigma(self, hidden_size):
        """Initial sigma is approximately sigma_init."""
        diffusion = LearnedDiffusion(hidden_size, sigma_init=0.5)

        assert torch.allclose(diffusion.sigma, torch.full((hidden_size,), 0.5), atol=0.1)

    def test_sigma_bounds(self, batch_size, hidden_size):
        """Sigma is bounded by min/max."""
        diffusion = LearnedDiffusion(
            hidden_size, sigma_init=0.1, min_sigma=0.01, max_sigma=1.0
        )

        # Manually set extreme log_sigma values
        with torch.no_grad():
            diffusion.log_sigma.fill_(-100)  # Very small

        state = torch.randn(batch_size, hidden_size)
        output = diffusion(state)
        assert torch.all(output >= 0.01)

        with torch.no_grad():
            diffusion.log_sigma.fill_(100)  # Very large

        output = diffusion(state)
        assert torch.all(output <= 1.0)

    def test_is_diagonal(self, hidden_size):
        """LearnedDiffusion is diagonal."""
        diffusion = LearnedDiffusion(hidden_size, sigma_init=0.1)
        assert diffusion.is_diagonal

    def test_learnable_parameters(self, hidden_size):
        """log_sigma is a learnable parameter."""
        diffusion = LearnedDiffusion(hidden_size, sigma_init=0.1)

        assert hasattr(diffusion, "log_sigma")
        assert isinstance(diffusion.log_sigma, nn.Parameter)

    def test_gradient_flow(self, hidden_size):
        """Gradients flow through learned parameters."""
        diffusion = LearnedDiffusion(hidden_size, sigma_init=0.1)
        state = torch.randn(4, hidden_size)

        output = diffusion(state)
        loss = output.sum()
        loss.backward()

        assert diffusion.log_sigma.grad is not None


# =============================================================================
# Tests for StateDependentDiffusion
# =============================================================================

class TestStateDependentDiffusion:
    """Tests for the StateDependentDiffusion class."""

    def test_output_shape(self, batch_size, hidden_size):
        """Output has shape [batch, hidden_size]."""
        diffusion = StateDependentDiffusion(hidden_size)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert output.shape == (batch_size, hidden_size)

    def test_state_dependent(self, batch_size, hidden_size):
        """Different states produce different outputs after MLP weights are non-zero."""
        diffusion = StateDependentDiffusion(hidden_size)

        # Re-initialize weights with non-zero values so state-dependence works
        # (default init sets last layer weights to zero for consistent initial output)
        for module in diffusion.mlp.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)

        state1 = torch.zeros(batch_size, hidden_size)
        state2 = torch.ones(batch_size, hidden_size) * 5

        output1 = diffusion(state1)
        output2 = diffusion(state2)

        # Should be different with non-zero MLP weights
        assert not torch.allclose(output1, output2)

    def test_sigma_bounds(self, batch_size, hidden_size):
        """Sigma is bounded by min/max."""
        diffusion = StateDependentDiffusion(
            hidden_size, min_sigma=0.01, max_sigma=1.0
        )
        state = torch.randn(batch_size, hidden_size) * 100  # Extreme state

        output = diffusion(state)
        assert torch.all(output >= 0.01)
        assert torch.all(output <= 1.0)

    def test_is_diagonal(self, hidden_size):
        """StateDependentDiffusion is diagonal."""
        diffusion = StateDependentDiffusion(hidden_size)
        assert diffusion.is_diagonal

    def test_positive_output(self, batch_size, hidden_size):
        """Output is always positive (due to Softplus)."""
        diffusion = StateDependentDiffusion(hidden_size)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert torch.all(output > 0)

    def test_custom_mlp_layers(self, batch_size, hidden_size):
        """Custom MLP configuration works."""
        diffusion = StateDependentDiffusion(
            hidden_size, mlp_hidden=128, mlp_layers=3
        )
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert output.shape == (batch_size, hidden_size)

    def test_different_activations(self, batch_size, hidden_size):
        """Different activation functions work."""
        for activation in ["tanh", "relu", "gelu"]:
            diffusion = StateDependentDiffusion(
                hidden_size, activation=activation
            )
            state = torch.randn(batch_size, hidden_size)

            output = diffusion(state)
            assert output.shape == (batch_size, hidden_size)

    def test_gradient_flow(self, hidden_size):
        """Gradients flow through MLP."""
        diffusion = StateDependentDiffusion(hidden_size)
        state = torch.randn(4, hidden_size, requires_grad=True)

        output = diffusion(state)
        loss = output.sum()
        loss.backward()

        assert state.grad is not None
        for param in diffusion.parameters():
            if param.requires_grad:
                assert param.grad is not None


# =============================================================================
# Tests for TimeVaryingDiffusion
# =============================================================================

class TestTimeVaryingDiffusion:
    """Tests for the TimeVaryingDiffusion class."""

    def test_output_shape(self, batch_size, hidden_size):
        """Output has shape [batch, hidden_size]."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.1)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state)
        assert output.shape == (batch_size, hidden_size)

    def test_sqrt_time_scaling(self, batch_size, hidden_size):
        """sqrt time scaling works correctly."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.5, time_scaling="sqrt")
        state = torch.randn(batch_size, hidden_size)

        t1 = torch.tensor(1.0)
        t4 = torch.tensor(4.0)

        output1 = diffusion(state, t1)
        output4 = diffusion(state, t4)

        # sqrt(4) = 2, so output4 should be ~2x output1
        assert torch.allclose(output4, output1 * 2, atol=0.1)

    def test_linear_time_scaling(self, batch_size, hidden_size):
        """Linear time scaling works correctly."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.5, time_scaling="linear")
        state = torch.randn(batch_size, hidden_size)

        t1 = torch.tensor(1.0)
        t2 = torch.tensor(2.0)

        output1 = diffusion(state, t1)
        output2 = diffusion(state, t2)

        # Output should scale linearly with time
        assert torch.allclose(output2, output1 * 2, atol=0.1)

    def test_no_time_scaling(self, batch_size, hidden_size):
        """No time scaling when time_scaling='none'."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.5, time_scaling="none")
        state = torch.randn(batch_size, hidden_size)

        t1 = torch.tensor(0.1)
        t10 = torch.tensor(10.0)

        output1 = diffusion(state, t1)
        output10 = diffusion(state, t10)

        # Should be the same regardless of time
        assert torch.allclose(output1, output10)

    def test_no_time_provided(self, batch_size, hidden_size):
        """Works when no time is provided."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.5)
        state = torch.randn(batch_size, hidden_size)

        output = diffusion(state, t=None)
        assert output.shape == (batch_size, hidden_size)

    def test_is_diagonal(self, hidden_size):
        """TimeVaryingDiffusion is diagonal."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.1)
        assert diffusion.is_diagonal

    def test_learnable_mode(self, hidden_size):
        """Learnable mode has trainable parameters."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.1, learnable=True)

        assert isinstance(diffusion.log_sigma, nn.Parameter)

    def test_fixed_mode(self, hidden_size):
        """Fixed mode has no trainable sigma."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.1, learnable=False)

        assert not isinstance(diffusion.log_sigma, nn.Parameter)

    def test_time_shape_handling(self, batch_size, hidden_size):
        """Handles different time tensor shapes."""
        diffusion = TimeVaryingDiffusion(hidden_size, sigma_init=0.1)
        state = torch.randn(batch_size, hidden_size)

        # Scalar time
        t_scalar = torch.tensor(0.5)
        output_scalar = diffusion(state, t_scalar)

        # 1D time
        t_1d = torch.full((batch_size,), 0.5)
        output_1d = diffusion(state, t_1d)

        # 2D time
        t_2d = torch.full((batch_size, 1), 0.5)
        output_2d = diffusion(state, t_2d)

        # All should produce same shape output
        assert output_scalar.shape == output_1d.shape == output_2d.shape


# =============================================================================
# Tests for create_diffusion factory
# =============================================================================

class TestCreateDiffusion:
    """Tests for the create_diffusion factory function."""

    def test_create_constant(self, hidden_size):
        """Creates ConstantDiffusion."""
        diffusion = create_diffusion("constant", hidden_size, sigma=0.2)
        assert isinstance(diffusion, ConstantDiffusion)

    def test_create_learned(self, hidden_size):
        """Creates LearnedDiffusion."""
        diffusion = create_diffusion("learned", hidden_size, sigma_init=0.2)
        assert isinstance(diffusion, LearnedDiffusion)

    def test_create_state_dependent(self, hidden_size):
        """Creates StateDependentDiffusion."""
        diffusion = create_diffusion("state_dependent", hidden_size)
        assert isinstance(diffusion, StateDependentDiffusion)

    def test_create_diagonal(self, hidden_size):
        """Creates diagonal diffusion (LearnedDiffusion)."""
        diffusion = create_diffusion("diagonal", hidden_size)
        assert isinstance(diffusion, LearnedDiffusion)

    def test_create_with_enum(self, hidden_size):
        """Works with DiffusionType enum."""
        diffusion = create_diffusion(DiffusionType.CONSTANT, hidden_size)
        assert isinstance(diffusion, ConstantDiffusion)

    def test_unknown_type_raises(self, hidden_size):
        """Unknown type raises error."""
        with pytest.raises(ValueError):
            create_diffusion("unknown_type", hidden_size)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestDiffusionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_batch(self, hidden_size):
        """Batch size 1 works."""
        diffusions = [
            ConstantDiffusion(hidden_size),
            LearnedDiffusion(hidden_size),
            StateDependentDiffusion(hidden_size),
            TimeVaryingDiffusion(hidden_size),
        ]
        state = torch.randn(1, hidden_size)

        for diffusion in diffusions:
            output = diffusion(state)
            assert output.shape == (1, hidden_size)

    def test_small_hidden_size(self, batch_size):
        """Small hidden size works."""
        hidden_size = 1
        diffusions = [
            ConstantDiffusion(hidden_size),
            LearnedDiffusion(hidden_size),
            StateDependentDiffusion(hidden_size),
            TimeVaryingDiffusion(hidden_size),
        ]
        state = torch.randn(batch_size, hidden_size)

        for diffusion in diffusions:
            output = diffusion(state)
            assert output.shape == (batch_size, hidden_size)

    def test_large_hidden_size(self, batch_size):
        """Large hidden size works."""
        hidden_size = 1024
        diffusions = [
            ConstantDiffusion(hidden_size),
            LearnedDiffusion(hidden_size),
            StateDependentDiffusion(hidden_size),
            TimeVaryingDiffusion(hidden_size),
        ]
        state = torch.randn(batch_size, hidden_size)

        for diffusion in diffusions:
            output = diffusion(state)
            assert output.shape == (batch_size, hidden_size)

    def test_extreme_state_values(self, batch_size, hidden_size):
        """Extreme state values don't cause NaN/Inf."""
        diffusion = StateDependentDiffusion(hidden_size)
        state = torch.randn(batch_size, hidden_size) * 1000

        output = diffusion(state)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_time(self, batch_size, hidden_size):
        """Zero time is handled correctly."""
        diffusion = TimeVaryingDiffusion(hidden_size, time_scaling="sqrt")
        state = torch.randn(batch_size, hidden_size)
        t = torch.tensor(0.0)

        output = diffusion(state, t)
        # With sqrt scaling and t=0, should be ~0
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)

    def test_very_small_time(self, batch_size, hidden_size):
        """Very small time doesn't cause numerical issues."""
        diffusion = TimeVaryingDiffusion(hidden_size, time_scaling="sqrt")
        state = torch.randn(batch_size, hidden_size)
        t = torch.tensor(1e-10)

        output = diffusion(state, t)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_large_time(self, batch_size, hidden_size):
        """Very large time is handled correctly."""
        diffusion = TimeVaryingDiffusion(hidden_size, time_scaling="sqrt")
        state = torch.randn(batch_size, hidden_size)
        t = torch.tensor(1e6)

        output = diffusion(state, t)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
