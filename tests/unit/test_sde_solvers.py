"""Unit tests for SDE solvers (nn/sde/solvers.py).

Tests for:
- EulerMaruyamaSolver
- MilsteinSolver
- SDEIntegrator
- create_solver factory function
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.sde.solvers import (
    SDESolverType,
    SDESolver,
    EulerMaruyamaSolver,
    MilsteinSolver,
    SDEIntegrator,
    create_solver,
)
from pfncps.nn.sde.diffusion import ConstantDiffusion


# =============================================================================
# Tests for EulerMaruyamaSolver
# =============================================================================

class TestEulerMaruyamaSolver:
    """Tests for the EulerMaruyamaSolver class."""

    def test_output_shape(self, batch_size, hidden_size):
        """Output has same shape as input state."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert new_state.shape == state.shape

    def test_deterministic_with_zero_noise(self, batch_size, hidden_size):
        """With zero diffusion, step is deterministic Euler."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.zeros(batch_size, hidden_size)
        dt = torch.tensor(0.01)
        dW = torch.zeros(batch_size, hidden_size)

        new_state = solver.step(state, drift, diffusion, dt, dW=dW)

        expected = state + drift * dt
        assert torch.allclose(new_state, expected)

    def test_drift_contribution(self, batch_size, hidden_size):
        """Drift contributes to state update."""
        solver = EulerMaruyamaSolver()
        state = torch.zeros(batch_size, hidden_size)
        drift = torch.ones(batch_size, hidden_size)
        diffusion = torch.zeros(batch_size, hidden_size)
        dt = torch.tensor(0.1)
        dW = torch.zeros(batch_size, hidden_size)

        new_state = solver.step(state, drift, diffusion, dt, dW=dW)

        # New state should be drift * dt = 0.1
        assert torch.allclose(new_state, torch.full_like(new_state, 0.1))

    def test_diffusion_contribution(self, batch_size, hidden_size):
        """Diffusion contributes to state update."""
        solver = EulerMaruyamaSolver()
        state = torch.zeros(batch_size, hidden_size)
        drift = torch.zeros(batch_size, hidden_size)
        diffusion = torch.ones(batch_size, hidden_size)
        dt = torch.tensor(0.01)
        dW = torch.ones(batch_size, hidden_size) * 0.1

        new_state = solver.step(state, drift, diffusion, dt, dW=dW)

        # New state should be diffusion * dW = 0.1
        assert torch.allclose(new_state, torch.full_like(new_state, 0.1))

    def test_handles_scalar_dt(self, batch_size, hidden_size):
        """Handles scalar dt."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.01)  # Scalar

        new_state = solver.step(state, drift, diffusion, dt)
        assert new_state.shape == state.shape

    def test_handles_1d_dt(self, batch_size, hidden_size):
        """Handles 1D dt [batch]."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.full((batch_size,), 0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert new_state.shape == state.shape

    def test_handles_2d_dt(self, batch_size, hidden_size):
        """Handles 2D dt [batch, 1]."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.full((batch_size, 1), 0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert new_state.shape == state.shape

    def test_no_nan_inf(self, batch_size, hidden_size):
        """No NaN or Inf in output."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert not torch.isnan(new_state).any()
        assert not torch.isinf(new_state).any()


# =============================================================================
# Tests for MilsteinSolver
# =============================================================================

class TestMilsteinSolver:
    """Tests for the MilsteinSolver class."""

    def test_output_shape(self, batch_size, hidden_size):
        """Output has same shape as input state."""
        solver = MilsteinSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert new_state.shape == state.shape

    def test_reduces_to_euler_without_gradient(self, batch_size, hidden_size):
        """Without diffusion gradient, reduces to Euler-Maruyama."""
        euler_solver = EulerMaruyamaSolver()
        milstein_solver = MilsteinSolver()

        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.01)
        dW = torch.randn(batch_size, hidden_size) * math.sqrt(0.01)

        euler_result = euler_solver.step(state, drift, diffusion, dt, dW=dW)
        milstein_result = milstein_solver.step(state, drift, diffusion, dt, dW=dW)

        # Without diffusion_gradient, should be same as Euler
        assert torch.allclose(euler_result, milstein_result)

    def test_milstein_correction_applied(self, batch_size, hidden_size):
        """Milstein correction is applied when gradient provided."""
        solver = MilsteinSolver()
        state = torch.zeros(batch_size, hidden_size)
        drift = torch.zeros(batch_size, hidden_size)
        diffusion = torch.ones(batch_size, hidden_size)
        diffusion_gradient = torch.ones(batch_size, hidden_size)  # dg/dx
        dt = torch.tensor(0.1)
        dW = torch.ones(batch_size, hidden_size) * 0.5

        result_without = solver.step(state, drift, diffusion, dt, dW=dW)
        result_with = solver.step(
            state, drift, diffusion, dt, dW=dW, diffusion_gradient=diffusion_gradient
        )

        # Should be different due to correction
        assert not torch.allclose(result_without, result_with)

    def test_no_nan_inf(self, batch_size, hidden_size):
        """No NaN or Inf in output."""
        solver = MilsteinSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert not torch.isnan(new_state).any()
        assert not torch.isinf(new_state).any()


# =============================================================================
# Tests for SDEIntegrator
# =============================================================================

class TestSDEIntegrator:
    """Tests for the SDEIntegrator class."""

    def test_output_shape(self, batch_size, hidden_size):
        """Output state has correct shape."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=5)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        final_state, trajectory = integrator.integrate(state, drift_fn, elapsed_time)

        assert final_state.shape == (batch_size, hidden_size)

    def test_trajectory_shape(self, batch_size, hidden_size):
        """Trajectory has correct shape [n_steps + 1, batch, hidden_size]."""
        n_steps = 5
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=n_steps)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        final_state, trajectory = integrator.integrate(state, drift_fn, elapsed_time)

        assert trajectory.shape == (n_steps + 1, batch_size, hidden_size)

    def test_trajectory_starts_with_initial_state(self, batch_size, hidden_size):
        """First element of trajectory is initial state."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=5)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        _, trajectory = integrator.integrate(state, drift_fn, elapsed_time)

        assert torch.allclose(trajectory[0], state)

    def test_forward_returns_final_state(self, batch_size, hidden_size):
        """forward() returns only final state."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=5)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        result = integrator(state, drift_fn, elapsed_time)

        assert result.shape == (batch_size, hidden_size)

    def test_euler_maruyama_solver(self, batch_size, hidden_size):
        """Works with Euler-Maruyama solver."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(
            hidden_size, diffusion, solver="euler_maruyama", n_steps=5
        )

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        final_state = integrator(state, drift_fn, elapsed_time)
        assert final_state.shape == (batch_size, hidden_size)

    def test_milstein_solver(self, batch_size, hidden_size):
        """Works with Milstein solver."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(
            hidden_size, diffusion, solver="milstein", n_steps=5
        )

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        final_state = integrator(state, drift_fn, elapsed_time)
        assert final_state.shape == (batch_size, hidden_size)

    def test_handles_scalar_time(self, batch_size, hidden_size):
        """Handles scalar elapsed_time."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=5)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        final_state = integrator(state, drift_fn, elapsed_time)
        assert final_state.shape == (batch_size, hidden_size)

    def test_handles_1d_time(self, batch_size, hidden_size):
        """Handles 1D elapsed_time [batch]."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=5)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.full((batch_size,), 1.0)

        final_state = integrator(state, drift_fn, elapsed_time)
        assert final_state.shape == (batch_size, hidden_size)

    def test_handles_2d_time(self, batch_size, hidden_size):
        """Handles 2D elapsed_time [batch, 1]."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=5)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.full((batch_size, 1), 1.0)

        final_state = integrator(state, drift_fn, elapsed_time)
        assert final_state.shape == (batch_size, hidden_size)

    def test_unknown_solver_raises(self, hidden_size):
        """Unknown solver raises error."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)

        with pytest.raises(ValueError, match="Unknown solver"):
            SDEIntegrator(hidden_size, diffusion, solver="unknown")

    def test_no_nan_inf(self, batch_size, hidden_size):
        """No NaN or Inf in output."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=10)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        final_state, trajectory = integrator.integrate(state, drift_fn, elapsed_time)

        assert not torch.isnan(final_state).any()
        assert not torch.isinf(final_state).any()
        assert not torch.isnan(trajectory).any()
        assert not torch.isinf(trajectory).any()


# =============================================================================
# Tests for create_solver factory
# =============================================================================

class TestCreateSolver:
    """Tests for the create_solver factory function."""

    def test_create_euler_maruyama(self):
        """Creates EulerMaruyamaSolver."""
        solver = create_solver("euler_maruyama")
        assert isinstance(solver, EulerMaruyamaSolver)

    def test_create_milstein(self):
        """Creates MilsteinSolver."""
        solver = create_solver("milstein")
        assert isinstance(solver, MilsteinSolver)

    def test_unknown_type_raises(self):
        """Unknown type raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            create_solver("unknown_solver")


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestSDESolverEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_batch(self, hidden_size):
        """Batch size 1 works."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(1, hidden_size)
        drift = torch.randn(1, hidden_size)
        diffusion = torch.full((1, hidden_size), 0.1)
        dt = torch.tensor(0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert new_state.shape == (1, hidden_size)

    def test_small_hidden_size(self, batch_size):
        """Small hidden size works."""
        hidden_size = 1
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.01)

        new_state = solver.step(state, drift, diffusion, dt)
        assert new_state.shape == (batch_size, hidden_size)

    def test_very_small_dt(self, batch_size, hidden_size):
        """Very small dt doesn't cause issues."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(1e-8)

        new_state = solver.step(state, drift, diffusion, dt)
        assert not torch.isnan(new_state).any()
        assert not torch.isinf(new_state).any()

    def test_zero_dt(self, batch_size, hidden_size):
        """Zero dt returns unchanged state."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(0.0)
        dW = torch.zeros(batch_size, hidden_size)

        new_state = solver.step(state, drift, diffusion, dt, dW=dW)
        # With dt=0 and dW=0, should be unchanged
        assert torch.allclose(new_state, state)

    def test_large_dt(self, batch_size, hidden_size):
        """Large dt doesn't cause issues."""
        solver = EulerMaruyamaSolver()
        state = torch.randn(batch_size, hidden_size)
        drift = torch.randn(batch_size, hidden_size)
        diffusion = torch.full((batch_size, hidden_size), 0.1)
        dt = torch.tensor(100.0)

        new_state = solver.step(state, drift, diffusion, dt)
        assert not torch.isnan(new_state).any()
        assert not torch.isinf(new_state).any()

    def test_single_integration_step(self, batch_size, hidden_size):
        """Single integration step works."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=1)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(0.1)

        final_state, trajectory = integrator.integrate(state, drift_fn, elapsed_time)

        assert trajectory.shape == (2, batch_size, hidden_size)

    def test_many_integration_steps(self, batch_size, hidden_size):
        """Many integration steps work."""
        diffusion = ConstantDiffusion(hidden_size, sigma=0.1)
        integrator = SDEIntegrator(hidden_size, diffusion, n_steps=100)

        state = torch.randn(batch_size, hidden_size)
        drift_fn = lambda s: -0.1 * s
        elapsed_time = torch.tensor(1.0)

        final_state, trajectory = integrator.integrate(state, drift_fn, elapsed_time)

        assert trajectory.shape == (101, batch_size, hidden_size)
        assert not torch.isnan(final_state).any()
