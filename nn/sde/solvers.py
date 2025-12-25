"""SDE solvers for stochastic particle filtering.

Provides numerical methods for solving SDEs:
    dX = f(X, t) dt + g(X, t) dW

Implemented solvers:
- Euler-Maruyama: First-order method, simple and widely used
- Milstein: Higher-order method for scalar noise
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from .diffusion import DiffusionCoefficient


class SDESolverType(Enum):
    """Types of SDE solvers."""
    EULER_MARUYAMA = "euler_maruyama"
    MILSTEIN = "milstein"


class SDESolver(ABC):
    """Abstract base class for SDE solvers."""

    @abstractmethod
    def step(
        self,
        state: Tensor,
        drift: Tensor,
        diffusion: Tensor,
        dt: Tensor,
        dW: Optional[Tensor] = None,
    ) -> Tensor:
        """Take one SDE integration step.

        Args:
            state: Current state [batch, hidden_size]
            drift: Drift term f(X, t) [batch, hidden_size]
            diffusion: Diffusion coefficient g(X, t) [batch, hidden_size]
            dt: Time step [batch, 1] or scalar
            dW: Optional pre-generated Wiener increment

        Returns:
            new_state: Updated state [batch, hidden_size]
        """
        pass


class EulerMaruyamaSolver(SDESolver):
    """Euler-Maruyama SDE solver.

    X_{n+1} = X_n + f(X_n) * dt + g(X_n) * dW

    where dW ~ N(0, dt).

    This is the simplest SDE solver, analogous to Euler method for ODEs.
    """

    def step(
        self,
        state: Tensor,
        drift: Tensor,
        diffusion: Tensor,
        dt: Tensor,
        dW: Optional[Tensor] = None,
    ) -> Tensor:
        """Take one Euler-Maruyama step."""
        # Ensure dt has correct shape for broadcasting
        if dt.dim() == 0:
            dt = dt.unsqueeze(0).unsqueeze(0)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        # Generate Wiener increment if not provided
        if dW is None:
            sqrt_dt = torch.sqrt(dt.clamp(min=1e-8))
            dW = sqrt_dt * torch.randn_like(state)

        # Euler-Maruyama update
        new_state = state + drift * dt + diffusion * dW

        return new_state


class MilsteinSolver(SDESolver):
    """Milstein SDE solver for improved accuracy.

    X_{n+1} = X_n + f(X_n) * dt + g(X_n) * dW + 0.5 * g(X_n) * g'(X_n) * (dW^2 - dt)

    Higher order accuracy than Euler-Maruyama when g has state dependence.
    Note: Requires gradient of diffusion, so only applicable when g is differentiable.
    """

    def __init__(self, compute_diffusion_gradient: bool = True):
        """Initialize Milstein solver.

        Args:
            compute_diffusion_gradient: Whether to compute g'(X) via autograd
        """
        self.compute_diffusion_gradient = compute_diffusion_gradient

    def step(
        self,
        state: Tensor,
        drift: Tensor,
        diffusion: Tensor,
        dt: Tensor,
        dW: Optional[Tensor] = None,
        diffusion_gradient: Optional[Tensor] = None,
    ) -> Tensor:
        """Take one Milstein step.

        Note: For diagonal diffusion, this reduces to:
        X_{n+1} = X_n + f * dt + g * dW + 0.5 * g * g' * (dW^2 - dt)
        """
        if dt.dim() == 0:
            dt = dt.unsqueeze(0).unsqueeze(0)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)

        if dW is None:
            sqrt_dt = torch.sqrt(dt.clamp(min=1e-8))
            dW = sqrt_dt * torch.randn_like(state)

        # Basic Euler-Maruyama part
        new_state = state + drift * dt + diffusion * dW

        # Milstein correction (for diagonal diffusion)
        if diffusion_gradient is not None:
            correction = 0.5 * diffusion * diffusion_gradient * (dW ** 2 - dt)
            new_state = new_state + correction

        return new_state


class SDEIntegrator(nn.Module):
    """High-level SDE integrator for LTC-style multi-step integration.

    Performs multiple SDE steps within a single time interval,
    analogous to ODE unfolding in standard LTC.
    """

    def __init__(
        self,
        hidden_size: int,
        diffusion: DiffusionCoefficient,
        solver: str = "euler_maruyama",
        n_steps: int = 6,
    ):
        """Initialize SDE integrator.

        Args:
            hidden_size: Dimension of state
            diffusion: Diffusion coefficient model
            solver: Solver type ('euler_maruyama', 'milstein')
            n_steps: Number of integration steps per time interval
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.diffusion = diffusion
        self.n_steps = n_steps

        # Create solver
        if solver == "euler_maruyama":
            self.solver = EulerMaruyamaSolver()
        elif solver == "milstein":
            self.solver = MilsteinSolver()
        else:
            raise ValueError(f"Unknown solver: {solver}")

    def integrate(
        self,
        state: Tensor,
        drift_fn: Callable[[Tensor], Tensor],
        elapsed_time: Tensor,
        inputs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Integrate SDE over elapsed time with multiple steps.

        Args:
            state: Initial state [batch, hidden_size]
            drift_fn: Function that computes drift given state
            elapsed_time: Total time to integrate [batch, 1] or scalar
            inputs: Optional inputs for context

        Returns:
            final_state: State after integration [batch, hidden_size]
            trajectory: All intermediate states [n_steps + 1, batch, hidden_size]
        """
        batch = state.shape[0]
        device = state.device
        dtype = state.dtype

        # Compute dt per step
        if elapsed_time.dim() == 0:
            dt = elapsed_time / self.n_steps
            dt = dt.expand(batch, 1)
        else:
            if elapsed_time.dim() == 1:
                elapsed_time = elapsed_time.unsqueeze(-1)
            dt = elapsed_time / self.n_steps

        # Store trajectory
        trajectory = [state]
        current_state = state

        # Integration loop
        for step in range(self.n_steps):
            # Compute drift at current state
            drift = drift_fn(current_state)

            # Compute diffusion coefficient
            g = self.diffusion(current_state)

            # Generate noise
            sqrt_dt = torch.sqrt(dt.clamp(min=1e-8))
            dW = sqrt_dt * torch.randn_like(current_state)

            # Take solver step
            current_state = self.solver.step(
                state=current_state,
                drift=drift,
                diffusion=g,
                dt=dt,
                dW=dW,
            )

            trajectory.append(current_state)

        # Stack trajectory
        trajectory = torch.stack(trajectory, dim=0)

        return current_state, trajectory

    def forward(
        self,
        state: Tensor,
        drift_fn: Callable[[Tensor], Tensor],
        elapsed_time: Tensor,
        inputs: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass - integrate and return final state."""
        final_state, _ = self.integrate(state, drift_fn, elapsed_time, inputs)
        return final_state


def create_solver(solver_type: str) -> SDESolver:
    """Create SDE solver by type.

    Args:
        solver_type: Type of solver

    Returns:
        SDESolver instance
    """
    if solver_type == "euler_maruyama":
        return EulerMaruyamaSolver()
    elif solver_type == "milstein":
        return MilsteinSolver()
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
