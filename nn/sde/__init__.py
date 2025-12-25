"""Stochastic ODE particle filter cells (Approach D).

Implements SDE-based particle filtering where noise is integrated
into the ODE dynamics at each unfold step, rather than added post-hoc.

This is specifically designed for LTC-style cells that naturally
discretize continuous-time dynamics.

Key difference from state-level (Approach A):
- Approach A: dh = f(h) dt, then h += noise
- Approach D: dh = f(h) dt + g(h) dW (noise is integral)

Provides:
- SDELTCCell: SDE particle filter over LTC dynamics
- SDEWiredLTCCell: SDE particle filter with NCP wiring
- Diffusion coefficient models (constant, learned, state-dependent)
- SDE solvers (Euler-Maruyama, Milstein)
"""

from .base import SDEPFCell
from .sde_ltc_cell import SDELTCCell
from .sde_wired_ltc_cell import SDEWiredLTCCell
from .diffusion import (
    DiffusionCoefficient,
    DiffusionType,
    ConstantDiffusion,
    LearnedDiffusion,
    StateDependentDiffusion,
    TimeVaryingDiffusion,
    create_diffusion,
)
from .solvers import (
    SDESolver,
    SDESolverType,
    EulerMaruyamaSolver,
    MilsteinSolver,
    SDEIntegrator,
    create_solver,
)

__all__ = [
    # Base
    "SDEPFCell",
    # Cells
    "SDELTCCell",
    "SDEWiredLTCCell",
    # Diffusion
    "DiffusionCoefficient",
    "DiffusionType",
    "ConstantDiffusion",
    "LearnedDiffusion",
    "StateDependentDiffusion",
    "TimeVaryingDiffusion",
    "create_diffusion",
    # Solvers
    "SDESolver",
    "SDESolverType",
    "EulerMaruyamaSolver",
    "MilsteinSolver",
    "SDEIntegrator",
    "create_solver",
]
