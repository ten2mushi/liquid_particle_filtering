"""Particle Filter Neural Circuit Policies (PF-NCPs).

A library for combining particle filtering with Neural Circuit Policies,
providing probabilistic inference capabilities for liquid time-constant
neural networks.

Four approaches are supported:
- **Approach A (State-Level)**: Particles track uncertainty over hidden states
- **Approach B (Parameter-Level)**: Particles track uncertainty over parameters
- **Approach C (Dual)**: Joint particles over both states and parameters
- **Approach D (SDE)**: Stochastic differential equation formulation (LTC only)

Example:
    >>> import pfncps.nn as pf
    >>> from ncps.wirings import AutoNCP
    >>>
    >>> # Simple particle filter CfC
    >>> model = pf.PFCfC(input_size=20, hidden_size=64, n_particles=32)
    >>>
    >>> # Wired NCP with dual approach
    >>> wiring = AutoNCP(units=64, output_size=10)
    >>> model = pf.PFNCP(
    ...     wiring=wiring,
    ...     input_size=20,
    ...     n_particles=16,
    ...     approach='dual',
    ... )
    >>>
    >>> # Forward pass
    >>> x = torch.randn(8, 100, 20)
    >>> outputs, (particles, log_weights) = model(x)

See Also:
    - `pfncps.nn`: PyTorch implementations
    - ncps: Original Neural Circuit Policies library
"""

__version__ = "0.1.0"
__author__ = "PF-NCP Authors"

# Subpackages available
__all__ = ["nn"]
