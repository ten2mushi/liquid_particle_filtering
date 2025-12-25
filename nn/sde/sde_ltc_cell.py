"""SDE LTC cell - Stochastic ODE formulation of LTC with particle filtering.

This implements Approach D: integrating noise directly into the LTC ODE solver
at each unfold step, rather than adding it post-hoc.
"""

from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from .base import SDEPFCell
from .diffusion import DiffusionType
from ..utils import AlphaMode
from ..observation import ObservationModel


class SDELTCCell(SDEPFCell):
    """SDE LTC (Liquid Time-Constant) cell with particle filtering.

    Implements the stochastic ODE formulation (Approach D) where noise
    is injected at each ODE unfold step:

    dv/dt = f(v, x, t) + g(v) * dW/dt

    This is more principled than post-hoc noise injection because:
    1. Noise scales correctly with time discretization
    2. Natural SDE interpretation of uncertainty
    3. Consistent with continuous-time dynamics

    The LTC dynamics are:
    f(v) = (1/cm) * [gleak * (vleak - v) + sum(w * sigmoid(v) * (erev - v))]

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> cell = SDELTCCell(wiring=wiring, in_features=20, n_particles=32)
        >>> x = torch.randn(8, 20)
        >>> output, (particles, log_weights) = cell(x)
    """

    def __init__(
        self,
        wiring,
        in_features: Optional[int] = None,
        n_particles: int = 32,
        # LTC specific parameters
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = False,
        # SDE configuration
        diffusion_type: Union[str, DiffusionType] = "learned",
        sigma_init: float = 0.1,
        solver: str = "euler_maruyama",
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
        # Stability configuration
        state_bounds: Optional[float] = 5.0,
        clamp_mode: str = "hard",
        diffusion_scale_by_dim: bool = False,
    ):
        """Initialize SDE LTC cell.

        Args:
            wiring: NCP wiring configuration
            in_features: Input dimension
            n_particles: Number of particles K
            input_mapping: Input mapping type
            output_mapping: Output mapping type
            ode_unfolds: Number of ODE/SDE unfolds per step
            epsilon: Numerical stability constant
            implicit_param_constraints: Use softplus for positive params
            diffusion_type: Type of diffusion coefficient
            sigma_init: Initial diffusion magnitude
            solver: SDE solver type
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
            state_bounds: Maximum absolute value for state clamping (None = no clamping)
            clamp_mode: Clamping mode ('hard' for torch.clamp, 'soft' for tanh-based)
            diffusion_scale_by_dim: If True, scale diffusion by 1/sqrt(hidden_size)
        """
        # Build wiring
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError("Wiring not built. Pass 'in_features' or call wiring.build().")

        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._epsilon = epsilon
        self._implicit_param_constraints = implicit_param_constraints

        # LTC parameter initialization ranges
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }

        super().__init__(
            hidden_size=wiring.units,
            n_particles=n_particles,
            diffusion_type=diffusion_type,
            sigma_init=sigma_init,
            n_unfolds=ode_unfolds,
            solver=solver,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
            observation_model=observation_model,
        )

        # Store stability configuration
        self._state_bounds = state_bounds
        self._clamp_mode = clamp_mode
        self._diffusion_scale_by_dim = diffusion_scale_by_dim

        self.input_size = wiring.input_dim

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    def _get_init_value(self, shape, param_name):
        """Get initialization value for LTC parameters."""
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _create_ltc_params(self):
        """Create LTC-specific parameters."""
        self.make_positive_fn = (
            nn.Softplus() if self._implicit_param_constraints else nn.Identity()
        )
        self._clip = nn.ReLU()

        # Per-neuron parameters
        self.gleak = nn.Parameter(self._get_init_value((self.state_size,), "gleak"))
        self.vleak = nn.Parameter(self._get_init_value((self.state_size,), "vleak"))
        self.cm = nn.Parameter(self._get_init_value((self.state_size,), "cm"))

        # Synapse parameters
        self.sigma = nn.Parameter(
            self._get_init_value((self.state_size, self.state_size), "sigma")
        )
        self.mu = nn.Parameter(
            self._get_init_value((self.state_size, self.state_size), "mu")
        )
        self.w = nn.Parameter(
            self._get_init_value((self.state_size, self.state_size), "w")
        )
        self.erev = nn.Parameter(
            torch.tensor(self._wiring.erev_initializer(), dtype=torch.float32)
        )

        # Sensory synapse parameters
        self.sensory_sigma = nn.Parameter(
            self._get_init_value((self.sensory_size, self.state_size), "sensory_sigma")
        )
        self.sensory_mu = nn.Parameter(
            self._get_init_value((self.sensory_size, self.state_size), "sensory_mu")
        )
        self.sensory_w = nn.Parameter(
            self._get_init_value((self.sensory_size, self.state_size), "sensory_w")
        )
        self.sensory_erev = nn.Parameter(
            torch.tensor(self._wiring.sensory_erev_initializer(), dtype=torch.float32)
        )

        # Sparsity masks
        self.register_buffer(
            "sparsity_mask",
            torch.tensor(np.abs(self._wiring.adjacency_matrix), dtype=torch.float32)
        )
        self.register_buffer(
            "sensory_sparsity_mask",
            torch.tensor(np.abs(self._wiring.sensory_adjacency_matrix), dtype=torch.float32)
        )

        # Input/output mappings
        if self._input_mapping in ["affine", "linear"]:
            self.input_w = nn.Parameter(torch.ones(self.sensory_size))
        if self._input_mapping == "affine":
            self.input_b = nn.Parameter(torch.zeros(self.sensory_size))

        if self._output_mapping in ["affine", "linear"]:
            self.output_w = nn.Parameter(torch.ones(self.motor_size))
        if self._output_mapping == "affine":
            self.output_b = nn.Parameter(torch.zeros(self.motor_size))

        # Store inputs for drift computation (will be set during forward)
        self._current_inputs = None
        self._current_elapsed = None

    def _ltc_sigmoid(self, v_pre, mu, sigma):
        """LTC sigmoid activation."""
        v_pre = v_pre.unsqueeze(-1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _map_inputs(self, inputs):
        """Apply input mapping."""
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self.input_w
        if self._input_mapping == "affine":
            inputs = inputs + self.input_b
        return inputs

    def _map_outputs(self, state: Tensor) -> Tensor:
        """Apply output mapping."""
        output = state
        if self.motor_size < self.state_size:
            output = output[:, :self.motor_size]

        if self._output_mapping in ["affine", "linear"]:
            output = output * self.output_w
        if self._output_mapping == "affine":
            output = output + self.output_b
        return output

    def _soft_clamp(self, x: Tensor, bounds: float) -> Tensor:
        """Soft clamping using tanh that preserves gradients.

        Maps (-inf, inf) to (-bounds, bounds) smoothly.
        At |x| << bounds, approximately identity.
        At |x| >> bounds, saturates to +/- bounds.

        Args:
            x: Input tensor
            bounds: Symmetric bound magnitude

        Returns:
            Clamped tensor with values in (-bounds, bounds)
        """
        return bounds * torch.tanh(x / bounds)

    def _compute_drift(
        self,
        state: Tensor,
        inputs: Tensor,
        elapsed_time: Tensor,
    ) -> Tensor:
        """Compute LTC drift term for SDE.

        The LTC drift is:
        f(v) = (1/tau) * [-v + w_num / w_den]

        where tau = cm / (gleak + sum(w))

        Args:
            state: Current voltage [batch, state_size]
            inputs: Mapped inputs [batch, sensory_size]
            elapsed_time: Time step [batch, 1]

        Returns:
            drift: Drift term [batch, state_size]
        """
        v_pre = state

        # Map inputs
        inputs = self._map_inputs(inputs)

        # Sensory contributions (pre-computed for efficiency)
        sensory_w_activation = self.make_positive_fn(self.sensory_w) * self._ltc_sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_w_activation = sensory_w_activation * self.sensory_sparsity_mask
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = sensory_rev_activation.sum(dim=1)
        w_denominator_sensory = sensory_w_activation.sum(dim=1)

        # Recurrent contributions
        w_param = self.make_positive_fn(self.w)
        w_activation = w_param * self._ltc_sigmoid(v_pre, self.mu, self.sigma)
        w_activation = w_activation * self.sparsity_mask
        rev_activation = w_activation * self.erev

        w_numerator = rev_activation.sum(dim=1) + w_numerator_sensory
        w_denominator = w_activation.sum(dim=1) + w_denominator_sensory

        # LTC drift
        gleak = self.make_positive_fn(self.gleak)
        cm = self.make_positive_fn(self.cm)

        numerator = gleak * self.vleak + w_numerator - (gleak + w_denominator) * v_pre
        denominator = cm

        drift = numerator / (denominator + self._epsilon)

        return drift

    def _propagate_particles(
        self,
        input: Tensor,
        particles: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate particles through SDE dynamics.

        Overrides base to handle LTC-specific integration where we
        need to integrate the full dynamics, not just drift.
        """
        batch, K, H = particles.shape
        device = particles.device
        dtype = particles.dtype

        # Default elapsed time
        if timespans is None:
            elapsed_time = torch.ones(batch, 1, device=device, dtype=dtype)
        elif timespans.dim() == 0:
            elapsed_time = timespans.expand(batch, 1)
        elif timespans.dim() == 1:
            elapsed_time = timespans.unsqueeze(-1)
        else:
            elapsed_time = timespans

        # Expand for particles
        input_expanded = input.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)
        elapsed_expanded = elapsed_time.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)

        # Flatten particles
        particles_flat = particles.reshape(batch * K, H)

        # Compute dt per unfold
        dt = elapsed_expanded / self.n_unfolds

        # SDE integration loop
        v_pre = particles_flat
        for step in range(self.n_unfolds):
            # Compute drift
            drift = self._compute_drift(v_pre, input_expanded, elapsed_expanded)

            # Compute diffusion
            g = self.diffusion(v_pre)

            # Scale diffusion by dimension if configured
            if self._diffusion_scale_by_dim:
                g = g / math.sqrt(self.state_size)

            # Generate noise
            sqrt_dt = torch.sqrt(dt.clamp(min=1e-8))
            dW = sqrt_dt * torch.randn_like(v_pre)

            # Euler-Maruyama step
            v_pre = v_pre + drift * dt + g * dW

            # Apply state bounds if configured
            if self._state_bounds is not None:
                if self._clamp_mode == "hard":
                    v_pre = torch.clamp(v_pre, min=-self._state_bounds, max=self._state_bounds)
                elif self._clamp_mode == "soft":
                    v_pre = self._soft_clamp(v_pre, self._state_bounds)

        # Reshape
        new_particles = v_pre.reshape(batch, K, H)

        # Compute outputs
        outputs_flat = self._map_outputs(v_pre)
        output_size = outputs_flat.shape[-1]
        outputs = outputs_flat.reshape(batch, K, output_size)

        return outputs, new_particles

    def apply_weight_constraints(self):
        """Apply weight constraints (for explicit mode)."""
        if not self._implicit_param_constraints:
            self.w.data = self._clip(self.w.data)
            self.sensory_w.data = self._clip(self.sensory_w.data)
            self.cm.data = self._clip(self.cm.data)
            self.gleak.data = self._clip(self.gleak.data)

    def extra_repr(self) -> str:
        stability_info = ""
        if self._state_bounds is not None:
            stability_info = f", state_bounds={self._state_bounds}, clamp_mode={self._clamp_mode}"
        if self._diffusion_scale_by_dim:
            stability_info += ", diffusion_scale_by_dim=True"
        return (
            f"state_size={self.state_size}, sensory_size={self.sensory_size}, "
            f"motor_size={self.motor_size}, n_particles={self.n_particles}, "
            f"n_unfolds={self.n_unfolds}{stability_info}"
        )
