"""Particle filter LTC cell - state-level particle filtering over LTC dynamics.

This is the state-level variant of LTC with particle filtering.
For stochastic ODE (SDE) formulation, see the sde/ module.
"""

from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from .base import StateLevelPFCell
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel


class PFLTCCell(StateLevelPFCell):
    """Particle filter LTC (Liquid Time-Constant) cell.

    Combines LTC dynamics with state-level particle filtering.
    Maintains K particles over the hidden state, each propagated
    through the LTC ODE solver with post-hoc noise injection.

    Note: This is the state-level approach (Approach A) which adds
    noise after each ODE solve. For the SDE formulation (Approach D)
    with per-unfold noise, see SDELTCCell.

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> cell = PFLTCCell(wiring=wiring, in_features=20, n_particles=32)
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
        # Noise configuration
        noise_type: Union[str, NoiseType] = "time_scaled",
        noise_init: float = 0.1,
        noise_learnable: bool = True,
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize particle filter LTC cell.

        Args:
            wiring: NCP wiring configuration
            in_features: Input dimension (optional if wiring is built)
            n_particles: Number of particles K
            input_mapping: Input mapping type ('affine', 'linear', 'none')
            output_mapping: Output mapping type
            ode_unfolds: Number of ODE solver unfolds
            epsilon: Numerical stability constant
            implicit_param_constraints: Use softplus for positive params
            noise_type: Type of noise injection
            noise_init: Initial noise scale
            noise_learnable: Whether noise is learnable
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
        """
        # Build wiring if in_features provided
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring not built. Pass 'in_features' or call wiring.build()."
            )

        # Store LTC-specific parameters
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._implicit_param_constraints = implicit_param_constraints

        # Define init ranges for LTC parameters
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
            input_size=wiring.input_dim,
            hidden_size=wiring.units,
            n_particles=n_particles,
            noise_type=noise_type,
            noise_init=noise_init,
            noise_learnable=noise_learnable,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
            observation_model=observation_model,
        )

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

    def _create_base_cell(self):
        """Create LTC parameters and mappings."""
        self.make_positive_fn = (
            nn.Softplus() if self._implicit_param_constraints else nn.Identity()
        )
        self._clip = nn.ReLU()

        # Allocate LTC parameters
        self._params = {}

        # Per-neuron parameters
        self._params["gleak"] = nn.Parameter(
            self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = nn.Parameter(
            self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = nn.Parameter(
            self._get_init_value((self.state_size,), "cm")
        )

        # Synapse parameters
        self._params["sigma"] = nn.Parameter(
            self._get_init_value((self.state_size, self.state_size), "sigma")
        )
        self._params["mu"] = nn.Parameter(
            self._get_init_value((self.state_size, self.state_size), "mu")
        )
        self._params["w"] = nn.Parameter(
            self._get_init_value((self.state_size, self.state_size), "w")
        )
        self._params["erev"] = nn.Parameter(
            torch.tensor(self._wiring.erev_initializer(), dtype=torch.float32)
        )

        # Sensory synapse parameters
        self._params["sensory_sigma"] = nn.Parameter(
            self._get_init_value((self.sensory_size, self.state_size), "sensory_sigma")
        )
        self._params["sensory_mu"] = nn.Parameter(
            self._get_init_value((self.sensory_size, self.state_size), "sensory_mu")
        )
        self._params["sensory_w"] = nn.Parameter(
            self._get_init_value((self.sensory_size, self.state_size), "sensory_w")
        )
        self._params["sensory_erev"] = nn.Parameter(
            torch.tensor(self._wiring.sensory_erev_initializer(), dtype=torch.float32)
        )

        # Sparsity masks (not learnable)
        self.register_buffer(
            "sparsity_mask",
            torch.tensor(np.abs(self._wiring.adjacency_matrix), dtype=torch.float32)
        )
        self.register_buffer(
            "sensory_sparsity_mask",
            torch.tensor(np.abs(self._wiring.sensory_adjacency_matrix), dtype=torch.float32)
        )

        # Register parameters
        for name, param in self._params.items():
            if isinstance(param, nn.Parameter):
                self.register_parameter(name, param)

        # Input/output mappings
        if self._input_mapping in ["affine", "linear"]:
            self.input_w = nn.Parameter(torch.ones(self.sensory_size))
        if self._input_mapping == "affine":
            self.input_b = nn.Parameter(torch.zeros(self.sensory_size))

        if self._output_mapping in ["affine", "linear"]:
            self.output_w = nn.Parameter(torch.ones(self.motor_size))
        if self._output_mapping == "affine":
            self.output_b = nn.Parameter(torch.zeros(self.motor_size))

    def _sigmoid(self, v_pre, mu, sigma):
        """LTC sigmoid activation."""
        v_pre = v_pre.unsqueeze(-1)  # For broadcasting
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

    def _map_outputs(self, state):
        """Apply output mapping."""
        output = state
        if self.motor_size < self.state_size:
            output = output[:, :self.motor_size]

        if self._output_mapping in ["affine", "linear"]:
            output = output * self.output_w
        if self._output_mapping == "affine":
            output = output + self.output_b
        return output

    def _ode_solver(self, inputs, state, elapsed_time):
        """LTC ODE solver with multiple unfolds."""
        v_pre = state

        # Sensory neuron effects (pre-computed)
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = sensory_w_activation * self.sensory_sparsity_mask
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        w_numerator_sensory = sensory_rev_activation.sum(dim=1)
        w_denominator_sensory = sensory_w_activation.sum(dim=1)

        # cm/t is loop invariant
        cm_t = self.make_positive_fn(self._params["cm"]) / (
            elapsed_time / self._ode_unfolds
        )

        # ODE unfolds
        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )
            w_activation = w_activation * self.sparsity_mask
            rev_activation = w_activation * self._params["erev"]

            w_numerator = rev_activation.sum(dim=1) + w_numerator_sensory
            w_denominator = w_activation.sum(dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self._params["gleak"])
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator

            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _propagate_single(
        self,
        input: Tensor,
        state: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate single state through LTC dynamics.

        Args:
            input: Input tensor [batch, input_size]
            state: Hidden state [batch, state_size]
            timespans: Time deltas [batch, 1] or scalar

        Returns:
            output: Motor output [batch, motor_size]
            new_state: New hidden state [batch, state_size]
        """
        # Default elapsed time
        if timespans is None:
            elapsed_time = 1.0
        elif timespans.dim() == 0:
            elapsed_time = timespans
        elif timespans.dim() == 1:
            elapsed_time = timespans.unsqueeze(-1)
        else:
            elapsed_time = timespans

        # Map inputs
        inputs = self._map_inputs(input)

        # Solve ODE
        new_state = self._ode_solver(inputs, state, elapsed_time)

        # Map outputs
        output = self._map_outputs(new_state)

        return output, new_state

    def apply_weight_constraints(self):
        """Apply weight constraints (for explicit mode)."""
        if not self._implicit_param_constraints:
            self._params["w"].data = self._clip(self._params["w"].data)
            self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
            self._params["cm"].data = self._clip(self._params["cm"].data)
            self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def extra_repr(self) -> str:
        return (
            f"state_size={self.state_size}, sensory_size={self.sensory_size}, "
            f"motor_size={self.motor_size}, n_particles={self.n_particles}, "
            f"ode_unfolds={self._ode_unfolds}"
        )
