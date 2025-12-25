"""Dual particle filter LTC cell."""

from typing import Tuple, Optional, Union, Dict, List

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from .base import DualPFCell
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel


class DualPFLTCCell(DualPFCell):
    """Dual particle filter LTC cell.

    Maintains K joint (state, parameter) particles over LTC dynamics.
    Enables uncertainty over both hidden voltages and synaptic parameters.

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> cell = DualPFLTCCell(
        ...     wiring=wiring,
        ...     in_features=20,
        ...     n_particles=16,
        ...     tracked_params=['w', 'sensory_w'],
        ... )
    """

    def __init__(
        self,
        wiring,
        in_features: Optional[int] = None,
        n_particles: int = 16,
        # LTC specific
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = True,
        # Parameter tracking
        tracked_params: Optional[List[str]] = None,
        param_evolution_noise: float = 0.01,
        # State noise
        state_noise_type: Union[str, NoiseType] = "time_scaled",
        state_noise_init: float = 0.1,
        # Resampling
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        use_rao_blackwell: bool = True,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize dual PF LTC cell."""
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError("Wiring not built.")

        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._implicit_param_constraints = implicit_param_constraints

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

        if tracked_params is None:
            tracked_params = ['w', 'sensory_w']

        super().__init__(
            input_size=wiring.input_dim,
            hidden_size=wiring.units,
            n_particles=n_particles,
            tracked_params=tracked_params,
            state_noise_type=state_noise_type,
            state_noise_init=state_noise_init,
            param_evolution_noise=param_evolution_noise,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
            use_rao_blackwell=use_rao_blackwell,
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

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        return torch.rand(*shape) * (maxval - minval) + minval

    def _create_base_cell(self):
        """Create LTC parameters."""
        self.make_positive_fn = (
            nn.Softplus() if self._implicit_param_constraints else nn.Identity()
        )

        # Parameters
        self.gleak = nn.Parameter(self._get_init_value((self.state_size,), "gleak"))
        self.vleak = nn.Parameter(self._get_init_value((self.state_size,), "vleak"))
        self.cm = nn.Parameter(self._get_init_value((self.state_size,), "cm"))

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

        self.register_buffer(
            "sparsity_mask",
            torch.tensor(np.abs(self._wiring.adjacency_matrix), dtype=torch.float32)
        )
        self.register_buffer(
            "sensory_sparsity_mask",
            torch.tensor(np.abs(self._wiring.sensory_adjacency_matrix), dtype=torch.float32)
        )

        if self._input_mapping in ["affine", "linear"]:
            self.input_w = nn.Parameter(torch.ones(self.sensory_size))
        if self._input_mapping == "affine":
            self.input_b = nn.Parameter(torch.zeros(self.sensory_size))

        if self._output_mapping in ["affine", "linear"]:
            self.output_w = nn.Parameter(torch.ones(self.motor_size))
        if self._output_mapping == "affine":
            self.output_b = nn.Parameter(torch.zeros(self.motor_size))

    def _get_trackable_params(self) -> Dict[str, Tensor]:
        return {
            'gleak': self.gleak,
            'vleak': self.vleak,
            'cm': self.cm,
            'w': self.w,
            'mu': self.mu,
            'sigma': self.sigma,
            'erev': self.erev,
            'sensory_w': self.sensory_w,
            'sensory_mu': self.sensory_mu,
            'sensory_sigma': self.sensory_sigma,
            'sensory_erev': self.sensory_erev,
        }

    def _ltc_sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.unsqueeze(-1)
        mues = v_pre - mu
        return torch.sigmoid(sigma * mues)

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self.input_w
        if self._input_mapping == "affine":
            inputs = inputs + self.input_b
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, :self.motor_size]
        if self._output_mapping in ["affine", "linear"]:
            output = output * self.output_w
        if self._output_mapping == "affine":
            output = output + self.output_b
        return output

    def _forward_with_params(
        self,
        input: Tensor,
        state: Tensor,
        params: Dict[str, Tensor],
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward with specific parameters."""
        if timespans is None:
            elapsed_time = 1.0
        elif timespans.dim() == 0:
            elapsed_time = timespans
        else:
            elapsed_time = timespans

        inputs = self._map_inputs(input)

        # Get params
        w = params.get('w', self.w)
        mu = params.get('mu', self.mu)
        sigma = params.get('sigma', self.sigma)
        erev = params.get('erev', self.erev)
        sensory_w = params.get('sensory_w', self.sensory_w)
        sensory_mu = params.get('sensory_mu', self.sensory_mu)
        sensory_sigma = params.get('sensory_sigma', self.sensory_sigma)
        sensory_erev = params.get('sensory_erev', self.sensory_erev)
        gleak = params.get('gleak', self.gleak)
        vleak = params.get('vleak', self.vleak)
        cm = params.get('cm', self.cm)

        v_pre = state

        # Sensory
        sensory_w_pos = self.make_positive_fn(sensory_w)
        sensory_activation = sensory_w_pos * self._ltc_sigmoid(
            inputs, sensory_mu, sensory_sigma
        )
        sensory_activation = sensory_activation * self.sensory_sparsity_mask
        sensory_rev = sensory_activation * sensory_erev

        w_num_sensory = sensory_rev.sum(dim=1)
        w_den_sensory = sensory_activation.sum(dim=1)

        cm_t = self.make_positive_fn(cm) / (elapsed_time / self._ode_unfolds)

        w_pos = self.make_positive_fn(w)
        for t in range(self._ode_unfolds):
            w_activation = w_pos * self._ltc_sigmoid(v_pre, mu, sigma)
            w_activation = w_activation * self.sparsity_mask
            rev_activation = w_activation * erev

            w_num = rev_activation.sum(dim=1) + w_num_sensory
            w_den = w_activation.sum(dim=1) + w_den_sensory

            gleak_pos = self.make_positive_fn(gleak)
            numerator = cm_t * v_pre + gleak_pos * vleak + w_num
            denominator = cm_t + gleak_pos + w_den

            v_pre = numerator / (denominator + self._epsilon)

        output = self._map_outputs(v_pre)
        return output, v_pre
