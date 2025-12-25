"""Particle filter Wired CfC cell - state-level particle filtering with NCP wiring."""

from typing import Tuple, Optional, Union, List
import math

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from .base import StateLevelPFCell
from .pf_cfc_cell import LeCun
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel


class PFWiredCfCCell(StateLevelPFCell):
    """Particle filter Wired CfC cell with NCP architecture.

    Combines wired CfC (multi-layer CfC with NCP sparsity) with
    state-level particle filtering. The wiring defines a sparse
    connectivity pattern inspired by neural circuits.

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> cell = PFWiredCfCCell(wiring=wiring, input_size=20, n_particles=32)
        >>> x = torch.randn(8, 20)
        >>> output, (particles, log_weights) = cell(x)
    """

    def __init__(
        self,
        wiring,
        input_size: Optional[int] = None,
        n_particles: int = 32,
        # CfC specific parameters
        mode: str = "default",
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
        """Initialize particle filter wired CfC cell.

        Args:
            wiring: NCP wiring configuration
            input_size: Input dimension (optional if wiring is built)
            n_particles: Number of particles K
            mode: CfC mode ('default', 'pure', 'no_gate')
            noise_type: Type of noise injection
            noise_init: Initial noise scale
            noise_learnable: Whether noise is learnable
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
        """
        # Build wiring if input_size provided
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring not built. Pass 'input_size' or call wiring.build()."
            )

        # Store wiring and mode
        self._wiring = wiring
        self._mode = mode

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
    def layer_sizes(self) -> List[int]:
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    def _create_base_cell(self):
        """Create wired CfC layers."""
        self._layers = nn.ModuleList()

        in_features = self._wiring.input_dim
        for l in range(self._wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)

            # Build sparsity mask
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]

            # Concatenate with recurrent connections
            input_sparsity = np.concatenate(
                [
                    input_sparsity,
                    np.ones((len(hidden_units), len(hidden_units))),
                ],
                axis=0,
            )

            # Create CfC layer
            layer = _WiredCfCLayer(
                in_features=in_features,
                hidden_size=len(hidden_units),
                mode=self._mode,
                sparsity_mask=input_sparsity,
            )
            self._layers.append(layer)
            in_features = len(hidden_units)

    def _propagate_single(
        self,
        input: Tensor,
        state: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate single state through wired CfC layers.

        Args:
            input: Input tensor [batch, input_size]
            state: Hidden state [batch, state_size]
            timespans: Time deltas [batch, 1] or scalar

        Returns:
            output: Last layer output [batch, motor_size]
            new_state: New hidden state [batch, state_size]
        """
        # Default timespans
        if timespans is None:
            ts = torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)
        elif timespans.dim() == 0:
            ts = timespans.expand(input.shape[0], 1)
        elif timespans.dim() == 1:
            ts = timespans.unsqueeze(-1)
        else:
            ts = timespans

        # Split state by layer
        h_state = torch.split(state, self.layer_sizes, dim=1)

        new_h_state = []
        inputs = input

        for i in range(self.num_layers):
            h, _ = self._layers[i](inputs, h_state[i], ts)
            inputs = h
            new_h_state.append(h)

        new_state = torch.cat(new_h_state, dim=1)
        output = h  # Last layer output

        return output, new_state

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize particles and log weights.

        For wired cells, we initialize states for the full state_size
        which spans all layers.
        """
        particles = torch.randn(
            batch_size, self.n_particles, self.state_size,
            device=device, dtype=dtype,
        ) * 0.1

        log_weights = torch.full(
            (batch_size, self.n_particles),
            -math.log(self.n_particles),
            device=device, dtype=dtype,
        )

        return particles, log_weights

    def extra_repr(self) -> str:
        return (
            f"state_size={self.state_size}, sensory_size={self.sensory_size}, "
            f"motor_size={self.motor_size}, n_particles={self.n_particles}, "
            f"num_layers={self.num_layers}, mode={self._mode}"
        )


class _WiredCfCLayer(nn.Module):
    """Single layer of wired CfC."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mode: str = "default",
        sparsity_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.mode = mode

        # Sparsity mask
        if sparsity_mask is not None:
            self.register_buffer(
                "sparsity_mask",
                torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32))
            )
        else:
            self.sparsity_mask = None

        # No backbone for wired variant
        cat_shape = in_features + hidden_size

        self.ff1 = nn.Linear(cat_shape, hidden_size)

        if mode == "pure":
            self.w_tau = nn.Parameter(torch.zeros(1, hidden_size))
            self.A = nn.Parameter(torch.ones(1, hidden_size))
        else:
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                nn.init.xavier_uniform_(w)

    def forward(
        self,
        input: Tensor,
        hx: Tensor,
        ts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward through single wired CfC layer."""
        x = torch.cat([input, hx], dim=1)

        # Apply sparsity mask to ff1
        if self.sparsity_mask is not None:
            ff1 = torch.nn.functional.linear(
                x, self.ff1.weight * self.sparsity_mask, self.ff1.bias
            )
        else:
            ff1 = self.ff1(x)

        if self.mode == "pure":
            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            if self.sparsity_mask is not None:
                ff2 = torch.nn.functional.linear(
                    x, self.ff2.weight * self.sparsity_mask, self.ff2.bias
                )
            else:
                ff2 = self.ff2(x)

            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)

            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)

            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, new_hidden
